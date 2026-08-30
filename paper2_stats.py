#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_stats.py — the statistical analysis behind every comparative claim in the paper.

WHY THIS FILE EXISTS
--------------------
The R2 draft compared configurations by their mean over seeds and left it there. A handful of seeds of
a deep RL controller are a small, heavy-tailed sample; a difference of means with no interval and no
test is not evidence, and a reviewer of a Q1 journal will say so. This module produces, for every
comparative statement the paper makes, an effect size and an interval, on two independent levels of
pairing, with a multiplicity correction applied within each metric family.

TWO LEVELS OF PAIRING, AND WHY BOTH ARE REPORTED
------------------------------------------------
1. SEED LEVEL (primary). Each configuration is trained with the same seeds. Seed s of
   configuration A and seed s of configuration B share the initialisation stream and the scene
   ordering, so they are naturally paired: one pair per seed. Test: Wilcoxon signed-rank (exact at
   these sample sizes), effect size: Cliff's delta, interval: percentile bootstrap on the paired mean
   difference. This level answers "does the change help across training runs?" and correctly treats
   the seed as the unit of replication, which is the mistake most RL papers make.

2. EPISODE LEVEL (secondary, and only valid because of common random numbers). Every configuration is
   evaluated on the identical 500 held-out scenarios, scenario i produced by reset seed
   EVAL_SEED_BASE + i for all of them. Averaging over seeds within a configuration gives one value
   per scenario, and scenario i is then directly comparable across configurations. n = 500 pairs.
   This level answers "on the same road situations, does the change help?" and is the only level at
   which the non-learned baselines — which have no training seed — can be compared at all.

The two levels can disagree. Where they do, the paper says so rather than quoting the friendlier one.

MULTIPLICITY
------------
Every configuration is compared against the proposed controller, so each metric carries a family of
comparisons. p-values are adjusted within each metric family by the Holm-Bonferroni step-down
procedure, which controls the family-wise error rate without assuming independence.

INTEGRITY: every number this module prints is computed from the per-seed and per-episode result files
written by the sweep. Nothing is entered by hand, no seed is excluded, and both the adjusted and the
raw p-values are reported so the correction cannot hide anything.
"""
import os, sys, glob, re, argparse, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# The reward of Eq. (15) charges for transmit power and for bandwidth. Anything in this file that
# reasons about what a return difference is *made of* needs those weights, and re-typing them here
# would let the accounting drift silently away from the environment that produced the numbers, so
# they are imported from it. A failure to import is left to surface as an ImportError rather than
# caught and replaced by defaults: an accounting that quietly falls back to a guess at the reward
# is worse than one that stops.
from paper2_env import W_PWR, W_SPEC, PT_MIN, PT_MAX, RB_MIN, RB_MAX

BOOT = 10000
BOOT_SEED = 20260726
REF = 'model'
METRICS = [('rlf_rate', 'RLF rate (%)', 'lower'),
           ('cvar25', 'CVaR$_{25}$ return', 'higher'),
           ('return_mean', 'mean return', 'higher')]
# `n_ho` is a COST, not a goal. It is tested here because the paper makes a quantitative claim about
# signalling load --- how many handovers a policy spends to reach the reliability it reaches --- and
# that claim deserves the same scene-clustered paired test as the reliability claim rather than a
# bare comparison of two means. The `lower` label sets the sign convention for the verdict engine and
# nothing else; on its own a low handover count is trivially achieved by never handing over, so any
# prose that quotes this metric must state the reliability verdict in the same breath. Holm is
# applied *within* each metric, so adding this family leaves the `rlf` and `ret` corrections exactly
# as they were --- this is not a route to significance that the previous families did not grant.
EP_METRICS = [('rlf', 'per-scenario RLF', 'lower'), ('ret', 'per-scenario return', 'higher'),
              ('n_ho', 'per-scenario handovers', 'lower')]


# --------------------------------------------------------------------------- primitives
def cliffs_delta(x, y):
    """(P(x>y) - P(x<y)); +1 means x dominates. Reported unpaired, the conventional form."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def cliff_label(d):
    a = abs(d)
    return 'negligible' if a < 0.147 else 'small' if a < 0.33 else 'medium' if a < 0.474 else 'large'


def hodges_lehmann(d):
    """Pseudomedian of the paired differences: the median of all Walsh averages (d_i + d_j)/2.

    This is the location parameter the Wilcoxon signed-rank test actually tests, so reporting it next
    to the mean difference keeps the estimate and the test statistic talking about the same quantity.
    They can differ sharply here and that difference is informative rather than an error: a controller
    that loses a little on most episodes and wins enormously on the few that would otherwise fail has
    a negative pseudomedian and a positive mean, which is the signature of a tail-optimizing policy."""
    d = np.asarray(d, float)
    if len(d) == 0:
        return float('nan')
    if len(d) == 1:
        return float(d[0])
    w = (d[:, None] + d[None, :])[np.triu_indices(len(d))] / 2.0
    return float(np.median(w))


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation: (W+ - W-)/(W+ + W-) over the non-tied pairs.

    This is the effect size that belongs with a Wilcoxon signed-rank test. Cliff's delta is an
    *unpaired* dominance measure and its conventional thresholds are calibrated for continuous,
    independent samples; applying them to a paired, heavily tied, near-binary outcome such as the
    per-scenario RLF flag produces the absurdity of a 10-percentage-point reliability difference
    labelled 'negligible'. Both are kept in the CSV; this one is what the report reads."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if len(d) == 0:
        return float('nan'), 0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm)), int(len(d))


def rb_label(r):
    """Cohen-style bands for a rank correlation, stated as bands and not as a verdict."""
    if not np.isfinite(r):
        return 'undefined (all pairs tied)'
    a = abs(r)
    return 'negligible' if a < 0.1 else 'small' if a < 0.3 else 'medium' if a < 0.5 else 'large'


def fmt_p(p):
    """Never print a p-value as exactly 0. Below the smallest value the test can resolve, say so."""
    if p is None or not np.isfinite(p):
        return 'n/a'
    if p < 1e-16:
        return '<1e-16'
    if p < 1e-3:
        return f'{p:.1e}'
    return f'{p:.3f}'


def boot_ci_paired(d, n=BOOT, seed=BOOT_SEED, level=95.0):
    """Percentile bootstrap CI for the mean of the paired differences, plus a two-sided bootstrap
    p-value (the achieved significance level: twice the smaller tail mass on the wrong side of 0)."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else float('nan'), float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [(100 - level) / 2, 100 - (100 - level) / 2])
    p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(max(p, 1.0 / n), 1.0))


def boot_ci_mean(x, n=BOOT, seed=BOOT_SEED, level=95.0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float('nan')), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


def boot_ci_cluster(d, clusters, n=BOOT, seed=BOOT_SEED, level=95.0):
    """Cluster (block) percentile bootstrap for the mean of the paired differences.

    The 500 held-out scenarios are not 500 independent draws. They are repeated resets over the 15
    held-out DAIR-V2X scenes, so every scenario that shares a scene also shares its road geometry,
    its blocker population and its gNB layout, and their outcomes are correlated. Resampling
    scenarios independently — the ordinary bootstrap — treats that shared structure as if each
    scenario were fresh evidence, and returns an interval that is too narrow by the design effect.
    This resamples whole SCENES with replacement and pools whatever scenarios the drawn scenes carry,
    so the resampling unit is the unit that is actually independent.

    Returns the mean, the interval, a two-sided achieved significance level, and the design effect —
    the clustered variance of the mean divided by the i.i.d. variance of the same mean. The design
    effect is returned rather than silently absorbed so that the inflation is reported in the paper
    instead of merely corrected for."""
    d = np.asarray(d, float)
    cl = np.asarray(clusters)
    keys = np.unique(cl)
    if len(d) < 2 or len(keys) < 2:
        return (float(d.mean()) if len(d) else float('nan'),
                float('nan'), float('nan'), float('nan'), float('nan'))
    sums = np.array([d[cl == k].sum() for k in keys], float)
    cnts = np.array([float((cl == k).sum()) for k in keys], float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n, len(keys)))
    bs = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)   # pooled mean over the scenes drawn
    lo, hi = np.percentile(bs, [(100 - level) / 2, 100 - (100 - level) / 2])
    p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    iid = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    vi = float(iid.var())
    deff = float(bs.var() / vi) if vi > 0 else float('nan')
    return float(d.mean()), float(lo), float(hi), float(min(max(p, 1.0 / n), 1.0)), deff


def cluster_level(a, b, clusters):
    """Collapse to one paired difference per cluster and test those.

    This is the conservative reading of the same data, and it is the one the paper's verdicts follow.
    The scene is the sampling unit, so the hypothesis test runs on the 15 scene-mean differences with
    an exact signed-rank test rather than on 500 correlated scenario differences. Ties are counted
    and reported: a scene on which two configurations behave identically carries no information about
    which is better, and with only 15 scenes there are few enough of them that hiding that fact
    inside a large n would be misleading."""
    a, b, cl = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    keys = np.unique(cl)
    am = np.array([a[cl == k].mean() for k in keys], float)
    bm = np.array([b[cl == k].mean() for k in keys], float)
    rb, ne = rank_biserial(am, bm)
    return (wilcoxon_p(am, bm), rb, ne, int(len(keys)), hodges_lehmann(am - bm),
            float((am - bm).mean()), signed_rank_floor(ne))


def signed_rank_floor(n_disc):
    """Smallest two-sided p the exact signed-rank test can return with `n_disc` untied pairs.

    Reported because with 15 scenes it is a binding constraint rather than a technicality. Failures
    concentrate in a minority of held-out scenes, so many scenes are exactly tied between two
    configurations and carry no information about which is better; the test then runs on only the
    untied ones. With eight untied scenes the most extreme possible outcome — every one of them
    favouring the same side — still yields p = 2/2^8 = 0.0078, which does not survive Holm correction
    over a fourteen-comparison family. Publishing that floor next to the p-value is what separates
    'the effect is absent' from 'this design cannot resolve an effect of any size', and only the
    second is true for several of the comparisons here."""
    if not n_disc or n_disc < 1:
        return float('nan')
    return float(min(1.0, 2.0 / (2.0 ** n_disc)))


def wilcoxon_p(a, b):
    """Paired signed-rank p. Returns 1.0 when every pair is tied (the test is undefined there)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method='wilcox',
                              method='exact' if len(d) <= 25 else 'auto').pvalue)
    except Exception:
        try:
            return float(wilcoxon(a, b).pvalue)
        except Exception:
            return float('nan')


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved, monotone, capped at 1."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    m, run = len(order), 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        out[i] = min(run, 1.0)
    return out


def stars(p):
    return '' if not np.isfinite(p) else ('***' if p < 0.001 else '**' if p < 0.01
                                          else '*' if p < 0.05 else 'n.s.')


def disc_mean(d):
    """Mean paired difference restricted to the pairs that are not tied.

    Reported beside the overall mean because the two together say where an effect comes from: an
    aggregate 10-percentage-point reliability gap that is produced by 55 scenarios out of 500 is a
    targeted intervention, and an identical aggregate produced by 500 small differences is a uniform
    shift. Those are different mechanisms and the paper distinguishes them."""
    d = np.asarray(d, float)
    d = d[d != 0]
    return float(d.mean()) if len(d) else float('nan')


def _finalize(df):
    """Post-processing shared by both pairing levels: multiplicity correction within each metric
    family, effect-size labels, printable p-values, and an explicit divergence flag.

    The flag exists because the two statistics in each row answer different questions. The bootstrap
    interval is on the MEAN paired difference; the signed-rank test and the rank-biserial correlation
    describe the PSEUDOMEDIAN, i.e. the typical pair. An interval that spans zero beside a rejecting
    rank test is therefore not an inconsistency to be explained away — it is the measurable signature
    of a policy that gives up a little on most scenarios to avoid catastrophe on a few. Reporting the
    two side by side without naming the divergence would read as an error, so it is named here and
    carried into the report rather than left for a reviewer to find."""
    if not len(df):
        return df
    df = df.copy()
    df['p_holm'] = np.nan
    for c in df.metric.unique():                          # correct within each metric family
        k = df.metric == c
        df.loc[k, 'p_holm'] = holm(df.loc[k, 'p_raw'].to_numpy())
    if 'p_floor' in df.columns:
        # The best adjusted p this comparison could ever attain, holding the rest of its family at
        # the values actually observed: substitute the comparison's own floor for its raw p and
        # re-run Holm. It has to be computed this way rather than as floor x family-size, because
        # Holm's multiplier depends on where the comparison ranks -- the smallest p in a family is
        # the one multiplied by m, and a comparison that ranks fifth is multiplied by m-4. Where the
        # result already exceeds 0.05 the comparison cannot be established by this design at any
        # effect size, and the paper says so instead of reporting it as a null result.
        df['p_floor_holm'] = np.nan
        for c in df.metric.unique():
            k = np.where((df.metric == c).to_numpy())[0]
            praw = df.p_raw.to_numpy(float)[k]
            pfl = df.p_floor.to_numpy(float)[k]
            for j in range(len(k)):
                if not np.isfinite(pfl[j]):
                    continue
                sub = praw.copy()
                sub[j] = pfl[j]
                df.iloc[k[j], df.columns.get_loc('p_floor_holm')] = holm(sub)[j]
        df['p_floor_holm'] = df.p_floor_holm.round(8)     # same scale as p_holm, so the two compare
        df['resolvable'] = df.p_floor_holm < 0.05
    df['cliff_label'] = df.cliff.map(cliff_label)         # kept in the CSV for completeness only
    df['rb_label'] = df.rb_corr.map(rb_label)
    df['sig'] = df.p_holm.map(stars)
    spans0 = (df.ci_lo <= 0) & (df.ci_hi >= 0)
    df['divergence'] = np.where(
        spans0 & (df.p_holm < 0.05), 'mean CI spans 0 while the rank test rejects',
        np.where((~spans0) & (df.p_holm >= 0.05), 'mean CI excludes 0 while the rank test does not '
                 'reject', ''))
    df['p_raw_s'] = df.p_raw.map(fmt_p)
    df['p_holm_s'] = df.p_holm.map(fmt_p)
    df['p_raw'] = df.p_raw.round(8)
    df['p_holm'] = df.p_holm.round(8)
    if 'p_raw_uncl' in df.columns:
        # The naive family is Holm-corrected too, so that the comparison the paper makes between the
        # two readings is like for like: the difference between them is then attributable to the
        # clustering alone and not to one of the two escaping multiplicity correction.
        df['p_holm_uncl'] = np.nan
        for c in df.metric.unique():
            k = df.metric == c
            df.loc[k, 'p_holm_uncl'] = holm(df.loc[k, 'p_raw_uncl'].to_numpy())
        df['p_raw_uncl_s'] = df.p_raw_uncl.map(fmt_p)
        df['p_holm_uncl_s'] = df.p_holm_uncl.map(fmt_p)
        df['p_raw_uncl'] = df.p_raw_uncl.round(8)
        df['p_holm_uncl'] = df.p_holm_uncl.round(8)
    return df


# ------------------------------------------------------------------------------- loading
def load(outdir):
    mf = os.path.join(outdir, 'all_metrics.csv')
    assert os.path.exists(mf), f'missing {mf} — run paper2_sweep_r3.py --aggregate_only first'
    per_seed = pd.read_csv(mf)

    eps = {}
    for f in sorted(glob.glob(os.path.join(outdir, '*_s*_eval.csv'))):
        m = re.match(r'(.+)_s(\d+)_eval\.csv$', os.path.basename(f))
        if not m:
            continue
        d = pd.read_csv(f)
        d['config'], d['seed'] = m.group(1), int(m.group(2))
        eps.setdefault(m.group(1), []).append(d)
    per_ep = {k: pd.concat(v, ignore_index=True) for k, v in eps.items()}

    # ---- the two sources must describe the same seeds, or the tables silently disagree.
    # A sweep job writes its per-episode eval CSV before its summary metrics row, so a run that is
    # interrupted -- or simply read while it is still going -- leaves configurations whose eval files
    # cover MORE seeds than all_metrics.csv does. Nothing downstream would notice: the per-config
    # table would average one seed set and the scenario-level table another, and the two would then
    # fail to reconcile by a few percentage points in a way that looks exactly like an injection bug
    # in the manuscript. That is not hypothetical -- it is how the -CVaR and -anticipation rows came
    # to disagree with their own summary table. all_metrics.csv is authoritative because it is the
    # completed-job record; eval files for seeds it does not list belong to jobs that had not
    # finished, and they are dropped here with the drop announced rather than hidden.
    if 'seed' in per_seed.columns:
        for cfg in sorted(per_ep):
            want = set(per_seed.loc[per_seed.config == cfg, 'seed'].astype(int))
            if not want:
                continue
            have = set(per_ep[cfg].seed.astype(int))
            missing = sorted(want - have)
            assert not missing, (f'{cfg}: all_metrics.csv lists seed(s) {missing} with no '
                                 f'*_s*_eval.csv — cannot pair those seeds scenario by scenario')
            extra = sorted(have - want)
            if extra:
                print(f'  [load] {cfg}: dropping eval seed(s) {extra} — no completed metrics row '
                      f'(keeping {sorted(want)}), so every table is built from one seed set')
                per_ep[cfg] = per_ep[cfg][per_ep[cfg].seed.astype(int).isin(want)].reset_index(
                    drop=True)

    base = {}
    # Reference-policy summary rows come from three writers. paper2_baselines.py produces the
    # standard set; paper2_lookahead_baseline.py produces the tuned anticipatory threshold rule;
    # paper2_antonly_baseline.py produces that same rule with the embedded A3 disjunct removed.
    # The third file is not an extra: antthr fires on the A3 event — or — the anticipatory one, so
    # it is a union of a reactive rule and an anticipatory one, and a comparison against it cannot
    # isolate anticipation from A3. The anticipation-only rule is the control that can. Each writer
    # keeps its own file rather than appending, because paper2_sweep_r3.py treats the existence of
    # baselines_metrics.csv as the signal that the reference policies have already been run. Merging
    # here instead of on disk keeps that signal meaningful and makes the merge idempotent: rerunning
    # any one writer cannot duplicate a row or silently drop one.
    frames = []
    for nm in ('baselines_metrics.csv', 'lookahead_metrics.csv', 'antonly_metrics.csv'):
        p = os.path.join(outdir, nm)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    base_m = pd.concat(frames, ignore_index=True) if frames else None
    if base_m is not None:
        dup = base_m.config.duplicated()
        assert not dup.any(), (f'reference policy listed twice across the reference-policy metrics '
                               f'files: {sorted(base_m.config[dup])}')
    for f in sorted(glob.glob(os.path.join(outdir, 'baseline_*_eval.csv'))):
        nm = re.match(r'baseline_(.+)_eval\.csv$', os.path.basename(f)).group(1)
        base[nm] = pd.read_csv(f)
    # Every discovered per-episode file must have a summary row, or panel (a) of the reference table
    # would silently omit a policy that panel (b) tests against.
    if base_m is not None:
        orphan = sorted(set(base) - set(base_m.config))
        assert not orphan, (f'baseline eval CSV(s) with no summary row: {orphan} — rerun the writer '
                            f'that produces them before building any table')
    return per_seed, per_ep, base_m, base


def ep_vector(df, metric):
    """One value per held-out scenario, averaged over training seeds. The index is the CRN scenario
    index, so the vectors of two configurations are aligned scenario by scenario."""
    return df.groupby('idx')[metric].mean().sort_index()


def ep_scene(df):
    """Scenario index -> held-out scene id.

    Common random numbers fix the scenario-to-scene map, so this is the same map in every
    configuration and every seed. That is asserted here rather than assumed, because the whole
    clustered analysis rests on it: if two configurations disagreed about which scene scenario 17
    belongs to, pairing them by scenario would already be meaningless."""
    assert 'scene' in df.columns, 'eval CSV has no scene column — clustered inference impossible'
    n = df.groupby('idx')['scene'].nunique()
    assert int(n.max()) == 1, 'a scenario index maps to more than one scene'
    return df.groupby('idx')['scene'].first().sort_index()


# ------------------------------------------------------------------------------ analyses
def per_config_table(per_seed):
    rows = []
    for cfg, g in per_seed.groupby('config'):
        r = dict(config=cfg, n_seeds=len(g))
        for c, _, _ in METRICS:
            if c not in g.columns:
                continue
            v = g[c].to_numpy(float)
            m, lo, hi = boot_ci_mean(v)
            r[f'{c}_mean'] = round(m, 4)
            r[f'{c}_std'] = round(float(v.std(ddof=1)) if len(v) > 1 else float('nan'), 4)
            r[f'{c}_median'] = round(float(np.median(v)), 4)
            r[f'{c}_iqr_lo'] = round(float(np.percentile(v, 25)), 4)
            r[f'{c}_iqr_hi'] = round(float(np.percentile(v, 75)), 4)
            r[f'{c}_ci_lo'] = round(lo, 4)
            r[f'{c}_ci_hi'] = round(hi, 4)
        for c in ('pt_dbm', 'rb', 'ho_per_ep', 'gap_return', 'gap_rlf',
                  'insample_rlf_rate', 'insample_cvar25', 'insample_return_mean',
                  'lat_mean_ms', 'lat_p95_ms', 'lat_p99_ms'):
            if c in g.columns:
                r[c] = round(float(g[c].mean()), 4)
        rows.append(r)
    return pd.DataFrame(rows).sort_values('config').reset_index(drop=True)


def seed_level(per_seed, ref=REF):
    """Paired-by-seed comparison of every configuration against the reference."""
    if ref not in set(per_seed['config']):
        return pd.DataFrame()
    R = per_seed[per_seed.config == ref].set_index('seed').sort_index()
    rows = []
    for cfg, g in per_seed.groupby('config'):
        if cfg == ref:
            continue
        G = g.set_index('seed').sort_index()
        common = sorted(set(R.index) & set(G.index))
        if len(common) < 3:
            continue
        for c, lbl, better in METRICS:
            if c not in R.columns or c not in G.columns:
                continue
            a = R.loc[common, c].to_numpy(float)      # reference
            b = G.loc[common, c].to_numpy(float)      # comparator
            diff = a - b
            m, lo, hi, bp = boot_ci_paired(diff)
            rb, ne = rank_biserial(a, b)
            rows.append(dict(level='seed', metric=c, metric_label=lbl, better=better,
                             comparison=f'{ref} vs {cfg}', n=len(common),
                             ref_mean=round(float(a.mean()), 4), cmp_mean=round(float(b.mean()), 4),
                             diff_mean=round(m, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             diff_hl=round(hodges_lehmann(diff), 4),
                             diff_disc=round(disc_mean(diff), 4),
                             p_raw=wilcoxon_p(a, b), p_boot=round(bp, 5),
                             rb_corr=round(rb, 4) if np.isfinite(rb) else float('nan'), n_eff=ne,
                             cliff=round(cliffs_delta(a, b), 4)))
    return _finalize(pd.DataFrame(rows))


def episode_level(per_ep, base, ref=REF):
    """Paired-by-scenario comparison, clustered on the held-out scene.

    Pairing is valid because common random numbers give every configuration the same 500 scenarios.
    Independence is a separate question, and it does not hold: those 500 scenarios are resets over
    only 15 held-out scenes, so the correct unit of evidence is the scene, not the scenario. Every
    row therefore carries two readings of the same data.

      * The reported reading (`p_raw`, `ci_lo`, `ci_hi`, `rb_corr`, `n_eff`) is clustered. The
        interval comes from a bootstrap that resamples scenes; the test is an exact signed-rank test
        on the 15 scene-mean differences. This is what the Holm correction, the significance stars,
        the divergence flag and every table downstream consume, so no clustered result can be quoted
        by accident as an unclustered one.
      * The naive reading is kept beside it with an `_uncl` suffix, together with `deff` and
        `n_eff_cluster`, so the size of the correction is auditable and can be stated in the paper
        rather than discovered by a referee.

    `diff_mean` stays the pooled mean over all scenarios, and the cluster bootstrap resamples scenes
    and pools their scenarios, so estimator and interval describe the same quantity: scenes are
    weighted by how many scenarios they contributed, exactly as in the point estimate."""
    if ref not in per_ep:
        return pd.DataFrame()
    smap = ep_scene(per_ep[ref])
    rows = []
    others = [(k, v) for k, v in per_ep.items() if k != ref] + \
             [(f'baseline:{k}', v) for k, v in sorted(base.items())]
    for cfg, df in others:
        for c, lbl, better in EP_METRICS:
            if c not in df.columns:
                continue
            a, b = ep_vector(per_ep[ref], c), ep_vector(df, c)
            common = a.index.intersection(b.index)
            if len(common) < 20:
                continue
            av, bv = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
            cl = smap.loc[common].to_numpy()
            assert (ep_scene(df).loc[common].to_numpy() == cl).all(), \
                f'{cfg} disagrees with {ref} about the scenario-to-scene map'
            sc = 100.0 if c == 'rlf' else 1.0                 # report RLF as a percentage
            d = (av - bv) * sc
            m, lo, hi, bp, deff = boot_ci_cluster(d, cl)
            _, lo_u, hi_u, bp_u = boot_ci_paired(d)
            p_sc, rb_sc, ne_sc, n_sc, hl_sc, _, pfloor = cluster_level(av * sc, bv * sc, cl)
            rb_u, ne_u = rank_biserial(av, bv)
            rows.append(dict(level='episode', metric=c, metric_label=lbl, better=better,
                             comparison=f'{ref} vs {cfg}', n=len(common), n_scene=n_sc,
                             ref_mean=round(float(av.mean() * sc), 4),
                             cmp_mean=round(float(bv.mean() * sc), 4),
                             diff_mean=round(m, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             diff_hl=round(hodges_lehmann(d), 4),
                             diff_hl_scene=round(hl_sc, 4),
                             diff_disc=round(disc_mean(d), 4),
                             p_raw=p_sc, p_boot=round(bp, 5),
                             rb_corr=round(rb_sc, 4) if np.isfinite(rb_sc) else float('nan'),
                             n_eff=ne_sc, p_floor=pfloor,
                             deff=round(deff, 3) if np.isfinite(deff) else float('nan'),
                             n_eff_cluster=(round(len(common) / deff, 1)
                                            if np.isfinite(deff) and deff > 0 else float('nan')),
                             ci_lo_uncl=round(lo_u, 4), ci_hi_uncl=round(hi_u, 4),
                             p_raw_uncl=wilcoxon_p(av, bv), p_boot_uncl=round(bp_u, 5),
                             rb_corr_uncl=round(rb_u, 4) if np.isfinite(rb_u) else float('nan'),
                             n_eff_uncl=ne_u,
                             cliff=round(cliffs_delta(av, bv), 4)))
    return _finalize(pd.DataFrame(rows))


QUIESCENCE_KEYS = ['ret', 'rlf', 'steps', 'mean_pt', 'mean_rb', 'n_ho']

# Every column the intervention profile reads. `econ` is derived rather than recorded, but it is
# carried alongside the recorded columns so that the idle policy's economy is averaged over exactly
# the same scenario subset as its return, by the same code path, instead of being reconstructed
# afterwards from a mean of means.
PROFILE_KEYS = QUIESCENCE_KEYS + ['econ']


def episode_econ(df):
    """The return points each episode paid for its resource usage, exactly, per Eq. (15).

    This is not an estimate and not a model. The per-step charge is linear in the two actuator
    settings, and the evaluation record stores each episode's step-mean of both actuators together
    with its length, so the episode's whole charge is a closed-form function of three recorded
    columns. Recovering it matters because a policy that economizes on power or bandwidth is paid
    for that in return points regardless of what it does about blockage, and a return difference
    between two policies at different operating points is therefore not attributable to the
    behaviour under study until this term is accounted for.
    """
    miss = [c for c in ('steps', 'mean_pt', 'mean_rb') if c not in df.columns]
    if miss:
        raise SystemExit(f'REFUSING TO MEASURE: the resource charge of Eq. (15) is recoverable only '
                         f'from the recorded episode length and actuator means; {miss} absent.')
    return (df['steps'] * (W_PWR * (df['mean_pt'] - PT_MIN) / (PT_MAX - PT_MIN)
                           + W_SPEC * (df['mean_rb'] - RB_MIN) / (RB_MAX - RB_MIN)))


# --- which do-nothing policy a row is measured against -------------------------------------------
# The counterfactual columns of Table XII ask what would have happened on these same scenarios had
# the policy not handed over. A single do-nothing policy answers that for every row only while every
# row shares a resource operating point, and the reference policies deliberately do not: each
# non-learned rule is run in two variants, one holding both resource actuators at their maxima
# ("hold") and one economizing on both ("eff"). Measuring an eff rule against the hold do-nothing
# policy changes two things at once --- whether the policy hands over, and what it spends --- so the
# difference it reports is not attributable to handing over. The confound is larger than most of the
# differences the column is used to read: the two do-nothing policies differ by roughly twelve
# return points on identical scenarios with zero handovers between them.
#
# Each row is therefore measured against the do-nothing policy in its own regime. The regime is read
# off the reference-policy naming, which is a designed factor of the experiment, and NOT inferred
# from the measured actuator means: inference would hand different rungs of the same ablation ladder
# different counterfactuals, and a column whose baseline changes down its own length cannot be read
# down its own length. The learned configurations have no do-nothing counterpart at their operating
# point --- no such policy was run, and none can be recovered from the record --- so they keep the
# maximum-power do-nothing policy and the residual is measured and disclosed instead of being
# imagined away. `a_econ_gap` below is that residual, in return points, per row.
IDLE_BY_REGIME = {'hold': 'static_hold', 'eff': 'static_eff'}
IDLE_DEFAULT = 'static_hold'


def matched_idle(cfg, base, default=IDLE_DEFAULT):
    """The do-nothing policy in `cfg`'s own resource regime, or the default where none was run."""
    nm = cfg.split(':', 1)[1] if cfg.startswith('baseline:') else None
    if nm is not None:
        for suf, idl in IDLE_BY_REGIME.items():
            if nm.endswith('_' + suf):
                if idl not in base:
                    raise SystemExit(
                        f'REFUSING TO MEASURE: {nm} is a "{suf}" reference policy, so its '
                        f'counterfactual is the do-nothing policy in that same regime, and '
                        f'{idl} was not evaluated. Measuring it against {default} instead would '
                        f'credit it with resource economy it did not buy by handing over. Run '
                        f'{idl} or drop the row; do not substitute.')
                return idl
    return default


def check_idle_regimes(base):
    """The regime suffix is a designed factor, but it is a *name*, and matching on a name is only
    sound while the name still describes the policy. A rule labelled "hold" that had stopped holding
    both actuators at their maxima, or a "static_eff" that had stopped economizing, would silently
    invert the matching and reintroduce exactly the confound the matching exists to remove --- with
    no symptom anywhere in the output. Both halves are therefore checked against the recorded
    actuator means before any counterfactual is drawn."""
    for k, df in sorted(base.items()):
        if not {'mean_pt', 'mean_rb'}.issubset(df.columns):
            continue
        if k.endswith('_hold'):
            lo_p, lo_b = float(df['mean_pt'].min()), float(df['mean_rb'].min())
            if abs(lo_p - PT_MAX) > 1e-6 or abs(lo_b - RB_MAX) > 1e-6:
                raise SystemExit(
                    f'REFUSING TO MEASURE: {k} is named a "hold" policy and the counterfactual '
                    f'matching takes that to mean both resource actuators are pinned at their '
                    f'maxima ({PT_MAX} dBm, {RB_MAX} RB), but it reaches down to '
                    f'{lo_p:.3f} dBm / {lo_b:.3f} RB. Either the policy or the name changed.')
        elif k.endswith('_eff'):
            hi_p = float(df['mean_pt'].max())
            if hi_p >= PT_MAX - 1e-6:
                raise SystemExit(
                    f'REFUSING TO MEASURE: {k} is named an "eff" policy, so it is matched against '
                    f'the economizing do-nothing policy, but it reaches the power ceiling '
                    f'({hi_p:.3f} dBm). Either the policy or the name changed.')


def _profile_one(g, I):
    """One intervention-profile row from ONE run's per-scenario summaries, aligned to the idle
    policy's on the same scenarios. Kept separate from the aggregation so that the split between
    acted-on and not-acted-on scenarios is made inside a single run, which is the only level at
    which the split means anything."""
    act = g['n_ho'] > 0                                     # primary split: did it hand over?
    # Behavioural identity is asked of the recorded quantities only. `econ` is a deterministic
    # function of three of them, so including it could not add information and could only subtract
    # it, by letting floating-point noise in a derived column break a match the record itself makes.
    same = (g[QUIESCENCE_KEYS].round(4) == I[QUIESCENCE_KEYS].round(4)).all(axis=1)
    r = dict(n=float(len(g)), n_active=float(act.sum()), frac_active=float(act.mean()),
             n_idle_like=float((~act).sum()), frac_same_as_idle=float(same.mean()),
             # The idle policy's own rate over EVERY common scenario, carried on the row rather than
             # left to be reconstructed. It is the base rate an enrichment factor divides by, and
             # since each row is now measured against the do-nothing policy in its own resource
             # regime, there is no longer one base rate for the table: there is one per regime, and
             # a reader who took the wrong one would be dividing by a policy the row was never
             # compared against. Recording it makes the denominator travel with its numerator.
             idle_rlf=float(I['rlf'].mean() * 100), idle_ret=float(I['ret'].mean()))
    for nm, k in (('a', act), ('q', ~act)):
        if k.sum():
            r[f'{nm}_rlf'] = float(g.loc[k, 'rlf'].mean() * 100)
            r[f'{nm}_rlf_idle'] = float(I.loc[k, 'rlf'].mean() * 100)
            r[f'{nm}_ret'] = float(g.loc[k, 'ret'].mean())
            r[f'{nm}_ret_idle'] = float(I.loc[k, 'ret'].mean())
            # How much of the return difference this row prints is resource-economy bookkeeping
            # rather than a consequence of handing over. Positive means the row spent less than its
            # counterfactual and was paid for it, so the printed gap flatters it by this many
            # points. For a row matched to a do-nothing policy at its own operating point this is
            # near zero by construction, and that is the check on the matching as much as it is a
            # correction; where it is not near zero, the row has no matched counterfactual and the
            # number is the disclosure.
            r[f'{nm}_econ'] = float(g.loc[k, 'econ'].mean())
            r[f'{nm}_econ_idle'] = float(I.loc[k, 'econ'].mean())
            r[f'{nm}_econ_gap'] = r[f'{nm}_econ_idle'] - r[f'{nm}_econ']
        else:
            for t in ('rlf', 'rlf_idle', 'ret', 'ret_idle', 'econ', 'econ_idle', 'econ_gap'):
                r[f'{nm}_{t}'] = float('nan')
    return r


def intervention_profile(per_ep, base, ref=REF, idle='static_hold'):
    """Where does a controller's advantage actually live?

    The split is made on the HANDOVER actuator: a held-out scenario is *active* for a configuration
    when that configuration issues at least one handover on it, and *idle-like* when it issues none.
    Handover is the mechanism the paper is about, and unlike a whole-trajectory comparison this split
    is not confounded by the resource actuators. That confound is not hypothetical: a configuration
    that economises bandwidth differs from the do-nothing policy on *every* scenario for a reason
    that has nothing to do with whether it intervened against blockage, so a trajectory-identity
    measure would score it as always-active and say nothing. `frac_same_as_idle` is kept as a
    secondary diagnostic for exactly that reason, and must be read with it: it is only meaningful
    between two policies that also drive the resource actuators alike, which regime matching makes
    approximately but not exactly true.

    Against each subset the do-nothing policy's own outcome on the *same* scenarios is reported, which
    is the counterfactual: it says how dangerous the scenarios the controller chose to act on actually
    were, and whether acting paid. WHICH do-nothing policy is matched to the row's own resource
    regime, because the return the counterfactual is compared against is not a property of handing
    over alone. Eq. (15) charges for transmit power and for bandwidth, so a policy that economizes on
    either is paid for it whatever it does about blockage; the two do-nothing policies on this page
    differ by about twelve return points on identical scenarios with zero handovers between them.
    Reading an economizing rule against the maximum-power do-nothing policy therefore reports that
    twelve-point payment as though handing over had earned it, which is large enough to reverse the
    sign of the comparison for a rule whose handovers are worth less than that. `hold` rules are
    matched to the `hold` do-nothing policy and `eff` rules to the `eff` one; `a_econ_gap` records
    what is left, exactly, in return points, so the residual is a published number rather than a
    hope. The learned configurations have no do-nothing counterpart at their own operating point ---
    none was run, and none is recoverable from the record --- so they keep the maximum-power one and
    `a_econ_gap` is their disclosure rather than their correction. This answers a question the aggregate cannot — whether a low mean
    failure rate comes from a controller that is quietly correct everywhere, or from one that is inert
    where nothing threatens the link and decisive where something does. The second is the behaviour a
    Near-RT RIC xApp should have, and it is a stronger and more falsifiable claim than the mean.

    Under common random numbers the environment is deterministic given the actions, so two policies
    emitting the same actions on a scenario record the same episode summary to the last decimal. The
    converse is not guaranteed — two different action sequences could in principle coincide on all six
    recorded quantities — so `frac_same_as_idle` is stated as behavioural indistinguishability at the
    level of the recorded summary, which is the limit of what the data supports.
    The split is made WITHIN a run and the resulting profiles are then averaged over runs; the
    reported value is a mean over seeds and `*_min` / `*_max` give its range. Doing it the other way
    round — pooling the seeds and splitting the pooled frame — is not a milder version of the same
    measurement, it is a different and much weaker one. `groupby('idx').mean()` over a multi-seed
    frame averages `n_ho` across seeds, so `> 0` marks a scenario active whenever ANY seed handed
    over on it. The active set is then a union over seeds: it can only grow as seeds are added, and a
    controller that acts on a quarter of the scenarios in every single run is reported as acting on
    all of them once enough runs are pooled. That is exactly what produced the 100% entries for the
    two ladder rungs whose handover branch collapses, whose per-run active fractions in fact span
    the whole interval from zero to one. Averaging profiles asks the question the table claims to
    ask: on a given run, how selective was this controller?

    A run in which the configuration never hands over contributes no `a_*` values, and one in which
    it always hands over contributes no `q_*` values. Those runs are skipped for that half of the
    row rather than counted as zero, and `*_nseed` records how many runs each mean rests on, so a
    mean over three surviving runs cannot be read as if it rested on twenty."""
    if idle not in base:
        return pd.DataFrame()
    check_idle_regimes(base)
    _cache = {}

    def _idle_frame(nm):
        if nm not in _cache:
            b = base[nm].copy()
            b['econ'] = episode_econ(b)
            _cache[nm] = b.groupby('idx')[PROFILE_KEYS].mean()
        return _cache[nm]

    rows = []
    # A do-nothing policy is the ruler the other rows are measured with, not a row to be measured:
    # against its own regime's counterfactual it is itself, and a row comparing a policy to itself
    # reports nothing. Both are dropped here rather than printed as a line of dashes. Their own
    # operating points and failure rates are in Table X(a), where they belong.
    src = [(k, v) for k, v in sorted(per_ep.items())] + \
          [(f'baseline:{k}', v) for k, v in sorted(base.items())
           if matched_idle(f'baseline:{k}', base, idle) != k]
    for cfg, df in src:
        if not set(QUIESCENCE_KEYS).issubset(df.columns):
            continue
        idl_name = matched_idle(cfg, base, idle)
        idl = _idle_frame(idl_name)
        runs = [d for _, d in df.groupby('seed')] if 'seed' in df.columns else [df]
        per = []
        for d in runs:
            d = d.copy()
            d['econ'] = episode_econ(d)
            g = d.groupby('idx')[PROFILE_KEYS].mean()
            common = g.index.intersection(idl.index)
            if len(common) < 20:
                continue
            per.append(_profile_one(g.loc[common], idl.loc[common]))
        if not per:
            continue
        P = pd.DataFrame(per)
        r = dict(config=cfg, idle_ref=idl_name, n_seeds=len(P))
        for k in P.columns:
            v = P[k].to_numpy(dtype=float)
            fin = v[np.isfinite(v)]
            r[f'{k}_nseed'] = int(len(fin))
            if not len(fin):
                r[k] = r[f'{k}_min'] = r[f'{k}_max'] = float('nan')
                continue
            nd = 4 if (k.startswith('frac') or k == 'n' or k.startswith('n_')) else 2
            r[k] = round(float(fin.mean()), nd)
            r[f'{k}_min'] = round(float(fin.min()), nd)
            r[f'{k}_max'] = round(float(fin.max()), nd)
        rows.append(r)
    df = pd.DataFrame(rows)
    # A yardstick with no free constant in it. The confound regime matching exists to remove is the
    # economy difference between the two do-nothing policies themselves; a row that was matched and
    # still carries a residual that large was not in fact matched, and the most likely cause is that
    # the regime suffixes have come to mean something other than what check_idle_regimes tests. The
    # comparison is against the thing being removed, so it needs no threshold to be chosen.
    if len(df) and len(_cache) > 1:
        _e = {k: float(v['econ'].mean()) for k, v in _cache.items()}
        _spread = max(_e.values()) - min(_e.values())
        for _, _r in df.iterrows():
            if _r['idle_ref'] == idle or not np.isfinite(float(_r.get('a_econ_gap', np.nan))):
                continue                                        # unmatched rows disclose, not claim
            if abs(float(_r['a_econ_gap'])) >= _spread:
                raise SystemExit(
                    f'REFUSING TO MEASURE: {_r["config"]} was matched to the do-nothing policy '
                    f'{_r["idle_ref"]} to remove a resource-economy confound worth {_spread:.2f} '
                    f'return points, and after matching it still carries '
                    f'{float(_r["a_econ_gap"]):.2f}. The regime matching is not doing what it '
                    'claims; do not publish a counterfactual drawn this way.')
    if len(df) and ref in set(df.config):                       # reference configuration first
        df['_o'] = (df.config != ref).astype(int)
        df = df.sort_values(['_o', 'config']).drop(columns='_o').reset_index(drop=True)
    return df


def seed_lottery(per_seed):
    """The companion study reported a reliability seed lottery. Quantify it here: spread of the
    held-out RLF rate across seeds within a configuration, and whether any single seed is an outlier
    in every configuration (which would indicate a shared cause rather than per-run variance)."""
    if 'rlf_rate' not in per_seed.columns:
        return pd.DataFrame(), pd.DataFrame()
    p = per_seed.pivot_table(index='seed', columns='config', values='rlf_rate')
    spread = pd.DataFrame({
        'config': p.columns,
        'rlf_min': p.min().values, 'rlf_max': p.max().values,
        'rlf_range': (p.max() - p.min()).values,
        'rlf_iqr': (p.quantile(0.75) - p.quantile(0.25)).values,
        'rlf_cv': (p.std(ddof=1) / p.mean().replace(0, np.nan)).values,
    }).round(4).reset_index(drop=True)
    z = ((p - p.mean()) / p.std(ddof=1).replace(0, np.nan))
    rank = pd.DataFrame({'seed': p.index, 'mean_z_rlf': z.mean(axis=1).values,
                         'n_worst': (p.rank(axis=0, ascending=False) == 1).sum(axis=1).values,
                         'n_best': (p.rank(axis=0, ascending=True) == 1).sum(axis=1).values}).round(4)
    return spread, rank.sort_values('mean_z_rlf', ascending=False).reset_index(drop=True)


def alpha_curve(per_config):
    rows = []
    for _, r in per_config.iterrows():
        c = str(r['config'])
        al = (0.25 if c == 'model' else 1.0 if c == 'no_cvar'
              else float(c[5:]) if c.startswith('alpha') else None)
        if al is None:
            continue
        rows.append(dict(alpha=al, config=c, n_seeds=r['n_seeds'],
                         rlf_rate=r.get('rlf_rate_mean'), rlf_ci_lo=r.get('rlf_rate_ci_lo'),
                         rlf_ci_hi=r.get('rlf_rate_ci_hi'), cvar25=r.get('cvar25_mean'),
                         return_mean=r.get('return_mean_mean')))
    return pd.DataFrame(rows).sort_values('alpha').reset_index(drop=True)


# -------------------------------------------------------------------------------- report
def fmt(df, cols=None):
    if df is None or not len(df):
        return '_(no rows)_\n'
    d = df[cols] if cols else df
    return d.to_markdown(index=False) + '\n'


PAIR_COLS = ['metric', 'comparison', 'n', 'n_scene', 'n_eff', 'ref_mean', 'cmp_mean', 'diff_mean',
             'ci_lo', 'ci_hi', 'deff', 'n_eff_cluster', 'diff_disc', 'diff_hl', 'p_raw', 'p_holm',
             'p_floor_holm', 'resolvable', 'sig', 'rb_corr', 'rb_label']
# The unclustered reading is printed in its own block rather than beside the clustered one, so that
# no reader can pick a narrow interval out of the same row as a wide one and quote it as the result.
UNCL_COLS = ['metric', 'comparison', 'n', 'diff_mean', 'ci_lo_uncl', 'ci_hi_uncl', 'p_raw_uncl',
             'p_holm_uncl', 'rb_corr_uncl', 'n_eff_uncl']
# The intervention profile carries a mean, a min, a max and a contributing-run count for every
# quantity, which is the right thing to keep on disk and the wrong thing to print. The report shows
# the means, the range of the one quantity whose range is the finding (how often the configuration
# acted), and the two run counts that say how much of the row is actually populated; the full frame
# is in stats_intervention.csv.
IV_COLS = ['config', 'n_seeds', 'frac_active', 'frac_active_min', 'frac_active_max',
           'frac_same_as_idle', 'a_rlf_idle', 'a_rlf', 'a_ret_idle', 'a_ret', 'a_rlf_nseed',
           'q_rlf', 'q_rlf_idle', 'q_rlf_nseed']


def disp(df, cols=PAIR_COLS):
    """Report view of a paired-comparison table. p-values are shown through fmt_p, so a value below
    the resolution of the test prints as `<1e-16` and never as an exact zero, which would claim a
    precision the test does not have. Cliff's delta stays in the CSV: it is an unpaired measure and
    its thresholds are not valid for these paired, heavily tied outcomes."""
    if df is None or not len(df):
        return '_(no rows)_\n'
    d = df.copy()
    for c in ('p_raw', 'p_holm', 'p_raw_uncl', 'p_holm_uncl'):
        if f'{c}_s' in d.columns:
            d[c] = d[f'{c}_s']
    return fmt(d, [c for c in cols if c in d.columns])


def divergence_note(df):
    """Name every row where the mean-difference interval and the rank test point different ways."""
    if df is None or not len(df) or 'divergence' not in df.columns:
        return ''
    f = df[df.divergence != '']
    if not len(f):
        return ('\nIn no comparison does the mean-difference interval disagree with the rank test.\n')
    L = ['\n**Where the mean and the rank test disagree, and why that is not a contradiction.** The '
         'bootstrap interval is on the *mean* paired difference; the signed-rank test and the '
         'rank-biserial correlation describe the *pseudomedian*, the typical pair. The two diverge '
         'precisely when a policy accepts a small loss on most scenarios in exchange for a large gain '
         'on the few that would otherwise fail — which is what a tail objective is for. The affected '
         'rows are listed rather than reconciled:\n\n']
    for _, r in f.iterrows():
        L.append(f"- `{r['comparison']}`, {r['metric']}: mean difference {r['diff_mean']:+.4g} "
                 f"(95% CI [{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]), pseudomedian {r['diff_hl']:+.4g}, "
                 f"Holm p = {fmt_p(r['p_holm'])} — {r['divergence']}.\n")
    return ''.join(L)


def report(outdir, per_seed, per_ep, base_m, base, pc, sl, el, spread, rank, ac, ip):
    L = []
    A = L.append
    # The report may never assert a seed count the run did not have --- and it may not assert a
    # single one when the sweep genuinely has two. n is the number of seeds behind the paired
    # seed-level tests, and the exact two-sided Wilcoxon floor 2/2**n follows from it rather than
    # from a remembered "10".
    #
    # Taking the minimum over *every* row of per_config silently picks up the risk-sweep arms of
    # section 7, which are deliberately run at fewer seeds than the ablation ladder. At twenty
    # ladder seeds beside five alpha seeds that minimum is five, and section 9 then told the reader
    # that no seed-level comparison anywhere could resolve past 2/2**5 = 0.0625 --- when the
    # ladder's own floor is 2/2**20, four seed-doublings finer. The error runs in the direction that
    # understates this study, which is the direction that gets believed, so it is worth being exact
    # about: the two groups differ by design, and both are named rather than collapsed into one
    # misleading number. The `alpha` name prefix is the same discriminator alpha_curve() already
    # uses, so this cannot drift from the arms section 7 actually reports.
    if 'n_seeds' in pc.columns and 'config' in pc.columns:
        _is_arm = pc.config.astype(str).str.startswith('alpha')
        _lad_n, _arm_n = pc.loc[~_is_arm, 'n_seeds'], pc.loc[_is_arm, 'n_seeds']
        n_seed = int(_lad_n.min()) if len(_lad_n) else int(pc.n_seeds.min())
        n_arm = int(_arm_n.min()) if len(_arm_n) else None
    else:
        n_seed, n_arm = int(per_seed.seed.nunique()), None
    p_floor = 2.0 / (2 ** n_seed)
    _split_n = n_arm is not None and n_arm != n_seed
    p_floor_arm = (2.0 / (2 ** n_arm)) if _split_n else None
    _spread_n = f'{n_seed} seeds (fewer for the risk-sweep arms)' if _split_n else f'{n_seed} seeds'
    A('# Paper 2 — R3 statistical analysis\n')
    A(f'Source: `{outdir}`. Reference configuration: `{REF}`. '
      f'Bootstrap: {BOOT} percentile resamples, seed {BOOT_SEED}. '
      'Paired Wilcoxon signed-rank, Holm-Bonferroni within each metric family. Every held-out number '
      'comes from the 15 scenes never seen in training, evaluated under common random numbers.\n')
    A('\nEach paired comparison is reported with three quantities that answer three different '
      'questions, because no single one of them is sufficient here. `diff_mean` with its bootstrap '
      'interval is the average difference, which is what a network operator ultimately pays or '
      'collects. `diff_hl` is the Hodges-Lehmann pseudomedian, the median of all Walsh averages of '
      'the paired differences; this is the location parameter the signed-rank test actually tests, so '
      'it is the quantity the p-value refers to. `rb_corr` is the matched-pairs rank-biserial '
      'correlation, the effect size that belongs with that test. Cliff\'s delta is retained in the '
      'CSV files but is deliberately not shown here: it is an *unpaired* dominance measure, and its '
      'conventional thresholds, applied to a paired and heavily tied near-binary outcome such as the '
      'per-scenario failure flag, label a ten-percentage-point reliability difference `negligible`. '
      'That is an artefact of borrowing a threshold, not a finding.\n')

    A('\n## 1. Per-configuration held-out performance\n')
    A('Mean over training seeds with a 95% percentile-bootstrap interval, and the median with the '
      f'inter-quartile range, because {_spread_n} is a small and skewed sample and the mean '
      'alone hides the spread.\n')
    keep = ['config', 'n_seeds']
    for c, _, _ in METRICS:
        keep += [f'{c}_mean', f'{c}_ci_lo', f'{c}_ci_hi', f'{c}_median', f'{c}_iqr_lo', f'{c}_iqr_hi']
    A(fmt(pc, [c for c in keep if c in pc.columns]))

    A('\n## 2. Measured resource usage and generalisation gap\n')
    A('What each controller actually spends, and how much of its in-sample advantage survives the '
      'move to unseen scenes.\n')
    A(fmt(pc, [c for c in ['config', 'pt_dbm', 'rb', 'ho_per_ep', 'gap_return', 'gap_rlf',
                           'lat_mean_ms', 'lat_p95_ms', 'lat_p99_ms'] if c in pc.columns]))

    A('\n## 3. Seed-level paired comparisons (the algorithm claim)\n')
    A('The unit of replication is the training run. Every difference is reference minus comparator, '
      'so a positive value favours the reference on the two metrics where higher is better '
      '(CVaR$_{25}$, mean return) and disfavours it on RLF rate, where lower is better.\n')
    A('\nThis level and the next answer different questions and neither substitutes for the other. '
      'Here the held-out scene set is held fixed and only the training seed varies, so a rejection '
      'says the design change beats its ablation reliably across training randomness *on these 15 '
      'scenes*. It says nothing about a sixteenth scene. Section 4 varies the scene instead and is '
      'the only place a claim about unseen road geometry can be made. Read together they are '
      'informative; read as one they would double-count, so no comparison is called established on '
      'the strength of both at once.\n')
    A(disp(sl))
    A(divergence_note(sl))

    A('\n## 4. Scene-clustered scenario-level comparisons (the generalisation claim; the only level '
      'available for the non-learned baselines)\n')
    A('Each of the held-out scenarios is identical across configurations by construction, so the '
      'difference is measured on the same road situations, and learned configurations are averaged '
      'over their training seeds first. Pairing is therefore exact. Independence is a separate '
      'question and it does not hold: the 500 held-out scenarios are repeated resets over only 15 '
      'held-out scenes, so scenarios drawn from one scene share its geometry, its blocker population '
      'and its gNB layout. Treating them as 500 independent observations would inflate every '
      'interval and every p-value in this table.\n')
    A('\nThe table below is therefore clustered on the scene throughout. `deff` is the measured '
      'design effect — the variance of the mean under a bootstrap that resamples scenes divided by '
      'its variance under a bootstrap that resamples scenarios — and `n_eff_cluster` is the '
      'corresponding effective sample size. `ci_lo`/`ci_hi` come from the scene bootstrap. `p_raw` '
      'is an exact signed-rank test on the 15 scene-mean differences, so the unit of the test is the '
      'unit that is actually independent, and `n_eff` is the number of scenes on which the two '
      'policies differ at all. `diff_mean` remains the pooled mean over all scenarios; the scene '
      'bootstrap pools the scenarios of whichever scenes it draws, so the point estimate and its '
      'interval weight scenes identically.\n')
    A('\n`p_floor_holm` is the smallest Holm-adjusted p this comparison could attain at any effect '
      'size, obtained by substituting the exact signed-rank floor 2/2^`n_eff` for its own raw p and '
      're-running the correction against the rest of the family as observed. Where `resolvable` is '
      'false the design cannot establish the comparison however large the true difference is, '
      'because too few scenes separate the two policies. Such a row is a statement about the '
      'resolution of a 15-scene test set and must never be read as evidence of no effect.\n')
    A('\nOn the failure-flag rows the mean difference *is* the risk difference in percentage points, '
      'which is the standard effect measure for a paired binary outcome and is the number the paper '
      'quotes. `diff_disc` is the mean difference over the scenarios that are not tied, so the pair '
      '(`diff_mean`, `diff_disc`) separates a small uniform shift from a large targeted one. The '
      'scenario-level pseudomedian `diff_hl` is degenerate on those rows — it is zero whenever fewer '
      'than half the scenarios differ, which is true of every such comparison — so it should be read '
      'as uninformative rather than as evidence of no effect. On the return rows, where the outcome '
      'is continuous, the pseudomedian is the informative one.\n')
    A(disp(el))
    A(divergence_note(el))
    A('\n### 4b. The same comparisons read without clustering (diagnostic only — not reportable)\n')
    A('Printed so that the size of the correction is auditable, and printed separately so that a '
      'narrow unclustered interval can never be lifted out of the same row as the clustered one. '
      'These are the numbers an analysis that ignored the scene structure would have produced. They '
      'are not the paper\'s results and no manuscript table is built from them.\n')
    A(disp(el, UNCL_COLS))

    if base_m is not None and len(base_m):
        A('\n## 5. Non-learned reference policies\n')
        A('`_hold` pins both actuators at maximum (best reliability, worst efficiency); `_eff` runs '
          'conventional closed-loop power control and minimum bandwidth (best return). Both are '
          'reported so that no weak setting is presented as the comparator. All of them are given '
          'ideal, zero-delay SINR reports for every gNB, which is stronger than deployed practice.\n')
        bc = [c for c in ['config', 'return_mean', 'cvar25', 'rlf_rate', 'pt_dbm', 'rb', 'ho_per_ep',
                          'insample_rlf_rate', 'n_eval'] if c in base_m.columns]
        A(fmt(base_m[bc]))

    A('\n## 6. Where the advantage lives: when the controller acts, and whether acting pays\n')
    A('A held-out scenario is *active* for a configuration when that configuration issues at least '
      'one handover on it, and *idle-like* when it issues none. The split is made on the handover '
      'actuator because handover is the mechanism under study and because, unlike a whole-trajectory '
      'comparison, it is not confounded by the resource actuators: a configuration that economises '
      'bandwidth differs from the do-nothing policy on every scenario for reasons unrelated to '
      'blockage. `a_*` columns describe the active subset and `q_*` the idle-like subset; `*_idle` is '
      'the do-nothing policy `static_hold` (maximum power, maximum bandwidth, never hand over) '
      'evaluated on those same scenarios, which is the counterfactual — it says how dangerous the '
      'scenarios the controller chose to act on actually were.\n')
    A(fmt(ip, [c for c in IV_COLS if c in ip.columns]))
    A('\nEvery value in this table is a mean over runs of a profile computed WITHIN a run, not a '
      'profile of the pooled runs. The distinction is not cosmetic. Pooling first averages the '
      'handover count across seeds before thresholding it at zero, which marks a scenario active '
      'whenever any single seed handed over on it; the active set is then a union that can only '
      'grow with the seed count, and a configuration that acts on a quarter of the scenarios in '
      'every run is reported as acting on all of them. `frac_active_min` and `frac_active_max` give '
      'the range across runs, and they are the honest reading for any configuration whose handover '
      'branch is unstable across seeds: a mean of one half over runs that individually sit at zero '
      'or at one describes a bimodal policy, not a moderate one. `a_rlf_nseed` and `q_rlf_nseed` '
      'count the runs that contributed to each half of the row, because a run that never hands over '
      'has no active subset and a run that always hands over has no idle-like subset; those runs '
      'are skipped for that half rather than counted as zero.\n')
    A('\n`frac_same_as_idle` is a secondary diagnostic: the fraction of scenarios on which the '
      'configuration\'s entire recorded episode summary — return, failure flag, step count, mean '
      'power, mean resource blocks, handover count — matches the do-nothing policy to four decimals. '
      'It is only interpretable for configurations that also pin both resource actuators at their '
      'maxima; for any configuration that moves them it is zero by construction and means nothing.\n')
    A('\nThis table is the direct answer to "what is the controller doing", and it is the one the '
      'paper should quote when describing behaviour. A controller whose active subset carries a high '
      'counterfactual failure rate is selecting the dangerous scenes correctly; the drop from '
      '`a_rlf_idle` to `a_rlf` is what its intervention buys on those scenes. A low failure rate on '
      'the idle-like subset says the scenes it declined to act on genuinely did not need action. The '
      'aggregate mean mixes the two and therefore understates both.\n')

    A('\n## 7. Risk level alpha\n')
    # Read the levels off the frame instead of asserting a remembered pair. The sensitivity arm runs
    # last in the sweep, so a report generated from a partial run would otherwise claim points that
    # do not exist yet -- exactly the failure this whole pipeline is built to make impossible.
    have = sorted(float(v) for v in ac.alpha.unique()) if ac is not None and len(ac) else []
    extra = [v for v in have if v not in (0.25, 1.0)]
    A(f'Levels present in this run: {", ".join(f"{v:g}" for v in have) if have else "none"}. '
      'alpha = 0.25 is the proposed controller and alpha = 1.0 is the risk-neutral ablation, so '
      'those two are already in the main ladder; the sensitivity arm adds the intermediate and '
      'aggressive levels.\n')
    if not extra:
        A('\n**The sensitivity arm is absent from this run.** With only the two ladder points there '
          'is no alpha curve, and any statement about how performance varies with the risk level '
          'would be an extrapolation from two configurations that differ in more than alpha. No '
          'such statement may be written into the manuscript from this report.\n')
    A(fmt(ac))

    A('\n## 8. Seed variance and the reliability lottery\n')
    A('Spread of the held-out RLF rate across training runs within each configuration.\n')
    A(fmt(spread))
    A('\nPer-seed standing across configurations. A seed that is worst almost everywhere indicates a '
      'shared cause rather than independent run-to-run noise.\n')
    A(fmt(rank))

    A('\n## 9. What this does and does not establish\n')
    A(f'- The seed-level tests on the ablation ladder have n = {n_seed} and the exact Wilcoxon '
      f'signed-rank test cannot return a two-sided p below {p_floor:.3g} at that sample size, so '
      'no seed-level comparison can be significant beyond that floor no matter how large the '
      'effect. The effect sizes and intervals carry the weight.'
      + (f' The risk-sweep arms of section 7 are run at n = {n_arm}, so their own floor is '
         f'{p_floor_arm:.3g}; they share a Holm family with the ladder, and that weaker floor is a '
         'property of those two arms rather than of the ladder comparisons. Quoting the smaller of '
         'the two counts for the whole family would understate the ladder by every seed that '
         'separates them.' if _split_n else '') + '\n')
    A('- The scenario-level tests have n = 500 and much more power, but their unit is a scenario, '
      'not a training run; they support statements about which situations a policy handles, not '
      'about how reliably training reproduces the policy. Both are reported for that reason.\n')
    A('- Holm-Bonferroni controls the family-wise error rate within a metric, not across the three '
      'metrics jointly.\n')
    A('- A significant rank test says the *typical* held-out scenario moves; it does not by itself '
      'say the average moves, and where the mean-difference interval spans zero the report says so '
      'explicitly instead of quoting whichever statistic is more flattering.\n')
    A('- The per-scenario failure flag is binary, so the scenario-level differences are dominated by '
      'ties. The rank-biserial correlation is computed over the non-tied pairs only, and `n_eff` in '
      'the CSV records how many pairs that leaves; a large correlation resting on few discordant '
      'pairs is a weaker statement than the same number resting on many.\n')
    A('- The comparators are the ablations and the reference policies of this study on this '
      'environment. No number here is compared against a figure taken from another paper, because no '
      'other paper evaluates on these scenes with this channel and this action space.\n')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/root/work/paper2/models/sweep_r3')
    ap.add_argument('--ref', default=REF)
    ap.add_argument('--report', default=None)
    a = ap.parse_args()
    globals()['REF'] = a.ref

    per_seed, per_ep, base_m, base = load(a.outdir)
    pc = per_config_table(per_seed)
    sl = seed_level(per_seed, a.ref)
    el = episode_level(per_ep, base, a.ref)
    spread, rank = seed_lottery(per_seed)
    ac = alpha_curve(pc)
    ip = intervention_profile(per_ep, base, a.ref)

    pc.to_csv(os.path.join(a.outdir, 'stats_per_config.csv'), index=False)
    if len(sl):
        sl.to_csv(os.path.join(a.outdir, 'stats_seed_level.csv'), index=False)
    if len(el):
        el.to_csv(os.path.join(a.outdir, 'stats_episode_level.csv'), index=False)
    if len(spread):
        spread.to_csv(os.path.join(a.outdir, 'stats_seed_spread.csv'), index=False)
        rank.to_csv(os.path.join(a.outdir, 'stats_seed_rank.csv'), index=False)
    if len(ac):
        ac.to_csv(os.path.join(a.outdir, 'stats_alpha.csv'), index=False)
    if len(ip):
        ip.to_csv(os.path.join(a.outdir, 'stats_intervention.csv'), index=False)

    rpt = report(a.outdir, per_seed, per_ep, base_m, base, pc, sl, el, spread, rank, ac, ip)
    path = a.report or os.path.join(a.outdir, 'stats_report.md')
    with open(path, 'w') as fh:
        fh.write(rpt)
    print(rpt)
    print(f'\nsaved -> {path} and stats_*.csv in {a.outdir}')


if __name__ == '__main__':
    main()
