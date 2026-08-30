#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_seed_power.py — how many seeds would it take to resolve each ladder contrast?

RECONSTRUCTED 2026-08-23. The original script was lost with the cloud container that produced the
first sweep_r3 record; it is absent from the working tree and from Paper2_cloud_snapshot.zip. This
is a rewrite from the method specified in PREREGISTRATION.md §3, which states it in enough detail to
reimplement exactly. Two consequences must be stated rather than glossed:

  * The numbers this produces are computed from whatever `models/sweep_r3/` currently holds. They
    will NOT match the figures the manuscript's §VIII-D presently quotes (empirical .03 -> .16,
    parametric .06 -> .11 for the tail-return contrast), because those were computed on a ten-seed
    record that no longer exists. Whatever this outputs is what the prose must be regenerated from.
  * Where the specification is silent on an implementation detail, this file makes the choice
    explicit and records it in the output JSON under "implementation", so a reader can see what was
    decided here rather than inherited.

METHOD (PREREGISTRATION.md §3, quoted obligations in brackets)

  [resample seeds with replacement from the observed paired differences to a target count N]
  [one resampled seed index is drawn per replicate and applied to *every* configuration, which
   preserves the common-random-number pairing of the real design]
      -> one index vector per replicate, shared across all contrasts and all metrics.

  [recompute the exact signed-rank p for every ladder contrast on that resampled seed set]
  [Holm-correct the full seven-comparison family]
  [count the fraction of 4000 replicates reaching Holm-adjusted p < 0.05]

  [the two alpha-sensitivity contrasts are held at their observed five-seed p-values, because the
   alpha sweep is a separate sensitivity check that will not be extended]
      -> alpha0.1 and alpha0.5 enter the Holm family at their fixed observed p and are never
         resampled. They occupy two of the seven slots and can therefore displace a ladder contrast
         in the Holm ordering, which is the point of including them.

  [a parametric paired-t power curve at each contrast's realistic Holm threshold is computed
   independently as a cross-check]
      -> noncentral-t power at alpha = the Holm threshold the contrast's OBSERVED rank earns,
         i.e. 0.05/(m - k + 1) for observed rank k in a family of m = 7.

  [stated limitation, carried into any manuscript text derived from this] Bootstrapping a sample of
  ten to estimate behaviour at fifty extrapolates well beyond the data. The resampled population is
  the empirical distribution of the observed seeds and inherits whatever they happened to show.
  These are order-of-magnitude guides to a required seed count, not precise power calculations.

USAGE
    python paper2_seed_power.py --json results/seed_power.json
    python paper2_seed_power.py --max_seeds 10 --json results/seed_power_10seed.json

`--max_seeds` caps the observed record the resampling draws from. Use 10 to reproduce the
pre-registration's reasoning from the first ten seeds of the rebuilt ladder; omit it to use every
seed present.
"""
import os, sys, json, glob, argparse, math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

REPORTED = 'model'
LADDER = ['p1_transplant', 'no_ant', 'no_cvar', 'no_per', 'plus_lstm']   # resampled
ALPHA_ARMS = ['alpha0.1', 'alpha0.5']                                    # held fixed
FAMILY_SIZE = 7                                                          # 5 ladder + 2 alpha
METRICS = [('rlf_rate', 'lower'), ('cvar25', 'higher'), ('return_mean', 'higher')]
N_GRID = [10, 15, 20, 30, 40, 50]
N_REPLICATES = 4000
ALPHA_LEVEL = 0.05
EXACT_MAX_N = 25          # above this the exact null is impractical; use the normal approximation


# ---------------------------------------------------------------- signed-rank machinery
@lru_cache(maxsize=None)
def _wplus_null(n):
    """Exact null distribution of W+ for n untied, non-zero pairs.

    Coefficient j of the returned array is the number of the 2**n sign assignments giving W+ = j.
    Built by polynomial multiplication over prod_{r=1..n} (1 + x**r).
    """
    dist = np.array([1.0])
    for r in range(1, n + 1):
        shifted = np.zeros(len(dist) + r)
        shifted[:len(dist)] += dist
        shifted[r:] += dist
        dist = shifted
    return dist / dist.sum()


def signed_rank_p(d):
    """Two-sided Wilcoxon signed-rank p for paired differences d.

    Zeros are dropped (Wilcoxon's own treatment; the pre-registration counts 'untied' pairs, which
    is what remains). The exact null is used when the surviving |d| carry no ties and n is small
    enough for the distribution to be built; otherwise the tie-corrected normal approximation with
    continuity correction is used. Returns (p, n_untied, method).
    """
    d = np.asarray(d, dtype=float)
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return 1.0, 0, 'degenerate'
    a = np.abs(d)
    ranks = stats.rankdata(a)
    w_plus = float(ranks[d > 0].sum())
    has_ties = len(np.unique(a)) != n

    if not has_ties and n <= EXACT_MAX_N:
        null = _wplus_null(n)
        support = np.arange(null.size)
        lo = null[support <= w_plus].sum()
        hi = null[support >= w_plus].sum()
        return float(min(1.0, 2.0 * min(lo, hi))), n, 'exact'

    # tie-corrected normal approximation with continuity correction
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(a, return_counts=True)
    tie_term = (counts ** 3 - counts).sum()
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return 1.0, n, 'degenerate'
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
    return float(min(1.0, 2.0 * stats.norm.sf(z))), n, 'approx'


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved. pvals is a 1-D array."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind='stable')
    adj = np.empty(m)
    running = 0.0
    for k, idx in enumerate(order):
        running = max(running, (m - k) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def holm_threshold_for_rank(k, m=FAMILY_SIZE, alpha=ALPHA_LEVEL):
    """The nominal level a contrast at observed rank k (1-based) must beat under Holm."""
    return alpha / (m - k + 1)


# ---------------------------------------------------------------- data
def load_ladder(outdir, max_seeds=None):
    """Return {config: {seed: {metric: value}}} from the per-seed metrics files."""
    data = {}
    for f in sorted(glob.glob(os.path.join(outdir, '*_s*_metrics.csv'))):
        base = os.path.basename(f)[:-len('_metrics.csv')]
        cfg, _, sd = base.rpartition('_s')
        if not sd.isdigit():
            continue
        row = pd.read_csv(f).iloc[0]
        data.setdefault(cfg, {})[int(sd)] = {m: float(row[m]) for m, _ in METRICS if m in row}
    if max_seeds is not None:
        data = {c: {s: v for s, v in d.items() if s < max_seeds} for c, d in data.items()}
    return data


def paired_matrix(data, cfg, metric):
    """Paired (reported - variant) differences over the seeds both configurations share."""
    seeds = sorted(set(data[REPORTED]) & set(data.get(cfg, {})))
    if not seeds:
        return np.array([]), []
    d = np.array([data[REPORTED][s][metric] - data[cfg][s][metric] for s in seeds])
    return d, seeds


# ---------------------------------------------------------------- power
def parametric_power(d_z, n, alpha):
    """Two-sided paired-t power at effect size d_z, n pairs, level alpha (noncentral t).

    scipy's nct returns nan for very large noncentrality (a big effect at a large n), where the
    answer is unambiguously 1. Fall back to the normal approximation rather than propagate nan
    into the report.
    """
    if n < 2 or not np.isfinite(d_z) or d_z == 0:
        return 0.0
    df = n - 1
    ncp = abs(d_z) * math.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2.0, df)
    val = float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))
    if not math.isfinite(val):
        val = float(stats.norm.sf(crit - ncp) + stats.norm.cdf(-crit - ncp))
    return min(1.0, max(0.0, val))


def run_metric(data, metric, direction, rng, replicates, n_grid, verbose=True):
    """Observed statistics and both power curves for one metric family."""
    # ---- observed
    diffs, seeds_used, observed = {}, {}, {}
    for cfg in LADDER:
        d, seeds = paired_matrix(data, cfg, metric)
        if d.size == 0:
            continue
        p, n_untied, how = signed_rank_p(d)
        sd = d.std(ddof=1)
        diffs[cfg] = d
        seeds_used[cfg] = seeds
        observed[cfg] = dict(
            n=int(d.size), n_untied=int(n_untied), method=how,
            mean_diff=float(d.mean()), sd_diff=float(sd),
            d_z=float(d.mean() / sd) if sd > 0 else float('inf'),
            sign_split=[int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
            p_raw=float(p))
    if not diffs:
        return None

    # ---- alpha arms: observed p only, held fixed forever after
    alpha_fixed = {}
    for cfg in ALPHA_ARMS:
        d, _ = paired_matrix(data, cfg, metric)
        if d.size:
            p, n_untied, how = signed_rank_p(d)
            alpha_fixed[cfg] = float(p)
            observed[cfg] = dict(n=int(d.size), n_untied=int(n_untied), method=how,
                                 mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)),
                                 d_z=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float('inf'),
                                 sign_split=[int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
                                 p_raw=float(p), held_fixed=True)

    # ---- observed Holm over the full family, in a fixed slot order
    fam = [c for c in LADDER if c in diffs] + [c for c in ALPHA_ARMS if c in alpha_fixed]
    fam_p = np.array([observed[c]['p_raw'] for c in fam])
    fam_adj = holm(fam_p)
    ranks = stats.rankdata(fam_p, method='ordinal')
    for c, adj, k in zip(fam, fam_adj, ranks):
        observed[c]['p_holm'] = float(adj)
        observed[c]['holm_rank'] = int(k)
        observed[c]['must_beat'] = float(holm_threshold_for_rank(int(k), m=len(fam)))
        observed[c]['resolved'] = bool(adj < ALPHA_LEVEL)

    # ---- bootstrap power. One shared index vector per replicate preserves CRN pairing.
    n_obs = len(next(iter(seeds_used.values())))
    ladder_here = [c for c in LADDER if c in diffs]
    fixed_tail = [alpha_fixed[c] for c in ALPHA_ARMS if c in alpha_fixed]
    emp = {c: {} for c in ladder_here}
    method_counts = {'exact': 0, 'approx': 0, 'degenerate': 0}

    for N in n_grid:
        hits = {c: 0 for c in ladder_here}
        for _ in range(replicates):
            idx = rng.integers(0, n_obs, size=N)          # shared across every configuration
            ps = []
            for c in ladder_here:
                p, _, how = signed_rank_p(diffs[c][idx])
                method_counts[how] += 1
                ps.append(p)
            adj = holm(np.array(ps + fixed_tail))
            for j, c in enumerate(ladder_here):
                if adj[j] < ALPHA_LEVEL:
                    hits[c] += 1
        for c in ladder_here:
            emp[c][N] = hits[c] / replicates
        if verbose:
            print(f'    N={N:3d} done', flush=True)

    # ---- parametric cross-check at each contrast's own observed Holm threshold
    par = {c: {N: parametric_power(observed[c]['d_z'], N, observed[c]['must_beat'])
               for N in n_grid} for c in ladder_here}

    def first_at(curve, target=0.80):
        for N in n_grid:
            if curve[N] >= target:
                return N
        return None

    return dict(
        metric=metric, direction=direction, n_observed_seeds=n_obs,
        observed=observed,
        empirical_power={c: emp[c] for c in ladder_here},
        parametric_power={c: par[c] for c in ladder_here},
        eighty_pct_at={c: dict(empirical=first_at(emp[c]), parametric=first_at(par[c]))
                       for c in ladder_here},
        signed_rank_methods_used=method_counts)


# ---------------------------------------------------------------- reporting
def print_report(res):
    m = res['metric']
    print(f"\n=== {m} ({'lower' if res['direction'] == 'lower' else 'higher'} is better) "
          f"| {res['n_observed_seeds']} observed seeds ===")
    print(f"{'contrast':<16}{'n':>4}{'untied':>8}{'d_z':>8}{'sign':>10}{'p_raw':>10}"
          f"{'p_Holm':>10}{'must beat':>11}  verdict")
    for c, o in res['observed'].items():
        pos, neg, zer = o['sign_split']
        if 'p_holm' not in o:
            continue
        print(f"{c:<16}{o['n']:>4}{o['n_untied']:>8}{o['d_z']:>8.2f}{f'{pos}/{neg}':>10}"
              f"{o['p_raw']:>10.5f}{o['p_holm']:>10.5f}{o['must_beat']:>11.5f}  "
              f"{'resolved' if o['resolved'] else 'NOT RESOLVED'}")
    grid = sorted(next(iter(res['empirical_power'].values())).keys())
    print(f"\n  power to resolve at Holm-corrected {ALPHA_LEVEL} (empirical / parametric)")
    print(f"  {'contrast':<16}{'d_z':>7}" + ''.join(f'{"N=" + str(N):>13}' for N in grid) + f"{'80% at':>10}")
    for c in res['empirical_power']:
        e, p = res['empirical_power'][c], res['parametric_power'][c]
        cells = ''.join(f'{f"{e[N]:.2f}/{p[N]:.2f}":>13}' for N in grid)
        at = res['eighty_pct_at'][c]['empirical']
        print(f"  {c:<16}{res['observed'][c]['d_z']:>7.2f}{cells}{(str(at) if at else '> ' + str(grid[-1])):>10}")


def emit_build_schema(out):
    """Add the shape `fix_r4b_numbers.py` reads, alongside the richer report structure.

    The manuscript build's Numbers.PW / Numbers.PW_TARGETS require exactly:

        targets                                      -> list of seed counts
        metrics.<metric>.rows.<cfg>.d_z              -> float
        metrics.<metric>.rows.<cfg>.sign             -> dict carrying 'n'
        metrics.<metric>.rows.<cfg>.n_for_80pct      -> int, or null when no target reaches 0.8
                                                        (the null is a finding: a flat curve)
        metrics.<metric>.rows.<cfg>.power[<target>]  -> {empirical, parametric}

    The build presence-checks these rather than value-checks them, so a key that goes missing stops
    the build. Both structures are written: `results` for humans, these for the build.
    """
    targets = out['n_grid']
    metrics = {}
    for metric, res in out['results'].items():
        rows = {}
        for cfg in res['empirical_power']:
            o = res['observed'][cfg]
            pos, neg, tied = o['sign_split']
            rows[cfg] = dict(
                d_z=o['d_z'],
                sign=dict(n=o['n'], untied=o['n_untied'], positive=pos, negative=neg, tied=tied),
                n_for_80pct=res['eighty_pct_at'][cfg]['empirical'],
                p_raw=o['p_raw'],
                p_holm=o.get('p_holm'),
                power={str(N): dict(empirical=res['empirical_power'][cfg][N],
                                    parametric=res['parametric_power'][cfg][N])
                       for N in targets},
            )
        metrics[metric] = dict(rows=rows)
    out['targets'] = targets
    out['metrics'] = metrics
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.path.join(HERE, 'models', 'sweep_r3'),
                    help='directory holding the per-seed *_metrics.csv files')
    ap.add_argument('--json', default=os.path.join(HERE, 'results', 'seed_power.json'))
    ap.add_argument('--replicates', type=int, default=N_REPLICATES)
    ap.add_argument('--max_seeds', type=int, default=None,
                    help='cap the observed record to seeds [0, max_seeds)')
    ap.add_argument('--n_grid', default=','.join(str(n) for n in N_GRID))
    ap.add_argument('--seed', type=int, default=20260726, help='bootstrap RNG seed (pinned)')
    a = ap.parse_args()

    n_grid = [int(x) for x in a.n_grid.split(',')]
    data = load_ladder(a.outdir, a.max_seeds)
    if REPORTED not in data:
        sys.exit(f'no "{REPORTED}" metrics found in {a.outdir} — run the sweep first')

    present = {c: len(v) for c, v in sorted(data.items())}
    print(f'seed-power analysis | outdir {a.outdir}')
    print(f'configurations present: ' + ', '.join(f'{c}({n})' for c, n in present.items()))
    missing = [c for c in LADDER if c not in data]
    if missing:
        print(f'WARNING: ladder contrasts absent, excluded from the family: {", ".join(missing)}')
    missing_alpha = [c for c in ALPHA_ARMS if c not in data]
    if missing_alpha:
        print(f'WARNING: alpha arms absent, Holm family is smaller than the pre-registered 7: '
              f'{", ".join(missing_alpha)}')
    print(f'replicates {a.replicates} | N grid {n_grid} | bootstrap seed {a.seed}')

    rng = np.random.default_rng(a.seed)
    out = dict(
        generated_by='paper2_seed_power.py (reconstructed 2026-08-23 from PREREGISTRATION.md §3)',
        outdir=os.path.abspath(a.outdir),
        configurations_present=present,
        max_seeds=a.max_seeds,
        replicates=a.replicates,
        n_grid=n_grid,
        bootstrap_seed=a.seed,
        alpha_level=ALPHA_LEVEL,
        holm_family_size=FAMILY_SIZE,
        implementation=dict(
            zero_differences='dropped before ranking (Wilcoxon); "untied" counts what survives',
            exact_null='built by polynomial DP over prod(1 + x**r); used when |d| has no ties and n <= %d' % EXACT_MAX_N,
            tied_or_large='tie-corrected normal approximation with continuity correction',
            why_ties_matter='rlf_rate is quantised to 0.2 pp (1/500 rollouts), so ties in |d| are common',
            pairing='one bootstrap index vector per replicate, shared across every configuration',
            alpha_arms='held at their observed p-values, never resampled (pre-registration §3)',
            parametric_alpha="each contrast's own observed Holm threshold, 0.05/(m-k+1)"),
        results={})

    for metric, direction in METRICS:
        if not any(metric in v for v in data[REPORTED].values()):
            continue
        print(f'\n[{metric}] bootstrapping ...', flush=True)
        res = run_metric(data, metric, direction, rng, a.replicates, n_grid)
        if res is None:
            continue
        out['results'][metric] = res
        print_report(res)

    emit_build_schema(out)
    missing = [k for k in ('targets', 'metrics') if not out.get(k)]
    if missing:
        sys.exit(f'refusing to write: build schema incomplete, missing {missing}')

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print(f'\nsaved -> {a.json}')
    print(f'  build schema: targets={out["targets"]}, metrics=' +
          ', '.join(f'{m}({len(v["rows"])} rows)' for m, v in out['metrics'].items()))


if __name__ == '__main__':
    main()
