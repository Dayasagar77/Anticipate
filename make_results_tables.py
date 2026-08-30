#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_results_tables.py — turn the sweep's statistics into the manuscript's results tables.

WHY THIS EXISTS
---------------
Every number in Section VIII has to come from a result file, and it has to come from the SAME code
path that produced the statistics report, or the two will drift and one of them will be wrong. This
module therefore does not re-implement anything: it imports paper2_stats and consumes exactly the
frames that module computes (per_config_table, seed_level, episode_level, alpha_curve,
intervention_profile). Its only job is presentation --- rounding, ordering, display names, and
markdown table assembly --- plus a JSON dump of every scalar the prose needs, so that the pass which
edits the manuscript quotes numbers from a file rather than from a screen.

WHAT IT EMITS (into <outdir>)
-----------------------------
  results_tables.md     the markdown table blocks, ready to splice into Paper2_manuscript.md
  results_numbers.json  every scalar the prose quotes, keyed so an edit script can look it up

TABLE NUMBERING. The blocks are emitted as Tables VIII--XII. Table VIII already exists in the
manuscript and is REPLACED by the version here (same position, more rows and columns). Tables
IX--XII are new and are appended within Section VIII, after the existing Table VIII, so no existing
table, equation, figure or citation number moves.

PARTIAL DATA. The sweep may still be running. Every block degrades gracefully: a configuration with
no completed seeds is simply absent, and each table prints the seed count it was computed on so a
half-finished run can never be mistaken for a finished one. Nothing here is written into the
manuscript automatically --- that is a separate, explicit pass.
"""
import os, sys, io, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper2_stats as S

# Display names. The ladder order is the argument of Section VIII-B: start from the earlier design,
# add one component at a time, end at the reported controller.
LADDER = [
    ('p1_transplant', 'P1 transplant (2-D design, retrained here)'),
    ('no_ant',        '$-$ anticipation (reactive)'),
    ('no_cvar',       '$-$ CVaR (risk-neutral)'),
    ('no_per',        '$-$ prioritized replay (uniform)'),
    ('plus_lstm',     '$+$ LSTM (recurrent)'),
    ('model',         '**Reported: FF + anticipation + CVaR + PER**'),
]
BASE_ORDER = [
    ('random',       'Random'),
    ('static_hold',  'Static (no handover), hold'),
    ('static_eff',   'Static (no handover), eff'),
    ('maxsinr_hold', 'Max-SINR, hold'),
    ('maxsinr_eff',  'Max-SINR, eff'),
    ('a3_hold',      '3GPP A3, hold'),
    ('a3_eff',       '3GPP A3, eff'),
    # The two anticipatory rules sit here deliberately: both are deployable on exactly the
    # information the learned controller uses, so they belong on the near side of the genie, and
    # they are the last pre-genie entries. They are ordered anticipation-alone first, then the
    # union: antthr fires on the A3 event *or* the anticipatory one, so it contains A3 and can
    # never be weaker than the A3 row above it, while antonly is that same anticipatory trigger
    # with the disjunct removed. Keeping both on the page is what makes either legible --- a
    # comparison against the union alone cannot say whether the anticipatory half or the A3 half
    # did the work.
    ('antonly_hold', 'Anticipation-only threshold (tuned on train), hold'),
    ('antonly_eff',  'Anticipation-only threshold (tuned on train), eff'),
    ('antthr_hold',  'Anticipatory threshold (tuned on train), hold'),
    ('antthr_eff',   'Anticipatory threshold (tuned on train), eff'),
    # 'not deployable' moves from the row label to the caption: it is the same warning in the same
    # table, and as a label it forced a four-line cell that pushed every other column narrower.
    # Not 'Genie (upper bound)'. The genie is handed perfect blockage geometry but applies a
    # greedy rule to it, and on the held-out scenarios deployable rules come in below it on
    # sustained RLF --- so it bounds what perfect PERCEPTION buys under that rule, not what any
    # policy can achieve. Section VIII-H derives which policies beat it rather than asserting
    # that none can. No fix_r4b_numbers.py anchor matches these label strings.
    ('genie_hold',   'Genie (perfect geometry), hold'),
    ('genie_eff',    'Genie (perfect geometry), eff'),
]
METRIC_NAME = {'rlf_rate': 'RLF rate', 'cvar25': 'CVaR$_{25}$', 'return_mean': 'mean return',
               'rlf': 'per-scenario RLF', 'ret': 'per-scenario return'}


def f(x, n=1):
    """Round for display; an absent number prints as an em dash rather than as 'nan'.

    A negative value gets a real minus sign rather than an ASCII hyphen, which typesets short, sits
    at the wrong height, and in a narrow cell reads as a word broken across a line. The Unicode
    character is used rather than an inline '$-$' because pandoc will not close a math span on a
    '$' that is followed by a digit, so '$-$13.60' emerges as two literal dollars.
    """
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '---'
    s = f'{x:.{n}f}'
    return ('−' + s[1:]) if s.startswith('-') else s


def pfmt(p):
    """Display a p-value in a manuscript table.

    paper2_stats.fmt_p is the *diagnostic* formatter and prints the raw magnitude ('<1e-16',
    '1.4e-09') because when reading a statistics report one wants to see how far into the tail a
    test landed. That is the wrong thing to print in a paper. At n = 500 scenario pairs the Wilcoxon
    statistic is evaluated through a normal approximation, so a tail probability of 10^-16 is a
    property of that approximation and not a resolvable measurement; quoting it reads as false
    precision. Everything below one in a thousand is therefore reported as such, which is the
    convention the reader expects, and nothing above it is altered.
    """
    if p is None or not np.isfinite(float(p)):
        return '---'
    p = float(p)
    return '$<0.001$' if p < 1e-3 else f'{p:.3f}'


def pm(row, col, n=1):
    m, s = row.get(f'{col}_mean'), row.get(f'{col}_std')
    if m is None or not np.isfinite(m):
        return '---'
    if s is None or not np.isfinite(s):
        return f(m, n)
    return f'{f(m, n)} $\\pm$ {f(s, n)}'


def sep(*w):
    """A pipe-table separator row whose dash counts set the relative column widths.

    Pandoc derives a table's column fractions from the number of dashes under each column, so the
    usual `|---|---|` makes every column equally wide. That is the wrong shape for these tables: one
    column holds 'P1 transplant (2-D design, retrained here)' and the next holds '4'. Equal columns
    force TeX to hyphenate the row labels down to two-syllable fragments -- 'P1 trans- plant',
    '- priori- tized re- play' -- and still leave header words such as 'Counterfactual' hanging past
    the right margin, which is what the overfull-hbox audit of the reading copy found. The weights
    are the widths the content actually needs; they are relative, so their sum is arbitrary.
    """
    return '|' + '|'.join('-' * max(3, int(x)) for x in w) + '|'


# ------------------------------------------------------------------ Table VIII: the ablation ladder
def table8(pc):
    idx = pc.set_index('config')
    lines = ['| Controller | Seeds | Sustained RLF (%) $\\downarrow$ | CVaR$_{25}$ return '
             '$\\uparrow$ | Mean return $\\uparrow$ | In-sample RLF (%) | Gap (pp) |',
             sep(26, 7, 15, 14, 13, 13, 12)]
    got = []
    for cfg, disp in LADDER:
        if cfg not in idx.index:
            continue
        r = idx.loc[cfg]
        ins = r.get('insample_rlf_rate')
        gap = r.get('gap_rlf')
        lines.append(f'| {disp} | {int(r["n_seeds"])} | {pm(r, "rlf_rate")} | {pm(r, "cvar25")} '
                     f'| {pm(r, "return_mean")} | {f(ins)} | {f(gap)} |')
        got.append(cfg)
    return '\n'.join(lines), got


# --------------------------------------------------- Table IX: paired seed-level tests vs reported
def table9(sl):
    if not len(sl):
        return '_(no seed-level rows yet: fewer than three completed seeds per configuration)_', []
    keep = sl[sl.level == 'seed'].copy()
    lines = ['| Comparison (reported $-$ variant) | Metric | $n$ | $\\Delta$ mean [95% CI] | '
             'Hodges--Lehmann | $r_{\\mathrm{rb}}$ ($n_{\\mathrm{eff}}$) | $p$ (Holm) |',
             sep(24, 12, 5, 21, 14, 14, 10)]
    order = {c: i for i, (c, _) in enumerate(LADDER)}
    keep['_cfg'] = keep['comparison'].str.split(' vs ').str[-1]
    keep['_o'] = keep['_cfg'].map(order).fillna(99)
    keep = keep.sort_values(['_o', 'metric'])
    for _, r in keep.iterrows():
        disp = dict(LADDER).get(r['_cfg'], r['_cfg'])
        p = r.get('p_holm', r.get('p_adj', r.get('p_raw')))
        lines.append(
            f'| {disp} | {METRIC_NAME.get(r["metric"], r["metric"])} | {int(r["n"])} '
            f'| {f(r["diff_mean"], 2)} [{f(r["ci_lo"], 2)}, {f(r["ci_hi"], 2)}] '
            f'| {f(r["diff_hl"], 2)} '
            f'| {f(r["rb_corr"], 2)} ({int(r["n_eff"]) if np.isfinite(r["n_eff"]) else 0}) '
            f'| {pfmt(p)} |')
    return '\n'.join(lines), sorted(set(keep['_cfg']))


# ------------------------- Table X: reference-policy operating points, and every scenario-level test
def table10(base_m, pc, el):
    """Two panels under one table number.

    Panel (a) is the reference-policy operating table. It gains the in-sample rate and the
    generalization gap, which an earlier version dropped even though paper2_stats prints them. That
    omission mattered: the A3 'hold' policy carries a large held-out-minus-in-sample offset while
    having no learnable parameters at all, so a gap of that size cannot be evidence of a learned
    controller overfitting, and a table that shows the gap only for the learned rows invites exactly
    that reading. The reported controller is repeated on the last row so the comparison is on the
    page rather than two tables away.

    Panel (b) is every scenario-level comparison against the reported controller --- the five
    ablation variants as well as the nine reference policies. Section VIII-G sends the reader to
    'the scenario-level test of Table X' for the ablations, and Section VII promises that no
    comparison is dropped; an earlier version looped over the reference policies only, so the five
    rows the prose points at were computed and then never printed.

    The numbers in panel (b) are the scene-clustered ones. paper2_stats.episode_level puts the
    clustered reading in the canonical column names precisely so that a presentation layer cannot
    quote the naive interval by accident, and the design-effect column publishes how much narrower
    the naive interval would have been rather than quietly absorbing the correction.
    """
    if base_m is None or not len(base_m):
        return '_(baseline metrics not present)_', []
    b = base_m.groupby('config').mean(numeric_only=True)
    e = el[(el.level == 'episode') & (el.metric == 'rlf')] if len(el) else el
    ptab = {r['comparison'].split(' vs ')[-1]: r for _, r in e.iterrows()} if len(e) else {}

    # ---- panel (a): held-out operating points of the non-learned policies -----------------------
    lines = ['**(a) Reference-policy operating points on the held-out scenes.**', '',
             '| Reference policy | RLF (%) $\\downarrow$ | In-sample RLF (%) | Gap (pp) | '
             'CVaR$_{25}$ $\\uparrow$ | Mean return $\\uparrow$ | HO/ep. | $\\bar{P}_T$ (dBm) | '
             '$\\bar{B}$ (RB) |',
             sep(72, 40, 56, 44, 44, 38, 40, 36, 30)]
    got = []
    for cfg, disp in BASE_ORDER:
        if cfg not in b.index:
            continue
        r = b.loc[cfg]
        ins = r.get('insample_rlf_rate')
        gap = (r['rlf_rate'] - ins) if ins is not None and np.isfinite(ins) else float('nan')
        lines.append(f'| {disp} | {f(r["rlf_rate"])} | {f(ins)} | {f(gap)} | {f(r["cvar25"])} '
                     f'| {f(r["return_mean"])} | {f(r["ho_per_ep"], 2)} | {f(r["pt_dbm"], 1)} '
                     f'| {f(r["rb"], 1)} |')
        got.append(cfg)
    # the reported controller on the same columns, so the gap column is read against something
    pci = pc.set_index('config')
    if S.REF in pci.index:
        r = pci.loc[S.REF]
        lines.append(f'| **Reported controller** | {f(r.get("rlf_rate_mean"))} '
                     f'| {f(r.get("insample_rlf_rate"))} | {f(r.get("gap_rlf"))} '
                     f'| {f(r.get("cvar25_mean"))} | {f(r.get("return_mean_mean"))} '
                     f'| {f(r.get("ho_per_ep"), 2)} | {f(r.get("pt_dbm"), 1)} '
                     f'| {f(r.get("rb"), 1)} |')

    # ---- panel (b): every scenario-level comparison, scene-clustered ----------------------------
    lines += ['', '**(b) Scene-clustered scenario-level comparisons against the reported '
              'controller.**', '',
              '| Comparison (reported $-$ variant) | Scenes (untied) | '
              '$\\Delta$RLF (pp) [95% CI] | Design effect | HL (pp) | $p$ (Holm) | '
              'Smallest attainable $p$ |',
              sep(96, 44, 92, 40, 42, 44, 58)]

    def prow(key, disp):
        r = ptab.get(key)
        if r is None:
            return None
        nsc = int(r['n_scene']) if np.isfinite(r.get('n_scene', np.nan)) else 0
        ne = int(r['n_eff']) if np.isfinite(r.get('n_eff', np.nan)) else 0
        p = r.get('p_holm', r.get('p_adj', r.get('p_raw')))
        return (f'| {disp} | {nsc} ({ne}) '
                f'| {f(r["diff_mean"], 2)} [{f(r["ci_lo"], 2)}, {f(r["ci_hi"], 2)}] '
                f'| {f(r.get("deff"), 1)} | {f(r.get("diff_hl_scene"), 2)} '
                f'| {pfmt(p)} | {pfmt(r.get("p_floor_holm"))} |')

    for title, items in [('Ablation variants', [(c, d) for c, d in LADDER if c != S.REF]),
                         ('Reference policies', [(f'baseline:{c}', d) for c, d in BASE_ORDER])]:
        body = [x for x in (prow(k, d) for k, d in items) if x]
        if not body:
            continue
        lines.append(f'| *{title}* | | | | | | |')
        lines += body
        got += [k for k, _ in items if k in ptab]
    return '\n'.join(lines), got


# ------------------------------------------------------------- Table XI: CVaR-alpha sensitivity
def table11(ac):
    if not len(ac):
        return '_(alpha sweep not present)_', []
    lines = ['| Risk level $\\alpha$ | Seeds | Sustained RLF (%) [95% CI] | CVaR$_{25}$ return | '
             'Mean return |', sep(18, 8, 30, 22, 22)]
    for _, r in ac.iterrows():
        a = 'risk-neutral' if float(r['alpha']) >= 1.0 else f'{float(r["alpha"]):.2f}'
        lines.append(f'| {a} | {int(r["n_seeds"])} | {f(r["rlf_rate"])} '
                     f'[{f(r["rlf_ci_lo"])}, {f(r["rlf_ci_hi"])}] '
                     f'| {f(r["cvar25"])} | {f(r["return_mean"])} |')
    return '\n'.join(lines), list(ac['config'])


# ------------------------------------------------------- Table XII: where the advantage lives
def table12(ip):
    if not len(ip):
        return '_(intervention profile not present)_', []
    lines = ['| Configuration | Acted on (%) | Counterfactual RLF (%) | Achieved RLF (%) | '
             'Counterfactual return | Achieved return | RLF, not acted on (%) |',
             sep(79, 34, 86, 53, 86, 53, 34)]
    names = dict(LADDER)
    names.update({f'baseline:{k}': v for k, v in BASE_ORDER})
    # ladder order first (reported controller last, as in Table VIII), then the reference policies
    order = {c: i for i, (c, _) in enumerate(LADDER)}
    order.update({f'baseline:{c}': 100 + i for i, (c, _) in enumerate(BASE_ORDER)})
    ip = ip.copy()
    ip['_o'] = ip['config'].map(order).fillna(999)
    ip = ip.sort_values('_o')
    got = []
    for _, r in ip.iterrows():
        c = str(r['config'])
        lines.append(f'| {names.get(c, c)} | {f(100 * r["frac_active"])} | {f(r["a_rlf_idle"])} '
                     f'| {f(r["a_rlf"])} | {f(r["a_ret_idle"])} | {f(r["a_ret"])} '
                     f'| {f(r["q_rlf"])} |')
        got.append(c)
    return '\n'.join(lines), got


# ------------------------------------------------- Table XIII: what the three actuators actually do
def table13(pc, per_seed, ref):
    """Two panels: actuator means per configuration, and the reported controller seed by seed.

    Section VIII-L promises the mean transmit power, the mean resource-block allocation and the
    handovers per episode 'for every configuration and every reference policy'. Table X panel (a)
    already carries those three columns for the reference policies; this table supplies the other
    half of the promise, which no table carried before, and adds the per-seed panel.

    Panel (b) exists because the bandwidth branch is bimodal and a mean over seeds therefore
    describes no seed that was run. A referee asked to accept 'most seeds never leave the maximum'
    is entitled to see the ten numbers rather than a mean and an adjective, and printing them costs
    ten rows. It is also the honest way to present an exploration failure: the split is visible, so
    a reader can judge for themselves whether the two low-allocation seeds are a lucky corner or a
    reachable optimum the other eight missed.

    This table is numbered XIII and appended after XII precisely so that adding it renumbers
    nothing: every cross-reference in the manuscript to Tables VIII through XII stays valid.
    """
    idx = pc.set_index('config')
    lines = ['**(a) Actuator settings by configuration, averaged over seeds and held-out '
             'scenarios.**', '',
             '| Controller | $\\bar{P}_T$ (dBm) | $\\bar{B}$ (RB) | HO/ep. |', sep(46, 16, 14, 12)]
    got = []
    for cfg, disp in LADDER:
        if cfg not in idx.index:
            continue
        r = idx.loc[cfg]
        lines.append(f'| {disp} | {f(r.get("pt_dbm"), 2)} | {f(r.get("rb"), 1)} '
                     f'| {f(r.get("ho_per_ep"), 2)} |')
        got.append(cfg)

    d = per_seed[per_seed.config == ref] if len(per_seed) else per_seed
    if len(d) and all(c in d.columns for c in ('seed', 'pt_dbm', 'rb', 'ho_per_ep')):
        d = d.sort_values('seed')
        lines += ['', '**(b) The reported controller seed by seed. The bandwidth branch, and only '
                  'the bandwidth branch, splits into two groups.**', '',
                  '| Seed | $\\bar{P}_T$ (dBm) | $\\bar{B}$ (RB) | HO/ep. | Sustained RLF (%) | '
                  'Mean return |', sep(10, 16, 14, 12, 20, 16)]
        for _, r in d.iterrows():
            lines.append(f'| {int(r["seed"])} | {f(r["pt_dbm"], 2)} | {f(r["rb"], 1)} '
                         f'| {f(r["ho_per_ep"], 3)} | {f(r.get("rlf_rate"), 1)} '
                         f'| {f(r.get("return_mean"), 2)} |')
        got.append(f'{ref}:per_seed')
    return '\n'.join(lines), got


# ------------------------------------------- per-seed actuator settings for the reported controller
def actuator_profile(per_seed, ref):
    """Per-seed actuator settings for the reported controller, and whether they are bimodal.

    Section VIII-L makes a claim about *where* the controller's return deficit against the eff-mode
    reference rules comes from, and that claim rests on the per-seed spread of the bandwidth
    actuator rather than on its mean. A mean of 54 resource blocks is equally consistent with every
    seed choosing 54 and with most seeds never moving off the maximum while a few descend to the
    floor, and those two worlds call for opposite sentences. The split is therefore computed here,
    from the data, and the prose that quotes it is generated from this dict.

    The cut is placed at the largest gap in the sorted per-seed values rather than at a threshold we
    pick, and `bimodal` is set only when that gap actually dominates: at least three times the next
    largest, and at least a quarter of the full range. On a unimodal sample the flag is False and
    the prose that consumes it must not describe two corners --- which is why the flag exists rather
    than a bare cut point, since a cut point alone would always manufacture one.
    """
    cols = ['seed', 'rb', 'pt_dbm', 'ho_per_ep', 'return_mean', 'rlf_rate']
    d = per_seed[per_seed.config == ref]
    if not len(d) or any(c not in d.columns for c in cols):
        return {}
    d = d[cols].sort_values('seed')
    v = np.sort(d.rb.to_numpy(float))
    # The action-space BOUNDS travel with the measured settings rather than being retyped in the
    # prose generator. Section VIII-L says how far the controller is from its own ceiling, and a
    # distance is meaningless without the ceiling that defines it; carrying both in the same dict
    # means the sentence cannot survive a change to the environment that moves either one. The
    # values are read from paper2_env, which main() separately asserts against the numbers the
    # caption quotes, so there is exactly one place where these constants live.
    import paper2_env as _E
    out = {'config': ref, 'n_seeds': int(len(d)),
           'per_seed': d.round(4).to_dict(orient='records'),
           'rb_mean': float(d.rb.mean()), 'pt_mean': float(d.pt_dbm.mean()),
           'ho_mean': float(d.ho_per_ep.mean()),
           'rb_min': float(v[0]), 'rb_max': float(v[-1]), 'bimodal': False,
           'pt_bound_lo': float(_E.PT_MIN), 'pt_bound_hi': float(_E.PT_MAX),
           'rb_bound_lo': float(_E.RB_MIN), 'rb_bound_hi': float(_E.RB_MAX),
           'pt_step': float(_E.PT_STEP), 'rb_step': float(_E.RB_STEP)}
    gaps = np.diff(v)
    if len(gaps) < 2 or not np.isfinite(gaps).all() or gaps.max() <= 0:
        return out
    i = int(np.argmax(gaps))
    second, rng = float(np.sort(gaps)[-2]), float(v[-1] - v[0])
    out['bimodal'] = bool(gaps[i] >= 3.0 * max(second, 1e-9) and gaps[i] >= 0.25 * max(rng, 1e-9))
    cut = 0.5 * (v[i] + v[i + 1])
    lo, hi = d[d.rb <= cut], d[d.rb > cut]
    # Rank 1 is the highest-return seed. Section VIII-L quotes these ranks to say whether the seeds
    # that economize on bandwidth are also the ones that score best, which is the whole of the
    # argument that this branch costs return without buying reliability.
    order = d.sort_values('return_mean', ascending=False).seed.tolist()
    out.update(cut=float(cut), n_lo=int(len(lo)), n_hi=int(len(hi)),
               rb_lo=float(lo.rb.mean()), rb_hi=float(hi.rb.mean()),
               ret_lo=float(lo.return_mean.mean()), ret_hi=float(hi.return_mean.mean()),
               rlf_lo=float(lo.rlf_rate.mean()), rlf_hi=float(hi.rlf_rate.mean()),
               pt_lo=float(lo.pt_dbm.mean()), pt_hi=float(hi.pt_dbm.mean()),
               seeds_lo=sorted(int(s) for s in lo.seed),
               seeds_hi=sorted(int(s) for s in hi.seed),
               ret_rank_lo=sorted(order.index(int(s)) + 1 for s in lo.seed))
    return out


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/root/work/paper2/models/sweep_r3')
    ap.add_argument('--ref', default='model')
    a = ap.parse_args()
    S.REF = a.ref

    per_seed, per_ep, base_m, base = S.load(a.outdir)
    pc = S.per_config_table(per_seed)
    sl = S.seed_level(per_seed, a.ref)
    el = S.episode_level(per_ep, base, a.ref)
    ac = S.alpha_curve(pc)
    ip = S.intervention_profile(per_ep, base, a.ref)
    spread, srank = S.seed_lottery(per_seed)

    # The Table XIII caption and the Section VIII-L prose both assert that an episode begins with
    # both resource actuators at their maxima, and both quote those maxima. That is a claim about
    # paper2_env.reset(), not about anything in the result files, so it is checked against the
    # environment here rather than trusted. If a future change to the environment moves either bound
    # or the reset convention, this stops the build instead of publishing a stale sentence.
    import paper2_env as ENV
    assert (ENV.PT_MAX, ENV.RB_MAX, ENV.PT_MIN, ENV.RB_MIN) == (30.0, 66, 23.0, 10), \
        f'actuator bounds moved: {(ENV.PT_MAX, ENV.RB_MAX, ENV.PT_MIN, ENV.RB_MIN)}'
    assert (ENV.PT_STEP, ENV.RB_STEP) == (1.0, 8), f'actuator steps moved: {(ENV.PT_STEP, ENV.RB_STEP)}'
    _rs = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper2_env.py'),
                  encoding='utf-8').read()
    assert 'self.pt = PT_MAX' in _rs and 'self.rb = RB_MAX' in _rs, \
        'reset() no longer starts both resource actuators at their maxima'

    blocks, got = [], {}
    for num, title, (body, g) in [
        ('VIII', 'Ablation ladder on the held-out scenes (mean $\\pm$ std over seeds). The last '
                 'column is the generalization gap, the held-out minus in-sample sustained-RLF '
                 'rate, in percentage points.', table8(pc)),
        # paper2_stats.seed_level computes diff = reference - comparator, i.e. reported - variant.
        # The reported controller is therefore favoured by a NEGATIVE difference on RLF (it fails
        # less often) and by a POSITIVE one on the two return metrics. An earlier draft of this
        # caption stated both directions the wrong way round.
        ('IX',   'Paired seed-level comparisons against the reported controller; '
                 # No space before a closing '$': pandoc will not close an inline math span
                 # on a '$' preceded by whitespace, so '$\\Delta = $' is emitted as two
                 # literal dollar signs and xelatex then fails with 'Missing $ inserted'.
                 '$\\Delta =$ reported $-$ variant on the same seed. A negative $\\Delta$ '
                 'therefore favours the reported controller on the RLF rate and a positive '
                 '$\\Delta$ favours it on the two return metrics; $p$ is Holm--Bonferroni '
                 'corrected within each metric family. This level varies the training seed with '
                 'the held-out scene set held fixed, so it asks whether a design change survives '
                 'training randomness on these 15 scenes; Table X(b) varies the scene instead and '
                 'asks whether it survives unseen road geometry. The two answer different '
                 'questions and neither substitutes for the other.',
         table9(sl)),
        ('X',    'Held-out operating points and scenario-level comparisons. (a) The non-learned '
                 'reference policies on the same held-out scenarios under common random numbers, '
                 'with the reported controller repeated on the last row; the gap column is the '
                 'held-out minus the in-sample sustained-RLF rate, in percentage points, and is '
                 'shown for the reference policies as well as the learned ones because a policy '
                 'with no learnable parameters can carry a gap of the same size. The genie rows are not '
                 'deployable and are not an upper bound: that policy is given the blockage the '
                 'controller has to infer but applies a greedy rule to it, and policies that see '
                 'strictly less come in below it here. (b) Every comparison against the reported '
                 'controller at the scenario level, as reported $-$ variant, so a negative '
                 '$\\Delta$ favours the reported controller. The held-out scenarios are repeated '
                 'resets over the same 15 held-out scenes and are therefore not independent draws: '
                 'the interval is a scene-clustered bootstrap, the test is an exact signed-rank '
                 'test on the 15 scene-mean differences, HL is the Hodges--Lehmann pseudomedian of '
                 'those differences, and the design effect is the ratio of the clustered to the '
                 'unclustered variance of the same mean. Scenes (untied) gives the number of '
                 'clusters and, in parentheses, how many of them are not exactly tied; the last '
                 'column is the smallest Holm-adjusted $p$ the comparison could attain at that '
                 'number of untied scenes, so a value above 0.05 means the design cannot resolve '
                 'an effect of any size, which is not the same as an absent effect.',
         table10(base_m, pc, el)),
        ('XI',   'Sensitivity to the CVaR risk level $\\alpha$.', table11(ac)),
        ('XII',  'Where the advantage lives. A scenario is *acted on* when the configuration issues '
                 'at least one handover; the counterfactual columns are what a never-hand-over '
                 'policy scores on those same scenarios. Which never-hand-over policy is matched '
                 'to the row\'s own resource regime, because the reward of Eq. (15) charges for '
                 'transmit power and for bandwidth and the two idle policies do not spend alike: '
                 'a rule tuned for efficiency is scored against the efficient idle policy and a '
                 'rule that holds the ceiling against the holding one, so that the counterfactual '
                 'return difference is the value of intervening and not the value of a cheaper '
                 'operating point. The two static policies are therefore the references for this '
                 'table and are not rows in it. The learned configurations have no idle '
                 'counterpart at their own operating point and are scored against the holding '
                 'idle policy; the residual resource-charge difference this leaves in their '
                 'counterfactual return column is reported in the text of Section VIII-J, and '
                 'per row in `stats_intervention.csv`, rather than absorbed into the number.',
         table12(ip)),
        ('XIII', 'What the three actuators actually do. Every episode begins with both resource '
                 'actuators at their maxima (30 dBm and 66 resource blocks), so economizing on '
                 'either is something the policy must actively do and holding the ceiling is the '
                 'default; the corresponding columns for the non-learned reference policies are in '
                 'Table X(a).', table13(pc, per_seed, a.ref)),
    ]:
        blocks.append(f'**Table {num}. {title}**\n\n{body}\n')
        got[num] = g

    md = ('<!-- generated by make_results_tables.py from the sweep result files; do not hand-edit '
          '-->\n\n' + '\n\n'.join(blocks) + '\n')
    mp = os.path.join(a.outdir, 'results_tables.md')
    open(mp, 'w').write(md)

    # ---- every scalar the prose needs, so the manuscript pass quotes a file, not a screen -------
    num = {'n_seeds': {r['config']: int(r['n_seeds']) for _, r in pc.iterrows()},
           'per_config': pc.round(4).to_dict(orient='records'),
           'seed_level': sl.round(5).to_dict(orient='records') if len(sl) else [],
           'episode_level': el.round(5).to_dict(orient='records') if len(el) else [],
           'alpha': ac.round(4).to_dict(orient='records') if len(ac) else [],
           'intervention': ip.round(4).to_dict(orient='records') if len(ip) else [],
           # The prose also quotes the reference policies (Section VIII-H, and the A3 cross-check in
           # Section IV-A) and the per-configuration seed spread (Section VIII-K). They are dumped
           # here for the same reason as everything else: so the manuscript pass reads a file.
           'baselines': (base_m.round(4).to_dict(orient='records')
                         if base_m is not None and len(base_m) else []),
           'seed_spread': (spread.round(4).to_dict(orient='records') if len(spread) else []),
           'seed_rank': (srank.round(4).to_dict(orient='records') if len(srank) else []),
           # Section VIII-L reports what the three actuators actually do, and the bandwidth branch
           # is only legible per seed: its mean over seeds describes no seed at all when the branch
           # is bimodal. Section VIII-H points at this subsection for that split, so it has to be
           # here rather than left to a reader's inference from a mean.
           'actuators': actuator_profile(per_seed, a.ref),
           'tables_populated': got}
    if a.ref in set(pc['config']):
        m = pc.set_index('config').loc[a.ref]
        num['latency_ms'] = {k: (float(m[k]) if k in m and np.isfinite(m[k]) else None)
                             for k in ('lat_mean_ms', 'lat_p95_ms', 'lat_p99_ms')}
    jp = os.path.join(a.outdir, 'results_numbers.json')
    json.dump(num, open(jp, 'w'), indent=1, default=float)

    print(md)
    print(f'saved -> {mp}\nsaved -> {jp}')
    print('seeds per config:', num['n_seeds'])


if __name__ == '__main__':
    main()
