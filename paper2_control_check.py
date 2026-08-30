#!/usr/bin/env python3
"""Re-derive, from result files only, the four findings that the control experiment turned up.

This script writes nothing into any result directory. It reads `*_eval.csv` and prints a report,
so it can be run at any time without disturbing the sweep, the published numbers, or the manuscript
build. It exists because the memo `Paper2_ControlExperiment_Findings_20260726.md` was written at
14--15 seeds and explicitly labelled interim, and the decision it asks for should rest on the
pre-registered 20-seed reading rather than on a mid-sweep one.

The anticipation-only control (`antonly_hold`) is the rule the paper's Section VIII-H comparison was
missing: `paper2_lookahead_baseline.LookAheadPolicy` fires on the 3GPP A3 event OR the anticipatory
one, so `antthr_hold` is a union of a reactive rule and an anticipatory rule and cannot isolate
anticipation. `paper2_antonly_baseline.py` is the same rule with the A3 disjunct removed, tuned on
the training scenes only and evaluated once on the held-out 500 under the same common random
numbers.

Every quantity below is measured. Nothing is projected, and nothing is carried over from the memo.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper2_stats as S                                     # noqa: E402  (path set above)

# A configuration is judged degenerate when its handover branch has collapsed rather than learned a
# worse policy. Episodes average ~98.6 steps, so more than ten handovers per episode is one per
# fewer than ten steps: not a policy, a saturated branch. Zero handovers across all 500 held-out
# scenarios is the other collapse. Both thresholds are stated here rather than inline so that the
# report can print them and a reader can disagree with them explicitly.
THRASH_HO_PER_EP = 10.0
NEVER_HO_PER_EP = 1e-3

LADDER = ['model', 'no_ant', 'no_cvar', 'no_per', 'plus_lstm', 'p1_transplant']


def cvar_of(x, alpha=0.25):
    """Mean of the worst alpha-fraction of scenario returns. Matches compare_lookahead.cvar_of."""
    x = np.sort(np.asarray(x, float))
    k = max(1, int(round(alpha * len(x))))
    return float(x[:k].mean())


def seed_of(path):
    return int(re.search(r'_s(\d+)_eval', path).group(1))


def seed_files(outdir, cfg):
    return sorted(glob.glob(os.path.join(outdir, f'{cfg}_s*_eval.csv')), key=seed_of)


def find_baseline(name, dirs):
    """First directory carrying this baseline's held-out eval, so a control run that has not yet
    been copied into the sweep directory is still found — and so that once it HAS been copied in,
    the sweep directory wins and the provisional copy stops being read."""
    for d in dirs:
        p = os.path.join(d, f'baseline_{name}_eval.csv')
        if os.path.exists(p):
            return p, d
    return None, None


def policy_row(df):
    """Held-out summary of one policy over its scenarios."""
    return dict(rlf=100.0 * float(df['rlf'].mean()),
                ho=float(df['n_ho'].mean()),
                ret=float(df['ret'].mean()),
                cvar25=cvar_of(df['ret'].to_numpy()),
                n=len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='models/sweep_r3')
    ap.add_argument('--antonly_dir', default='models/sweep_r3_antonly_PROVISIONAL',
                    help='searched only if the sweep directory does not carry antonly itself')
    ap.add_argument('--expect_seeds', type=int, default=0,
                    help='refuse unless every ladder rung has exactly this many seeds (0 = no gate)')
    a = ap.parse_args()

    D = a.outdir
    dirs = [D, a.antonly_dir]

    counts = {c: len(seed_files(D, c)) for c in LADDER}
    print('seeds on disk:', '  '.join(f'{c}:{n}' for c, n in counts.items()))
    if a.expect_seeds:
        bad = {c: n for c, n in counts.items() if n != a.expect_seeds}
        if bad:
            raise SystemExit(f'REFUSING TO REPORT: --expect_seeds {a.expect_seeds} was requested but '
                             f'{bad} do not match. A comparison drawn across unequal seed counts '
                             f'weights the configurations differently and is not the pre-registered '
                             f'analysis.')
    print()

    # ---- 1. the policy table -------------------------------------------------------------------
    # Reference policies are deterministic given the common random numbers, so they carry no seed
    # dimension; the learned configurations are averaged over seeds AFTER each seed is summarised,
    # never by pooling episodes across seeds.
    rows = []
    for name in ['antonly_hold', 'antthr_hold', 'genie_hold', 'a3_hold', 'static_hold',
                 'maxsinr_hold']:
        p, src = find_baseline(name, dirs)
        if p is None:
            print(f'  (skip {name}: no held-out eval found in {dirs})')
            continue
        r = policy_row(pd.read_csv(p))
        r.update(policy=name, kind='reference', nseed=1,
                 src=os.path.basename(os.path.dirname(p)) if src != D else 'sweep')
        rows.append(r)

    per_seed = {}
    for cfg in LADDER:
        fs = seed_files(D, cfg)
        if not fs:
            continue
        rs = [dict(seed=seed_of(f), **policy_row(pd.read_csv(f))) for f in fs]
        per_seed[cfg] = pd.DataFrame(rs).set_index('seed').sort_index()
        m = per_seed[cfg]
        rows.append(dict(policy=cfg, kind='learned', nseed=len(m), src='sweep',
                         rlf=float(m['rlf'].mean()), ho=float(m['ho'].mean()),
                         ret=float(m['ret'].mean()), cvar25=float(m['cvar25'].mean()),
                         n=int(m['n'].iloc[0])))

    T = pd.DataFrame(rows).sort_values('rlf').reset_index(drop=True)
    print('--- held-out performance, all policies on the same scenarios ---')
    print('    (learned rows are the mean over seeds of each seed\'s own held-out summary)')
    print(T[['policy', 'kind', 'nseed', 'rlf', 'ho', 'ret', 'cvar25', 'n', 'src']]
          .to_string(index=False,
                     formatters={'rlf': '{:.2f}'.format, 'ho': '{:.3f}'.format,
                                 'ret': '{:.2f}'.format, 'cvar25': '{:.2f}'.format}))
    print()

    # ---- 2. the paired comparison against the control ------------------------------------------
    ap_path, _ = find_baseline('antonly_hold', dirs)
    if ap_path is None or 'model' not in per_seed:
        print('--- paired comparison skipped: control or model seeds absent ---\n')
    else:
        A = pd.read_csv(ap_path)
        a_rlf = 100.0 * float(A['rlf'].mean())
        m = per_seed['model']['rlf']
        win = int((m < a_rlf - 1e-9).sum())
        tie = int((np.abs(m - a_rlf) <= 1e-9).sum())
        loss = int((m > a_rlf + 1e-9).sum())
        d_seed = (m - a_rlf).to_numpy()

        # Two intervals, because they answer two different questions and disagreeing about which one
        # to report is not a reason to report neither. The seed-level interval asks how much of the
        # gap is training-run noise: the unit resampled is the seed, and the control contributes no
        # variance because it has no seeds. The scenario-level interval asks whether the gap holds
        # across road geometry: it pairs each seed's per-scenario outcomes against the control's on
        # the same scenario index and resamples whole SCENES, since scenarios sharing a scene share
        # their blocker population and are not independent draws.
        rng = np.random.default_rng(20260726)
        bs = rng.choice(d_seed, size=(10000, len(d_seed)), replace=True).mean(axis=1)
        lo_s, hi_s = np.percentile(bs, [2.5, 97.5])

        Ai = A.set_index('idx').sort_index()
        diffs, clusters = [], []
        for f in seed_files(D, 'model'):
            g = pd.read_csv(f).set_index('idx').sort_index()
            common = g.index.intersection(Ai.index)
            diffs.append(100.0 * (g.loc[common, 'rlf'].to_numpy() - Ai.loc[common, 'rlf'].to_numpy()))
            clusters.append(Ai.loc[common, 'scene'].to_numpy())
        d_ep = np.concatenate(diffs)
        cl_ep = np.concatenate(clusters)
        mn_e, lo_e, hi_e, p_e, deff = S.boot_ci_cluster(d_ep, cl_ep)

        print('--- reported controller vs the anticipation-only control, on sustained RLF ---')
        print(f'    control (antonly_hold, deterministic): {a_rlf:.3f}%')
        print(f'    controller, per seed: n={len(m)}  min={m.min():.3f}%  '
              f'median={m.median():.3f}%  mean={m.mean():.3f}%  max={m.max():.3f}%')
        print(f'    seeds beating the control: {win}   tying: {tie}   losing: {loss}')
        print(f'    seed-level paired mean difference: {d_seed.mean():+.3f} pp   '
              f'95% bootstrap CI over seeds [{lo_s:+.3f}, {hi_s:+.3f}]')
        print(f'    scenario-level paired mean difference: {mn_e:+.3f} pp   '
              f'95% scene-clustered CI [{lo_e:+.3f}, {hi_e:+.3f}]   '
              f'p={p_e:.4g}   design effect {deff:.2f}')
        print('    (a positive difference is the controller failing MORE often than the hand rule)')
        for k in ('ho', 'ret', 'cvar25'):
            av = policy_row(A)[k]
            mv = float(per_seed['model'][k].mean())
            label = {'ho': 'handovers/episode', 'ret': 'mean return', 'cvar25': 'CVaR25'}[k]
            print(f'    {label:<20} control {av:8.3f}   controller {mv:8.3f}   '
                  f'difference {mv - av:+.3f}')
        print()

    # ---- 3. behavioural modes ------------------------------------------------------------------
    print(f'--- per-seed behavioural mode  (never: <{NEVER_HO_PER_EP} HO/ep; '
          f'thrash: >{THRASH_HO_PER_EP:.0f} HO/ep) ---')
    print(f'{"config":<15}{"never":>7}{"thrash":>8}{"normal":>8}{"degenerate":>13}'
          f'   normal-seed HO/ep range')
    for cfg in LADDER:
        if cfg not in per_seed:
            continue
        hs = per_seed[cfg]['ho'].to_numpy()
        never = int((hs < NEVER_HO_PER_EP).sum())
        thrash = int((hs > THRASH_HO_PER_EP).sum())
        normal = len(hs) - never - thrash
        nm = hs[(hs >= NEVER_HO_PER_EP) & (hs <= THRASH_HO_PER_EP)]
        rng_s = f'{nm.min():.3f}-{nm.max():.3f}' if len(nm) else '--'
        print(f'{cfg:<15}{never:>7}{thrash:>8}{normal:>8}{never + thrash:>8}/{len(hs):<4}   {rng_s}')
    print()

    # ---- 4. where the handover-budget gap comes from -------------------------------------------
    ho = {}
    for name in ['a3_hold', 'antthr_hold', 'antonly_hold']:
        p, _ = find_baseline(name, dirs)
        if p is not None:
            ho[name] = float(pd.read_csv(p)['n_ho'].mean())
    if 'model' in per_seed:
        ho['model'] = float(per_seed['model']['ho'].mean())
    if {'a3_hold', 'antthr_hold', 'model'} <= set(ho):
        gap = ho['antthr_hold'] - ho['model']
        print('--- decomposition of the handover-budget gap over the union rule ---')
        for k in ['a3_hold', 'antthr_hold', 'antonly_hold', 'model']:
            if k in ho:
                print(f'    {k:<14} {ho[k]:.3f} handovers/episode')
        print(f'    antthr_hold - model = {gap:.3f}/ep, of which the embedded A3 disjunct is '
              f'{ho["a3_hold"]:.3f} ({100.0 * ho["a3_hold"] / gap:.1f}%)')
        if 'antonly_hold' in ho:
            print(f'    against the anticipation-only control the saving is '
                  f'{ho["antonly_hold"] - ho["model"]:+.3f}/ep')
        print()

    # ---- 5. how the headline ratio moves with seed count ----------------------------------------
    if 'model' in per_seed and 'no_ant' in per_seed:
        p, _ = find_baseline('static_hold', dirs)
        base = 100.0 * float(pd.read_csv(p)['rlf'].mean()) if p else float('nan')
        r = per_seed['model']['rlf'].to_numpy()
        print('--- prefix means: how the reported controller\'s RLF moved as seeds were added ---')
        for n in sorted({5, 10, 15, len(r)}):
            if 0 < n <= len(r):
                mu = float(np.mean(r[:n]))
                print(f'    first {n:2d} seeds: mean RLF {mu:.3f}%   '
                      f'(vs never-hand-over {base:.1f}% -> {base / mu:.1f}x)')
        print('    NOTE: prefix means are shown to expose drift, not to license stopping at a '
              'favourable prefix. PREREGISTRATION.md 4.3 fixes the analysis at exactly 20 seeds.')
        print()


if __name__ == '__main__':
    main()
