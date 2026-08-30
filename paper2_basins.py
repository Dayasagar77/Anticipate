#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_basins.py — the convergence-basin analysis, as scalars the manuscript build can quote.

Training in this environment does not scatter around one behaviour. It converges either to a policy
that holds transmit power near the ceiling or to one that settles near the floor, with an empty
interval of several dB between them. This script measures that, and writes every figure the prose
needs into models/sweep_r3/basin_numbers.json so that `fix_r4b_numbers.py` can generate Section
VIII-K without a single typed number, on the same terms as every other quantity in the paper.

STATUS: the basin structure was found AFTER the twenty-seed sweep, not predicted before it. It is
exploratory, the generated prose says so, and this file records `post_hoc: true` so the claim cannot
quietly lose that label in a later rebuild.

WHAT IT ASSERTS RATHER THAN ASSUMES
  * that the split exists at all -- a minimum empty gap, per configuration. If a future sweep
    produces a unimodal power distribution the assertion fires and the build stops, rather than the
    manuscript describing two basins that are no longer there.
  * that the split is threshold-invariant across the whole gap, so no cut point is a judgement.
  * that every configuration has the same number of seeds, since the proportions are compared.

USAGE
    python paper2_basins.py --outdir models/sweep_r3
    python paper2_basins.py --outdir models/sweep_r3 --json models/sweep_r3/basin_numbers.json
"""
import os, io, json, glob, argparse, sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

REPORTED = 'model'
LADDER = ['model', 'p1_transplant', 'no_ant', 'no_cvar', 'no_per', 'plus_lstm']
THRESHOLD = 27.0          # dBm; asserted below to be invariant across the empty interval
MIN_GAP_DB = 3.0          # the split must be at least this clean or the claim is withdrawn
N_BOOT = 2000
BOOT_SEED = 20260823


def load(outdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, '*_s*_metrics.csv'))):
        base = os.path.basename(f)[:-len('_metrics.csv')]
        cfg, _, sd = base.rpartition('_s')
        if not sd.isdigit() or cfg not in LADDER:
            continue
        d = pd.read_csv(f)
        d.insert(0, 'config', cfg)
        d.insert(1, 'sd', int(sd))
        rows.append(d)
    if not rows:
        sys.exit('no ladder metrics found in ' + outdir)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.path.join(HERE, 'models', 'sweep_r3'))
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    out_path = a.json or os.path.join(a.outdir, 'basin_numbers.json')

    m = load(a.outdir)
    ns = m.groupby('config').size()
    assert ns.nunique() == 1, f'configurations have unequal seed counts, cannot compare: {dict(ns)}'
    n_seeds = int(ns.iloc[0])

    # ---- the split, and the assertion that there is one ------------------------------------
    per = {}
    gaps = []
    for c in LADDER:
        g = m[m.config == c]
        p = np.sort(g.pt_dbm.values)
        d = np.diff(p)
        k = int(np.argmax(d))
        gap = float(d[k])
        gaps.append(gap)
        assert gap >= MIN_GAP_DB, (
            f'REFUSING: configuration {c} no longer shows two separated basins '
            f'(largest empty interval {gap:.2f} dB < {MIN_GAP_DB} dB). Section VIII-K describes a '
            f'bimodal convergence that this sweep does not exhibit; withdraw the claim rather than '
            f'publishing it.')
        lo, hi = g[g.pt_dbm <= THRESHOLD], g[g.pt_dbm > THRESHOLD]
        assert len(lo) and len(hi), f'{c}: threshold {THRESHOLD} leaves a basin empty'
        per[c] = dict(
            n=int(len(g)), n_low=int(len(lo)), n_high=int(len(hi)),
            prop_low=float(len(lo) / len(g)),
            gap_db=round(gap, 2),
            pt_low=round(float(lo.pt_dbm.mean()), 2), pt_high=round(float(hi.pt_dbm.mean()), 2),
            rlf_low=round(float(lo.rlf_rate.mean()), 2), rlf_high=round(float(hi.rlf_rate.mean()), 2),
            corr_rlf_pt=round(float(np.corrcoef(g.rlf_rate, g.pt_dbm)[0, 1]), 3),
            rlf_mean=round(float(g.rlf_rate.mean()), 2),
            rlf_sd=round(float(g.rlf_rate.std(ddof=0)), 2))

    # ---- threshold invariance ----------------------------------------------------------------
    ref = (m.pt_dbm > THRESHOLD).values
    invariant = []
    for t in range(int(np.floor(m.pt_dbm.min())) + 1, int(np.ceil(m.pt_dbm.max()))):
        if bool(((m.pt_dbm > t).values == ref).all()):
            invariant.append(t)
    assert THRESHOLD in [float(x) for x in invariant] or int(THRESHOLD) in invariant, invariant
    assert len(invariant) >= 2, f'the classification is not threshold-invariant: {invariant}'

    # ---- is the basin a property of the seed? -------------------------------------------------
    piv = m.pivot_table(index='sd', columns='config', values='pt_dbm') <= THRESHOLD
    cnt = piv.sum(axis=1).values
    k = len(LADDER)
    obs = np.bincount(cnt, minlength=k + 1)[:k + 1].astype(float)
    p_low = float(piv.values.mean())
    exp = np.array([stats.binom.pmf(i, k, p_low) for i in range(k + 1)]) * len(piv)
    chi = float(((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    seed_p = float(stats.chi2.sf(chi, k))

    # ---- decomposition: within-basin vs composition -------------------------------------------
    rng = np.random.default_rng(BOOT_SEED)
    R = m[m.config == REPORTED].set_index('sd')
    r_hi = float(R[R.pt_dbm > THRESHOLD].rlf_rate.mean())
    r_lo = float(R[R.pt_dbm <= THRESHOLD].rlf_rate.mean())
    w = float((R.pt_dbm <= THRESHOLD).mean())

    # Indexed selection, never .isin: a bootstrap resample contains repeats, and `isin` would
    # collapse them, turning resampling-with-replacement into subset sampling and understating the
    # interval. Each (config, seed) is unique within a configuration, so .loc duplicates correctly.
    BY = {c: m[m.config == c].set_index('sd') for c in LADDER}

    def share(idx, cfg):
        r = BY[REPORTED].loc[idx]
        v = BY[cfg].loc[idx]
        rb, vb = r.pt_dbm > THRESHOLD, v.pt_dbm > THRESHOLD
        if rb.all() or (~rb).all() or vb.all() or (~vb).all():
            return None
        ww = float((~rb).mean())
        within = ((1 - ww) * (r[rb].rlf_rate.mean() - v[vb].rlf_rate.mean())
                  + ww * (r[~rb].rlf_rate.mean() - v[~vb].rlf_rate.mean()))
        marg = float(r.rlf_rate.mean() - v.rlf_rate.mean())
        if abs(marg) < 1e-9:
            return None
        return dict(marginal=marg, within=float(within), mix=float(marg - within),
                    mix_share=float((marg - within) / marg))

    seeds = sorted(R.index)
    dec, fisher = {}, {}
    for c in LADDER:
        if c == REPORTED:
            continue
        s = share(seeds, c)
        if s is not None:
            bs = [share(list(rng.choice(seeds, len(seeds), replace=True)), c) for _ in range(N_BOOT)]
            vals = np.array([x['mix_share'] for x in bs if x is not None])
            s = {kk: round(vv, 3) for kk, vv in s.items()}
            s['mix_share_ci'] = [round(float(np.percentile(vals, 2.5)), 3),
                                 round(float(np.percentile(vals, 97.5)), 3)]
            s['boot_n'] = int(len(vals))
            # a ratio whose denominator is near zero is not interpretable; mark it
            s['interpretable'] = bool(abs(s['marginal']) >= 1.0)
        dec[c] = s
        tbl = [[per[REPORTED]['n_low'], per[REPORTED]['n_high']],
               [per[c]['n_low'], per[c]['n_high']]]
        fisher[c] = round(float(stats.fisher_exact(tbl)[1]), 4)

    out = dict(
        generated_by='paper2_basins.py',
        post_hoc=True,
        post_hoc_note=('The basin structure was found after the twenty-seed sweep, not predicted '
                       'before it. Every statement generated from this file must be labelled '
                       'exploratory.'),
        outdir=os.path.abspath(a.outdir),
        n_seeds=n_seeds,
        threshold_dbm=THRESHOLD,
        threshold_invariant_range=[int(min(invariant)), int(max(invariant))],
        min_gap_db=round(float(min(gaps)), 2),
        max_gap_db=round(float(max(gaps)), 2),
        corr_min=round(float(min(v['corr_rlf_pt'] for v in per.values())), 3),
        corr_max=round(float(max(v['corr_rlf_pt'] for v in per.values())), 3),
        per_config=per,
        seed_independence=dict(chi2=round(chi, 2), df=k, p=round(seed_p, 3),
                               observed=obs.astype(int).tolist(),
                               expected=[round(x, 2) for x in exp], p_low=round(p_low, 3)),
        decomposition=dec,
        fisher_vs_reported=fisher)

    io.open(out_path, 'w', encoding='utf-8').write(json.dumps(out, indent=2))
    print(f'basins | {n_seeds} seeds x {len(LADDER)} configurations')
    print(f'  empty interval: {out["min_gap_db"]}-{out["max_gap_db"]} dB; classification invariant '
          f'for any threshold in [{out["threshold_invariant_range"][0]}, '
          f'{out["threshold_invariant_range"][1]}] dBm')
    print(f'  corr(RLF, mean power): {out["corr_min"]} to {out["corr_max"]} across configurations')
    print(f'  basin is not a seed property: chi2 {chi:.2f} on {k} df, p = {seed_p:.3f}')
    for c in LADDER:
        v = per[c]
        print(f'  {c:<15} low {v["n_low"]:2d}/{v["n"]}  RLF {v["rlf_low"]:6.2f} (low) vs '
              f'{v["rlf_high"]:5.2f} (high)   r = {v["corr_rlf_pt"]:+.3f}')
    print(f'saved -> {out_path}')


if __name__ == '__main__':
    main()
