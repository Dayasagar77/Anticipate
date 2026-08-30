#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_antonly_baseline.py --- the anticipation trigger ALONE, with the A3 disjunct removed.

WHY THIS FILE EXISTS
--------------------
`paper2_lookahead_baseline.py` defines the anticipatory threshold rule as a UNION:

    hand over if (3GPP event-A3 fires)  OR  (the anticipatory condition fires)

That union is the right construction for the RELIABILITY comparison, and the manuscript says so:
a rule that contains A3 can never be less reliable than A3, so beating it on sustained RLF is a
strictly harder test than beating A3. But it is exactly the wrong construction for the SIGNALLING
comparison, and the two were being read off the same policy. Measured on the held-out set:

    a3_hold      3.840 handovers/episode      <-- the embedded reactive rule
    antthr_hold  4.192 handovers/episode      <-- A3 OR anticipation
    model        0.304 handovers/episode      <-- the learned controller

Of the 3.888 handover gap between the learned controller and the anticipatory rule, 3.840 --- 98.8%
--- is the reactive rule the anticipatory rule contains by construction. The anticipation trigger
itself adds 0.352 handovers per episode on top of A3, which is the same order as the controller's
own 0.304. A referee reading the manuscript's signalling claim against the code will see that the
comparison is really "learned controller versus 3GPP A3 with extra steps", and that the obvious
control --- the anticipatory condition WITHOUT the reactive disjunct --- was never run.

This file runs it. It is a reference policy, not a ladder rung: PREREGISTRATION.md §4.4 fixes the
seed-level Holm family at 7 and that is not touched. This enters the episode-level reference-policy
family, which grows accordingly, and the family is NOT split to protect anything.

WHAT IT ISOLATES, AND WHAT IT STILL DOES NOT
--------------------------------------------
Removing the A3 disjunct also removes an information asymmetry that the union had and that the
manuscript did not disclose. A3 reads the instantaneous SINR of ALL FOUR links
(paper2_baselines.A3Policy.ho); the learned controller's 11-dimensional observation carries only the
SERVING link's SINR and its margin (paper2_env._obs). So the union rule was reading strictly more
instantaneous channel state than the controller it was being compared against, in the opposite
direction from the anticipation features. `antonly` reads no SINR at all in `hold` mode.

It still does not read the whole anticipation state. Two policies are provided:

  antonly --- the three features the original rule used: the serving link's predicted time-to-blockage
      `ttb`, its predicted knife-edge depth `projdepth`, and the best alternative link's `cand_ttb`.
      This is the minimal edit to the published rule, so the handover-count decomposition above is
      attributable to the removal of A3 and to nothing else.

  antfull --- the same, plus optional conjuncts on the two features the original rule never read:
      corridor occupancy `ncorr` and closing rate `closing`. Both grids include an "ignore this
      feature" level, so selection on the training scenes can decline them; if it does, that is
      evidence the two features carry nothing a threshold rule can use, which is itself worth
      reporting. With both declined, antfull is antonly. This is the control for the claim that the
      hand rule is deployable "on exactly the information the learned controller consumes".

PROTOCOL --- identical to every other reference policy, deliberately
-------------------------------------------------------------------
Thresholds are chosen by exhaustive grid search on the TRAINING scenes only, maximizing CVaR_0.25 of
episode return --- the same objective the controller optimizes and the same 30 scenes it trains on.
The held-out scenes are scored exactly once, after selection is frozen, under the same common random
numbers (EVAL_SEED_BASE) as every learned configuration. The full grid is written to disk so the
selection is auditable rather than asserted. Both resource modes are run: `hold` pins power and
bandwidth at their maxima, `eff` runs conventional closed-loop power control at minimum bandwidth.

INTEGRITY: no number here is copied from any other paper, and no threshold is chosen on test.
"""
import os, sys, time, json, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import paper2_baselines as BL

# --- the grid searched on the TRAINING scenes -------------------------------------------------
# TAU/DEPTH/MARGIN are the grids of paper2_lookahead_baseline.py, unchanged, so that antonly and
# antthr differ in the A3 disjunct and in nothing else.
TAU_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]     # s; 3.0 = the full anticipation horizon
DEPTH_GRID = [1.0, 5.0, 10.0, 20.0]                  # dB of predicted knife-edge loss
MARGIN_GRID = [0.0, 0.5]                             # s the alternative must stay clear beyond serving

# The two features the published rule never read. `None` is the "ignore this feature" level and is
# what makes antfull a superset of antonly rather than a different rule.
NCORR_GRID = [None, 1.0, 3.0]                        # require at least this many corridor vehicles
CLOSING_GRID = [None, -1.0, -3.0]                    # require approach at least this fast (m/s, <0)


class AntOnlyPolicy(BL.Policy):
    """The anticipatory condition, with no reactive disjunct and no SINR read in `hold` mode.

    Subclasses Policy rather than A3Policy on purpose: inheriting A3Policy and merely declining to
    call super().ho() would leave the reactive machinery one edit away from being reintroduced, and
    the whole point of this policy is that the reactive rule is absent.
    """
    base = 'antonly'

    def __init__(self, mode='hold', tau=1.0, depth=1.0, margin=0.0,
                 ncorr=None, closing=None, **kw):
        super().__init__(mode)
        self.tau, self.depth, self.margin = float(tau), float(depth), float(margin)
        self.ncorr = None if ncorr is None else float(ncorr)
        self.closing = None if closing is None else float(closing)
        self.name = f'{self.base}_{mode}'

    def ho(self, env, obs):
        g = env.frames[env.t]['per_g'][env.serv]
        if g['ttb'] > self.tau or g['projdepth'] < self.depth:
            return 0
        if self.ncorr is not None and g['ncorr'] < self.ncorr:
            return 0
        if self.closing is not None and g['closing'] > self.closing:
            return 0
        alt = max((env.frames[env.t]['per_g'][j]['ttb'] for j in range(4) if j != env.serv),
                  default=0.0)
        return int(alt > g['ttb'] + self.margin)


class AntFullPolicy(AntOnlyPolicy):
    """antonly plus the two unread features; identical class, different name and grid."""
    base = 'antfull'


def grid_for(full):
    base = list(itertools.product(TAU_GRID, DEPTH_GRID, MARGIN_GRID))
    if not full:
        return [dict(tau=t, depth=d, margin=m, ncorr=None, closing=None) for t, d, m in base]
    return [dict(tau=t, depth=d, margin=m, ncorr=nc, closing=cl)
            for t, d, m in base for nc in NCORR_GRID for cl in CLOSING_GRID]


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--eval_n', type=int, default=500)
    ap.add_argument('--eval_n_train', type=int, default=200)
    ap.add_argument('--tune_n', type=int, default=200, help='training-scene episodes per grid point')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--full', action='store_true',
                    help='also run antfull (all five anticipation features); 9x the tuning cost')
    ap.add_argument('--only_full', action='store_true', help='run antfull and skip antonly')
    ap.add_argument('--outdir', default='/root/work/paper2/models/sweep_r3')
    a = ap.parse_args()

    import paper2_env as PE
    from paper2_controller import EVAL_SEED_BASE, summarize, latency_stats
    os.makedirs(a.outdir, exist_ok=True)
    pool = pd.read_csv(a.pool)
    env_tr = PE.MultiRSUBlockageEnv(pool, seed=a.seed, split='train')
    env_te = PE.MultiRSUBlockageEnv(pool, seed=a.seed, split='test')
    print(f'train scenes={len(env_tr.scene_ids)} held-out scenes={len(env_te.scene_ids)} | '
          f'tuning on {a.tune_n} TRAINING episodes, CRN base {EVAL_SEED_BASE}', flush=True)

    jobs = []
    if not a.only_full:
        jobs.append((AntOnlyPolicy, False))
    if a.full or a.only_full:
        jobs.append((AntFullPolicy, True))

    rows = []
    for cls, full in jobs:
        grid = grid_for(full)
        print(f'\n===== {cls.base}: {len(grid)} grid points x 2 modes on training scenes =====',
              flush=True)
        chosen, tune_rows = {}, []
        for mode in ('hold', 'eff'):
            best = None
            for k, gp in enumerate(grid):
                pol = cls(mode=mode, **gp)
                m = summarize(BL.run_policy(env_tr, pol, a.tune_n, EVAL_SEED_BASE))
                rec = dict(mode=mode, **gp, cvar25=m['cvar25'], return_mean=m['return_mean'],
                           rlf_rate=m['rlf_rate'], ho_per_ep=m['ho_per_ep'])
                tune_rows.append(rec)
                if best is None or rec['cvar25'] > best['cvar25']:
                    best = rec
                if full and (k + 1) % 25:
                    continue
                print(f"  [tune {cls.base} {mode} {k + 1}/{len(grid)}] tau={gp['tau']:<5} "
                      f"depth={gp['depth']:<5} margin={gp['margin']:<4} ncorr={str(gp['ncorr']):<5} "
                      f"closing={str(gp['closing']):<5} CVaR25={m['cvar25']:7.2f} "
                      f"RLF={m['rlf_rate']:6.2f}% HO/ep={m['ho_per_ep']:5.2f}", flush=True)
            chosen[mode] = best
            print(f"  == selected [{cls.base} {mode}] " +
                  ' '.join(f'{k}={best[k]}' for k in ('tau', 'depth', 'margin', 'ncorr', 'closing')) +
                  f" (train CVaR25={best['cvar25']:.2f})", flush=True)

        pd.DataFrame(tune_rows).to_csv(
            os.path.join(a.outdir, f'{cls.base}_tuning_grid.csv'), index=False)
        json.dump(chosen, open(os.path.join(a.outdir, f'{cls.base}_selected.json'), 'w'),
                  indent=2, default=str)

        # --- held-out scoring: touched exactly once, after selection is frozen -----------------
        for mode in ('hold', 'eff'):
            b = chosen[mode]
            pol = cls(mode=mode, **{k: b[k] for k in ('tau', 'depth', 'margin', 'ncorr', 'closing')})
            t0, lat = time.time(), []
            te = BL.run_policy(env_te, pol, a.eval_n, EVAL_SEED_BASE, latency=lat)
            tr = BL.run_policy(env_tr, pol, a.eval_n_train, EVAL_SEED_BASE)
            te.to_csv(os.path.join(a.outdir, f'baseline_{pol.name}_eval.csv'), index=False)
            tr.to_csv(os.path.join(a.outdir, f'baseline_{pol.name}_eval_train.csv'), index=False)
            m = dict(config=pol.name, seed=a.seed, learned=0, wall_s=round(time.time() - t0, 1))
            m.update(summarize(te)); m.update(summarize(tr, 'insample')); m.update(latency_stats(lat))
            m.update(sel_tau=b['tau'], sel_depth=b['depth'], sel_margin=b['margin'],
                     sel_ncorr=b['ncorr'], sel_closing=b['closing'])
            rows.append(m)
            print(f"  {pol.name:14s} return={m['return_mean']:7.2f}  CVaR25={m['cvar25']:7.2f}  "
                  f"RLF={m['rlf_rate']:6.2f}%  pt={m['pt_dbm']:5.2f}dBm  rb={m['rb']:5.1f}  "
                  f"HO/ep={m['ho_per_ep']:5.2f}  | {m['wall_s']:.0f}s", flush=True)

    out = os.path.join(a.outdir, 'antonly_metrics.csv')
    df = pd.DataFrame(rows)
    if os.path.exists(out):                       # a second invocation (e.g. --only_full) appends
        prev = pd.read_csv(out)
        df = pd.concat([prev[~prev.config.isin(df.config)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f'\nsaved -> {out}')
    print('NOTE: paper2_stats.py must load this file alongside baselines_metrics.csv and '
          'lookahead_metrics.csv. The episode-level Holm family grows by the number of rows added; '
          'it is NOT split to protect any existing comparison.')


if __name__ == '__main__':
    main()
