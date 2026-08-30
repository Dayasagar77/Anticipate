#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_lookahead_baseline.py — a DEPLOYABLE anticipatory handover rule, tuned on training scenes.

WHY THIS FILE EXISTS
--------------------
The reference set in paper2_baselines.py brackets the learned controller from below (random, static,
max-SINR, 3GPP A3 --- all reactive) and from above (the geometry genie, which is granted perfect
future geometry and is therefore not deployable). Nothing in that set answers the sharpest question a
referee can ask:

    the anticipation state is itself the contribution; how much of the gain comes from HAVING that
    state, and how much from LEARNING a policy over it?

A hand-written threshold rule that reads exactly the same anticipation features the controller reads,
and is deployable on exactly the same information, isolates the answer. If it matches the learned
controller, the paper's finding is "the state matters, the learning does not", and that is what will
be written. If it does not, the margin is the value of learning, measured rather than asserted.

THE RULE
--------
Hand over at step t if EITHER of the following fires:

  (a) the conventional 3GPP A3 event (neighbour better than serving by a 3 dB hysteresis, sustained
      for a 2-frame time-to-trigger) --- identical to the a3 policy, so this rule is never weaker
      than the standard reactive one it extends; or
  (b) the anticipatory event:  ttb_serv <= TAU  AND  projdepth_serv >= DEPTH  AND
      max_j!=serv ttb_j > ttb_serv + MARGIN
      i.e. the serving link is predicted to be occluded within TAU seconds, the predicted occlusion
      is deep enough to matter, and some alternative link stays clear longer than the serving one.

Every quantity in (b) is read from anticipate.anticipation_features, which forward-projects the
CURRENTLY OBSERVED vehicle boxes at constant velocity. No future annotation is consulted. The rule is
as deployable as A3 is, given a sensing pipeline that reports neighbouring vehicle boxes --- the same
pipeline the learned controller presupposes.

HOW THE THRESHOLDS ARE CHOSEN --- AND WHY IT IS DONE THIS WAY
-------------------------------------------------------------
(TAU, DEPTH, MARGIN) are selected by exhaustive grid search on the TRAINING scenes ONLY, maximizing
CVaR_0.25 of episode return --- the same 30 scenes the controller trains on and the same objective the
controller optimizes. The held-out scenes are evaluated exactly once, after selection is frozen.

This matters. Tuning a baseline on the held-out set and then reporting it against a controller that
never saw it would flatter the baseline; tuning it on nothing at all (fixed guesses) would flatter the
controller. Selecting on train and scoring on test gives both sides the same protocol. The selected
thresholds and the full grid are written to disk so the choice is auditable.

Both efficiency modes are run, exactly as for every other reference policy: `hold` pins power and
bandwidth at maximum (best for RLF, worst for return) and `eff` runs conventional closed-loop power
control with minimum bandwidth (best for return).

INTEGRITY: reference policies run on the same environment, same held-out scenes, same common random
numbers as every learned configuration. No number here is copied from any other paper.
"""
import os, sys, time, json, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import paper2_baselines as BL

# --- the grid searched on the TRAINING scenes -------------------------------------------------
TAU_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]     # s; 3.0 = the full anticipation horizon
DEPTH_GRID = [1.0, 5.0, 10.0, 20.0]                  # dB of predicted knife-edge loss
MARGIN_GRID = [0.0, 0.5]                             # s the alternative must stay clear beyond serving


class LookAheadPolicy(BL.A3Policy):
    """A3 (inherited, unchanged) OR the anticipatory threshold event."""
    base = 'antthr'

    def __init__(self, mode='hold', tau=1.0, depth=1.0, margin=0.0, **kw):
        super().__init__(mode=mode, **kw)
        self.tau, self.depth, self.margin = float(tau), float(depth), float(margin)
        self.name = f'{self.base}_{mode}'

    def ho(self, env, obs):
        if super().ho(env, obs):                      # conventional reactive trigger, unchanged
            return 1
        g = env.frames[env.t]['per_g'][env.serv]
        if g['ttb'] > self.tau or g['projdepth'] < self.depth:
            return 0
        alt = max((env.frames[env.t]['per_g'][j]['ttb'] for j in range(4) if j != env.serv),
                  default=0.0)
        return int(alt > g['ttb'] + self.margin)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--eval_n', type=int, default=500)
    ap.add_argument('--eval_n_train', type=int, default=200)
    ap.add_argument('--tune_n', type=int, default=200, help='training-scene episodes per grid point')
    ap.add_argument('--seed', type=int, default=0)
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

    grid = list(itertools.product(TAU_GRID, DEPTH_GRID, MARGIN_GRID))
    chosen, rows, tune_rows = {}, [], []

    for mode in ('hold', 'eff'):
        best = None
        for tau, dep, mar in grid:
            pol = LookAheadPolicy(mode=mode, tau=tau, depth=dep, margin=mar)
            tr = BL.run_policy(env_tr, pol, a.tune_n, EVAL_SEED_BASE)
            m = summarize(tr)
            rec = dict(mode=mode, tau=tau, depth=dep, margin=mar, cvar25=m['cvar25'],
                       return_mean=m['return_mean'], rlf_rate=m['rlf_rate'])
            tune_rows.append(rec)
            if best is None or rec['cvar25'] > best['cvar25']:
                best = rec
            print(f"  [tune {mode}] tau={tau:<5} depth={dep:<5} margin={mar:<4} "
                  f"CVaR25={m['cvar25']:7.2f} ret={m['return_mean']:7.2f} "
                  f"RLF={m['rlf_rate']:6.2f}%", flush=True)
        chosen[mode] = best
        print(f"  == selected [{mode}] tau={best['tau']} depth={best['depth']} "
              f"margin={best['margin']} (train CVaR25={best['cvar25']:.2f})", flush=True)

    pd.DataFrame(tune_rows).to_csv(os.path.join(a.outdir, 'lookahead_tuning_grid.csv'), index=False)
    json.dump(chosen, open(os.path.join(a.outdir, 'lookahead_selected.json'), 'w'), indent=2)

    # --- held-out scoring: touched exactly once, after selection is frozen ---------------------
    for mode in ('hold', 'eff'):
        b = chosen[mode]
        pol = LookAheadPolicy(mode=mode, tau=b['tau'], depth=b['depth'], margin=b['margin'])
        t0, lat = time.time(), []
        te = BL.run_policy(env_te, pol, a.eval_n, EVAL_SEED_BASE, latency=lat)
        tr = BL.run_policy(env_tr, pol, a.eval_n_train, EVAL_SEED_BASE)
        te.to_csv(os.path.join(a.outdir, f'baseline_{pol.name}_eval.csv'), index=False)
        tr.to_csv(os.path.join(a.outdir, f'baseline_{pol.name}_eval_train.csv'), index=False)
        m = dict(config=pol.name, seed=a.seed, learned=0, wall_s=round(time.time() - t0, 1))
        m.update(summarize(te)); m.update(summarize(tr, 'insample')); m.update(latency_stats(lat))
        m.update(sel_tau=b['tau'], sel_depth=b['depth'], sel_margin=b['margin'])
        rows.append(m)
        print(f"  {pol.name:14s} return={m['return_mean']:7.2f}  CVaR25={m['cvar25']:7.2f}  "
              f"RLF={m['rlf_rate']:6.2f}%  pt={m['pt_dbm']:5.2f}dBm  rb={m['rb']:5.1f}  "
              f"HO/ep={m['ho_per_ep']:5.2f}  | {m['wall_s']:.0f}s", flush=True)

    out = os.path.join(a.outdir, 'lookahead_metrics.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'saved -> {out}')
    print('NOTE: merge into baselines_metrics.csv with --merge once the sweep is idle.')


if __name__ == '__main__':
    main()
