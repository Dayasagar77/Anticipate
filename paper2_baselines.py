#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_baselines.py — non-learned comparators for the Paper-2 held-out evaluation.

WHY THIS FILE EXISTS
--------------------
The R2 results compared the proposed controller only against its own ablations. A reviewer will ask
what a conventional handover rule does on the same scenes, and the R2 draft could not answer: the
only external reference point was a uniform-random policy whose RLF rate and tail return were never
even recorded. This module supplies reference policies evaluated on exactly the same held-out scenes
with exactly the same reset seeds (common random numbers) as every learned configuration, so the
comparison is paired episode by episode.

HANDOVER RULES
--------------
  random   uniform over the branched action space; the trivial lower reference.
  static   never hand over.
  maxsinr  hand over whenever any other gNB currently offers a higher SINR (greedy, no hysteresis).
  a3       3GPP-style A3 event: trigger when the best neighbour exceeds the serving cell by a
           hysteresis margin of 3 dB for a time-to-trigger of 2 steps (200 ms at the 10 Hz frame
           rate). This is the conventional measurement-report rule the proposed controller must beat.
  genie    an UPPER BOUND, not a deployable scheme: granted perfect knowledge of future GEOMETRY
           (though not of fading), it looks ahead over the window [t+L_HO, t+L_HO+W] and hands over
           when the best gNB's mean noiseless SINR over that window exceeds the serving gNB's by
           more than 1 dB. It measures how much of the remaining gap is reachable at all.

POWER / SPECTRUM MODES — why every rule is run twice
----------------------------------------------------
The episode return contains efficiency terms that charge for transmit power and for the resource
blocks held. A rule that simply pins both actuators at maximum therefore buys its reliability at the
worst possible cost and looks weak on return for a reason that has nothing to do with its handover
logic. To keep the comparison about handover, each rule is run in two modes and BOTH are reported:

  hold  maximum transmit power, maximum bandwidth. Most favourable setting for the rule's RLF rate,
        least favourable for its return.
  eff   conventional closed-loop power control toward an SINR target of SINR_out + 10 dB in 1 dB
        steps, and minimum bandwidth. Most favourable for return.

`eff` drives the bandwidth to its floor because in this environment — inherited unchanged from the
companion study — the resource-block count enters the objective only as an efficiency cost and does
not enter the SINR at all. That is a property of the model, and it is stated in the paper rather
than papered over: a bandwidth-quantity actuator cannot orthogonalize an interferer; doing so would
require control of resource-block *placement*, which this action space does not have.

FAIRNESS NOTE, WHICH IS STATED IN THE PAPER
-------------------------------------------
Every rule-based policy is handed the *exact* instantaneous SINR of every link, i.e. ideal,
noise-free, zero-delay measurement reports for all four gNBs at every step. Real A3 operates on
filtered, quantised, periodically reported RSRP. The baselines are therefore stronger than their
deployed counterparts, which makes any margin the learned controller holds over them conservative.

INTEGRITY: these are reference policies run on the same environment and the same episodes; no number
here is copied from any other paper.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

A3_HYST_DB = 3.0          # 3GPP A3 offset
A3_TTT_STEPS = 2          # time-to-trigger: 2 frames = 200 ms at 10 Hz
GENIE_WINDOW = 5          # look-ahead window length, in frames, after handover completion
GENIE_MARGIN_DB = 1.0     # do not churn for less than this
TPC_MARGIN_DB = 10.0      # closed-loop power-control target, above the outage threshold
TPC_DEADBAND_DB = 3.0     # hysteresis band, so the loop does not oscillate every frame


def _resource_action(env, mode):
    """(power action, spectrum action) for the chosen efficiency mode. 0 = down, 1 = hold, 2 = up."""
    if mode == 'hold':
        return 2, 2                                   # both clipped at their maxima
    import paper2_env as PE
    s = env._sinr_db(env.serv, env.t, env.pt)
    tgt = PE.SINR_OUT_DB + TPC_MARGIN_DB
    if s < tgt:
        a_p = 2
    elif s > tgt + TPC_DEADBAND_DB:
        a_p = 0
    else:
        a_p = 1
    return a_p, 0                                     # spectrum: economise down to the floor


# --------------------------------------------------------------------------- policies
class Policy:
    base = 'base'

    def __init__(self, mode='hold', **kw):
        self.mode = mode
        self.name = f'{self.base}_{mode}'

    def reset(self, env):
        pass

    def ho(self, env, obs):
        return 0

    def act(self, env, obs):
        a_p, a_s = _resource_action(env, self.mode)
        return [int(self.ho(env, obs)), a_p, a_s]


class RandomPolicy(Policy):
    base = 'random'

    def __init__(self, mode='hold', seed=0, **kw):
        super().__init__(mode)
        self.name = 'random'                          # unaffected by the resource mode
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        return [int(self.rng.integers(n)) for n in env.action_space.nvec]


class StaticPolicy(Policy):
    base = 'static'


class MaxSinrPolicy(Policy):
    base = 'maxsinr'

    def ho(self, env, obs):
        s = [env._sinr_db(j, env.t, env.pt) for j in range(4)]
        return int(int(np.argmax(s)) != env.serv)


class A3Policy(Policy):
    """3GPP A3: a neighbour becomes offset-better than serving, sustained for the time-to-trigger."""
    base = 'a3'

    def __init__(self, mode='hold', hyst=A3_HYST_DB, ttt=A3_TTT_STEPS, **kw):
        super().__init__(mode)
        self.hyst, self.ttt = float(hyst), int(ttt)
        self.count, self.cand = 0, None

    def reset(self, env):
        self.count, self.cand = 0, None

    def ho(self, env, obs):
        s = [env._sinr_db(j, env.t, env.pt) for j in range(4)]
        v, j = max((v, j) for j, v in enumerate(s) if j != env.serv)
        if v > s[env.serv] + self.hyst:
            self.count = self.count + 1 if self.cand == j else 1
            self.cand = j
        else:
            self.count, self.cand = 0, None
        if self.count >= self.ttt:
            self.count, self.cand = 0, None
            return 1
        return 0


class GeniePolicy(Policy):
    """Upper bound: perfect future GEOMETRY (not fading), over the post-handover window."""
    base = 'genie'

    def __init__(self, mode='hold', window=GENIE_WINDOW, margin=GENIE_MARGIN_DB, **kw):
        super().__init__(mode)
        self.w, self.margin = int(window), float(margin)

    def ho(self, env, obs):
        T = len(env.frames)
        lo = min(env.t + env.ho_lat, T - 1)
        hi = min(lo + self.w, T)
        idx = list(range(lo, hi)) or [lo]
        m = [float(np.mean([env._sinr_noiseless_db(j, u, env.pt) for u in idx])) for j in range(4)]
        best = int(np.argmax(m))
        return int(best != env.serv and m[best] > m[env.serv] + self.margin)


CLASSES = [RandomPolicy, StaticPolicy, MaxSinrPolicy, A3Policy, GeniePolicy]


def build_policies(seed=0):
    pols = [RandomPolicy(seed=seed)]
    for cls in CLASSES[1:]:
        for mode in ('hold', 'eff'):
            pols.append(cls(mode=mode))
    return pols


# ------------------------------------------------------------------------- evaluation
def run_policy(env, pol, n, seed_base, latency=None):
    """Evaluate one policy over n episodes under common random numbers. Same schema as
    paper2_controller.evaluate, so the two are directly comparable episode by episode."""
    import pandas as pd
    rows = []
    for i in range(n):
        obs, info = env.reset(seed=seed_base + i)
        pol.reset(env)
        R, steps, nho = 0.0, 0, 0
        pts, rbs = [], []
        done = trunc = False
        while not (done or trunc):
            if latency is None:
                a = pol.act(env, obs)
            else:
                t0 = time.perf_counter()
                a = pol.act(env, obs)
                latency.append((time.perf_counter() - t0) * 1000.0)
            obs, r, done, trunc, i2 = env.step(a)
            R += r; steps += 1
            pts.append(i2['pt']); rbs.append(i2['rb']); nho += int(i2['ho'])
        rows.append(dict(idx=i, eval_seed=seed_base + i, scene=info['scene'],
                         ret=round(float(R), 6), rlf=int(bool(done)), steps=steps,
                         mean_pt=round(float(np.mean(pts)), 4), mean_rb=round(float(np.mean(rbs)), 4),
                         n_ho=nho))
    return pd.DataFrame(rows)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--p1', default=None)
    ap.add_argument('--eval_n', type=int, default=500)
    ap.add_argument('--eval_n_train', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0, help='only affects the random policy')
    ap.add_argument('--outdir', default='/root/work/paper2/models/sweep_r3')
    a = ap.parse_args()
    if a.p1:
        sys.path.insert(0, a.p1)
    import paper2_env as PE
    from paper2_controller import EVAL_SEED_BASE, summarize, latency_stats
    os.makedirs(a.outdir, exist_ok=True)
    pool = pd.read_csv(a.pool)
    env_te = PE.MultiRSUBlockageEnv(pool, seed=a.seed, split='test')
    env_tr = PE.MultiRSUBlockageEnv(pool, seed=a.seed, split='train')
    print(f'held-out scenes={len(env_te.scene_ids)} train scenes={len(env_tr.scene_ids)} | '
          f'eval {a.eval_n} held-out / {a.eval_n_train} in-sample, CRN base {EVAL_SEED_BASE}', flush=True)

    rows = []
    for pol in build_policies(a.seed):
        t0, lat = time.time(), []
        te = run_policy(env_te, pol, a.eval_n, EVAL_SEED_BASE, latency=lat)
        tr = run_policy(env_tr, pol, a.eval_n_train, EVAL_SEED_BASE)
        te.to_csv(os.path.join(a.outdir, f'baseline_{pol.name}_eval.csv'), index=False)
        tr.to_csv(os.path.join(a.outdir, f'baseline_{pol.name}_eval_train.csv'), index=False)
        m = dict(config=pol.name, seed=a.seed, learned=0, wall_s=round(time.time() - t0, 1))
        m.update(summarize(te)); m.update(summarize(tr, 'insample')); m.update(latency_stats(lat))
        rows.append(m)
        print(f"  {pol.name:14s} return={m['return_mean']:7.2f}  CVaR25={m['cvar25']:7.2f}  "
              f"RLF={m['rlf_rate']:6.2f}%  pt={m['pt_dbm']:5.2f}dBm  rb={m['rb']:5.1f}  "
              f"HO/ep={m['ho_per_ep']:5.2f}  | {m['wall_s']:.0f}s", flush=True)

    out = os.path.join(a.outdir, 'baselines_metrics.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
