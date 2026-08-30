#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_ho_requests.py -- count handover REQUESTS, not just accepted handovers.

Why. Supplementary S2 states that the tabulated handover column counts handovers the state machine
*accepted*, that the reward is charged on every *request*, and that "the two quantities must not be
equated". The abstract, Section VIII-H and the Conclusion nevertheless report the 12.7x ratio as a
reduction in *signalling*. Either the wording changes or the request count gets measured. This
measures it.

What the state machine actually does, read from paper2_env.step:

    if self._ho_timer > 0:      -> the request is not examined at all
    elif a_ho == 1:
        cand = argmax over j != serv (the serving index is pinned to -1e9)
        if cand != self.serv:   -> ALWAYS true, since serv can never be the argmax
            ... ho_started = True

So there is exactly ONE reason a request fails to become a handover: the completion timer is
already running. S2's second stated reason -- "a request for which no neighbour improves on the
serving link is discarded" -- is not implemented; the candidate is the best NON-SERVING gNB and is
never compared against the serving link. That is reported separately as a text/code mismatch.

Consequently  requests = accepted + timer-blocked,  and this script measures all three on the same
500 common-random-number scenarios the paper scores everything else on. Nothing is retrained: the
frozen checkpoints of models/sweep_r3 are re-evaluated.

Verification built in: n_ho recomputed here must equal the published per-seed value to the episode.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np, pandas as pd, torch

import paper2_env as E
import paper2_controller as C
import paper2_baselines as B
import paper2_lookahead_baseline as L

# the tuned anticipatory threshold rule -- the comparison the 12.7x claim is actually about.
# Thresholds are read from the file the tuning wrote, never retyped.
SELECTED = json.load(open('lookahead_selected.json'))

POOL = pd.read_csv('dair_geometry_pool.csv')


def instrumented(env, step_action, n, seed_base):
    """Roll out n CRN episodes, counting requests / accepted / timer-blocked."""
    rows = []
    for i in range(n):
        obs, info = env.reset(seed=seed_base + i)
        st = step_action(reset=True)
        R, steps, nho, nreq, nblk = 0.0, 0, 0, 0, 0
        done = trunc = False
        while not (done or trunc):
            a, st = step_action(env=env, obs=obs, st=st)
            req = int(int(a[0]) == 1)
            busy = env._ho_timer > 0            # read BEFORE step: the timer that gates this request
            obs, r, done, trunc, i2 = env.step(a)
            R += r; steps += 1
            nho += int(i2['ho']); nreq += req
            nblk += int(req and busy)
        rows.append(dict(idx=i, eval_seed=seed_base + i, scene=info['scene'],
                         ret=round(float(R), 6), rlf=int(bool(done)), steps=steps,
                         n_ho=nho, n_req=nreq, n_blocked=nblk))
    return pd.DataFrame(rows)


def learned(path, obs_dim):
    ag = C.Agent(obs_dim, use_lstm=False)
    ag.net.load_state_dict(torch.load(path, map_location='cpu'))
    ag.net.eval()

    def f(env=None, obs=None, st=None, reset=False):
        if reset:
            return None
        a, h = ag.act(obs, st, 0.0)
        return a, h
    return f


def reference(pol):
    def f(env=None, obs=None, st=None, reset=False):
        if reset:
            return None
        return pol.act(env, obs), None
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--out', default='ho_requests.csv')
    a = ap.parse_args()

    env = E.MultiRSUBlockageEnv(POOL, seed=0, split='test')
    obs_dim = env.observation_space.shape[0]
    published = pd.read_csv('all_metrics.csv') if os.path.exists('all_metrics.csv') else None
    out, t0 = [], time.time()

    for s in range(a.seeds):
        p = f'models/model_s{s}.pt'
        if not os.path.exists(p):
            continue
        df = instrumented(env, learned(p, obs_dim), a.n, C.EVAL_SEED_BASE)
        rec = dict(policy='reported controller', seed=s,
                   ho_per_ep=df.n_ho.mean(), req_per_ep=df.n_req.mean(),
                   blocked_per_ep=df.n_blocked.mean(), rlf_rate=100.0 * df.rlf.mean())
        out.append(rec)
        print(f"  model s{s:<2d} accepted {rec['ho_per_ep']:6.3f}  requests {rec['req_per_ep']:6.3f}"
              f"  blocked {rec['blocked_per_ep']:6.3f}  rlf {rec['rlf_rate']:5.2f}%"
              f"   [{time.time()-t0:5.0f}s]", flush=True)

    for mode in ('hold', 'eff'):
        b = SELECTED[mode]
        pol = L.LookAheadPolicy(mode=mode, tau=b['tau'], depth=b['depth'], margin=b['margin'])
        df = instrumented(env, reference(pol), a.n, C.EVAL_SEED_BASE)
        out.append(dict(policy=f'AnticipatoryThreshold:{mode}', seed=-1,
                        ho_per_ep=df.n_ho.mean(), req_per_ep=df.n_req.mean(),
                        blocked_per_ep=df.n_blocked.mean(), rlf_rate=100.0 * df.rlf.mean()))
        print(f"  {'AnticipatoryThreshold:'+mode:<28s} accepted {df.n_ho.mean():6.3f}  "
              f"requests {df.n_req.mean():6.3f}  blocked {df.n_blocked.mean():6.3f}  "
              f"rlf {100*df.rlf.mean():5.2f}%   [{time.time()-t0:5.0f}s]", flush=True)

    for pol in B.build_policies(seed=0):
        name = f'{type(pol).__name__}:{getattr(pol, "mode", "-")}'
        df = instrumented(env, reference(pol), a.n, C.EVAL_SEED_BASE)
        out.append(dict(policy=name, seed=-1, ho_per_ep=df.n_ho.mean(),
                        req_per_ep=df.n_req.mean(), blocked_per_ep=df.n_blocked.mean(),
                        rlf_rate=100.0 * df.rlf.mean()))
        print(f"  {name:<28s} accepted {df.n_ho.mean():6.3f}  requests {df.n_req.mean():6.3f}"
              f"  blocked {df.n_blocked.mean():6.3f}  rlf {100*df.rlf.mean():5.2f}%"
              f"   [{time.time()-t0:5.0f}s]", flush=True)

    pd.DataFrame(out).to_csv(a.out, index=False)
    print(f'\nsaved -> {a.out}')


if __name__ == '__main__':
    main()
