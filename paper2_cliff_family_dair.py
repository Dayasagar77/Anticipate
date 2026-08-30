#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_cliff_family_dair.py — the cliff family, re-measured on THIS paper's own environment.

WHY THIS FILE EXISTS
--------------------
paper2_cliff_family.py produced results/cliff_family.csv, which feeds Section IV-A, Table II,
Fig. 3, Fig. 4, the abstract's "roughly sevenfold" and "about 18 dB", Contribution 1 and the
Conclusion. Reading it end to end shows what it actually runs:

    sys.path.insert(0, '/root/work/exp')
    from env_6g_nyu import SubTHzEnv6G_NYU              <- Paper 1's FLAT, SINGLE-LINK env
    FLAGS = [... 'rederived_blockage_%s.csv' % r ...]   <- Paper 1's HighD trajectory pool
    class BlockedFlatEnv(SubTHzEnv6G_NYU): ...          <- one gNB, RSRP, no interference

So that surface was measured on HighD trajectories with 2-D binary blocked flags inside Paper 1's
single-link environment --- NOT on DAIR-V2X, NOT with the 3-D knife-edge blockage model, and NOT in
the four-gNB interference-limited environment that Section III of this paper defines. Meanwhile the
manuscript said the depth-0 cliff was "measured here on this environment" and Fig. 3's caption said
"measured on our environment". A reader arriving from Section III --- which is entirely DAIR-V2X ---
reads both as the Paper 2 environment. That is a material provenance defect, and no amount of
rewording fixes it as well as simply measuring the surface where the paper says it is measured.

WHAT THIS FILE MEASURES
-----------------------
The same 2-D (transmit power x blockage depth) policy-free sustained-RLF surface, on
MultiRSUBlockageEnv: real DAIR-V2X intersection geometry, four roadside gNBs, ITU-R P.526 3-D
knife-edge blockage on every UE->gNB ray, K = 4 nearest in-scene vehicles as co-channel
interferers, the NYU-calibrated CI channel at 140 GHz, sustained-RLF over N310 frames.

HOW DEPTH IS SWEPT, STATED EXACTLY
-----------------------------------
The 3-D model already returns a CONTINUOUS per-link diffraction loss; unlike Paper 1's binary
flags there is no "how deep" left free. The swept quantity is therefore excess attenuation ON TOP
of the modelled loss, and it is applied uniformly to every link the geometry declares blocked:

    depth_db >= BLOCK_THR_DB (1 dB)   ->   depth_db + Delta

  * applied to all FOUR UE->gNB links, not only the serving one, so the spatial-diversity
    structure survives: a handover helps exactly when the alternative ray is geometrically clear;
  * applied to the interferer->gNB links as well, so the channel stays internally consistent. A
    sweep that attenuated the desired signal but left interference untouched would inflate the
    effect and invite the obvious objection. Attenuating the interferers too makes the measured
    degradation the conservative one.
  * the 40 dB cap in blockage3d applies to the modelled diffraction term; Delta is added above it.

Delta = 0 is therefore the untouched physical environment, and the depth-0 row is a genuine
measurement of this paper's own channel rather than an import.

PROTOCOL
--------
Policy-free: the 3GPP event-A3 rule of paper2_baselines.A3Policy (3 dB hysteresis, 2-frame
time-to-trigger), reused unchanged, with the power and spectrum branches pinned to hold so that
transmit power is the swept variable and not a policy output. All 45 scenes: this is an
environment characterization with no learning, so the 30/15 train/held-out split does not apply
(and Section IV-C's multi-UE measurement already reports over all 45).

COMMON RANDOM NUMBERS ACROSS THE WHOLE GRID. Episode i of seed s is drawn with
reset(seed = SEED_BASE + 10000*s + i) in EVERY cell, so all 56 cells face byte-identical scenes
and byte-identical shadow-fading realisations. The surface is thus a paired measurement: a
difference between two cells is a difference in outcome on the same episodes, not a difference
between two independent samples. Paper 1's version did not have this property.

INTEGRITY: real geometry, real channel constants, no number copied from anywhere.

Usage: python3 paper2_cliff_family_dair.py [--nep 150] [--powers 23,..,30] [--depths 0,..,30]
"""
import os, sys, csv, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

import blockage3d as B
import paper2_env as PE
from paper2_baselines import A3Policy

POOL_CSV = 'dair_geometry_pool.csv'
SEED_BASE = 700_000_000          # distinct from the controller's EVAL_SEED_BASE (900_000_000)
SEEDS = [0, 1, 2, 3, 4]

# Physical geometry is identical for every cell of the grid, so it is unpickled once per scene for
# the whole sweep rather than once per environment instance (35 instances x 45 scenes otherwise).
_PHYS = {}


class DepthSweptEnv(PE.MultiRSUBlockageEnv):
    """MultiRSUBlockageEnv with (a) Delta dB of excess loss on every geometrically blocked link and
    (b) the transmit power pinned, so power is a swept parameter rather than a policy output."""

    def __init__(self, pool_df, extra_db=0.0, pin_pt=PE.PT_MAX, **kw):
        self.extra_db = float(extra_db)
        self.pin_pt = float(pin_pt)
        self._swept = {}
        super().__init__(pool_df, **kw)

    def _precompute(self, sid):
        """Parent caches the PHYSICAL geometry once (shared, and written to geom_cache/); the
        depth-augmented view is derived from it and memoized separately, so the on-disk cache is
        never contaminated with swept values."""
        if sid in _PHYS:
            frames = _PHYS[sid]
        else:
            frames = _PHYS[sid] = super()._precompute(sid)
        if self.extra_db <= 0.0:
            return frames
        if sid in self._swept:
            return self._swept[sid]
        d = self.extra_db
        out = []
        for fr in frames:
            per_g = []
            for g in fr['per_g']:
                gg = dict(g)
                if g['depth'] >= B.BLOCK_THR_DB:
                    gg['depth'] = g['depth'] + d
                gg['intf'] = [(dist, dep + d if dep >= B.BLOCK_THR_DB else dep)
                              for dist, dep in g['intf']]
                per_g.append(gg)
            out.append(dict(fr, per_g=per_g))
        self._swept[sid] = out
        return out

    def reset(self, *, seed=None, options=None):
        """Identical to the parent, except the transmit power is the pinned sweep level from the
        first frame onward --- including in the initial serving-cell selection."""
        _, info = super().reset(seed=seed, options=options)
        self.pt = self.pin_pt
        self.serv = int(np.argmax([self._sinr_db(j, 0, self.pt) for j in range(4)]))
        self._ho_target = self.serv
        return self._obs(self._sinr_db(self.serv, 0, self.pt)), info


def run_episode(env, pol, seed):
    """One episode under the A3 handover rule with both resource branches pinned to hold."""
    obs, _ = env.reset(seed=seed)
    pol.reset(env)
    pin = env.pt
    done = trunc = False
    rlf = False
    nho = 0
    while not (done or trunc):
        a_ho = int(pol.ho(env, obs))
        nho += a_ho
        obs, _, done, trunc, info = env.step([a_ho, 1, 1])   # 1 = hold on power and on spectrum
        assert abs(env.pt - pin) < 1e-9, 'transmit power drifted off the swept level'
        rlf = bool(info['rlf'])
    return rlf, nho


def depth_row(pool_df, delta, powers, nep):
    """Every cell of one depth row, sharing one environment per seed.

    The environment is constructed once per (depth, seed) and the pinned power is changed between
    episodes; because reset() re-applies self.pin_pt and re-selects the serving cell, the power
    levels remain fully independent runs while the depth-augmented geometry is derived only once.
    """
    rlf = np.zeros((len(powers), len(SEEDS)))
    ho = np.zeros((len(powers), len(SEEDS)))
    for si, s in enumerate(SEEDS):
        env = DepthSweptEnv(pool_df, extra_db=delta, seed=s, split=None)
        pol = A3Policy(mode='hold')
        for pi, pt in enumerate(powers):
            env.pin_pt = float(pt)
            n_rlf, n_ho = 0, 0
            for i in range(nep):
                r, h = run_episode(env, pol, SEED_BASE + 10_000 * s + i)
                n_rlf += int(r); n_ho += h
            rlf[pi, si] = 100.0 * n_rlf / nep
            ho[pi, si] = n_ho / nep
    return rlf.mean(1), rlf.std(1), ho.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nep', type=int, default=150)
    ap.add_argument('--powers', default='23,24,25,26,27,28,29,30')
    ap.add_argument('--depths', default='0,5,10,15,20,25,30')
    ap.add_argument('--pool', default=POOL_CSV)
    ap.add_argument('--out', default='/root/work/paper2/results/cliff_family_dair.csv')
    a = ap.parse_args()
    powers = [float(x) for x in a.powers.split(',')]
    depths = [float(x) for x in a.depths.split(',')]

    pool_df = pd.read_csv(a.pool)
    nsc = pool_df['scene_id'].astype(str).nunique()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print(f"CLIFF FAMILY (DAIR-V2X, this paper's environment)", flush=True)
    print(f"  scenes={nsc} (all, no split) | K={4} interferers | grid {len(powers)}x{len(depths)}"
          f" | {len(SEEDS)} seeds x {a.nep} episodes | A3 rule, power pinned", flush=True)
    print("       pt(dBm): " + " ".join(f"{p:5.0f}" for p in powers), flush=True)

    rows, t0 = [], time.time()
    for d in depths:
        m, sd, nho = depth_row(pool_df, d, powers, a.nep)
        for pi, pt in enumerate(powers):
            rows.append({'block_db': d, 'pt_dbm': pt, 'RLF_mean': f'{m[pi]:.2f}',
                         'RLF_std': f'{sd[pi]:.2f}', 'ho_per_ep': f'{nho[pi]:.2f}'})
        print(f"  {d:4.0f} dB : " + " ".join(f'{v:5.1f}' for v in m) +
              f"   (RLF %)  [{(time.time()-t0)/60:.1f} min]", flush=True)
        with open(a.out, 'w', newline='') as f:                 # checkpoint after every row
            w = csv.DictWriter(f, fieldnames=['block_db', 'pt_dbm', 'RLF_mean', 'RLF_std',
                                              'ho_per_ep'])
            w.writeheader(); w.writerows(rows)
    print(f"-> {a.out}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
