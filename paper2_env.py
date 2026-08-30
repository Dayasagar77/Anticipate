#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_env.py — Paper-2 integrated training environment: MultiRSUBlockageEnv.

Composes everything validated in Results #1–M3 into one gym environment:
  * DAIR-V2X geometry: a real intersection scene (moving UE + surrounding vehicles, per frame).
  * Multiple roadside gNBs/RSUs (the 4 intersection corners) — enables HANDOVER.
  * Real 3D blockage on every UE->RSU link (blockage3d, ITU-R P.526 knife-edge).
  * Multi-UE interference: SINR with the K nearest vehicles as co-channel interferers.
  * Joint branched action: handover x power x spectrum (Paper 1 backbone).
  * Geometry ANTICIPATION state in the observation (anticipate.py).
  * Reliability-tilted reward; sustained-RLF (SINR < SINR_out for N310) termination.

R3 REVISION — three methodological fixes, all of which change reported numbers:

  S1  HELD-OUT SCENE SPLIT.  The 45 scenes are partitioned deterministically (independent of the
      training seed) into 30 train / 15 test.  `split='train'` draws only training scenes;
      `split='test'` only held-out ones.  Previously training and evaluation both sampled from all
      45 scenes, i.e. the controller was evaluated in-sample.

  S2  COMMON RANDOM NUMBERS.  Shadow fading is now pre-drawn once per episode as a standardised
      array z[t, gnb, link] and scaled by sigma(LOS/NLOS) at use.  Two consequences:
        (a) reset(seed=k) fully determines the scene AND the whole fading realisation, so every
            configuration can be evaluated on byte-identical scenarios (paired comparison);
        (b) one fading realisation per link per step, which is what the manuscript states.  The
            previous code re-drew fading on every _sinr_db() call, so the same link at the same
            step received different fading in the handover-candidate scan than in the reward
            computation.
        Fading remains independent ACROSS steps (no temporal correlation), as documented.

  S3  _sinr_db now takes (gnb_index, frame_index) rather than a frame dict, so the fading lookup
      is well defined.  Callers outside this file must use the new signature.

INTEGRITY: real geometry + Paper 1 channel/noise/RLF constants; nothing fabricated.
"""
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/root/work/exp')
import numpy as np, pandas as pd
import gymnasium as gym
from gymnasium import spaces
import blockage3d as B
import anticipate as A


def _p1():
    import channel_model_nyu as ch
    import channel_model as cm
    try:
        from env_6g_v5 import RLF_THRESH_DBM, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX
        from env_6g_v6 import N310_DEFAULT
    except Exception:
        RLF_THRESH_DBM, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX, N310_DEFAULT = -92., 23., 30., 10, 66, 3
    return ch, cm, RLF_THRESH_DBM, PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX, N310_DEFAULT

CH, CM, RLF_THRESH_DBM, PT_MIN, PT_MAX, RB_MIN, RB_MAX, N310 = _p1()
SINR_OUT_DB = RLF_THRESH_DBM - CM.N0_DBM          # -13.8 dB : K=0 reproduces Paper 1's RSRP RLF
N0_MW = 10 ** (CM.N0_DBM / 10.0)
INTERF_PT_DBM, G_SERV, G_SIDE = 30.0, CH.ARRAY_GAIN_DBI, 0.0
PT_STEP, RB_STEP = 1.0, 8
HORIZON = A.HORIZON_S
# reward weights (reliability-tilted)
W_REL, RLF_PEN, W_PWR, W_SPEC, W_HO = 1.0, 20.0, 0.15, 0.05, 0.05
# --- W_PWR override, added 30 Aug 2026 for the reward-weight sensitivity sweep. ---------------
# The transmit-power penalty weight is the one reward term a reviewer can argue produces the
# bimodal convergence structure (a linear power cost against a step-shaped reliability cliff).
# Testing that needs W_PWR varied WITHOUT touching anything else, so it is read here, at module
# scope, where no import order can defeat it. Unset -> 0.15, the published value, unchanged.
_W_PWR_ENV = os.environ.get('P2_W_PWR')
if _W_PWR_ENV is not None:
    W_PWR = float(_W_PWR_ENV)
# ----------------------------------------------------------------------------------------------

# ---------------------------------------------------------------- S1: the scene split
SPLIT_SEED = 20260725          # fixed forever; the split must not move with the training seed
N_TRAIN_SCENES = 30


def scene_split(scene_ids, n_train=N_TRAIN_SCENES, seed=SPLIT_SEED):
    """Deterministic train/held-out partition of the scene pool.

    Sorted first so the result depends only on the SET of scene ids, not on their order in the
    csv; then a fixed-seed permutation. Identical for every configuration and every training seed.
    """
    ids = sorted(set(map(str, scene_ids)))
    perm = np.random.default_rng(seed).permutation(len(ids))
    tr = sorted(ids[i] for i in perm[:n_train])
    te = sorted(ids[i] for i in perm[n_train:])
    return tr, te


class MultiRSUBlockageEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, pool_df, n_interferers=4, seed=0, mask_anticipation=False, scene_subset=None,
                 ho_latency=2, split=None):
        super().__init__()
        self.mask_ant = mask_anticipation          # anticipation ablation: zero the N1 features
        self.ho_lat = int(ho_latency)              # handover completion latency (steps); too-late HO fails
        self._subset = set(map(str, scene_subset)) if scene_subset is not None else None
        pool_df = pool_df.copy(); pool_df['scene_id'] = pool_df['scene_id'].astype(str)
        # clean incomplete annotations: zero-fill missing velocities; drop non-UE vehicles lacking box
        # dimensions (cannot be modelled as blockers). UE box dims are unused (it is a ray endpoint).
        pool_df['v_x'] = pool_df['v_x'].fillna(0.0); pool_df['v_y'] = pool_df['v_y'].fillna(0.0)
        nodim = pool_df[['length', 'width', 'height', 'theta']].isna().any(axis=1)
        pool_df = pool_df[~(nodim & ~pool_df['is_ue'])].copy()
        pool_df[['length', 'width', 'height', 'theta']] = \
            pool_df[['length', 'width', 'height', 'theta']].fillna(0.0)
        self.scene_ids = list(dict.fromkeys(pool_df['scene_id']))
        # -- S1: restrict to the train or held-out half of the deterministic split
        self.split = split
        if split is not None:
            tr, te = scene_split(self.scene_ids)
            keep = set(tr if split == 'train' else te)
            assert keep, f'empty split {split}'
            self.scene_ids = [s for s in self.scene_ids if s in keep]
        if self._subset is not None:               # restrict to a scene subset (e.g. hard/blocked scenes)
            self.scene_ids = [s for s in self.scene_ids if s in self._subset] or self.scene_ids
        self.pool = {s: pool_df[pool_df['scene_id'] == s] for s in self.scene_ids}
        self.K = n_interferers
        self.rng = np.random.default_rng(seed)
        self._cache = {}                              # scene_id -> precomputed geometry
        # obs: 5 physics + 5 anticipation + 1 handover-in-progress ; action: branched [HO(2),pwr(3),spec(3)]
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 3, 3])

    # ---------- per-scene geometry precompute (cached in-memory AND on disk) ----------
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geom_cache')

    def _precompute(self, sid):
        if sid in self._cache:
            return self._cache[sid]
        import pickle
        cf = os.path.join(self.CACHE_DIR, f'{sid}.pkl')            # geometry is config-independent
        if os.path.exists(cf):
            try:
                frames = pickle.load(open(cf, 'rb')); self._cache[sid] = frames; return frames
            except Exception:
                pass
        sdf = self.pool[sid]
        gnbs = [B.scene_gnb(sdf, B.H_GNB_DEFAULT, mode=m) for m in ['SW', 'SE', 'NW', 'NE']]
        frames = []
        for ts, fr in sdf.groupby('timestamp'):
            if not fr['is_ue'].any():
                continue
            ue = fr[fr['is_ue']].iloc[0]; others = fr[~fr['is_ue']]
            boxes = others[['x', 'y', 'length', 'width', 'height', 'theta']].to_dict('records')
            vehs = others[['x', 'y', 'length', 'width', 'height', 'theta', 'v_x', 'v_y']].to_dict('records')
            per_g = []
            for g in gnbs:
                r = B.link_blockage((ue['x'], ue['y']), B.H_UE_DEFAULT, g[:2], g[2], boxes)
                af = A.anticipation_features(dict(x=ue['x'], y=ue['y'], v_x=ue['v_x'], v_y=ue['v_y']), g, vehs)
                # K nearest vehicles to this gNB as interferers: (dist, depth)
                ds = sorted(((math.hypot(o['x'] - g[0], o['y'] - g[1]), o) for o in boxes), key=lambda z: z[0])
                intf = []
                for d, o in ds[:self.K]:
                    rr = B.link_blockage((o['x'], o['y']), B.H_UE_DEFAULT, g[:2], g[2],
                                         [b for b in boxes if b is not o])
                    intf.append((d, rr['depth_db']))
                per_g.append(dict(dist=math.hypot(ue['x'] - g[0], ue['y'] - g[1]), depth=r['depth_db'],
                                  ttb=af['ttb_s'], projdepth=af['proj_depth_db'], ncorr=af['n_corridor'],
                                  closing=af['closing_rate_mps'], intf=intf))
            frames.append(dict(per_g=per_g, speed=float(math.hypot(ue['v_x'], ue['v_y']))))
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            pickle.dump(frames, open(cf, 'wb'))
        except Exception:
            pass
        self._cache[sid] = frames
        return frames

    # ---------- SINR (S2/S3: fading is a lookup into the per-episode CRN array) ----------
    @staticmethod
    def _sigma(depth_db):
        return CH.SIG_NLOS if depth_db > 0 else CH.SIG_LOS

    def _sinr_db(self, gi, t, pt):
        """SINR (dB) on link gi at frame t with transmit power pt (dBm).

        Shadow fading for (t, gi) is read from the pre-drawn standardised array self._z, so it is
        the SAME realisation wherever this link/step is evaluated within the episode, and the SAME
        realisation across configurations evaluated with the same reset seed.
        """
        g = self.frames[t]['per_g'][gi]
        z = self._z[t, gi]
        p_des = 10 ** ((pt + G_SERV - CH.ci_pathloss_db(max(g['dist'], 1.0), True)
                        - z[0] * self._sigma(g['depth']) - g['depth']) / 10.0)
        inti = 0.0
        for j, (d, dep) in enumerate(g['intf']):
            inti += 10 ** ((INTERF_PT_DBM + G_SIDE - CH.ci_pathloss_db(max(d, 1.0), True)
                            - z[1 + j] * self._sigma(dep) - dep) / 10.0)
        return 10 * math.log10(p_des / (N0_MW + inti))

    def _sinr_noiseless_db(self, gi, t, pt):
        """Expected SINR with the fading realisation set to zero — used only by the genie
        upper-bound baseline, which is granted perfect knowledge of GEOMETRY but not of fading."""
        g = self.frames[t]['per_g'][gi]
        p_des = 10 ** ((pt + G_SERV - CH.ci_pathloss_db(max(g['dist'], 1.0), True) - g['depth']) / 10.0)
        inti = 0.0
        for d, dep in g['intf']:
            inti += 10 ** ((INTERF_PT_DBM + G_SIDE - CH.ci_pathloss_db(max(d, 1.0), True) - dep) / 10.0)
        return 10 * math.log10(p_des / (N0_MW + inti))

    def _obs(self, sinr):
        g = self.frames[self.t]['per_g'][self.serv]
        cand_ttb = max((self.frames[self.t]['per_g'][j]['ttb'] for j in range(4) if j != self.serv), default=HORIZON)
        ant = [g['ttb'] / HORIZON, min(g['projdepth'] / 40.0, 2.0), min(g['ncorr'] / 10.0, 2.0),
               np.clip(g['closing'] / 10.0, -2, 2), cand_ttb / HORIZON]
        if self.mask_ant:
            ant = [0.0] * 5                            # anticipation ablation: controller is reactive
        ho = [float(self._ho_timer > 0)]               # handover-in-progress flag (physics, never masked)
        return np.array([
            np.clip(sinr / 30.0, -2, 2), np.clip((sinr - SINR_OUT_DB) / 20.0, -2, 2),
            (self.pt - PT_MIN) / (PT_MAX - PT_MIN), (self.rb - RB_MIN) / (RB_MAX - RB_MIN),
            np.clip(self.frames[self.t]['speed'] / 30.0, 0, 2)] + ant + ho, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """reset(seed=k) is fully deterministic: it fixes BOTH the scene and the entire shadow-fading
        realisation of the episode, so two controllers reset with the same k face byte-identical
        conditions (common random numbers)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        sid = self.scene_ids[self.rng.integers(len(self.scene_ids))]
        self.frames = self._precompute(sid)
        self.scene_id = sid
        # -- S2: pre-draw standardised fading for every (frame, gNB, [serving link | K interferers])
        self._z = self.rng.normal(0.0, 1.0, size=(len(self.frames), 4, 1 + self.K))
        self.t = 0; self.pt = PT_MAX; self.rb = RB_MAX; self._consec = 0
        # serving = gNB with best initial SINR
        self.serv = int(np.argmax([self._sinr_db(j, 0, self.pt) for j in range(4)]))
        self._ho_timer = 0; self._ho_target = self.serv
        sinr = self._sinr_db(self.serv, 0, self.pt)
        return self._obs(sinr), {'scene': sid}

    def step(self, action):
        a_ho, a_pwr, a_rb = int(action[0]), int(action[1]), int(action[2])
        ho_started = False
        # handover FSM: takes ho_lat steps; the UE stays on the OLD gNB during transit, so a handover
        # initiated AFTER the serving link blocks arrives too late, while an anticipatory one completes.
        if self._ho_timer > 0:
            self._ho_timer -= 1
            if self._ho_timer == 0:
                self.serv = self._ho_target            # handover completes -> new gNB
        elif a_ho == 1:                                # initiate handover to the best other gNB
            cand = int(np.argmax([self._sinr_db(j, self.t, self.pt) if j != self.serv else -1e9
                                  for j in range(4)]))
            if cand != self.serv:
                self._ho_target = cand; self._ho_timer = self.ho_lat; ho_started = True
        self.pt = float(np.clip(self.pt + (a_pwr - 1) * PT_STEP, PT_MIN, PT_MAX))
        self.rb = int(np.clip(self.rb + (a_rb - 1) * RB_STEP, RB_MIN, RB_MAX))
        self.t += 1
        done = self.t >= len(self.frames) - 1
        sinr = self._sinr_db(self.serv, self.t, self.pt)
        link_ok = sinr >= SINR_OUT_DB
        self._consec = 0 if link_ok else self._consec + 1
        rlf = self._consec >= N310
        r = (W_REL * (1.0 if link_ok else 0.0) - RLF_PEN * (1.0 if rlf else 0.0)
             - W_PWR * (self.pt - PT_MIN) / (PT_MAX - PT_MIN) - W_SPEC * (self.rb - RB_MIN) / (RB_MAX - RB_MIN)
             - W_HO * (1.0 if a_ho == 1 else 0.0))
        terminated = bool(rlf)
        return self._obs(sinr), float(r), terminated, bool(done and not terminated), \
            {'sinr': sinr, 'rlf': rlf, 'serv': self.serv, 'pt': self.pt, 'rb': self.rb,
             'ho': ho_started}


def _smoke():
    pool = pd.read_csv('dair_geometry_pool.csv')
    tr, te = scene_split(pool['scene_id'].astype(str).unique())
    assert not (set(tr) & set(te)) and len(tr) + len(te) == 45, 'split must partition the pool'
    print(f'split: {len(tr)} train / {len(te)} held-out, disjoint OK')
    print(f'  train head: {tr[:6]}')
    print(f'  test  all : {te}')

    e_tr = MultiRSUBlockageEnv(pool, seed=0, split='train')
    e_te = MultiRSUBlockageEnv(pool, seed=0, split='test')
    assert not (set(e_tr.scene_ids) & set(e_te.scene_ids))
    print(f'env scenes: train={len(e_tr.scene_ids)} test={len(e_te.scene_ids)} '
          f'obs={e_tr.observation_space.shape} act={list(e_tr.action_space.nvec)} '
          f'SINR_out={SINR_OUT_DB:.1f} dB N310={N310}')

    # -- CRN determinism: same reset seed must give the identical scene, fading and trajectory
    def fixed_rollout(env, k):
        obs, info = env.reset(seed=k)
        rng = np.random.default_rng(7)                 # identical action stream both times
        trace, done, trunc = [float(obs[0])], False, False
        while not (done or trunc):
            obs, r, done, trunc, i2 = env.step([rng.integers(2), rng.integers(3), rng.integers(3)])
            trace.append(float(i2['sinr']))
        return info['scene'], np.array(trace)

    s1, t1 = fixed_rollout(e_te, 1_000_000)
    s2, t2 = fixed_rollout(e_te, 1_000_000)
    assert s1 == s2 and t1.shape == t2.shape and np.allclose(t1, t2), 'CRN reset is not deterministic'
    s3, t3 = fixed_rollout(e_te, 1_000_001)
    print(f'CRN: seed 1e6 -> scene {s1}, {len(t1)} steps, reproducible bit-for-bit; '
          f'seed 1e6+1 -> scene {s3} (differs: {s1 != s3 or not np.allclose(t1, t3[:len(t1)])})')

    # -- one fading realisation per link per step
    e_te.reset(seed=1_000_000)
    a = e_te._sinr_db(0, 0, 30.0); b = e_te._sinr_db(0, 0, 30.0)
    assert a == b, 'fading must not be re-drawn within a step'
    print(f'fading lookup stable within a step: {a:.3f} == {b:.3f}')
    print('SMOKE OK — split disjoint, CRN deterministic, fading fixed per link per step.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--p1', default=None)
    a = ap.parse_args()
    if a.p1:
        sys.path.insert(0, a.p1)
    if a.smoke:
        _smoke()


if __name__ == '__main__':
    main()
