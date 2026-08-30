#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2ma_env.py — Paper-2 (multi-agent) environment: MultiAgentRSUBlockageEnv.

The single new variable over paper2_env.py: single-UE -> MULTI-AGENT. M vehicles per scene are
promoted to learning agents, each managing its own link (handover x power x sub-band). Everything
else -- DAIR geometry, ITU-R P.526 3D blockage, NYU-calibrated 140 GHz channel, CRN fading, the
30/15 scene split -- is inherited from paper2_env so the ONLY thing that changes is the number of
agents.

What becomes real by going multi-agent (all disclosed limitations of the single-UE paper):
  * INTERFERENCE is now endogenous: agent j's chosen transmit power, on an OVERLAPPING sub-band,
    lands on agent i's serving gNB along j's own blockage-attenuated path. No fixed-power proxy for
    the learning agents (static non-agent vehicles keep the loaded-cell proxy as background).
  * SPECTRUM is now load-bearing: the sub-band action selects one of N_SUBBANDS orthogonal channels;
    two agents on the SAME sub-band whose transmissions reach the same gNB interfere, on DIFFERENT
    sub-bands they do not. Spectrum choice therefore determines co-channel interference -- the actuator
    that did nothing in the single-UE model now matters.
  * MUTUAL BLOCKAGE: every agent's UE->gNB ray is tested against all other vehicles' 3D boxes, the
    other agents included. Blockage is fixed by the replayed trajectory (an agent cannot act its body
    out of another's line of sight), so blockage-driven handover demand is CORRELATED across agents.

Reward / reliability: to avoid the episode-truncation reward trap diagnosed in the single-UE study,
the episode is NOT terminated on RLF. It runs the whole scene. step() returns, per agent, an
EFFICIENCY reward and -- separately, in info -- the per-agent sustained-RLF cost, so a constrained
(Lagrangian) trainer sets the reliability price explicitly instead of inheriting it from a truncation
accident. r_eff = W_REL*link_ok - W_PWR*power_cost - W_SPEC*spectrum_cost - W_HO*handover.

INTEGRITY: real geometry + Paper-1 channel constants (reused from paper2_env); nothing fabricated.
This is a PROTOTYPE: its first job is to validate the couplings and measure per-episode cost on the
45 example scenes before any training budget is committed.
"""
import os, sys, math, argparse, time, pickle, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import blockage3d as B
import anticipate as A
import paper2_env as PE            # reuse channel constants, scene_split, reward weights

# --- inherited constants (single source of truth = paper2_env) ---
CH, CM = PE.CH, PE.CM
SINR_OUT_DB, N0_MW = PE.SINR_OUT_DB, PE.N0_MW
G_SERV, G_SIDE = PE.G_SERV, PE.G_SIDE
INTERF_PT_DBM = PE.INTERF_PT_DBM
PT_MIN, PT_MAX, PT_STEP = PE.PT_MIN, PE.PT_MAX, PE.PT_STEP
N310 = PE.N310
HORIZON = PE.HORIZON
W_REL, RLF_PEN, W_PWR, W_SPEC, W_HO = PE.W_REL, PE.RLF_PEN, PE.W_PWR, PE.W_SPEC, PE.W_HO
scene_split = PE.scene_split

GNBS = ['SW', 'SE', 'NW', 'NE']    # the four roadside gNBs, shared by all agents
N_GNB = len(GNBS)
N_SUBBANDS = 3                     # orthogonal spectrum channels the spectrum action selects among
K_BG = 4                           # static (non-agent) vehicles as background interferers per gNB


def _sigma(depth_db):
    return CH.SIG_NLOS if depth_db > 0 else CH.SIG_LOS


class MultiAgentRSUBlockageEnv(gym.Env):
    """M vehicles as simultaneous learning users and mutual blockers over one DAIR scene.

    Agents act synchronously each frame. Per-agent branched action [handover(2), power(3), band(3)];
    per-agent observation is the single-UE 11-vector plus two local-contention features (co-agents on
    the serving gNB, co-agents on the serving sub-band), i.e. 13-d. Agents absent from a frame are
    masked (inactive: no signal, no interference, no reward).
    """
    metadata = {'render_modes': []}
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geom_cache_ma')

    def __init__(self, pool_df, n_agents=8, seed=0, mask_anticipation=False, split=None,
                 ho_latency=2, n_subbands=N_SUBBANDS, k_bg=K_BG, gnb_modes=None,
                 compute_anticipation=True, gnb_cap=None):
        super().__init__()
        self.M = int(n_agents)
        self.mask_ant = mask_anticipation
        self.want_ant = bool(compute_anticipation)   # skip the costly forecast for reference-policy runs
        # gNB admission capacity: max agents a tower can schedule per frame. None = unlimited (legacy).
        # When a blockage wave forces many agents onto the few clear towers at once, an over-subscribed
        # tower serves only its top-capacity agents (by desired signal); the surplus are unserved that
        # frame (link fails). This is the association-congestion coupling that makes coordination matter
        # at sub-THz, where pencil beams make co-channel interference too weak to be the lever.
        self.gnb_cap = None if gnb_cap is None else int(gnb_cap)
        self.ho_lat = int(ho_latency)
        self.n_sb = int(n_subbands)
        self.k_bg = int(k_bg)
        self.gnb_modes = list(gnb_modes) if gnb_modes else list(GNBS)
        self.n_gnb = len(self.gnb_modes)
        pool_df = pool_df.copy(); pool_df['scene_id'] = pool_df['scene_id'].astype(str)
        pool_df['v_x'] = pool_df['v_x'].fillna(0.0); pool_df['v_y'] = pool_df['v_y'].fillna(0.0)
        nodim = pool_df[['length', 'width', 'height', 'theta']].isna().any(axis=1)
        pool_df = pool_df[~(nodim & ~pool_df['is_ue'])].copy()
        pool_df[['length', 'width', 'height', 'theta']] = \
            pool_df[['length', 'width', 'height', 'theta']].fillna(0.0)
        self.scene_ids = list(dict.fromkeys(pool_df['scene_id']))
        self.split = split
        if split is not None:
            tr, te = scene_split(self.scene_ids)
            keep = set(tr if split == 'train' else te)
            assert keep, f'empty split {split}'
            self.scene_ids = [s for s in self.scene_ids if s in keep]
        self.pool = {s: pool_df[pool_df['scene_id'] == s] for s in self.scene_ids}
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        # per-agent obs: 5 physics + 5 anticipation + 1 ho-flag + 2 contention + n_gnb tower-load = 13+n_gnb.
        # The per-tower load vector is the coordination signal: each agent sees how full every gNB is, so
        # it can flee an over-subscribed tower toward a clear one instead of colliding blindly.
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(13 + self.n_gnb,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 3, 3])   # per agent

    # ---------- agent selection: the M vehicles present in the most frames (UE always included) ----------
    def _select_agents(self, sdf):
        counts = sdf.groupby('agent_id')['timestamp'].nunique().sort_values(ascending=False)
        ue_ids = list(sdf[sdf['is_ue']]['agent_id'].unique())
        ordered = ue_ids + [a for a in counts.index if a not in ue_ids]
        sel = ordered[:self.M]
        # sparse scenes may hold fewer than M vehicles; pad with None so every scene has exactly M
        # agent slots (the padded ones are never present -> permanently inactive/masked).
        sel += [None] * (self.M - len(sel))
        return sel

    # ---------- per-scene geometry precompute (config-independent; cached on disk) ----------
    def _precompute(self, sid):
        key = (sid, self.M, self.n_sb, self.k_bg, tuple(self.gnb_modes), self.want_ant)
        if key in self._cache:
            return self._cache[key]
        h = hashlib.md5(repr(key).encode()).hexdigest()[:10]
        cf = os.path.join(self.CACHE_DIR, f'{sid}_{h}.pkl')
        if os.path.exists(cf):
            try:
                obj = pickle.load(open(cf, 'rb')); self._cache[key] = obj; return obj
            except Exception:
                pass
        sdf = self.pool[sid]
        agents = self._select_agents(sdf)
        gnbs = [B.scene_gnb(sdf, B.H_GNB_DEFAULT, mode=m) for m in self.gnb_modes]
        NG = self.n_gnb
        frames = []
        for ts, fr in sdf.groupby('timestamp'):
            present = {r['agent_id']: r for _, r in fr.iterrows()}
            all_boxes = fr[['x', 'y', 'length', 'width', 'height', 'theta']].to_dict('records')
            # per-agent geometry vs every gNB
            ag = []
            for aid in agents:
                if aid not in present:
                    ag.append(None); continue
                v = present[aid]
                # blockers for this agent's rays = all OTHER vehicles (agents included -> mutual blockage)
                boxes = [b for b in all_boxes
                         if not (abs(b['x'] - v['x']) < 1e-9 and abs(b['y'] - v['y']) < 1e-9)]
                vehs = [dict(x=b['x'], y=b['y'], length=b['length'], width=b['width'],
                             height=b['height'], theta=b['theta'], v_x=0.0, v_y=0.0) for b in boxes]
                dist = np.zeros(NG); depth = np.zeros(NG)
                ttb = np.zeros(NG); projd = np.zeros(NG); ncorr = np.zeros(NG); clos = np.zeros(NG)
                for gi, g in enumerate(gnbs):
                    r = B.link_blockage((v['x'], v['y']), B.H_UE_DEFAULT, g[:2], g[2], boxes)
                    dist[gi] = math.hypot(v['x'] - g[0], v['y'] - g[1]); depth[gi] = r['depth_db']
                    if self.want_ant:
                        af = A.anticipation_features(dict(x=v['x'], y=v['y'], v_x=v['v_x'], v_y=v['v_y']), g, vehs)
                        ttb[gi] = af['ttb_s']; projd[gi] = af['proj_depth_db']
                        ncorr[gi] = af['n_corridor']; clos[gi] = af['closing_rate_mps']
                    else:
                        ttb[gi] = HORIZON
                ag.append(dict(dist=dist, depth=depth, ttb=ttb, projd=projd, ncorr=ncorr, clos=clos,
                               speed=float(math.hypot(v['v_x'], v['v_y']))))
            # background interference per gNB: K nearest STATIC (non-agent) vehicles
            agent_xy = set((round(present[a]['x'], 3), round(present[a]['y'], 3))
                           for a in agents if a in present)
            statics = [b for b in all_boxes if (round(b['x'], 3), round(b['y'], 3)) not in agent_xy]
            bg = np.zeros(NG)
            for gi, g in enumerate(gnbs):
                ds = sorted((math.hypot(b['x'] - g[0], b['y'] - g[1]), b) for b in statics)
                acc = 0.0
                for d, b in ds[:self.k_bg]:
                    rr = B.link_blockage((b['x'], b['y']), B.H_UE_DEFAULT, g[:2], g[2],
                                         [c for c in statics if c is not b])
                    acc += 10 ** ((INTERF_PT_DBM + G_SIDE - CH.ci_pathloss_db(max(d, 1.0), True)
                                   - rr['depth_db']) / 10.0)
                bg[gi] = acc
            active = np.array([a is not None for a in ag], dtype=bool)
            frames.append(dict(ag=ag, bg=bg, active=active))
        obj = dict(agents=[str(a) for a in agents], frames=frames)
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            pickle.dump(obj, open(cf, 'wb'))
        except Exception:
            pass
        self._cache[key] = obj
        return obj

    # ---------- gNB admission (finite capacity) ----------
    def _desired_dbm(self, i, g):
        a = self.frames[self.t]['ag'][i]
        return self.pt[i] + G_SERV - CH.ci_pathloss_db(max(a['dist'][g], 1.0), True) - a['depth'][g]

    def _compute_admission(self):
        """Which active agents their serving gNB can schedule this frame. Unlimited if gnb_cap is None;
        otherwise each tower admits its top-capacity agents by desired signal, surplus are unserved."""
        fr = self.frames[self.t]
        adm = fr['active'].copy()
        if self.gnb_cap is None:
            return adm
        for g in range(self.n_gnb):
            on_g = [i for i in range(self.M) if fr['active'][i] and self.serv[i] == g]
            if len(on_g) <= self.gnb_cap:
                continue
            ranked = sorted(on_g, key=lambda i: self._desired_dbm(i, g), reverse=True)
            for i in ranked[self.gnb_cap:]:
                adm[i] = False                       # surplus over capacity: unserved this frame
        return adm

    # ---------- SINR for agent i at the current frame ----------
    def _sinr_db(self, i):
        fr = self.frames[self.t]
        a = fr['ag'][i]
        if a is None:
            return SINR_OUT_DB - 30.0
        g = self.serv[i]
        z_des = self._z[self.t, i, g]
        p_des = 10 ** ((self.pt[i] + G_SERV - CH.ci_pathloss_db(max(a['dist'][g], 1.0), True)
                        - z_des * _sigma(a['depth'][g]) - a['depth'][g]) / 10.0)
        inti = float(fr['bg'][g])
        for j in range(self.M):
            if j == i or not fr['active'][j] or self.sb[j] != self.sb[i]:
                continue
            if not self._admitted[j]:                # unserved agents do not transmit -> no interference
                continue
            aj = fr['ag'][j]
            zj = self._z[self.t, j, g]
            inti += 10 ** ((self.pt[j] + G_SIDE - CH.ci_pathloss_db(max(aj['dist'][g], 1.0), True)
                            - zj * _sigma(aj['depth'][g]) - aj['depth'][g]) / 10.0)
        return 10 * math.log10(p_des / (N0_MW + inti))

    def _obs(self, i, sinr):
        fr = self.frames[self.t]
        a = fr['ag'][i]
        if a is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        g = self.serv[i]
        cand_ttb = max((a['ttb'][j] for j in range(self.n_gnb) if j != g), default=HORIZON)
        ant = [a['ttb'][g] / HORIZON, min(a['projd'][g] / 40.0, 2.0), min(a['ncorr'][g] / 10.0, 2.0),
               np.clip(a['clos'][g] / 10.0, -2, 2), cand_ttb / HORIZON]
        if self.mask_ant:
            ant = [0.0] * 5
        ho = [float(self._ho_timer[i] > 0)]
        # contention features: co-agents on my serving gNB, and on my sub-band
        co_g = sum(1 for j in range(self.M) if j != i and fr['active'][j] and self.serv[j] == g)
        co_b = sum(1 for j in range(self.M) if j != i and fr['active'][j] and self.sb[j] == self.sb[i])
        cont = [min(co_g / 4.0, 2.0), min(co_b / 4.0, 2.0)]
        # per-tower load (the coordination signal): agents currently served by each gNB, /capacity
        cap = float(self.gnb_cap) if self.gnb_cap else float(self.M)
        occ = [0] * self.n_gnb
        for j in range(self.M):
            if fr['active'][j]:
                occ[self.serv[j]] += 1
        load = [min(o / cap, 2.0) for o in occ]
        return np.array([
            np.clip(sinr / 30.0, -2, 2), np.clip((sinr - SINR_OUT_DB) / 20.0, -2, 2),
            (self.pt[i] - PT_MIN) / (PT_MAX - PT_MIN), self.sb[i] / max(self.n_sb - 1, 1),
            np.clip(a['speed'] / 30.0, 0, 2)] + ant + ho + cont + load, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        sid = self.scene_ids[self.rng.integers(len(self.scene_ids))]
        obj = self._precompute(sid)
        self.frames = obj['frames']; self.agent_ids = obj['agents']; self.scene_id = sid
        F = len(self.frames)
        # CRN fading: one standardised draw per (frame, agent, gNB)
        self._z = self.rng.normal(0.0, 1.0, size=(F, self.M, self.n_gnb))
        self.t = 0
        self.pt = np.full(self.M, PT_MAX); self.sb = np.zeros(self.M, dtype=int)
        self._consec = np.zeros(self.M, dtype=int)
        self._ho_timer = np.zeros(self.M, dtype=int); self._ho_target = np.zeros(self.M, dtype=int)
        # initial serving gNB = best instantaneous SINR (per agent, ignoring mutual interference)
        self.serv = np.zeros(self.M, dtype=int)
        for i in range(self.M):
            a = self.frames[0]['ag'][i]
            if a is None:
                continue
            self.serv[i] = int(np.argmax([
                self.pt[i] + G_SERV - CH.ci_pathloss_db(max(a['dist'][gi], 1.0), True) - a['depth'][gi]
                for gi in range(self.n_gnb)]))
        self._ho_target = self.serv.copy()
        self._admitted = self._compute_admission()
        obs = np.stack([self._obs(i, self._sinr_db(i)) for i in range(self.M)])
        return obs, {'scene': sid, 'agents': self.agent_ids}

    def step(self, action):
        action = np.asarray(action).reshape(self.M, 3)
        fr = self.frames[self.t]
        ho_started = np.zeros(self.M, dtype=bool)
        for i in range(self.M):
            if not fr['active'][i]:
                continue
            a_ho, a_pwr, a_sb = int(action[i, 0]), int(action[i, 1]), int(action[i, 2])
            if self._ho_timer[i] > 0:
                self._ho_timer[i] -= 1
                if self._ho_timer[i] == 0:
                    self.serv[i] = self._ho_target[i]
            elif a_ho == 1:
                a = fr['ag'][i]
                cand = int(np.argmax([
                    (self.pt[i] + G_SERV - CH.ci_pathloss_db(max(a['dist'][gi], 1.0), True) - a['depth'][gi])
                    if gi != self.serv[i] else -1e9 for gi in range(self.n_gnb)]))
                if cand != self.serv[i]:
                    self._ho_target[i] = cand; self._ho_timer[i] = self.ho_lat; ho_started[i] = True
            self.pt[i] = float(np.clip(self.pt[i] + (a_pwr - 1) * PT_STEP, PT_MIN, PT_MAX))
            self.sb[i] = int(np.clip(self.sb[i] + (a_sb - 1), 0, self.n_sb - 1))
        self.t += 1
        done = self.t >= len(self.frames) - 1
        fr = self.frames[self.t]
        self._admitted = self._compute_admission()
        obs = np.zeros((self.M, self.observation_space.shape[0]), dtype=np.float32)
        rew = np.zeros(self.M, dtype=np.float32)
        rlf = np.zeros(self.M, dtype=bool)
        link_ok = np.zeros(self.M, dtype=bool)
        for i in range(self.M):
            if not fr['active'][i]:
                self._consec[i] = 0
                continue
            a_ho = int(action[i, 0])
            if not self._admitted[i]:                 # over-subscribed tower could not schedule this agent
                self._consec[i] += 1
                rlf[i] = self._consec[i] >= N310
                rew[i] = -W_HO * (1.0 if a_ho == 1 else 0.0)   # unserved: no availability, not transmitting
                obs[i] = self._obs(i, SINR_OUT_DB - 20.0)
                continue
            sinr = self._sinr_db(i)
            ok = sinr >= SINR_OUT_DB
            link_ok[i] = ok
            self._consec[i] = 0 if ok else self._consec[i] + 1
            rlf[i] = self._consec[i] >= N310
            rew[i] = (W_REL * (1.0 if ok else 0.0)
                      - W_PWR * (self.pt[i] - PT_MIN) / (PT_MAX - PT_MIN)
                      - W_SPEC * (self.sb[i] / max(self.n_sb - 1, 1))
                      - W_HO * (1.0 if a_ho == 1 else 0.0))
            obs[i] = self._obs(i, sinr)
        info = {'rlf': rlf, 'link_ok': link_ok, 'active': fr['active'].copy(),
                'admitted': self._admitted.copy(), 'serv': self.serv.copy(),
                'sb': self.sb.copy(), 'pt': self.pt.copy(), 'ho': ho_started}
        # NO termination on RLF (avoids the truncation reward-trap); episode runs the whole scene.
        return obs, rew, False, bool(done), info


def _smoke(pool_path, n_agents, seed=0):
    pool = pd.read_csv(pool_path)
    t0 = time.time()
    env = MultiAgentRSUBlockageEnv(pool, n_agents=n_agents, seed=seed, split='train')
    print(f'built env: {len(env.scene_ids)} train scenes, M={env.M} agents, '
          f'{env.n_sb} sub-bands, obs={env.observation_space.shape}, act={list(env.action_space.nvec)}')
    rng = np.random.default_rng(7)
    # first episode (cold: triggers geometry precompute), then a warm one for step-rate
    for label in ('cold', 'warm'):
        te = time.time()
        obs, info = env.reset(seed=1000 + (0 if label == 'cold' else 1))
        assert obs.shape == (env.M, 13), obs.shape
        steps = 0; tot_rlf = np.zeros(env.M); active_steps = np.zeros(env.M)
        done = False
        while not done:
            act = rng.integers([2, 3, 3], size=(env.M, 3))
            obs, rew, term, done, info = env.step(act)
            tot_rlf += info['rlf'].astype(float); active_steps += info['active'].astype(float)
            steps += 1
        dt = time.time() - te
        rate = steps / dt if dt > 0 else float('inf')
        print(f'  [{label}] scene {info["serv"].shape[0]}-agent, {steps} steps in {dt:.2f}s '
              f'({rate:.0f} steps/s); per-agent RLF steps={tot_rlf.astype(int).tolist()} '
              f'active={active_steps.astype(int).tolist()}')
    # coupling check: same-subband interference must lower SINR vs orthogonal bands
    env.reset(seed=1000)
    for i in range(env.M):
        env.sb[i] = 0                         # force all agents onto one sub-band
    s_same = env._sinr_db(0)
    for i in range(env.M):
        env.sb[i] = i % env.n_sb              # spread across sub-bands
    s_diff = env._sinr_db(0)
    print(f'  coupling: agent0 SINR all-same-band={s_same:.2f} dB, spread-bands={s_diff:.2f} dB '
          f'-> interference {"REDUCES" if s_diff > s_same else "does NOT reduce"} SINR when bands differ')
    print(f'total smoke time {time.time() - t0:.1f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--agents', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        _smoke(a.pool, a.agents, a.seed)


if __name__ == '__main__':
    main()
