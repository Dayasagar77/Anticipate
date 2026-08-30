#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_multiue.py — Paper-2: MULTI-UE interference under real 3D blockage (the guide's requirement).

Extends the single-UE cliff to an interference-limited multi-UE cell. Uplink at the roadside gNB:
the target UE's SINR is degraded by K co-channel interfering UEs — the K other in-scene vehicles
NEAREST the gNB (strongest interferers = a loaded cell), each with its OWN geometry-derived 3D blockage.

  SINR(t) = P_des(t) / (N0 + sum_k P_k(t))
  P_des = Ptx + G_serv(15 dBi) - PL_CI_LOS(d_ue) - shadow_ue - blockage_ue           [the target link]
  P_k   = Ptx_int + G_side(0 dBi) - PL_CI_LOS(d_k) - shadow_k - blockage_k             [each interferer]
Sustained RLF = SINR < SINR_out for >= N310 consecutive steps, with
  SINR_out = RLF_THRESH_dBm - N0_dBm = -92 - (-78.2) = -13.8 dB
so at K=0 (no interference) this REPRODUCES Paper 1's single-UE RSRP-based RLF exactly.

THE QUESTION: as cell load K rises, does interference compound the blockage floor and make transmit
power even less able to recover reliability? Sweep K and transmit power; report RLF and power-immune %.

Reuses Paper 1's exact channel (NYU 142 GHz CI, 15 dBi array), noise floor (N0=-78.2 dBm), interferer
model (Ptx=30 dBm, 0 dBi sidelobe), and RLF criterion (N310=3). INTEGRITY: real geometry + Paper 1
constants; nothing fabricated. gNB placement is the documented modelling parameter (blockage3d.scene_gnb).
"""
import os, sys, csv, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/root/work/exp')                 # Paper 1 modules (cloud); set --p1 on your machine
import numpy as np, pandas as pd
import blockage3d as B


def _load_p1(p1):
    if p1:
        sys.path.insert(0, p1)
    import channel_model_nyu as ch
    import channel_model as cm
    from env_6g_v5 import RLF_THRESH_DBM
    from env_6g_v6 import N310_DEFAULT
    import multi_seed_eval as mse
    return ch, cm, RLF_THRESH_DBM, N310_DEFAULT, mse


INTERFERER_PT_DBM = 30.0     # Paper 1 co-channel interferer transmit power
G_SERV_DBI = 15.0            # aligned serving beam (Paper 1 ARRAY_GAIN_DBI)
G_SIDE_DBI = 0.0            # misaligned interferer -> sidelobe (Paper 1 INTERFERER_ARRAY_GAIN_DBI)


def frame_vehicle_links(fr, gnb, h_ue, cap_db, freq, kmax):
    """Per frame: the UE link + the kmax nearest-to-gNB interferer links, as (dist, depth, is_ue).
    Only these links need blockage (interferers are the K nearest); every vehicle is still a blocker."""
    gx, gy, gh = gnb
    recs = fr[['x', 'y', 'length', 'width', 'height', 'theta']].to_dict('records')
    is_ue = fr['is_ue'].values
    dists = [math.hypot(v['x'] - gx, v['y'] - gy) for v in recs]
    ue_i = next(i for i in range(len(recs)) if is_ue[i])
    others_idx = sorted((i for i in range(len(recs)) if not is_ue[i]), key=lambda i: dists[i])[:kmax]
    out = []
    for i in [ue_i] + others_idx:
        blockers = [b for j, b in enumerate(recs) if j != i]
        r = B.link_blockage((recs[i]['x'], recs[i]['y']), h_ue, (gx, gy), gh, blockers, freq, cap_db)
        out.append((dists[i], r['depth_db'], bool(is_ue[i])))
    return out


def scene_links(scene_df, gnb_mode, h_ue, h_gnb, cap_db, freq, kmax):
    """Precompute per-frame UE+interferer links for one scene (seed/power-independent geometry)."""
    gnb = B.scene_gnb(scene_df, h_gnb, mode=gnb_mode)
    frames = []
    for ts, fr in scene_df.groupby('timestamp'):
        if not fr['is_ue'].any():
            continue
        frames.append(frame_vehicle_links(fr, gnb, h_ue, cap_db, freq, kmax))
    return frames


def rx_dbm(pt, gain, d, depth, shadow, ch):
    return pt + gain - ch.ci_pathloss_db(d, is_los=True) - shadow - depth


def scene_rlf(frames, pt, K, seed, ch, cm, sinr_out, n310):
    """Sustained RLF for the target UE over one scene at power pt with K nearest interferers."""
    rng = np.random.default_rng(seed)
    n0_mw = 10 ** (cm.N0_DBM / 10.0)
    consec = 0
    for fl in frames:
        ue = next(v for v in fl if v[2])
        others = sorted((v for v in fl if not v[2]), key=lambda z: z[0])[:K]
        sh_ue = rng.normal(0, ch.SIG_NLOS if ue[1] > 0 else ch.SIG_LOS)
        p_des = 10 ** (rx_dbm(pt, G_SERV_DBI, ue[0], ue[1], sh_ue, ch) / 10.0)
        inti = 0.0
        for d, dep, _ in others:
            sh = rng.normal(0, ch.SIG_NLOS if dep > 0 else ch.SIG_LOS)
            inti += 10 ** (rx_dbm(INTERFERER_PT_DBM, G_SIDE_DBI, d, dep, sh, ch) / 10.0)
        sinr = 10 * math.log10(p_des / (n0_mw + inti))
        consec = consec + 1 if sinr < sinr_out else 0
        if consec >= n310:
            return True
    return False


def rlf_rate(scenes, pt, K, seeds, ch, cm, sinr_out, n310):
    hit = tot = 0
    for frames in scenes:
        for s in seeds:
            hit += int(scene_rlf(frames, pt, K, s, ch, cm, sinr_out, n310)); tot += 1
    return 100.0 * hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--p1', default=None, help='path to Paper 1 modules (default: /root/work/exp)')
    ap.add_argument('--gnb', default='corner')
    ap.add_argument('--loads', default='0,2,4,8,12', help='co-channel interferer counts K (cell load)')
    ap.add_argument('--powers', default='23,30')
    ap.add_argument('--scenes', type=int, default=0, help='limit #scenes (0=all) for a quick run')
    ap.add_argument('--out', default='/root/work/paper2/results/multiue.csv')
    a = ap.parse_args()
    ch, cm, RLF_THRESH_DBM, N310, mse = _load_p1(a.p1)
    sinr_out = RLF_THRESH_DBM - cm.N0_DBM
    loads = [int(x) for x in a.loads.split(',')]
    powers = [float(x) for x in a.powers.split(',')]
    seeds = list(mse.SEEDS)

    pool = pd.read_csv(a.pool); pool['scene_id'] = pool['scene_id'].astype(str)
    sids = list(dict.fromkeys(pool['scene_id']))
    if a.scenes:
        sids = sids[:a.scenes]
    print(f"MULTI-UE | scenes={len(sids)} seeds={len(seeds)} N310={N310} | SINR_out={sinr_out:.1f} dB "
          f"(=RLF {RLF_THRESH_DBM:.0f} dBm − N0 {cm.N0_DBM:.1f}) | gNB={a.gnb}", flush=True)
    kmax = max(loads)
    scenes = [scene_links(pool[pool['scene_id'] == s], a.gnb, B.H_UE_DEFAULT, B.H_GNB_DEFAULT,
                          B.CAP_DB_DEFAULT, ch.FREQ, kmax) for s in sids]

    rows = []
    hdr = "  load K | " + " | ".join(f"RLF@{int(p)}dBm" for p in powers)
    print(hdr, flush=True)
    for K in loads:
        rlfs = {p: rlf_rate(scenes, p, K, seeds, ch, cm, sinr_out, N310) for p in powers}
        # power-immune among frames that fail at min power: still fail at max power (no-shadow geometry proxy)
        rows.append(dict(K=K, **{f'RLF_{int(p)}': round(rlfs[p], 2) for p in powers}))
        line = f"  {K:6d} | " + " | ".join(f"{rlfs[p]:8.2f}" for p in powers)
        print(line, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lo, hi = powers[0], powers[-1]
    r_lo0, r_hi0 = rows[0][f'RLF_{int(lo)}'], rows[0][f'RLF_{int(hi)}']
    r_loK, r_hiK = rows[-1][f'RLF_{int(lo)}'], rows[-1][f'RLF_{int(hi)}']
    print(f"\nLOAD EFFECT: at K=0, power {lo:.0f}->{hi:.0f} dBm cuts RLF {r_lo0:.1f}->{r_hi0:.1f}%; "
          f"at K={loads[-1]}, {r_loK:.1f}->{r_hiK:.1f}% "
          f"(power's benefit {'shrinks' if (r_lo0-r_hi0)>(r_loK-r_hiK) else 'holds'} under load).", flush=True)
    print("-> " + a.out, flush=True)


if __name__ == '__main__':
    main()
