#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""blockage3d.py — Paper-2 M3: real 3D vehicle-blockage kernel over the DAIR geometry pool.
Cast the UE->gNB 3D ray and test it against each surrounding vehicle's real 3D box; a vehicle blocks
only if its box TOP rises above the ray (1.25 m car clears a ray a 3.87 m bus blocks). Blockage depth
(dB) from ITU-R P.526 single knife-edge diffraction. Model choices documented/ablatable; --selftest
verifies the maths. INTEGRITY: real geometry only; synthetic selftest is never a research-number source.
"""
import os, sys, csv, math, argparse
import numpy as np
import pandas as pd

C_LIGHT = 299_792_458.0
FREQ_HZ = 140e9
H_UE_DEFAULT = 1.5
H_GNB_DEFAULT = 6.0
CAP_DB_DEFAULT = 40.0
BLOCK_THR_DB = 1.0


def knife_edge_loss_db(v):
    if v <= -0.78:
        return 0.0
    return 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)


def _seg_rect_trange(ue_xy, gnb_xy, cx, cy, L, W, theta):
    ct, st = math.cos(-theta), math.sin(-theta)
    def to_local(px, py):
        dx, dy = px - cx, py - cy
        return (dx * ct - dy * st, dx * st + dy * ct)
    x0, y0 = to_local(*ue_xy); x1, y1 = to_local(*gnb_xy)
    dx, dy = x1 - x0, y1 - y0
    hL, hW = L / 2.0, W / 2.0
    p = (-dx, dx, -dy, dy); q = (x0 + hL, hL - x0, y0 + hW, hW - y0)
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        r = qi / pi
        if pi < 0:
            if r > t1: return None
            if r > t0: t0 = r
        else:
            if r < t0: return None
            if r < t1: t1 = r
    return (t0, t1) if t0 <= t1 else None


def link_blockage(ue_xy, h_ue, gnb_xy, h_gnb, boxes, freq_hz=FREQ_HZ, cap_db=CAP_DB_DEFAULT):
    """Blockage of the UE->gNB link by vehicle boxes. Returns dict(blocked, depth_db, blocker, ...)."""
    D = math.hypot(gnb_xy[0] - ue_xy[0], gnb_xy[1] - ue_xy[1])
    lam = C_LIGHT / freq_hz
    best = dict(blocked=False, depth_db=0.0, blocker=None, dh=0.0, d1=0.0, d2=0.0, n_cross=0)
    if D < 1e-3:
        return best
    rising = h_gnb >= h_ue; n_cross = 0
    for k, b in enumerate(boxes):
        tr = _seg_rect_trange(ue_xy, gnb_xy, b['x'], b['y'], b['length'], b['width'], b['theta'])
        if tr is None:
            continue
        n_cross += 1; t0, t1 = tr
        t_eval = t0 if rising else t1
        t_eval = min(max(t_eval, 1e-3), 1.0 - 1e-3)
        h_ray = h_ue + t_eval * (h_gnb - h_ue)
        dh = float(b['height']) - h_ray
        if dh <= 0:
            continue
        d1 = max(D * t_eval, 0.5); d2 = max(D * (1.0 - t_eval), 0.5)
        v = dh * math.sqrt(2.0 / lam * (1.0 / d1 + 1.0 / d2))
        loss = min(knife_edge_loss_db(v), cap_db)
        if loss > best['depth_db']:
            best.update(depth_db=loss, blocker=k, dh=dh, d1=d1, d2=d2)
    best['n_cross'] = n_cross
    best['blocked'] = best['depth_db'] >= BLOCK_THR_DB
    return best


def scene_gnb(scene_df, h_gnb, mode='corner', margin=5.0):
    """Roadside-gNB placement grounded in the scene's traffic extent. 8 peripheral placements
    (4 corners + 4 edge midpoints) span where a roadside pole could sit; sweeping them is the
    robustness test. 'centroid' kept for comparison."""
    x0, x1 = float(scene_df['x'].min()), float(scene_df['x'].max())
    y0, y1 = float(scene_df['y'].min()), float(scene_df['y'].max())
    xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    P = {'SW': (x0 - margin, y0 - margin), 'SE': (x1 + margin, y0 - margin),
         'NW': (x0 - margin, y1 + margin), 'NE': (x1 + margin, y1 + margin),
         'S': (xm, y0 - margin), 'N': (xm, y1 + margin), 'W': (x0 - margin, ym), 'E': (x1 + margin, ym),
         'corner': (x0 - margin, y0 - margin),
         'centroid': (float(scene_df['x'].mean()), float(scene_df['y'].mean()))}
    gx, gy = P[mode]
    return (gx, gy, float(h_gnb))


ROADSIDE_PLACEMENTS = ['SW', 'SE', 'NW', 'NE', 'S', 'N', 'W', 'E']


def scene_trace(scene_df, h_ue, h_gnb, freq_hz, cap_db, gnb_mode='corner'):
    gx, gy, gh = scene_gnb(scene_df, h_gnb, mode=gnb_mode)
    out = []
    for ts, fr in scene_df.groupby('timestamp'):
        ue = fr[fr['is_ue']]
        if len(ue) == 0:
            continue
        ue = ue.iloc[0]; others = fr[~fr['is_ue']]
        boxes = others[['x', 'y', 'length', 'width', 'height', 'theta']].to_dict('records')
        r = link_blockage((ue['x'], ue['y']), h_ue, (gx, gy), gh, boxes, freq_hz, cap_db)
        blk_h = float(others.iloc[r['blocker']]['height']) if r['blocker'] is not None else np.nan
        out.append(dict(timestamp=ts, dist=math.hypot(ue['x'] - gx, ue['y'] - gy),
                        blocked=r['blocked'], depth_db=r['depth_db'], blocker_h=blk_h, n_between=r['n_cross']))
    return out


def _selftest():
    ue = (0.0, 0.0); gnb = (50.0, 0.0); h_ue, h_gnb = 1.5, 6.0
    def box(x, y, L, W, h, th=0.0): return dict(x=x, y=y, length=L, width=W, height=h, theta=th)
    assert not link_blockage(ue, h_ue, gnb, h_gnb, [])['blocked']
    assert not link_blockage(ue, h_ue, gnb, h_gnb, [box(25, 0, 5, 2, 3.0)])['blocked'], "3m clears"
    rB = link_blockage(ue, h_ue, gnb, h_gnb, [box(25, 0, 5, 2, 4.5)]); assert rB['blocked']
    rC = link_blockage(ue, h_ue, gnb, h_gnb, [box(10, 0, 5, 2, 4.0)]); assert rC['depth_db'] >= rB['depth_db'] - 1e-6
    assert not link_blockage(ue, h_ue, gnb, h_gnb, [box(25, 10, 5, 2, 5.0)])['blocked'], "side misses"
    assert link_blockage(ue, h_ue, gnb, h_gnb, [box(25, 0, 8, 2.5, 4.5, th=0.6)])['blocked'], "rotated blocks"
    print(f"SELFTEST OK — mid-4.5m={rB['depth_db']:.1f} dB near-4.0m={rC['depth_db']:.1f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        _selftest()


if __name__ == '__main__':
    main()
