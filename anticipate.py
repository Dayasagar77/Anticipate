#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""anticipate.py — Paper-2 M4 (N1): geometry-derived blockage-ANTICIPATION state.

The novel, interpretable feature that separates Paper 2 from reactive baselines. From the surrounding
vehicles' current positions AND velocities (v_x, v_y in the DAIR pool), forward-simulate the geometry
under constant velocity and predict, for the UE->serving-gNB link:
  * time-to-blockage (TTB): earliest future time the link goes NLOSv (a vehicle box top rises above
    the ray at the crossing), capped at a horizon; = horizon if no blockage is predicted.
  * projected blockage depth at TTB (ITU-R P.526 knife-edge, via blockage3d).
  * blockers in the LOS corridor now: count + nearest lateral distance to the ray.
  * closing rate: how fast the nearest corridor vehicle approaches the ray (m/s; <0 = approaching).
These are physics, not a black box: the controller can pre-emptively hand over / adjust power before
the block, which is the ONLY lever left once power loses reliability authority under deep blockage.

INTEGRITY: pure geometry on real annotations; --selftest uses hand-crafted motion to verify TTB
monotonicity and is never a research-number source.
"""
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import blockage3d as B

HORIZON_S = 3.0        # anticipation horizon (s)
DT_S = 0.25            # forward-sim step (s); horizon/DT geometry evals per query
CORRIDOR_W = 4.0       # lateral half-width (m) defining the LOS corridor


def _seg_point_dist(ax, ay, bx, by, px, py):
    """Distance from point P to segment AB, and the closest-point parameter t in [0,1]."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy), t


def anticipation_features(ue, gnb, vehicles, h_ue=B.H_UE_DEFAULT, cap_db=B.CAP_DB_DEFAULT,
                          freq=B.FREQ_HZ, horizon=HORIZON_S, dt=DT_S, corridor_w=CORRIDOR_W):
    """ue: dict with x,y,v_x,v_y. gnb: (gx,gy,gh) fixed. vehicles: list of dicts x,y,length,width,
    height,theta,v_x,v_y. Returns the anticipation feature dict."""
    gx, gy, gh = gnb
    # --- current corridor occupancy + closing rate (constant-velocity radial approach to the ray) ---
    n_corridor, near_d, near_close = 0, corridor_w * 3, 0.0
    for v in vehicles:
        d, _ = _seg_point_dist(ue['x'], ue['y'], gx, gy, v['x'], v['y'])
        if d <= corridor_w:
            n_corridor += 1
        if d < near_d:
            near_d = d
            # closing rate: d(dist-to-ray)/dt approximated by a small forward step of THIS vehicle
            vx, vy = v.get('v_x', 0.0), v.get('v_y', 0.0)
            d2, _ = _seg_point_dist(ue['x'] + ue.get('v_x', 0.0) * dt, ue['y'] + ue.get('v_y', 0.0) * dt,
                                    gx, gy, v['x'] + vx * dt, v['y'] + vy * dt)
            near_close = (d2 - d) / dt          # <0 => approaching the ray
    # --- forward-simulate the geometry to find time-to-blockage + projected depth ---
    ttb, proj_depth = horizon, 0.0
    n = int(round(horizon / dt))
    for k in range(n + 1):
        t = k * dt
        ue_xy = (ue['x'] + ue.get('v_x', 0.0) * t, ue['y'] + ue.get('v_y', 0.0) * t)
        boxes = [dict(x=v['x'] + v.get('v_x', 0.0) * t, y=v['y'] + v.get('v_y', 0.0) * t,
                      length=v['length'], width=v['width'], height=v['height'], theta=v['theta'])
                 for v in vehicles]
        r = B.link_blockage(ue_xy, h_ue, (gx, gy), gh, boxes, freq, cap_db)
        if r['blocked']:
            ttb, proj_depth = t, r['depth_db']
            break
    return dict(ttb_s=ttb, proj_depth_db=proj_depth, n_corridor=n_corridor,
                nearest_ray_dist_m=near_d, closing_rate_mps=near_close)


def _selftest():
    gnb = (50.0, 0.0, 6.0)
    ue = dict(x=0.0, y=0.0, v_x=0.0, v_y=0.0)
    tall = lambda x, y, vx, vy: dict(x=x, y=y, length=5.0, width=2.0, height=4.5, theta=0.0, v_x=vx, v_y=vy)
    # 1) no vehicles -> no predicted blockage
    f0 = anticipation_features(ue, gnb, [])
    assert f0['ttb_s'] == HORIZON_S and f0['proj_depth_db'] == 0.0, f0
    # 2) tall vehicle sitting ON the ray now -> TTB ~ 0
    f1 = anticipation_features(ue, gnb, [tall(25, 0, 0, 0)])
    assert f1['ttb_s'] == 0.0 and f1['proj_depth_db'] > 1.0, f1
    # 3) tall vehicle off to the side approaching the ray -> finite TTB in (0, horizon), approaching
    f2 = anticipation_features(ue, gnb, [tall(25, 8, 0, -3.0)])   # 8 m off, closing at 3 m/s
    assert 0.0 < f2['ttb_s'] < HORIZON_S, f2
    assert f2['closing_rate_mps'] < 0, f2
    # 4) closer approacher blocks sooner than a farther one (TTB monotone in distance)
    f_near = anticipation_features(ue, gnb, [tall(25, 5, 0, -3.0)])
    f_far = anticipation_features(ue, gnb, [tall(25, 9, 0, -3.0)])
    assert f_near['ttb_s'] <= f_far['ttb_s'], (f_near, f_far)
    # 5) vehicle moving AWAY from the ray -> no blockage predicted
    f3 = anticipation_features(ue, gnb, [tall(25, 6, 0, +5.0)])
    assert f3['ttb_s'] == HORIZON_S, f3
    print("SELFTEST OK — TTB=0 on-ray, finite+monotone for approachers, horizon for none/receding.")
    print(f"  approaching(8m,3m/s): TTB={f2['ttb_s']:.2f}s depth={f2['proj_depth_db']:.1f}dB "
          f"close={f2['closing_rate_mps']:.1f}m/s | near(5m) TTB={f_near['ttb_s']:.2f} far(9m) TTB={f_far['ttb_s']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        _selftest()


if __name__ == '__main__':
    main()
