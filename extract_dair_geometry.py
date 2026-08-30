#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_dair_geometry.py — Paper-2 M2: DAIR-V2X-Seq (TFD) -> compact 3D vehicle-geometry pool.

WHY (the one new variable, 2D -> 3D):
  Paper 1 used static 2D geometric LOS/NLOS. Paper 2 needs REAL 3D dynamic vehicle
  blockage: surrounding cars (with height!) as obstacles between the UE and the roadside
  gNB. DAIR-V2X-Seq's Trajectory Forecasting Dataset (TFD) already gives continuous 3D
  bounding boxes (x,y,z, length,width,height, heading theta) per vehicle per frame at
  10 Hz, in compact per-scene CSVs — no point clouds, no images. That is exactly the
  geometry we need and it stays tractable on a 16 GB laptop.

WHAT THIS DOES:
  Reads TFD trajectory CSVs and emits ONE consolidated geometry pool
  (scene_id, timestamp, agent_id, is_ue, sub_type, x, y, z, length, width, height,
   theta, v_x, v_y) restricted to type == Vehicle, with the target agent flagged as the
   UE. It also prints a SANITY REPORT (coordinate ranges, box-dim distributions by
   sub_type, vehicles/frame, frame rate, and the observed `tag` vocabulary) so we can
   confirm the coordinate frame/units and the target-agent convention against the real
   data before any downstream modelling.

  gNB PLACEMENT IS DEFERRED to the M3 blockage env — this step only produces the
  vehicle-geometry pool. (For the cooperative view, --infra_ref also records each
  scene's infrastructure-sensor centroid as a candidate roadside-gNB anchor.)

TFD single-view trajectory columns (from the DAIR-V2X-Seq dataset README, verbatim):
  city, timestamp, id, type, sub_type, tag, x, y, z, length, width, height,
  theta, v_x, v_y, intersect_id
  type      in [Vehicle, Bicycle, Pedestrian]
  sub_type  in [Car, Truck, Van, Bus, Pedestrian, Cyclist, Tricyclist, Motorcyclist]
  tag       = flag marking the target agent (exact vocab confirmed from data by --report)

INTEGRITY: this script only transforms real dataset annotations. It fabricates nothing.
  A tiny synthetic sample (make_synthetic_sample) exists solely to unit-test the code path
  and is NEVER a source of any reported research number.

Usage:
  python3 extract_dair_geometry.py --tfd_root <V2X-Seq-TFD> --view single-vehicle --report
  python3 extract_dair_geometry.py --tfd_root <V2X-Seq-TFD> --view cooperative --infra_ref \
          --out results/dair_geometry_pool.csv
"""
import os, sys, glob, argparse, json
import numpy as np
import pandas as pd

# Canonical TFD columns we rely on (read by NAME, so extra cooperative columns are fine).
CORE_COLS = ['timestamp', 'id', 'type', 'sub_type', 'tag',
             'x', 'y', 'z', 'length', 'width', 'height', 'theta', 'v_x', 'v_y']
VEHICLE_TYPE = 'Vehicle'
# sub_types that are physical vehicles (blockers with real height); pedestrians/cyclists excluded.
VEHICLE_SUBTYPES = {'Car', 'Truck', 'Van', 'Bus'}
# tag tokens that, if present as strings, mark the forecast target agent (Argoverse-lineage).
TARGET_TAG_TOKENS = {'target', 'target_agent', 'agent', 'predict', 'tgt', '1', 'true'}


def _view_dirs(root, view):
    """Return (label, glob) list of trajectory-CSV globs for the requested view."""
    if view == 'single-vehicle':
        return [('single-vehicle', os.path.join(root, 'single-vehicle', 'trajectories', '*.csv'))]
    if view == 'single-infrastructure':
        return [('single-infrastructure', os.path.join(root, 'single-infrastructure', 'trajectories', '*.csv'))]
    if view == 'cooperative':
        base = os.path.join(root, 'cooperative-vehicle-infrastructure')
        return [('cooperative', os.path.join(base, 'vehicle-trajectories', '*.csv'))]
    raise ValueError(f"unknown view: {view}")


def _infra_glob(root):
    base = os.path.join(root, 'cooperative-vehicle-infrastructure')
    return os.path.join(base, 'infrastructure-trajectories', '*.csv')


def _is_target(tag_series, ue_tag=None):
    """Boolean mask: which rows are the target agent. Robust to string/numeric tag encodings.
    If ue_tag is given, it overrides the default vocabulary (match that exact token, case-insensitive)."""
    s = tag_series
    if ue_tag is not None:
        return s.astype(str).str.strip().str.lower().eq(str(ue_tag).strip().lower())
    # numeric encoding (0/1)
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().any():
        m = num.fillna(0) != 0
        if m.any():
            return m
    # string encoding
    st = s.astype(str).str.strip().str.lower()
    return st.isin(TARGET_TAG_TOKENS)


def load_scene(path):
    """Load one scene CSV -> DataFrame with CORE_COLS present (missing -> NaN), scene_id attached."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    scene_id = os.path.splitext(os.path.basename(path))[0]
    for c in CORE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df['scene_id'] = scene_id
    return df


def build_pool(paths, keep_all_types=False, ue_tag=None):
    """Concatenate scenes -> vehicle-geometry pool. Flags is_ue on the target agent (if a vehicle)."""
    frames, tag_vocab, type_vocab = [], set(), set()
    for p in paths:
        df = load_scene(p)
        tag_vocab.update(map(str, pd.unique(df['tag'].dropna())[:20]))
        type_vocab.update(map(str, pd.unique(df['type'].dropna())))
        tgt = _is_target(df['tag'], ue_tag=ue_tag)
        # case-insensitive: the example subset is title-case ('Vehicle','Car') while the full
        # dataset is upper-case ('VEHICLE','CAR'); accept both so the same extractor serves both.
        _ty = df['type'].astype(str).str.strip().str.lower()
        _st = df['sub_type'].astype(str).str.strip().str.lower()
        is_veh = _ty.eq(VEHICLE_TYPE.lower()) | _st.isin({s.lower() for s in VEHICLE_SUBTYPES})
        df['is_ue'] = (tgt & is_veh)
        keep = df if keep_all_types else df[is_veh].copy()
        out = keep[['scene_id', 'timestamp', 'id', 'is_ue', 'type', 'sub_type',
                    'x', 'y', 'z', 'length', 'width', 'height', 'theta', 'v_x', 'v_y']]
        frames.append(out.rename(columns={'id': 'agent_id'}))
    pool = pd.concat(frames, ignore_index=True)
    return pool, sorted(tag_vocab), sorted(type_vocab)


def infra_reference(root):
    """Per-scene infrastructure-sensor centroid (cooperative view) — candidate roadside-gNB anchor."""
    rows = []
    for p in sorted(glob.glob(_infra_glob(root))):
        df = load_scene(p)
        rows.append({'scene_id': df['scene_id'].iloc[0],
                     'infra_x': float(pd.to_numeric(df['x'], errors='coerce').mean()),
                     'infra_y': float(pd.to_numeric(df['y'], errors='coerce').mean()),
                     'infra_z': float(pd.to_numeric(df['z'], errors='coerce').mean())})
    return pd.DataFrame(rows)


def report(pool, tag_vocab, type_vocab):
    print("=== DAIR-V2X-Seq GEOMETRY POOL — SANITY REPORT ===", flush=True)
    n_scene = pool['scene_id'].nunique()
    print(f"scenes: {n_scene} | vehicle rows: {len(pool):,} | UE rows: {int(pool['is_ue'].sum()):,}")
    print(f"observed `tag` vocabulary (<=20/scene sampled): {tag_vocab}")
    print(f"observed `type` vocabulary: {type_vocab}")
    ue_scenes = pool[pool['is_ue']]['scene_id'].nunique()
    print(f"scenes with a detected vehicle-UE: {ue_scenes}/{n_scene}"
          + ("   <-- CHECK tag convention if this is 0" if ue_scenes == 0 else ""))
    # per-frame vehicle count
    per = pool.groupby(['scene_id', 'timestamp']).size()
    print(f"vehicles/frame: mean {per.mean():.1f}  median {per.median():.0f}  max {per.max()}")
    # frame rate (median timestamp step within a scene)
    def _step(g):
        t = np.sort(pd.to_numeric(g['timestamp'], errors='coerce').dropna().unique())
        return np.median(np.diff(t)) if len(t) > 1 else np.nan
    steps = pool.groupby('scene_id').apply(_step, include_groups=False).dropna()
    if len(steps):
        print(f"median timestamp step: {np.median(steps):.4g} (expect ~0.1 s @10Hz, "
              f"or ~1e8 if ns epochs)")
    # coordinate ranges (frame/units check)
    for c in ['x', 'y', 'z']:
        v = pd.to_numeric(pool[c], errors='coerce')
        print(f"  {c}: [{v.min():.2f}, {v.max():.2f}]  span {v.max()-v.min():.1f}")
    # box dims by sub_type (plausibility: Car ~4-5m L, Truck/Bus taller H)
    print("  box dims by sub_type (median L x W x H, metres):")
    for st, g in pool.groupby('sub_type'):
        L = pd.to_numeric(g['length'], errors='coerce').median()
        W = pd.to_numeric(g['width'], errors='coerce').median()
        H = pd.to_numeric(g['height'], errors='coerce').median()
        print(f"    {str(st):12s} n={len(g):7d}  {L:.2f} x {W:.2f} x {H:.2f}")
    print("=== end report ===", flush=True)


def make_synthetic_sample(path):
    """SYNTHETIC TFD-format CSV for CODE TESTING ONLY — never a research-data source."""
    rng = np.random.default_rng(0)
    rows = []
    T = np.arange(0.0, 1.0, 0.1)              # 10 frames @10Hz
    agents = [('a0', 'Vehicle', 'Car', 'target'),
              ('a1', 'Vehicle', 'Truck', 'others'),
              ('a2', 'Vehicle', 'Bus', 'others'),
              ('a3', 'Pedestrian', 'Pedestrian', 'others')]
    dims = {'Car': (4.5, 1.9, 1.5), 'Truck': (10.0, 2.5, 3.5),
            'Bus': (12.0, 2.6, 3.2), 'Pedestrian': (0.6, 0.6, 1.7)}
    for aid, typ, sub, tag in agents:
        L, W, H = dims[sub]
        x0, y0 = rng.uniform(-30, 30, 2)
        for t in T:
            rows.append(dict(city='yizhuang', timestamp=round(float(t), 1), id=aid, type=typ,
                             sub_type=sub, tag=tag, x=x0 + 8*t, y=y0 + 2*t, z=H/2,
                             length=L, width=W, height=H, theta=0.3, v_x=8.0, v_y=2.0,
                             intersect_id='I1'))
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfd_root', help='path to V2X-Seq-TFD root')
    ap.add_argument('--view', default='single-vehicle',
                    choices=['single-vehicle', 'single-infrastructure', 'cooperative'])
    ap.add_argument('--out', default='dair_geometry_pool.csv')
    ap.add_argument('--infra_ref', action='store_true',
                    help='(cooperative) also emit per-scene infrastructure-sensor centroid')
    ap.add_argument('--report', action='store_true', help='print sanity report')
    ap.add_argument('--keep_all_types', action='store_true',
                    help='keep non-vehicles too (default: vehicles only)')
    ap.add_argument('--ue_tag', default=None,
                    help='override the target-agent tag token (case-insensitive) if the report '
                         'shows 0 detected vehicle-UEs with the defaults')
    ap.add_argument('--selftest', action='store_true',
                    help='run on a synthetic sample (code test only) and exit')
    a = ap.parse_args()

    if a.selftest:
        os.makedirs('/root/work/paper2/results', exist_ok=True)
        sp = '/root/work/paper2/results/_synthetic_tfd_sample.csv'
        make_synthetic_sample(sp)
        pool, tagv, typev = build_pool([sp])
        report(pool, tagv, typev)
        assert pool['is_ue'].sum() > 0, "self-test: UE not detected"
        assert set(pool['sub_type']) <= VEHICLE_SUBTYPES, "self-test: non-vehicle leaked"
        assert (pool.groupby(['scene_id', 'timestamp']).size() >= 3).all(), "self-test: frame vehicle count"
        print("\nSELFTEST OK — parsing, UE flagging, vehicle filter, report all pass.")
        return

    if not a.tfd_root:
        ap.error('--tfd_root is required (or use --selftest)')
    paths = []
    for label, g in _view_dirs(a.tfd_root, a.view):
        paths = sorted(glob.glob(g))
        print(f"view={label}: {len(paths)} scene CSVs under {g}", flush=True)
    if not paths:
        ap.error(f"no scene CSVs found for view={a.view} under {a.tfd_root}. "
                 f"Check the path and that trajectory CSVs are present.")
    pool, tagv, typev = build_pool(paths, keep_all_types=a.keep_all_types, ue_tag=a.ue_tag)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pool.to_csv(a.out, index=False)
    print(f"-> wrote {len(pool):,} rows across {pool['scene_id'].nunique()} scenes to {a.out}", flush=True)
    if a.infra_ref and a.view == 'cooperative':
        ref = infra_reference(a.tfd_root)
        rp = a.out.replace('.csv', '_infra_ref.csv')
        ref.to_csv(rp, index=False)
        print(f"-> wrote infrastructure references for {len(ref)} scenes to {rp}", flush=True)
    if a.report:
        report(pool, tagv, typev)


if __name__ == '__main__':
    main()
