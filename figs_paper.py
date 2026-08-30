#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figs_paper.py — publication figure set for the Paper-2 manuscript.
Real-data figures (cliff, learning curves, per-seed spread, a real DAIR-V2X scene) + schematics
(system, blockage geometry, controller architecture). Saves each as PDF (vector, for LaTeX) + PNG.
INTEGRITY: data figures use only real result/geometry files; schematics are clearly illustrative.

Typesetting policy (v2): no in-figure titles that duplicate the manuscript caption; panel
identifiers "(a)/(b)" only. Every text label is placed with explicit head-room or a white
bbox so nothing overlaps a curve, marker, axis or another label."""
import os, math, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import blockage3d as B

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 10,
                     'axes.titlesize': 10.5, 'axes.labelsize': 10, 'legend.fontsize': 9,
                     'savefig.dpi': 300, 'figure.dpi': 130, 'axes.linewidth': 0.8})
# PROVENANCE. Figs. 7 and 9 must depict the SAME experiment as Table VIII. That table is now the
# six-configuration ladder measured in models/sweep_r3 (prioritized replay added, Paper-1 design
# transplanted in as the bottom rung); the four-configuration models/sweep_final run it used to read
# is superseded. Both the directory and the configuration list are therefore declared once, here.
SWEEP = 'models/sweep_r3'
CFG = ['p1_transplant', 'no_ant', 'no_cvar', 'no_per', 'plus_lstm', 'model']
LBL = {'p1_transplant': 'P1 transplant', 'no_ant': '$-$anticipation', 'no_cvar': '$-$CVaR',
       'no_per': '$-$PER', 'plus_lstm': '$+$LSTM', 'model': 'reported'}
COL = {'p1_transplant': '#7f7f7f', 'no_ant': '#d62728', 'no_cvar': '#ff7f0e',
       'no_per': '#9467bd', 'plus_lstm': '#1f77b4', 'model': '#2ca02c'}
WBOX = dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.4)   # halo for labels over ink


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + '.png'), bbox_inches='tight')
    plt.close(fig); print('  ->', name)


# ---------------------------------------------------------------- Fig: reliability cliff (REAL)
def fig_cliff():
    # DAIR provenance repair: the manuscript's Section IV is measured on THIS paper's
    # environment (paper2_cliff_family_dair.py), not on the HighD single-link sweep.
    d = pd.read_csv(os.path.join(HERE, 'results/cliff_family_dair.csv'))
    piv = d.pivot(index='block_db', columns='pt_dbm', values='RLF_mean')
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={'width_ratios': [1.15, 1]})
    px, py = piv.columns.values.astype(float), piv.index.values.astype(float)
    im = axA.pcolormesh(px, py, piv.values, cmap='RdYlGn_r',
                        shading='auto', vmin=0, vmax=max(1, piv.values.max()))
    lo_v, hi_v = float(piv.values.min()), float(piv.values.max())
    levels = [l for l in (5, 10, 15, 20, 25, 30, 40, 50) if lo_v < l < hi_v]
    cs = axA.contour(px, py, piv.values, levels=levels, colors='k', linewidths=0.7)
    # F8 fix: pad the panel so no contour label is clipped at the frame, and label away from edges
    dx, dy = (px[-1] - px[0]), (py[-1] - py[0])
    axA.set_xlim(px[0] - 0.09 * dx, px[-1] + 0.09 * dx)
    axA.set_ylim(py[0] - 0.09 * dy, py[-1] + 0.09 * dy)
    axA.clabel(cs, fmt='%d%%', fontsize=7, inline_spacing=4)
    axA.set_xlabel('transmit power $P_t$ (dBm)'); axA.set_ylabel('excess blockage depth (dB)')
    axA.set_title('(a) Sustained-RLF surface'); fig.colorbar(im, ax=axA, label='RLF rate (%)')
    # The deepest rows of the DAIR surface SATURATE: the 25 and 30 dB rows are numerically
    # identical and the 20 dB row is within a fraction of a point of them, so drawing all seven
    # curves solid hides three of them under one another. Distinct dash patterns for the top
    # three keep every measured curve visible, and the coincidence is stated from the data
    # rather than asserted.
    ytop = 0.0
    depths_all = sorted(d['block_db'].unique())
    LS = ['-', '-', '-', '-', (0, (6, 1.6)), (0, (1.3, 1.3)), (0, (6, 1.6, 1.3, 1.6))]
    for k, bd in enumerate(depths_all):
        s = d[d['block_db'] == bd].sort_values('pt_dbm')
        axB.plot(s['pt_dbm'], s['RLF_mean'], marker='o', ms=3, lw=1.3,
                 ls=LS[k % len(LS)], label=f'{bd:.0f} dB')
        ytop = max(ytop, float(s['RLF_mean'].max()))
    axB.set_xlabel('transmit power $P_t$ (dBm)'); axB.set_ylabel('sustained-RLF rate (%)')
    axB.set_title('(b) Power cuts at each blockage depth'); axB.grid(alpha=0.3, lw=0.5)
    # F8 fix (b): the 7-curve legend used to be dropped on top of the 25/30 dB curves at upper
    # right; reserve a band above every curve and lay the legend out flat inside it instead
    axB.set_ylim(0, ytop * 1.42)
    axB.legend(title='blockage depth', fontsize=7.5, title_fontsize=7.5, ncol=4,
               loc='upper center', framealpha=0.95, handlelength=1.4,
               columnspacing=1.0, handletextpad=0.4, borderpad=0.4)
    sat = piv.loc[[dd for dd in depths_all if dd >= 20.0]]
    spread = float((sat.max(axis=0) - sat.min(axis=0)).max())
    axB.text(0.985, 0.035,
             'the 20, 25 and 30 dB curves lie within %.2f pp of one another at every\n'
             'power (the 25 and 30 dB rows are identical): the surface saturates' % spread,
             transform=axB.transAxes, ha='right', va='bottom', fontsize=6.8, color='#333',
             bbox=WBOX)
    fig.tight_layout(); save(fig, 'fig_cliff')


# ---------------------------------------------------------------- Fig: learning curves (REAL)
def fig_learning():
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ymin, ymax = np.inf, -np.inf
    for c in CFG:
        curves = []
        for f in sorted(glob.glob(os.path.join(HERE, f'{SWEEP}/{c}_s*_returns.csv'))):
            r = pd.read_csv(f)['return'].values; curves.append(r)
        if not curves:
            continue
        n = min(map(len, curves)); A = np.vstack([c[:n] for c in curves])
        w = 41                                            # smooth each seed, then aggregate
        sm = np.vstack([np.convolve(a, np.ones(w) / w, mode='valid') for a in A])
        x = np.arange(sm.shape[1]); m, s = sm.mean(0), sm.std(0)
        ax.plot(x, m, color=COL[c], lw=1.6, label=LBL[c])
        ax.fill_between(x, m - s, m + s, color=COL[c], alpha=0.15, lw=0)
        ymin = min(ymin, (m - s).min()); ymax = max(ymax, (m + s).max())
    ax.set_xlabel('training episode')
    # F4 fix: short y-label (the mean/std detail lives in the caption) so it is never clipped
    ax.set_ylabel('episode return (smoothed)')
    ax.set_xlim(left=0)
    # F4 fix: reserve head-room and park the legend in it, so no curve is ever covered
    span = ymax - ymin
    # six configurations need two legend rows, so the reserved band is deeper than the four-curve
    # version's 0.30 --- the legend must never come down onto a curve
    ax.set_ylim(ymin - 0.06 * span, ymax + 0.46 * span)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc='upper center', ncol=3, fontsize=8.5, framealpha=0.95,
              handlelength=1.6, columnspacing=1.1, borderpad=0.4)
    fig.tight_layout(); save(fig, 'fig_learning')


# ---------------------------------------------------------------- Fig: per-seed RLF spread (REAL)
def fig_perseed():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    rng = np.random.default_rng(0)
    cfgs, allv = [], []
    for c in CFG:
        v = np.array([float(pd.read_csv(f)['rlf_rate'].iloc[0])
                      for f in sorted(glob.glob(os.path.join(HERE, f'{SWEEP}/{c}_s*_metrics.csv')))])
        if len(v):                       # a configuration with no completed seed is simply absent
            cfgs.append(c); allv.append(v)
    gmax = max(v.max() for v in allv)
    # F2 fix: fix the axis first, then place every mean label inside reserved head-room
    ax.set_ylim(-1.5, gmax * 1.20)
    for i, (c, vals) in enumerate(zip(cfgs, allv)):
        xj = i + (rng.random(len(vals)) - 0.5) * 0.28
        ax.scatter(xj, vals, s=26, color=COL[c], edgecolor='k', lw=0.4, zorder=3, alpha=0.9)
        ax.hlines(vals.mean(), i - 0.24, i + 0.24, color='k', lw=2, zorder=4)
        ax.text(i, vals.max() + 0.045 * gmax, f'{vals.mean():.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', zorder=5, bbox=WBOX)
    ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([LBL[c] for c in cfgs], fontsize=9)
    ax.set_xlim(-0.6, len(cfgs) - 0.4)
    ax.set_ylabel('sustained-RLF rate per seed (%)')
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    fig.tight_layout(); save(fig, 'fig_perseed')


# ---------------------------------------------------------------- Fig: a real DAIR-V2X scene (REAL)
def _rot_rect(ax, cx, cy, L, W, th, **kw):
    c, s = math.cos(th), math.sin(th)
    corners = [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)]
    pts = [(cx + x * c - y * s, cy + x * s + y * c) for x, y in corners]
    ax.add_patch(plt.Polygon(pts, closed=True, **kw))


def fig_scene():
    pool = pd.read_csv(os.path.join(HERE, 'results/dair_geometry_pool.csv'))
    pool['scene_id'] = pool['scene_id'].astype(str)
    # search for a frame in which at least one corner-gNB link is actually blocked (so the red link
    # is visible); score = 100*(#blocked corner links) + (#vehicles). UE box dims are legitimately NaN.
    modes = ['NW', 'NE', 'SW', 'SE']
    pick = None
    for sid, sc in pool.groupby('scene_id'):
        gnbs = [B.scene_gnb(sc, B.H_GNB_DEFAULT, mode=m) for m in modes]
        for ts, fr in sc.groupby('timestamp'):
            if not fr['is_ue'].any():
                continue
            oth = fr[~fr['is_ue']]
            oth = oth[~oth[['length', 'width', 'height', 'theta']].isna().any(axis=1)]
            if len(oth) < 4:
                continue
            ue0 = fr[fr['is_ue']].iloc[0]
            boxes0 = oth[['x', 'y', 'length', 'width', 'height', 'theta']].to_dict('records')
            res0 = [B.link_blockage((ue0['x'], ue0['y']), B.H_UE_DEFAULT, (g[0], g[1]), g[2], boxes0) for g in gnbs]
            score = 100 * sum(r['blocked'] for r in res0) + len(oth)
            if pick is None or score > pick[0]:
                pick = (score, sid, ts, fr, oth, gnbs, res0)
        if pick is not None and pick[0] >= 100:
            break
    _, sid, ts, fr, others, gnbs, res = pick
    ue = fr[fr['is_ue']].iloc[0]

    # F5 fix: shift to a local origin so the axes never print a "+4.731e6" offset block
    ox = min(others['x'].min(), ue['x'], min(g[0] for g in gnbs))
    oy = min(others['y'].min(), ue['y'], min(g[1] for g in gnbs))
    uex, uey = ue['x'] - ox, ue['y'] - oy
    gxy = [(g[0] - ox, g[1] - oy) for g in gnbs]
    ueL, ueW = max(float(ue['length']), 4.4), max(float(ue['width']), 1.9)

    # the vehicle that actually causes the deepest blockage, for the zoom panel
    bi = next((r['blocker'] for r in res if r['blocked'] and r['blocker'] is not None), None)
    blk = others.iloc[bi] if bi is not None else None
    dblk = math.hypot(blk['x'] - ox - uex, blk['y'] - oy - uey) if blk is not None else 0.0

    # F5 fix: two panels — the 300 m scene made every 5 m vehicle box invisible, so the overview
    # now carries the topology and a zoom carries the geometry that produces the blockage.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw={'width_ratios': [1.5, 1]})
    axA.set_anchor('N'); axB.set_anchor('N')      # equal-aspect panels: align their tops, not centres
    R = max(28.0, dblk + 14.0)

    for ax, zoom in ((axA, False), (axB, True)):
        for j, (_, v) in enumerate(others.iterrows()):
            is_blk = (bi is not None and j == bi)
            _rot_rect(ax, v['x'] - ox, v['y'] - oy, v['length'], v['width'], v['theta'],
                      facecolor=('#ff7f0e' if is_blk else '#b0b8c4'), edgecolor='k',
                      lw=0.8 if is_blk else 0.6, alpha=0.95, zorder=4 if is_blk else 3)
        _rot_rect(ax, uex, uey, ueL, ueW, float(ue['theta']),
                  facecolor='#2ca02c', edgecolor='k', lw=1.0, zorder=6)
        for (gx, gy), r in zip(gxy, res):
            col = '#d62728' if r['blocked'] else '#2ca02c'
            ax.plot([uex, gx], [uey, gy], color=col, lw=1.7 if r['blocked'] else 1.2,
                    ls='--', alpha=0.85, zorder=2)
        ax.set_aspect('equal'); ax.grid(alpha=0.25, lw=0.5)
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_xlabel('x (m, scene-local)')

    # ---- (a) overview: gNB labels pushed *inward*, where the scene corners are empty
    IN = {'NW': (1, -1), 'NE': (-1, -1), 'SW': (1, 1), 'SE': (-1, 1)}
    for (gx, gy), r, mode in zip(gxy, res, modes):
        axA.scatter([gx], [gy], marker='^', s=95, color='#1f77b4', edgecolor='k', lw=0.6, zorder=5)
        lab = f'gNB$_\\mathrm{{{mode}}}$' + (f'\n{r["depth_db"]:.0f} dB blocked' if r['blocked'] else '')
        sx, sy = IN[mode]
        axA.annotate(lab, (gx, gy), xytext=(10 * sx, 10 * sy), textcoords='offset points',
                     fontsize=8, ha='left' if sx > 0 else 'right', va='bottom' if sy > 0 else 'top',
                     color=('#d62728' if r['blocked'] else 'k'), zorder=8, bbox=WBOX)
    axA.add_patch(Rectangle((uex - R, uey - R), 2 * R, 2 * R, fill=False, ec='#333',
                            ls='-', lw=1.0, zorder=7))
    axA.annotate('zoom (b)', (uex + R, uey + R), xytext=(4, 4), textcoords='offset points',
                 fontsize=8, ha='left', va='bottom', color='#333', zorder=8, bbox=WBOX)
    axA.set_ylabel('y (m, scene-local)')
    axA.margins(0.16)
    axA.set_title('(a) Full intersection scene')

    # ---- (b) zoom on the UE: this is where the 3D boxes are legible
    axB.set_xlim(uex - R, uex + R); axB.set_ylim(uey - R, uey + R)
    axB.annotate('UE', xy=(uex, uey), xytext=(0, -26), textcoords='offset points',
                 ha='center', va='top', fontsize=9, fontweight='bold', color='#12520f', zorder=9,
                 bbox=WBOX, arrowprops=dict(arrowstyle='-', lw=0.8, color='#12520f', shrinkA=0, shrinkB=7))
    # The symbol $h$ is reserved by Section III-F for the depth by which the box top penetrates the
    # LOS ray --- the quantity that enters the knife-edge parameter. Labelling the box HEIGHT with it
    # attributes the wrong quantity to the wrong symbol, and the two differ here by more than a
    # factor of two. Both are shown, each under its own name, and both are read out of the same
    # link_blockage result the red link is drawn from.
    rblk = next((r for r in res if r['blocked'] and r['blocker'] is not None), None)
    if blk is not None:
        bxp, byp = blk['x'] - ox, blk['y'] - oy
        sgn = 1 if byp >= uey else -1
        axB.annotate(f'blocking vehicle\nbox height {float(blk["height"]):.2f} m\n'
                     f'LOS penetration $h={rblk["dh"]:.2f}$ m', xy=(bxp, byp),
                     xytext=(0, 30 * sgn), textcoords='offset points', ha='center',
                     va='bottom' if sgn > 0 else 'top', fontsize=8, color='#8a4b00', zorder=9,
                     bbox=WBOX, arrowprops=dict(arrowstyle='-', lw=0.8, color='#8a4b00',
                                                shrinkA=0, shrinkB=7))
    axB.set_ylabel('y (m, scene-local)')
    axB.set_title(f'(b) Zoom on the UE ($\\pm${R:.0f} m)')

    leg = [Line2D([0], [0], color='#2ca02c', ls='--', label='clear link'),
           Line2D([0], [0], color='#d62728', ls='--', label='blocked link'),
           Line2D([0], [0], marker='^', color='w', markerfacecolor='#1f77b4', markeredgecolor='k', label='gNB', ms=9),
           Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c', markeredgecolor='k', label='UE', ms=9),
           Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', markeredgecolor='k', label='blocking vehicle', ms=9),
           Line2D([0], [0], marker='s', color='w', markerfacecolor='#b0b8c4', markeredgecolor='k', label='other vehicle', ms=9)]
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(handles=leg, fontsize=8.5, loc='lower center', ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.3, handletextpad=0.5)
    save(fig, 'fig_scene')

    # The caption quotes this figure's geometry, and a caption that quotes a figure by hand is a
    # caption that drifts from it: the previous one named the truck's box height under the symbol
    # Section III-F reserves for the LOS penetration. The numbers are therefore published here, by
    # the code that draws them, and the manuscript pass reads them out of this file rather than
    # re-deriving or retyping them. If the scene selection ever picks a different frame, the caption
    # follows automatically.
    facts = dict(scene_id=str(sid), timestamp=float(ts), n_vehicles=int(len(others)),
                 blocked_mode=[m for m, r in zip(modes, res) if r['blocked']],
                 depth_db=float(rblk['depth_db']) if rblk else 0.0,
                 blocker_height_m=float(blk['height']) if blk is not None else float('nan'),
                 los_penetration_m=float(rblk['dh']) if rblk else 0.0,
                 d1_m=float(rblk['d1']) if rblk else 0.0, d2_m=float(rblk['d2']) if rblk else 0.0)
    with open(os.path.join(FIG, 'fig_scene_facts.json'), 'w') as fh:
        json.dump(facts, fh, indent=1)


# ---------------------------------------------------------------- Fig: system schematic
def fig_system():
    # layout fix: the frame used to run from y=0.2 with nothing below y=1.3 but bare road, which
    # printed as a wide empty band; the RIC column is also moved right so the red control-loop
    # label no longer touches the RIC box border
    fig, ax = plt.subplots(figsize=(8.8, 3.6)); ax.set_xlim(0, 11.0); ax.set_ylim(1.1, 5.2)
    ax.axis('off')
    # roads
    ax.add_patch(Rectangle((0, 2.4), 6.2, 1.4, color='#e9ecef'))
    ax.add_patch(Rectangle((2.4, 0.2), 1.4, 5.4, color='#e9ecef'))
    ax.plot([0, 6.2], [3.1, 3.1], 'w--', lw=1); ax.plot([3.1, 3.1], [0.2, 5.6], 'w--', lw=1)
    # gNBs at the four intersection corners
    for (gx, gy) in [(2.2, 3.9), (3.9, 3.9), (2.2, 2.2), (3.9, 2.2)]:
        ax.scatter([gx], [gy], marker='^', s=140, color='#1f77b4', edgecolor='k', zorder=5)
    # F9-class fix: label offset off the marker, with a halo
    ax.annotate('roadside gNBs', (3.9, 3.9), xytext=(12, 10), textcoords='offset points',
                fontsize=8, ha='left', va='bottom', bbox=WBOX, zorder=6,
                arrowprops=dict(arrowstyle='-', lw=0.6, color='#555', shrinkA=0, shrinkB=5))
    # UE and blocking truck
    ax.add_patch(Rectangle((1.0, 2.7), 0.7, 0.35, color='#2ca02c', ec='k', zorder=4))
    ax.text(1.35, 2.52, 'UE', color='#12520f', fontweight='bold', fontsize=9, ha='center', va='top', zorder=6)
    ax.add_patch(Rectangle((2.55, 2.62), 1.0, 0.5, color='#6c757d', ec='k', zorder=4))
    ax.text(3.05, 2.87, 'truck', fontsize=7, ha='center', va='center', color='w', zorder=6)
    ax.plot([1.35, 3.9], [2.95, 3.9], color='#d62728', lw=1.6, ls='--', zorder=3)   # blocked
    ax.plot([1.35, 2.2], [2.95, 2.2], color='#2ca02c', lw=1.4, ls='--', zorder=3)   # clear
    ax.text(2.35, 3.44, 'blocked', color='#d62728', fontsize=7.5, rotation=20.4,
            ha='center', va='bottom', zorder=6, bbox=WBOX)
    for cx in [4.7, 5.4]:
        ax.add_patch(Rectangle((cx, 2.75), 0.55, 0.3, color='#b0b8c4', ec='k'))
    # O-RAN RIC box
    ax.add_patch(FancyBboxPatch((7.6, 1.45), 3.0, 3.4, boxstyle='round,pad=0.05',
                                fc='#f1f3f5', ec='#1f77b4', lw=1.4))
    ax.text(9.10, 4.52, 'O-RAN Near-RT RIC', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#1f4b7a')
    ax.add_patch(FancyBboxPatch((7.8, 2.8), 2.6, 1.4, boxstyle='round,pad=0.04', fc='#dbe7f3', ec='#1f77b4'))
    ax.text(9.10, 3.50, 'Blockage-anticipating\nrisk-sensitive RL xApp\n(QR-DQN + CVaR)',
            ha='center', va='center', fontsize=7.5, linespacing=1.5)
    ax.text(9.10, 2.15, 'geometry state\n(3D vehicle boxes)', ha='center', va='center',
            fontsize=7, style='italic')
    # control loop arrows
    ax.add_patch(FancyArrowPatch((6.25, 3.7), (7.65, 3.7), arrowstyle='-|>', mutation_scale=13, color='k'))
    ax.text(6.95, 3.84, 'measure', fontsize=7, ha='center', va='bottom', bbox=WBOX)
    ax.add_patch(FancyArrowPatch((7.65, 2.9), (6.25, 2.9), arrowstyle='-|>', mutation_scale=13, color='#d62728'))
    ax.text(6.95, 2.76, 'HO / power /\nspectrum', fontsize=7, ha='center', va='top',
            color='#d62728', bbox=WBOX)
    fig.tight_layout(); save(fig, 'fig_system')


# ---------------------------------------------------------------- Fig: blockage geometry schematic
def fig_geometry():
    fig, ax = plt.subplots(figsize=(7.4, 3.0)); ax.axis('off')
    ue, gnb = (0.5, 1.0), (9.0, 2.4)
    ax.plot([-0.4, 10.4], [0, 0], color='#adb5bd', lw=1.2, zorder=1)              # ground
    ax.plot([ue[0], gnb[0]], [ue[1], gnb[1]], 'k-', lw=1.4, zorder=3)             # LOS ray
    slope_deg = math.degrees(math.atan2(gnb[1] - ue[1], gnb[0] - ue[0]))

    # Fresnel ellipse around the LOS
    mx, my = (ue[0] + gnb[0]) / 2, (ue[1] + gnb[1]) / 2
    D = math.hypot(gnb[0] - ue[0], gnb[1] - ue[1])
    ax.add_patch(Ellipse((mx, my), D, 1.5, angle=slope_deg, fill=False, ec='#1f77b4', ls=':', lw=1.1))

    # terminals
    ax.scatter(*ue, s=60, color='#2ca02c', ec='k', zorder=6)
    ax.text(ue[0], ue[1] + 0.30, 'UE\n$h_{\\mathrm{UE}}=1.5$ m', fontsize=8, ha='center', va='bottom')
    ax.scatter(*gnb, marker='^', s=140, color='#1f77b4', ec='k', zorder=6)
    ax.text(gnb[0], gnb[1] + 0.26, 'gNB\n$h_{\\mathrm{gNB}}=6$ m', fontsize=8, ha='center', va='bottom')

    # F6 fix: LOS label carries a white halo so the ray no longer strikes through it
    ax.text(2.35, 1.36, 'line-of-sight ray', fontsize=8, rotation=slope_deg,
            rotation_mode='anchor', ha='center', va='bottom', zorder=5, bbox=WBOX)
    # F6 fix: Fresnel label moved clear of the vehicle, tied to the ellipse by a leader
    ax.annotate('first Fresnel zone', xy=(3.05, 2.11), xytext=(2.5, 2.62), color='#1f77b4',
                fontsize=8, ha='center', va='bottom', zorder=5, bbox=WBOX,
                arrowprops=dict(arrowstyle='-', lw=0.7, color='#1f77b4', shrinkA=2, shrinkB=0))

    # blocking vehicle box penetrating the LOS
    bx = 5.6
    t = (bx - ue[0]) / (gnb[0] - ue[0]); ray_h = ue[1] + t * (gnb[1] - ue[1])
    ax.add_patch(Rectangle((bx - 0.6, 0.0), 1.2, ray_h + 0.55, facecolor='#6c757d', ec='k', zorder=4))
    ax.text(bx, 0.62, 'blocking\nvehicle', fontsize=7.5, ha='center', va='center',
            color='w', zorder=5, linespacing=1.4)
    # F6 fix: penetration gauge on the clear (left) side; label no longer sits on the Fresnel edge
    ax.annotate('', xy=(bx - 0.74, ray_h), xytext=(bx - 0.74, ray_h + 0.55),
                arrowprops=dict(arrowstyle='<->', color='#d62728', lw=1.0), zorder=5)
    ax.text(bx - 0.88, ray_h + 0.275, 'penetration $h$', color='#d62728', fontsize=8,
            ha='right', va='center', zorder=5, bbox=WBOX)
    # F6 fix: d1/d2 dimension arrows moved below the ground line, so no arrow hides inside the vehicle
    for xx in (ue[0], bx, gnb[0]):
        ax.plot([xx, xx], [-0.52, 0.0], color='#adb5bd', ls=':', lw=0.8, zorder=1)
    for xa, xb, lab in ((ue[0], bx, '$d_1$'), (bx, gnb[0], '$d_2$')):
        ax.annotate('', xy=(xa, -0.42), xytext=(xb, -0.42),
                    arrowprops=dict(arrowstyle='<->', color='k', lw=0.9), zorder=3)
        ax.text((xa + xb) / 2, -0.42, lab, fontsize=9, ha='center', va='center', zorder=5, bbox=WBOX)
    ax.set_xlim(-0.6, 10.6); ax.set_ylim(-0.85, 3.35)
    fig.tight_layout(); save(fig, 'fig_geometry')


# ---------------------------------------------------------------- Fig: controller architecture
def _box(ax, x, y, w, h, text, fc, ec='k', fs=8, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03', fc=fc, ec=ec, lw=1.1, **kw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, linespacing=1.5)


def _arrow(ax, x0, y0, x1, y1, color='k'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=12, color=color, lw=1.1))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 3.3)); ax.set_xlim(0, 15.2); ax.set_ylim(0.35, 5.65); ax.axis('off')
    # F9 fix: every block is centred on the same y=3.0 spine and heights are brought closer together
    _box(ax, 0.1, 1.5, 2.3, 3.0, 'State $s_t\\in\\mathbb{R}^{11}$\n\n5 physics\n5 anticipation\n1 HO flag',
         '#e7f5e9', fs=7.8)
    _box(ax, 2.9, 2.3, 2.0, 1.4, 'FC encoder\n$2\\times128$, ReLU', '#f1f3f5', fs=8)
    _box(ax, 5.4, 2.3, 2.2, 1.4, 'LSTM (128)\n[ablation only]', '#fdecea', ec='#d62728', ls='--', fs=8)
    _box(ax, 8.1, 1.2, 3.0, 3.6, 'Branched dueling\nQR-DQN heads\n\nHO (2) · Power (3)\nSpectrum (3)\n\n$N=32$ quantiles',
         '#e7f0fb', fs=8)
    _box(ax, 11.6, 2.2, 3.4, 1.6, 'CVaR$_{0.25}$ action selection\n(argmax of worst-25%\nquantile mean)',
         '#fff8e1', ec='#e0c060', fs=8)
    _arrow(ax, 2.4, 3.0, 2.9, 3.0)
    _arrow(ax, 4.9, 3.0, 5.4, 3.0)
    _arrow(ax, 7.6, 3.0, 8.1, 3.0)
    _arrow(ax, 11.1, 3.0, 11.6, 3.0)
    # F9 fix: the feed-forward bypass is routed well below the LSTM and its label sits off the arrow
    _arrow(ax, 4.9, 2.45, 8.1, 1.75, color='#888')
    # the label is right-anchored just short of the QR-DQN box (x=8.1) so it can never touch it
    ax.text(7.85, 1.28, 'feed-forward path (reported model)', fontsize=7, color='#555',
            ha='right', va='top', bbox=WBOX)
    _arrow(ax, 13.3, 2.2, 13.3, 1.35)
    ax.text(13.3, 1.20, 'action $a_t$ = (HO, power, spectrum)', ha='center', va='top', fontsize=8)
    fig.tight_layout(); save(fig, 'fig_architecture')


if __name__ == '__main__':
    print('generating figures...')
    for fn in [fig_cliff, fig_learning, fig_perseed, fig_scene, fig_system, fig_geometry, fig_architecture]:
        try:
            fn()
        except Exception as e:
            print('  !! FAILED', fn.__name__, '::', repr(e))
    print('done ->', FIG)
