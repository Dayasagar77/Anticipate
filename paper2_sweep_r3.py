#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_sweep_r3.py — the R3 confirmatory sweep: every reported number, regenerated.

WHY R3 EXISTS
-------------
The R2 sweep is superseded, not merely extended. Five defects in the R2 apparatus each on their own
invalidate the numbers it produced, and they have all now been repaired in the environment, the
controller and the replay module:

  D1  the manuscript claimed n-step *prioritized* experience replay; the code replayed the last eight
      episodes in collection order. Fixed: paper2_per.SequencePER.
  D2  the n-step target treated time-limit truncation as termination, so it bootstrapped a zero
      continuation value at the end of every non-failing episode. Fixed: Agent._seq_loss.
  D3  training and evaluation drew from the same 45 scenes, so every reported number was in-sample.
      Fixed: a fixed 30/15 scene split; all headline numbers come from the 15 held-out scenes.
  D4  evaluation scenarios were drawn independently per configuration, so configurations were compared
      on different episodes. Fixed: common random numbers, EVAL_SEED_BASE + i for every configuration.
  D5  60 evaluation rollouts quantise the RLF rate to 1.67 pp, which is coarser than the effects being
      claimed. Fixed: 500 held-out rollouts, quantum 0.2 pp.

Because D1 and D2 change the learning dynamics and D3 changes what is measured, no R2 number can be
carried forward. Everything is re-run.

ARMS
----
  main       5 configurations x 10 seeds. The ablation ladder from the Paper-1 controller transplanted
             into 3D up to the full proposed controller, each step adding exactly one component.
  replay     the PER ablation (uniform sampling, no importance weights) x 10 seeds. This exists
             because the paper now claims prioritized replay; a claimed component must be ablated.
  alpha      CVaR risk level sensitivity, alpha in {0.1, 0.5} x 5 seeds. alpha = 0.25 is `model` and
             alpha = 1.0 (risk-neutral) is `no_cvar`, so only two new points are needed.
  baselines  the non-learned comparators, run once (they have no training seed).

RESUMABILITY
------------
A job is skipped if its `_metrics.csv` already exists, so the sweep can be killed and relaunched
without losing completed runs. Each job's console output is kept next to its results.

INTEGRITY: this driver only schedules subprocesses and concatenates their result files. It performs
no selection: every seed that runs is aggregated, none is dropped, and the per-seed files are kept so
any aggregate in the paper can be recomputed from them.
"""
import os, sys, time, argparse, subprocess, itertools
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# name -> extra CLI flags for paper2_controller.py
MAIN = {
    # the proposed controller: anticipation + CVaR, feed-forward. Scheduled first inside each seed so
    # that an interrupted sweep still has the headline configuration for every seed that started.
    'model':         ['--ff'],
    # the Paper-1 controller dropped into 3D: feed-forward, risk-neutral, no anticipation features
    'p1_transplant': ['--ff', '--no_ant', '--alpha', '1.0'],
    # + anticipation features, still risk-neutral
    'no_cvar':       ['--ff', '--alpha', '1.0'],
    # + CVaR objective, no anticipation features
    'no_ant':        ['--ff', '--no_ant'],
    # + recurrence, to test whether memory adds anything over the explicit anticipation features
    'plus_lstm':     [],
}
REPLAY = {'no_per': ['--ff', '--no_per']}
ALPHAS = [0.1, 0.5]


def jobs(a):
    """Seed-major order, on purpose. The paired statistics compare configurations seed by seed, so a
    sweep interrupted at 60% is far more useful as six complete seeds of every configuration than as
    ten seeds of the first three. Within a seed the headline configuration runs first."""
    out = []
    for s in range(a.seeds):
        for cfg, flags in MAIN.items():
            out.append((cfg, s, flags))
        if not a.no_replay_arm:
            for cfg, flags in REPLAY.items():
                out.append((cfg, s, flags))
    if not a.no_alpha_arm:
        for s in range(a.alpha_seeds):
            for al in ALPHAS:
                out.append((f'alpha{al}', s, ['--ff', '--alpha', str(al)]))
    return out


def run_one(job, a):
    cfg, seed, flags = job
    out = os.path.join(a.outdir, f'{cfg}_s{seed}')
    if os.path.exists(out + '_metrics.csv') and not a.force:
        return cfg, seed, 'skip', 0.0
    cmd = [PY, os.path.join(HERE, 'paper2_controller.py'), '--train',
           '--episodes', str(a.episodes), '--seed', str(seed), '--out', out,
           '--eval_n', str(a.eval_n), '--eval_n_train', str(a.eval_n_train),
           '--threads', '1'] + flags
    t0 = time.time()
    with open(out + '.log', 'w') as fh:
        fh.write(' '.join(cmd) + '\n\n')
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    dt = time.time() - t0
    ok = (rc == 0) and os.path.exists(out + '_metrics.csv')
    return cfg, seed, ('ok' if ok else f'FAIL rc={rc}'), dt


def run_baselines(a):
    out = os.path.join(a.outdir, 'baselines_metrics.csv')
    if os.path.exists(out) and not a.force:
        print('baselines: already present, skipping', flush=True)
        return
    cmd = [PY, os.path.join(HERE, 'paper2_baselines.py'), '--eval_n', str(a.eval_n),
           '--eval_n_train', str(a.eval_n_train), '--outdir', a.outdir]
    with open(os.path.join(a.outdir, 'baselines.log'), 'w') as fh:
        fh.write(' '.join(cmd) + '\n\n'); fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    print(f'baselines: {"ok" if rc == 0 else f"FAIL rc={rc}"}', flush=True)


def aggregate(a):
    import pandas as pd, glob, re
    rows = []
    for f in sorted(glob.glob(os.path.join(a.outdir, '*_s*_metrics.csv'))):
        m = re.match(r'(.+)_s(\d+)_metrics\.csv$', os.path.basename(f))
        d = pd.read_csv(f)
        d.insert(0, 'config', m.group(1))
        rows.append(d)
    if not rows:
        print('nothing to aggregate yet')
        return None
    all_m = pd.concat(rows, ignore_index=True)
    all_m.to_csv(os.path.join(a.outdir, 'all_metrics.csv'), index=False)

    agg = {}
    for c in ('return_mean', 'cvar25', 'rlf_rate', 'insample_rlf_rate', 'gap_return', 'gap_rlf',
              'pt_dbm', 'rb', 'ho_per_ep', 'lat_mean_ms', 'lat_p95_ms', 'lat_p99_ms', 'train_s'):
        if c in all_m.columns:
            agg[c] = ['mean', 'std']
    s = all_m.groupby('config').agg(agg)
    s.columns = ['_'.join(c) for c in s.columns]
    s.insert(0, 'n_seeds', all_m.groupby('config').size())
    s = s.round(4).reset_index()
    s.to_csv(os.path.join(a.outdir, 'summary.csv'), index=False)
    print('\n' + s[['config', 'n_seeds', 'return_mean_mean', 'cvar25_mean', 'rlf_rate_mean',
                    'rlf_rate_std']].to_string(index=False))
    print(f"\nsaved -> {os.path.join(a.outdir, 'summary.csv')} and all_metrics.csv")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.path.join(HERE, 'models', 'sweep_r3'))
    ap.add_argument('--episodes', type=int, default=1500)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--alpha_seeds', type=int, default=5)
    ap.add_argument('--eval_n', type=int, default=500)
    ap.add_argument('--eval_n_train', type=int, default=200)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--no_alpha_arm', action='store_true')
    ap.add_argument('--no_replay_arm', action='store_true')
    ap.add_argument('--no_baselines', action='store_true')
    ap.add_argument('--aggregate_only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.aggregate_only:
        aggregate(a); return

    js = jobs(a)
    todo = [j for j in js
            if a.force or not os.path.exists(os.path.join(a.outdir, f'{j[0]}_s{j[1]}_metrics.csv'))]
    print(f'R3 sweep | {len(js)} jobs, {len(todo)} to run, {a.workers} workers, '
          f'{a.episodes} episodes, eval {a.eval_n} held-out / {a.eval_n_train} in-sample', flush=True)
    print(f'outdir: {a.outdir}', flush=True)

    if not a.no_baselines:
        run_baselines(a)

    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for cfg, seed, st, dt in ex.map(lambda j: run_one(j, a), js):
            if st == 'skip':
                continue
            done += 1
            el = time.time() - t0
            eta = el / done * (len(todo) - done)
            print(f'[{done}/{len(todo)}] {cfg}_s{seed}: {st} in {dt/60:.1f} min | '
                  f'elapsed {el/60:.0f} min, eta {eta/60:.0f} min', flush=True)
            aggregate(a) if done % 5 == 0 else None

    print(f'\nsweep finished in {(time.time()-t0)/60:.1f} min', flush=True)
    aggregate(a)


if __name__ == '__main__':
    main()
