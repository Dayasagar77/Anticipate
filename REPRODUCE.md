# Reproducing the results

Everything below runs single-threaded on CPU. The reported results were produced on an
Intel i9-12900H with 16 GB of RAM under Windows 11, Python 3.11.9, torch 2.5.1,
numpy 2.4.4, pandas 3.0.3, gymnasium 1.2.3, scipy 1.17.1.

## 1. Build the geometry pool

DAIR-V2X is **not** redistributed here. Obtain it from its authors, then:

```bash
python extract_dair_geometry.py --root <path-to-DAIR-V2X> --out dair_geometry_pool.csv
```

This keeps box annotations for physical vehicles (Car, Truck, Van, Bus), drops rows with
no box dimensions, and tags the target agent as the UE. The reported pool retains
**119,015** rows across **45 scenes** of 100 frames at 10 Hz, a median of **22** surrounding
vehicles per frame (mean 25.5, range 5–61) and a median of 60 distinct vehicles per scene.
If your extraction does not reproduce those figures, stop and reconcile before training.

## 2. Train

```bash
python paper2_sweep_r3.py --episodes 1500 --seeds 20
```

Trains the ablation ladder and the replay and risk-level arms, 1500 episodes per run,
evaluating on 500 held-out rollouts under common random numbers. Resumable: a job whose
`_metrics.csv` exists is skipped, so the sweep can be killed and relaunched.

Scenes are split 30 training / 15 held-out by a fixed seed that never moves with the
training seed, so the partition is identical for every configuration and every run.

## 3. Statistics and tables

```bash
python paper2_stats.py
python make_results_tables.py
python measure_ho_requests.py     # handover requests vs accepted handovers
python paper2_basins.py           # the transmit-power and bandwidth convergence split
```

## Determinism

Training is a deterministic function of (seed, configuration): re-running a seed under an
unchanged configuration reproduces sustained RLF, mean return, tail return and mean
transmit power to four decimal places. A consequence worth stating, because it is easy to
get wrong: **re-running the same seeds is not an independent replication.** Any
confirmatory run must use seeds that have not been trained before.

## Pre-registration

`PREREGISTRATION.md` fixes the twenty-seed analysis in advance — the seed count, the
commitment to analyse once at that size, and the rule that a result is called established
only at the unit under which it was tested. It was timestamped before the additional seeds
were trained.
