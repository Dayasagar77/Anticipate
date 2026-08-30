# Anticipate

Code for the paper *"Anticipating the Blockage Tail: Geometry-Aware, Risk-Sensitive
Reinforcement Learning for Joint Handover and Power Control in 140 GHz Vehicular Networks."*
A risk-sensitive, blockage-anticipating controller for a vehicular user equipment at
140 GHz that jointly manages handover, transmit power and sub-band allocation, reading a
geometry-derived 3D-blockage-anticipation state computed from the bounding boxes of
surrounding vehicles on real **DAIR-V2X** trajectories through an ITU-R P.526 knife-edge
model, and optimising Conditional Value-at-Risk with a quantile-regression critic.

Daya Sagar G and Deepa Nivethika S., School of Computer Science and Engineering,
Vellore Institute of Technology, Chennai.

## What the paper reports

**The measurement comes first, policy-free.** Sweeping the full 23–30 dBm transmit-power
budget removes at most 8.7 percentage points of sustained radio-link failure, while the
failure surviving maximum power rises from 9.3% to 22.0% as occlusion deepens. At every
depth more failure survives maximum power than the sweep removes. What power cannot reach
is a *tail*, not a shift of the mean — which is what motivates a CVaR objective rather
than assuming one.

**The headline is a negative result.** A tuned threshold rule reading the same anticipation
state reaches 1.2% sustained radio-link failure against the learned controller's 3.3%.
What learning contributes is **selectivity** — it acts on 28.3% of held-out scenarios
against the rule's 98.8%, and issues 12.7× fewer accepted handovers — not a lower failure
rate. For an operator who needs only the reliability, the rule is the cheaper and more
certifiable choice.

**And that 3.3% is a mixture.** The transmit-power branch converges bimodally: fourteen of
twenty runs average 0.96% sustained failure and six average 8.90%, with nothing in between.
The mean describes no run that was trained.

## Repository layout

| | |
|---|---|
| `paper2_env.py` | the single-UE environment: 140 GHz channel, four gNBs, sustained-RLF criterion, reward |
| `paper2ma_env.py` | the multi-UE variant used for the interference-load measurement |
| `blockage3d.py` | 3D ray–box occlusion and ITU-R P.526 single knife-edge diffraction |
| `anticipate.py` | the five anticipation features: time-to-blockage, projected depth, corridor occupancy, closing rate |
| `channel_model.py`, `channel_model_nyu.py` | close-in path loss with measurement-calibrated 142 GHz urban-microcell parameters |
| `paper2_controller.py` | branched QR-DQN, dueling heads, CVaR-greedy selection, n-step Double-DQN targets |
| `paper2_per.py` | n-step prioritised **sequence** replay with importance weights |
| `paper2_baselines.py`, `paper2_antonly_baseline.py`, `paper2_lookahead_baseline.py` | the non-learned reference policies, including the tuned anticipatory threshold rule and the perfect-geometry genie |
| `paper2_sweep_r3.py` | the sweep driver — ablation ladder, replay arm, risk-level arm; resumable |
| `paper2_stats.py`, `make_results_tables.py` | paired seed-level and scenario-level statistics, Holm correction, every table |
| `paper2_basins.py` | the transmit-power and bandwidth convergence split |
| `measure_ho_requests.py` | handover *requests* against *accepted* handovers |
| `extract_dair_geometry.py` | builds the geometry pool from the released DAIR-V2X annotations |
| `figs_paper.py` | every figure |
| `PREREGISTRATION.md` | the twenty-seed analysis, fixed in advance |

## Requirements

Python 3.11, `torch`, `numpy`, `pandas`, `gymnasium`, `scipy`, `matplotlib`. Single-threaded
CPU throughout — no GPU is used or needed. The reported results were produced on an Intel
i9-12900H under Windows 11 with torch 2.5.1, numpy 2.4.4, pandas 3.0.3, gymnasium 1.2.3,
scipy 1.17.1.

## Reproducing

See [REPRODUCE.md](REPRODUCE.md).

```bash
python extract_dair_geometry.py --root <path-to-DAIR-V2X> --out dair_geometry_pool.csv
python paper2_sweep_r3.py --episodes 1500 --seeds 20
python paper2_stats.py
```

A twenty-seed sweep is roughly five minutes per run. Training is a deterministic function
of (seed, configuration) — which has a consequence worth stating, because it is easy to get
wrong: **re-running the same seeds is not an independent replication.**

## Data — what is not here, and why

This repository carries **no DAIR-V2X data and nothing derived from it**. The extracted
geometry pool is built from annotations released by the DAIR-V2X authors under their own
licence, and is not ours to redistribute. `extract_dair_geometry.py` rebuilds it; the
paper's Section 3.5 statistics are reproduced in REPRODUCE.md so an independent extraction
can be checked against ours (**119,015** retained rows, a median of **22** surrounding
vehicles per frame, mean 25.5, range 5–61).

Trained checkpoints, learning curves and the per-episode result files behind every interval
in the paper are archived by the authors and available on request to the corresponding
author.

**DAIR-V2X** — Yu et al., *DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure
Cooperative 3D Object Detection*. Obtain it from its authors and follow their licence.

## Licence

MIT — see [LICENSE](LICENSE). This covers the code in this repository only, not DAIR-V2X.
