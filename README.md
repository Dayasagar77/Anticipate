# Anticipate

Code for **"Anticipating the Blockage Tail: Geometry-Aware, Risk-Sensitive Reinforcement
Learning for Joint Handover and Power Control in 140 GHz Vehicular Networks"**
— Daya Sagar G and Deepa Nivethika S., School of Computer Science and Engineering,
Vellore Institute of Technology, Chennai.

A risk-sensitive, blockage-anticipating reinforcement-learning controller for a vehicular
user equipment at 140 GHz, evaluated on real DAIR-V2X trajectories with three-dimensional
dynamic occlusion computed from vehicle bounding boxes through an ITU-R P.526 knife-edge
model.

## What the paper reports

- A **policy-free measurement** of the blockage–reliability geometry: sweeping the full
  23–30 dBm transmit-power budget removes at most 8.7 percentage points of sustained
  radio-link failure, while the failure surviving maximum power rises from 9.3% to 22.0%
  as occlusion deepens. What power cannot reach is a *tail*, which is what motivates a
  CVaR objective.
- A branched **QR-DQN controller** with CVaR-greedy action selection over handover,
  transmit power and sub-band, with n-step prioritised sequence replay.
- A **negative result, reported as the headline**: a tuned threshold rule reading the same
  anticipation state reaches 1.2% sustained radio-link failure against the learned
  controller's 3.3%. What learning contributes is selectivity — 12.7× fewer accepted
  handovers and 11.1× fewer requests — not a lower failure rate.
- The 3.3% is a **mixture**: the transmit-power branch converges bimodally, with fourteen
  of twenty runs averaging 0.96% and six averaging 8.90%. That structure is analysed
  separately.

## What is in this repository, and what is not

**Included** — everything in this project that is ours: the environment, the channel and
3D-blockage models, the anticipation features, the controller and replay, every baseline
and reference policy, the sweep driver, and the statistics and figure code.

**Not included, deliberately:**

| | why |
|---|---|
| `dair_geometry_pool.csv` | Derived from **DAIR-V2X**, which is released by its own authors under its own licence. It is not ours to redistribute. Rebuild it locally with `extract_dair_geometry.py` from the released annotations. |
| `models/`, `*.pt` | Trained checkpoints — regenerable from the sweep driver, and large. |
| `results/`, `figures/` | Regenerable outputs. |

## Reproducing

See [REPRODUCE.md](REPRODUCE.md). In short: obtain DAIR-V2X, build the geometry pool,
run the sweep, then the statistics.

```bash
python extract_dair_geometry.py --root <path-to-DAIR-V2X> --out dair_geometry_pool.csv
python paper2_sweep_r3.py --episodes 1500 --seeds 20
python paper2_stats.py
```

Single-threaded CPU throughout; a 20-seed sweep is roughly five minutes per run.

## Data

**DAIR-V2X** — Yu et al., *DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure
Cooperative 3D Object Detection*. Obtain it from its authors and follow their licence.
This repository redistributes none of it.

## Licence

MIT — see [LICENSE](LICENSE). The licence covers this code only, not DAIR-V2X.
