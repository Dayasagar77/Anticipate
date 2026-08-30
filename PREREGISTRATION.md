# Pre-registration — increasing the ablation ladder from 10 to 20 seeds

**Written 2026-07-26T03:12Z, BEFORE any seed ≥ 10 was trained.** The ordering matters and is the
point of this document. Increasing a sample size after seeing a null result is only legitimate if
the new target, the stopping rule and the reporting commitment are fixed in advance and the record
is timestamped. Everything below was decided from the ten-seed data and a power analysis computed
on it; no data from seeds 10–19 existed when this was written.

The ten-seed record this document reasons about is retained unaltered in the artefact as
`models/sweep_r3/*_10seed_FROZEN.csv`, so every table below can be recomputed from it after the
sweep has been extended.

`fix_r4b_numbers.py` enforces §4.3 mechanically: the manuscript build refuses to write at any seed
count other than the pre-registered start of 10 or the pre-registered target of 20. The stopping
rule therefore cannot be relaxed after the fact by letting the sweep run further and rebuilding.

---

## 1. The state of evidence at the moment of the decision

Seed-level paired tests, ten seeds, exact Wilcoxon signed-rank on paired-by-seed differences,
Holm-corrected over the seven-comparison family for each metric. Source:
`models/sweep_r3/stats_seed_level_10seed_FROZEN.csv`.

### Reliability (sustained RLF rate, lower is better)

| contrast | n | untied | sign split (maj/min) | p_raw | p_Holm | must beat | verdict |
|---|---|---|---|---|---|---|---|
| p1_transplant | 10 | 10 | 10 / 0 | 0.00195 | 0.01367 | 0.00714 | resolved |
| no_ant | 10 | 10 | 10 / 0 | 0.00195 | 0.01367 | 0.00833 | resolved |
| no_cvar | 10 | 10 | 7 / 3 | 0.13086 | 0.50000 | 0.01667 | **not resolved** |
| no_per | 10 | 9 | 7 / 2 | 0.02734 | 0.13672 | 0.01000 | **not resolved** |
| plus_lstm | 10 | 8 | 5 / 3 | 0.19531 | 0.50000 | 0.02500 | **not resolved** |

### Worst-case tail return (CVaR₂₅, higher is better)

| contrast | sign split | p_raw | p_Holm | verdict |
|---|---|---|---|---|
| p1_transplant | 10 / 0 | 0.00195 | 0.01367 | resolved |
| no_ant | 10 / 0 | 0.00195 | 0.01367 | resolved |
| no_cvar | 6 / 4 | 0.55664 | 0.82617 | **not resolved** |
| no_per | 6 / 4 | 0.10547 | 0.52734 | **not resolved** |
| plus_lstm | 7 / 3 | 0.27539 | 0.82617 | **not resolved** |

### Mean return

All four ladder contrasts resolve; only `plus_lstm` (5 / 5 split) does not.

---

## 2. Why the three contrasts fail, and it is not the test's resolution limit

Two different things can stop an exact signed-rank test resolving an effect, and they call for
opposite responses. They must not be conflated.

**Resolution floor.** With *n* untied pairs the smallest two-sided p the exact test can return is
2/2ⁿ. At ten seeds that is 0.001953, and after Holm over seven comparisons the best attainable
adjusted p is 0.01367 — which clears 0.05. **So the floor is not binding here.** A perfectly
consistent ten-seed effect *is* resolvable at this design, and indeed two of them are. (The floor
*is* binding at the episode level, which is why `signed_rank_floor` is reported there; that is a
different family and a different constraint.)

**Sign inconsistency.** This is what actually blocks all three. `no_per` splits 7/2 across untied
seeds, `no_cvar` 7/3, `plus_lstm` 5/3. The paired bootstrap intervals tell the same story from the
other side: `no_per` reliability is [−10.72, −2.82], excluding zero comfortably, with a
rank-biserial of −0.82. A large, mostly-consistent effect that one or two seeds contradict.

Sign inconsistency is exactly the failure mode more seeds can fix — **if** the inconsistency is
sampling noise rather than a genuinely bimodal effect. Whether it is, is an empirical question, and
that is what the power analysis answers.

---

## 3. The power analysis

Script: `paper2_seed_power.py`. Output: `results/seed_power.json`.

Method: resample seeds with replacement from the observed ten paired differences to a target count
N; recompute the exact signed-rank p for every ladder contrast on that resampled seed set;
Holm-correct the full seven-comparison family; count the fraction of 4000 replicates reaching
Holm-adjusted p < 0.05. One resampled seed index is drawn per replicate and applied to *every*
configuration, which preserves the common-random-number pairing of the real design. The two
alpha-sensitivity contrasts are held at their observed five-seed p-values, because the alpha sweep
is a separate sensitivity check that will not be extended. A parametric paired-t power curve at
each contrast's realistic Holm threshold is computed independently as a cross-check.

**Stated limitation, carried into any manuscript text derived from this.** Bootstrapping a sample
of ten to estimate behaviour at fifty extrapolates well beyond the data. The resampled population
is the empirical distribution of ten seeds and inherits whatever those ten happened to show. These
are order-of-magnitude guides to a required seed count, not precise power calculations.

### Power to resolve at Holm-corrected 0.05 (empirical / parametric)

**Reliability**

| contrast | d_z | N=10 | N=15 | N=20 | N=30 | N=40 | N=50 | 80% at |
|---|---|---|---|---|---|---|---|---|
| no_per | 0.99 | .26/.49 | .70/.78 | **.89/.93** | .99/.99 | 1.00/1.00 | 1.00/1.00 | **20** |
| no_cvar | 0.76 | .12/.36 | .29/.60 | .47/.78 | .72/.94 | .87/.99 | .94/1.00 | 40 |
| plus_lstm | 0.56 | .03/.24 | .16/.39 | .32/.54 | .54/.75 | .72/.88 | .85/.95 | 50 |

**Tail return (CVaR₂₅)**

| contrast | d_z | N=10 | N=20 | N=30 | N=50 | 80% at |
|---|---|---|---|---|---|---|
| no_cvar | **0.10** | .03/.06 | .06/.07 | .08/.08 | **.16/.11** | > 50 |
| no_per | 0.54 | .12/.12 | .46/.36 | .72/.59 | .95/.87 | 50 |
| plus_lstm | 0.48 | .04/.13 | .17/.34 | .30/.54 | .56/.81 | > 50 |

The `no_cvar` tail row is the one that matters most for interpretation. Its power curve is
essentially **flat in N** — 0.03 at ten seeds, 0.16 at fifty. A flat power curve is the signature of
an effect near zero, not of an under-powered design. The observed paired difference is −0.59 CVaR₂₅
points (d_z = 0.10): the risk-neutral variant is, if anything, marginally *better* on the very tail
metric the CVaR objective exists to improve. **No achievable seed count will change this, and the
manuscript must not imply otherwise.**

---

## 4. What is pre-specified

1. **Target.** Extend the six ladder configurations — `model`, `p1_transplant`, `no_ant`,
   `no_cvar`, `no_per`, `plus_lstm` — from 10 to **20 seeds**, i.e. add seeds 10–19. Chosen because
   20 is the smallest target at which the leading unresolved contrast (`no_per` on reliability)
   reaches ≥ 0.8 power on **both** estimators. It is not chosen to be the count that makes something
   significant.
2. **The alpha arm is not extended.** It stays at five seeds and remains a sensitivity check.
   Baselines are unseeded and unchanged.
3. **Stopping rule, fixed now.** Analyse once, at exactly 20 seeds. **No further extension will be
   made regardless of the outcome.** If a contrast is still unresolved at 20, it is reported as
   unresolved and the power table above is what explains why.
4. **No other change.** Same environment, same controller, same 30/15 scene split, same
   common-random-number evaluation, same 500 held-out rollouts, same statistics code, same Holm
   family size of 7. Only the seed count moves. This is the single-variable discipline the whole
   three-paper arc is built on.
5. **Reporting commitment.** The manuscript will state that the seed count was increased from ten
   to twenty, that the increase was decided from a pre-run power analysis, and that this document
   fixed the target and the stopping rule beforehand. The ten-seed table will not be quietly
   replaced without that disclosure.
6. **Verdict words stay generated.** Every claim-strength word in the manuscript is produced by
   `Numbers.verdict()` from the result files, never typed. Whatever the twenty-seed data says
   rewrites the prose automatically — including if a currently-resolved contrast stops resolving.

## 5. Predictions recorded in advance

So that the record shows what was expected before the data existed:

- `no_per` on reliability — **expected to resolve** (power ≈ 0.9).
- `no_cvar` on reliability — **genuinely uncertain**, roughly a coin flip (power ≈ 0.5–0.8).
- `plus_lstm` on reliability — **expected to remain unresolved** (power ≈ 0.3–0.5).
- `no_cvar` on tail return — **expected to remain unresolved**, and expected to stay near zero.
- `no_ant` and `p1_transplant` — expected to stay resolved on all three metrics.

If `no_ant` or `p1_transplant` were to *stop* resolving at twenty seeds, that would be a material
finding about the paper's headline claims and would be reported as such, not re-run away.

## 6. The finding this does not rescue, and should not

Independent of seed count, the ten-seed data already establishes something the paper must state
plainly, because a reviewer will find it otherwise:

- The risk-neutral variant (`no_cvar`) achieves **significantly higher mean return** than the CVaR
  controller (Holm p = 0.0137, 86.0 vs 79.5).
- It achieves **statistically indistinguishable worst-case tail return** (77.3 vs 76.7, d_z = 0.10,
  power flat in N).
- The CVaR objective's only benefit is a directional reliability gain (4.34% → 1.18% RLF) that ten
  seeds do not resolve, and twenty may not either.

The honest characterisation is therefore: **the CVaR objective trades established mean return for a
reliability gain that this design does not establish, and it does not measurably improve the return
tail it was introduced to improve.** More seeds may resolve the reliability leg. Nothing will
resolve the tail leg, because the effect is not there. Section VIII-D must say this in those terms.

## 7. Reproduction

```
python3 paper2_seed_power.py --json results/seed_power.json
python3 paper2_sweep_r3.py --seeds 20 --no_alpha_arm --no_baselines --workers 2
python3 paper2_stats.py --outdir models/sweep_r3
```

The sweep skips any job whose `_metrics.csv` already exists, so seeds 0–9 are reused unchanged and
only seeds 10–19 are trained. Estimated cost: 60 training runs, ~3,518 s of CPU per seed across the
six configurations, ≈ 5–7 h wall-clock on two cores.
