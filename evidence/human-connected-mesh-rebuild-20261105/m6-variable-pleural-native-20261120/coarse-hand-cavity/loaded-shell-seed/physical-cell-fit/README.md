# Author the two limiting bed-load cells

The matched control and candidate use the published offline preload solver
with one numerical correction. The passive mattress potential is evaluated
at the actual indentation. The contact equilibrium is solved by a
48-iteration bracket, then the normal load comes from the foundation
reaction. This prevents the contact root's force rounding from dominating
the energy near Newton convergence. The equations and residual gate are
unchanged. The 10 kPa, 120 mm foundation and prescribed deep-boundary
assumptions are the same as in the first checkpoint.

The authoring fit moves two middle-layer nodes of the exact native coarse
reference. The two local linear programs preserve reference volume for
each affected material, anatomical region and donor. Shape checks evaluate
the actual Float32 emitted geometry. The remapping recomputes lumped mass
increments from the change in reference volume while conserving each
regional quota. Independent validation counted all 28,632 positive cells
in both frames, no drop in either global minimum shape measure and a
maximum mass difference of 2.22e-16 kg across 628 typed banks. The verified
skin/fat mass is 13.939679828 kg. Exterior and deep surface nodes, element
connectivity and existing cell ancestry stay fixed. Interior movement was
limited to 0.515 mm. This is an offline mesh and mass candidate.

| Static setup | Last converged fraction of gravity | First failed stage |
| --- | ---: | --- |
| Matched 10 kPa mattress control | None | 2.5%, cell 20754 reaches expansion limit |
| 10 kPa mattress with repaired internal nodes | 6% | 7.5%, new expansion witness 22659 |
| Rigid bed control | 5% | 6%, cell 22244 reaches compression limit |
| Rigid bed with repaired internal nodes | 5% | 6%, new compression witness 22240 |

The repaired compliant run balances 57.1719 N of the model's 952.8644 N
weight at 6% gravity. At 7.5% gravity it spreads the resultant over 263
nodes but fails the nodal force residual before convergence. The opposed
witnesses show where local layer quality limits the static solve. Neither
fixture establishes full-load equilibrium. Native loaded ligaments, motion
clearance, variable pleural pressure and throughput still need combined
verification. The fit does not replace the separate scalp-clearance
candidate and has not been applied to runtime assets.

Reproduce beside the baseline inputs pinned in the receipts.

```sh
python3 test_predict.py
python3 predict.py mattress-exact-control-20 --mattress-kpa 10 --loads .025 .05 .06 .075 .1 --max-iterations 145 --redirect
python3 fit_preload_cells.py fit-exact-21
```
