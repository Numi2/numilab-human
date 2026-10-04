# Complete source-skin motion screen — 2026-10-04

This CPU-only screen carried the retained closed FJ2810 source-skin surface
through the current ABI 5 linear-blend skin field at three fixed bilateral
knee poses. The outer-source weights were held exactly, and unknown inner and
connector weights were extended with one inverse-edge-length graph-harmonic
solve. The plan and source identities were pinned before the pose predicates in
[`plan.json`](media/skin-full-solid-motion-20261004/plan.json).

The source-rest reconstruction passed at `1.53e-7 m`; the complete emitted
surface remained a closed oriented manifold in all three samples. Its outer
vertices matched the retained native ABI 5 packs within `1.434 µm`, below the
fixed `2 µm` control limit. The full solid nevertheless failed embeddedness:

| Bilateral knee angle | Full-solid exact pairs | Outer-to-outer pairs | Inner/connector-to-outer pairs | Inner/connector-to-inner pairs |
| ---: | ---: | ---: | ---: | ---: |
| 0.4 rad | 2,760 | 0 | 2,760 | 0 |
| 0.8 rad | 2,403 | 0 | 2,403 | 0 |
| 1.2 rad | 2,098 | 75 | 2,023 | 0 |

At 1.2 rad the 75 outer-to-outer pairs equal the independently retained
outer-shell baseline. The other pairs arise where the inferred interior or
connector sheet crosses the outer sheet. Counts fall across these three
snapshots, so the screen does not support a monotonic flexion trend. The exact
pair lists, emitted positions, per-pose rows and aggregate summary are retained
under [`results/`](media/skin-full-solid-motion-20261004/results/). The
independent [pair-sheet classification](media/skin-full-solid-motion-20261004/results/crossing-classes.json)
reclassifies only those retained witnesses; it runs no new geometric predicate.

This rejects the global harmonic extension as a complete-surface motion
candidate. It does not refute the full source geometry or the outer ABI 5
deformation, and it does not diagnose physical skin mechanics, material,
collision, clinical anatomy, or continuous motion. No runtime payload changed.
The next skin step needs source-local constraints for hidden-sheet motion;
further arbitrary damping of the inherited outer field would risk hiding the
known flexion crossings by suppressing knee response.

The first three implementation attempts are retained in
[`attempts/`](media/skin-full-solid-motion-20261004/attempts/). The first
applied a partition tolerance that was too strict for inherited Float32 rows;
the second could not save the diagnostic because the output directory had not
yet been created; the third compared the empty-override pose result instead of
the explicit `qpos0` reconstruction. Those observations are marked invalid.
The corrected run passed the original `20 µm` rest gate before running any pose
predicate. Focused tests and Ruff pass for the audit implementation.

Reproduce the screen and the pair-sheet classification with:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:Sources/myosim/checkout \
  .venv-mujoco312/bin/python -m numilab_human.skin_full_solid_motion \
  --output Docs/media/skin-full-solid-motion-20261004/results

OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:Sources/myosim/checkout \
  .venv-mujoco312/bin/python -m numilab_human.skin_full_solid_crossing_classes \
  --results Docs/media/skin-full-solid-motion-20261004/results \
  --output Docs/media/skin-full-solid-motion-20261004/results/crossing-classes.json
```
