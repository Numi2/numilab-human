# Feasible volume directions in the registered skin and fat

All seven native releases use the same anatomy, bed, muscle recruitment, histories, pressure coupling, time step, physical equations and acceptance tolerances on SSH Numi, Apple M6. The solver can alter its proposed Newton correction before the original whole-scene line search. A failed physical step still rolls back. This is an opt-in solver experiment.

| Coupled run | GPU time advances | Fresh residual at rejection | Last root Krylov columns | Last full-trial volume witnesses | Wall time including preparation, s |
|---|---:|---:|---:|---|---:|
| Registered skin, ordinary Newton direction | 2 | 0.035921149 | 768 | 950449, 1035269 | 261.29 |
| Two measured cells | 2 | 0.000602195039 | 848 | 874304 | 265.32 |
| Two cells, conservative Float32 output | 2 | 0.000602394983 | 928 | 874304 | 267.93 |
| GPU-selected cells with one owner per node | 2 | 0.023617046 | 768 | 950449 | 264.21 |
| GPU-selected cells with simultaneous private-node corrections | 3 | 9.53673807e-05 (terminal geometry failed) | 688 | 950449 | 255.42 |
| Same directions, loaded-volume certificate | 2 | 9.53673807e-05 | 688 | 950449 | 257.64 |
| Loaded-volume search and certificate | 2 | not assembled | 528 | none recorded | 254.73 |

The fixed-patch test reduced the third-step residual by a factor of 59.63. Its two original fat cells cleared that full-trial volume check, exposing a different limiting cell. Both accepted FEM load records and the coupled physiology trace reproduced the control.

A separate implementation obtains the sparse active set from the previous Newton iteration's full-trial geometry records. It retains the last determinant witness per 256-cell tile within the current physical solve, assigns shared node claims by minimum global element ID, then tests each disjoint proposal against the current geometry. Records reset at Newton iteration zero. It modifies only the search vector. Geometry validation still tests every active element. Constitutive failures, contact, material history and the independently assembled final momentum balance retain their existing authority.

In the priority-only version, native rows 874304 and 950449 share a node. The lower-ID row could reserve that node even when its current step was feasible. The simultaneous version keeps the original Newton motion on shared nodes and projects corrections on the private nodes of each implicated cell. A mass-orthogonal projection keeps the private correction balanced. This changes the solver search only. Another 1,536 paired GPU cases separately exercised shared-node ownership, simultaneous volume changes and exact preservation of the shared directions.

The production Metal kernel passed 3,072 independently evaluated trials, including compression, dilation, varied masses, legacy and compensated coordinates, admissible directions and fixed corners. The first implementation lost up to 4.78e-5 of the magnitude of the proposed momentum adjustment when small corrections were added to large Float32 velocities. Balancing the emitted search vector reduced this to 7.34e-8. The dynamic selection test also passed conflicting and disjoint cells, invalid indices within a different tile, nonvolume failure reasons and iteration resets.

The independent terminal audit exposed a disagreement about the volume reference. Element 1035269 started at 1.41864 times its material-reference volume. Its candidate near 0.35 of material volume had fallen to 0.246715 of its initially loaded volume. The runtime had already advanced the controller and mechanical clock to step three when the terminal exporter rejected it. This row in the table therefore has no qualified third endpoint. The subsequent GPU certificate checks the original terminal criterion on registered physical geometry before commit, using compensated signed determinants and a deterministic lowest-index witness. The release-aware search also uses the tighter of the material and loaded-volume bounds. The separate terminal audit is retained. In 4,608 M6 checks the largest GPU-to-binary64 ratio difference was 3.41e-8. The recorded 1.25e-5 is the tightened inner nonlinear-latch target. The fresh object certificate applies its separately configured relative residual tolerance.

The certificate-only integrated replay rejected element 1035269 before step three committed. The reported loaded-volume ratio was 0.246715158. Both accepted force records and coupled physiology records matched the control. The release-aware solver committed 2 steps. Its next search reached the registered volume floor on element 1035269 and stopped before the final force assembly. Its last measured inner residual was 9.9197037e-05.

The same dense body on the captured two-step control still exhibits 10,895 exterior skin/tissue intersections and excessive bed load. The parent physical registration contains those measured witnesses. The 16-second resting-body qualification and its 0.1× performance requirement remain open.

Restore the registered-fold parent release, build the base experiment with build.py, then the Float32 correction with quantized/compile.py, the active selector with dynamic/build.py and the simultaneous projection with coupled/build.py. Build the release-relative GPU certificate with release/build.py and use its parent-1 or parent-2 launch as recorded. Each receipt pins build commands, inputs and native outputs. For the release-aware replay, use the registered root, cavity tools and recruitment paths from the recovery manifest. The native entry point is:

```bash
"$PY" "$z/run_cavity_motion.py" "$a/material-direction/release/parent-2" "$a/material-direction/release/owner" "$z/typed-fans/six-margin.npz" "$fresh_output" --steps 3 --trace-root 2 --capture-fem-body-load --recruitment-reference "$h/drive-control-60"
```

Choose a fresh output directory for each trial. qualify.sh checks the identical input set and conditional endpoint and independently restores the solver evidence from its GitHub release.
