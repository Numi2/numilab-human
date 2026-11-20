# Parallel Metal operator launch for the cleared skin/fat scene

SSH Numi, Apple M6. The previously published body, skin/fat registration,
material programs, contacts, 24-iteration budget, timestep, and coupled
acceptance thresholds are the same. The launch uses three existing ordered
Metal kernels:

- `NM_FEM_GATHER_SIMD=1` evaluates incident FEM and contact contributions
  across lanes and applies them in the original row order.
- `NM_FEM_CACHED_NODE_SIMD=1` distributes cached nodal block construction.
- `NM_VASCULAR_COOPERATIVE_CAVITY_OPERATOR=1` distributes cavity face work
  and retains compensated arithmetic and face order.

For the 64-root matched test, body time decreased from 157.492 to
125.808 seconds, a 20.1% reduction at the same accepted simulated time.
Wall time from launch decreased from 174.503 to 142.706 seconds. The
measured body real-time factor was 0.00000795. An eight-root comparison
reduced body time from 19.900 to 16.241 seconds. Adding the parallel base
preconditioner provided no additional improvement in the eight-root tests.

Both 64-root runs used 625 Krylov columns in 171 cycles with work.
Mechanical and vascular residual maxima were 79.96% and 4.58% of the
same limits. The coupled traces and emitted initial mesh/mass CSVs are
byte-identical. All 91 captured regional force and strain comparisons
pass the 1% thresholds. The maximum accepted FEM coordinate difference
is 1.14e-16 m using the existing endpoint comparator. All 28,632
skin/fat cells have positive terminal Jacobians, with minimum 0.98885.
Independent exact shell triangle checks found zero self or cross-layer
intersections at either endpoint. This audit does not infer geometry
at intervening times or test all other tissue surfaces.

The launch retains variable external pleural pressure. This 1 ms interval
does not encompass a breath. It remains an unbalanced release with real
bed contact and no root assistance. Equilibrium, shoulder loading under
equilibrium, 16-second clearance, native presentation, and 0.1 times
real-time performance are not qualified.

The full core binary, source asset and initial pack hashes are pinned in
the native launch receipt. Match the pack SHA when restoring from the
included private recovery receipt. On SSH Numi, the validated workspace
driver is `coarse-hand-cavity-024/operator-resident-030/replay.py`.
From that folder, run `python3 replay.py parallel-operators <new-output>
--steps 64` with the pinned pack at `initial.mrvpack`, using the existing
preparation venv. Both control and candidate write to a new directory.
