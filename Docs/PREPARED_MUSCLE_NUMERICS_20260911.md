# Prepared muscle numerical repairs

Native `8aa3ad1b971618100552cda01c96f315a810d722`, published on `origin/coupled`,
repairs inside-sphere wrapping, cancellation in the implicit fibre update, and
an undocumented ten-microsecond timestep floor. The preceding
[initial-pose repair](COUPLED_INITIAL_POSES_20260910.md) remains qualified.

| Fixed-state comparison | Before | After |
|---|---:|---:|
| Maximum path error, 416 muscles versus native FP64 | 274.847 µm | 0.572 µm |
| Same-path fibre force error, 100 µs | 0.272733 N | 0.002789 N |
| Normalized fibre residual, 100 µs | 0.00216488 | 2.34228e-7 |
| Same-path fibre force error, 1 µs | 2.043753 N | 0.002760 N |
| Fibre publication discrepancy, 1 µs | 26.1 length ULPs | 0.495 length ULPs |

The two path outliers were EDC5_l and EDC4_l. The inside-wrap Newton solve used
a coordinate whose derivative becomes singular near one; FP32 overshoot could
choose the midpoint fallback. The same source equation now uses an angle
coordinate and a safeguarded bracket. The independent pinned MuJoCo 3.12 audit
confirms maximum path error **0.576 µm** at the same FP32 initial pose, with all
57 source files verified against the source archive. This is numerical source
agreement, not anatomical or experimental calibration.

The fibre solver now searches for the small displacement about the accepted
length. This preserves the velocity and tendon extension before rounding the
published absolute length. Tendon strain is calculated from extension, and the
residual is recomputed at the returned implicit displacement. The existing
FP32 state layout, source constitutive laws and force ownership are unchanged.
The authored positive timestep is used directly, including below ten
microseconds. The probe's timestep override is a reference calculation at a
fixed state; it neither steps the body nor admits a different live transaction.

All 416 muscles pass at **1, 5, 10 and 100 µs**. Maximum normalized same-path
force error is below 1.81e-6 and normalized equilibrium residual below 2.90e-7,
against fixed 1e-5 numerical budgets. Published length agrees with previous
length plus timestep times published velocity within 0.501 length ULPs. The
old solver fails the same one- and 100-microsecond gates; those controls remain
in the evidence bundle.

Ten native checks and the default-pose source/path/force/activation probe pass
with Metal API validation. The allocator address-reuse subcase remains explicitly
`not_observed`; the adapter's transaction and rollback checks still execute.
The exact published stack with Brain `c7db5de` passes four scenarios of 64 roots,
**6.4 ms each**, with bitwise recruited replay and dropout/zero physical equality
across all ten trace kinds. An independent check over the complete cohort finds
maximum fibre publication discrepancy 0.50000019 ULPs. Terminal recruited root
speed is 0.0371961 m/s; this is unheld motion, not standing qualification.

Full source-force convergence remains open. Using the independent FP64 path
instead of the GPU path gives a maximum force difference of **0.780 N at
100 µs**, and **4.844 N at 1 µs**. Sub-micrometre geometry differences are
amplified by these stiff fitted architectures. The same-path scalar result
does not close geometry/force consistency, loaded equilibrium or timestep
convergence. The next mechanical work must resolve that consistency before
claiming settled anatomical loading or selecting a stance controller.

The validation-enabled cohort took 445.835 seconds and included a brief Metal
System Trace attachment. It is not a performance qualification. The trace
contains 2.071 seconds of unioned owned active-compute intervals, including
0.617 seconds in nine named groups containing candidate kinematics; the longest
such group is 80.6 ms. Application encoding intervals cover 54.3 ms. No graphics
compiler or individual shader-profiler intervals were recorded. Intervals can
extend across the requested two-second capture boundary. These grouped events
identify work to inspect, but cannot establish an individual kernel's cost.
The exported timing tables are retained; the raw trace and environment-bearing
table of contents stay on the Mac mini.

A separate four-root diagnostic with API validation disabled took 30.517
seconds, versus 31.640 seconds with validation enabled, and all 160 physical
trace records are byte-identical. This small comparison does not establish a
speedup; disabling validation does not remove the large execution cost.

The [receipt](media/prepared-muscle-numerics-20260911/receipt.json) preserves
failed controls, intermediate numerics, exact source snapshots, native logs,
independent source checks and the coupled cohort. Its
[verifier](media/prepared-muscle-numerics-20260911/verify_receipt.py) recomputes
force normalization from the frozen NHMYO2 payload and audits complete physical
traces. Registered anatomical tissue, independent calibration, sustained
standing/recovery/walking and the five performance workloads remain open.
