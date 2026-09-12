# Directional valve resistance, 12 September 2026

This increment adds an explicit native flow law with nonnegative forward
resistance and finite positive reverse resistance. Zero forward resistance
becomes a pressure-equality constraint in Matter's coupled solve. Reverse
leakage remains finite. All 29 selected native checks pass on the physical
Apple M4 Pro with zero failures or skips. Native commit
[`6f4a47ae`](https://github.com/Numi2/numi-lab/commit/6f4a47ae5102235aa5d078947216eb48e488f97c)
is published on `coupled`; the [qualification receipt](media/cardiac-directional-valve-20260912/qualification.json)
binds 22 source files, rebuilt executables, the production Metal library,
43 retained artifacts and the verified published revision.

The final rebuild takes 8.5586 seconds and the serial 29-check selection takes
74.8470 seconds. It reruns the prior 27 selected material, reference,
vascular/cavity, snapshot, transaction, mixed-physics and identification checks
plus the new compiler and Metal valve checks. Source and binary hashes are
unchanged across execution. Historical receipts keep their original ABI and
source identities.

## Source coefficients and formulation boundary

Rodero S4 page 3, Table F supplies the resistance pairs below. The
[coefficient record](media/cardiac-directional-valve-20260912/source-coefficients.json)
pins the exact supplement and unchanged parent Human configuration. The
supplement gives peak resistance values but does not specify a complete valve
switching equation. This increment qualifies an explicitly declared native
formulation using those coefficients; it does not establish exact CARP
switching behavior or measured leaflet mechanics.
[S4 supplement](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1008851.s004&type=supplementary).

| Source valves | Forward resistance | Reverse resistance |
| --- | ---: | ---: |
| Mitral and tricuspid | 0.05 mmHg s/mL | 1000 mmHg s/mL |
| Aortic and pulmonary | 0 mmHg s/mL | 10000 mmHg s/mL |

The declared unit conversion is 133.322387415 Pa per mmHg, so the resistance
multiplier is 133322387.415 Pa s/m³ per mmHg s/mL. The source's valve rows print
`mmHg/mL/s`; the normalized resistance dimension follows its Windkessel model.
This convention is explicit and separate from Shi–Hose's literal pressure
conversion in its own source reproduction.

## Native equation

With signed flow Q from compartment a to b, the physical residual is

```
r = R(Q) Q - (Pa - Pb)
R(Q) = Rforward for Q > 0, Rreverse for Q < 0
dr = Rbranch dQ - (dPa - dPb)
```

At Q=0, the residual is independent of the chosen resistance. The directional
linearization chooses the pressure-driven branch, with a fixed forward tie
at zero pressure difference. Species transport must use the same branch
choice at that tie.

For an ideal forward branch the flow equation contains no artificial
resistance or division by zero. The existing volume balances and pressure
relations determine Q together with the rest of the system. A numerical scale
used only in the preconditioner does not change this physical residual.

The C++ `WorldSource` interface owns this bounded increment. Older Human JSON
schemas retain their previous law sets; no old payload silently acquires a
different valve interpretation.

## Admission and evidence gates

The admitted coefficient range is 0 <= Rforward <= Rreverse with finite
Rreverse > 0. Inertance, orifice coefficients and pressure floors are excluded
from this law. An authored positive forward resistance may not round to zero
and silently become an ideal constraint. Only the existing unilateral laws
retain nonnegative-flow limiters and zero-flow working sets.

The representation uses Matter ABI33, package18, snapshot archive11 and accepted
proof manifest11. The reverse coefficient participates in compiled identity and
package layout validation; incompatible historical packages remain versioned.

Potential ideal forward edges must form an undirected forest. This is a
conservative structural restriction that excludes redundant ideal-pressure
cycles. It does not prove invertibility for every mechanically coupled cavity
network. Other rank or convergence failures remain ordinary rejected native
transactions.

For two compliant reservoirs, an independent backward-Euler reference is

```
Qnew = (Pa_old - Pb_old) / (Rbranch + dt (1/Ca + 1/Cb))
Va_new = Va_old - dt Qnew
Vb_new = Vb_old + dt Qnew
```

Six runtime cases pass on the physical Apple M4 Pro, each with two environments
and 16 primary steps at 0.01 seconds (0.16 seconds of synthetic dynamics),
followed by replay, isolated rollback, signed-flow restore and episode reset.
The cases include nonzero-flow crossings in both directions at each source
resistance pair, plus reverse opening from zero and ideal forward opening.
Volumes, compliance, pressures, tissue exchange and tracer amounts are synthetic.
The coefficient record retains the declared decimal SI conversion; the FP64
runtime oracle uses serialized FP32 coefficients.

| Quantity | Maximum observed | Unchanged acceptance bound |
| --- | ---: | ---: |
| State error, normalized by its variable scale | 1.8341e-6 | 8e-5 |
| Signed flow relative error, reference magnitude > 1e-10 m³/s | 2.7105e-5 | 3e-4 |
| Closed volume or species relative error | 4.7684e-7 | 3e-5 |
| Pressure equation error / 1000 Pa | 7.3616e-7 | 3e-5 |
| Direct residual/Jv error / max(1, reference magnitude) | 7.9535e-8 | 2e-5 |
| One-sided or interior finite-difference error / max(1, Jv magnitude) | 5.1392e-4 | 3e-3 |
| Equal-resistance legacy state difference, normalized | 1.1921e-7 | 3e-5 |

Four direct production-kernel controls in two environments check the residual,
selected Jacobian, unequal-concentration reverse donor at Q=0, forward pressure
tie, and an exactly zero ideal forward diagonal. The direct harness invokes
production pressure-coefficient preparation. Two additional two-edge controls
check the earliest common breakpoint, including FP32 rounding, 28 rows of
bitwise trial/apply agreement and unchanged accepted state, correction and masks.
Their synthetic mechanical-owner slots verify shared alpha publication; they
do not establish a coupled anatomical FEM trajectory.

Forty CPU admission controls cover source and cooked coefficients, resealed
packages, ideal-edge cycles, underflow, signed flow and atomic writer rejection.
They execute no physical steps. Direct production residual/Jv checks and native
transactions remain distinct from those admission checks.

## Retained failure and globalization repair

The first physical attempt failed on the first forward-to-reverse case at the
source mitral resistance ratio. It exhausted all eight line-search trials and
rejected the transaction with native status 10. The last tried step was
0.0078125; diagnostics recorded the next halved step, 0.00390625, alongside
initial merit 1.126669 and last trial merit 1.431498. The exact log, source and
binary hashes remain under `native-runs/metal-attempt-001`.

The repair passes at the same 12 Newton iterations, 64 FGMRES
iterations and eight merit trials. It limits the common update to the earliest
directional-flow breakpoint and reevaluates the branch on the next Newton
iteration. Every state owner consumes that same step. A flow-only clip, an
artificial positive resistance or a looser physical tolerance would change the
problem and is not part of this repair.

The second physical attempt passes all six runtime cases after the core fix,
then aborts in the direct harness because its shader lookup omitted the
production namespace. Independent review also found missing pressure-coefficient
preparation in that harness. Both harness defects were corrected without changing
the fixtures or acceptance gates; attempt003 passes the complete focused check.
Both earlier physical attempts and all four CPU build/admission attempts are
retained, including their exact source, executable and log hashes.

Anatomical pressure-port assignment, quantitative supports, unloaded/loading
data, source activation timing, blood mass transfer and subject calibration
remain separate gates. A two-reservoir valve control does not simulate the
1,470,083-tetrahedron cardiac wall or a complete cardiac cycle.
