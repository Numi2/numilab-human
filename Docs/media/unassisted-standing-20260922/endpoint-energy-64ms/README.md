# Accepted-endpoint energy, 64 ms

Three native Apple M4 Pro runs accepted all 64 × 1 ms steps and exited zero.
Energy accounting on/off produced **bitwise identical terminal 129 q and 128 v**.
All three started from identical q/v and used the same unchanged executable,
library, metallib and five source payloads; hashes and raw logs are retained.
Wall time was 63.82 s (feedback + energy), 63.75 s (feedback), and 67.19 s
(feedback disabled + energy), including initial recruitment.

Native source: [9433b5c4](https://github.com/Numi2/numi-lab/commit/9433b5c4e120468a343532ed84708ba77e587196).
The read-only diagnostic uses actual body masses, world inertias and armatures,
compensated root translation, and the two accepted endpoints of each step.
Constraint work uses native final impulses and pre-step Jacobians/linearization
against the accepted endpoint midpoint velocity. No alternate physics run is used.

| Quantity (J) | Feedback 10 / 1 | Feedback disabled |
|---|---:|---:|
| Kinetic energy change | 7.778081845e-5 | 4.250087397e-5 |
| Gravity potential work | -3.010837496e-3 | -3.251659818e-3 |
| Passive potential work | 3.162055584e-7 | -5.335302552e-7 |
| Muscle midpoint work | 1.522986992e-3 | 1.706524016e-3 |
| Joint damping midpoint work | -1.267481179e-5 | -1.187668706e-5 |
| Contact normal endpoint work | 1.554491922e-3 | 1.593432297e-3 |
| Contact tangent endpoint work | 2.138163452e-5 | 3.953856589e-6 |
| Equality endpoint work | 2.064577008e-12 | -1.718421866e-12 |
| Equality solver-iterate correction work | -273.5236111 | -273.2703161 |
| Unclosed available-terms residual | 2.116369844e-6 | 2.660740734e-6 |

Body damping work was zero. Source-limit endpoint work, exact projection work
and internal musculotendon energy remain unavailable; integration/bias/quadrature
remainders remain unresolved. **This is not energy closure or a ten-second energy
run.** The short disabled-controller comparison establishes changed motion, not
sustained balance benefit: maximum q difference 6.931787357e-5 at coordinate 42,
maximum v difference 0.001313039626 at DoF 55 (mixed generalized units).

The first attempt reached 64 native accepted steps but failed final borrowed
tendon publication checking: the diagnostic used one-step batches while the
check expected eight. `failed-batch-count/` retains its exit 1 and logs. The fix
matches the expected final batch count to the actual submission size, preserving
all transfer, byte equality, abort and physical gates. The corrected runs above
passed those gates. The original ten-second standing artifacts remain separate.

Reproduce with an isolated build of the linked native commit and the existing
`native-runtime-source-package-20260915/input` package:

```sh
sh reproduce.sh /path/to/native-build /path/to/input /new/output-directory
python3 compare.py /new/output-directory
```

`compare.py .` also validates the compressed retained outputs in this directory.
For ordinary Human command use, `--execute --endpoint-energy` enables accounting;
use a short prefix for diagnosis. The default sustained standing path is unchanged.
