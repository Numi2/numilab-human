# Native whole-body force audit — 2026-09-14

The native audit reader now keeps the six floating-root coordinates separate
from the 122 internal coordinates. It requires every generalized-force owner
to declare availability before a static handoff can be admitted. A small root
wrench therefore cannot hide an unbalanced knee, tendon, fibre, joint-limit,
or contact coordinate.

The retained physical-M4 experiment was run from the isolated
`MetalRobo-human-static-contact-experiment-20260914` worktree at source
`86c24d8` plus its uncommitted source patch. The NHTENDON2 v4 payload was
`4ede760b55218831f4c423b6c00b0119b4141922fee085d2fced228b0e8ebcf2`. The
native audit input and the derived receipt are preserved under
[`Docs/media/native-whole-body-force-audit-20260914/`](media/native-whole-body-force-audit-20260914/).
The experiment is diagnostic evidence only; the shared source checkout was
restored unchanged after the run.

The audit reports:

| Quantity | Result |
| --- | ---: |
| generalized coordinates | `128` |
| floating-root coordinates | `6` |
| internal coordinates | `122` |
| root maximum absolute residual | `30.9265819982` |
| internal maximum absolute residual | `1315.41475411` |
| all-coordinate maximum normalized residual | `1.00000000000` |
| fibre/tendon candidate state | `solved=true` |
| complete force-term ownership | `false` |
| whole-body generalized equilibrium | `false` |

The missing native terms are tendon, ligament/limit, contact, joint equality,
and damping. The largest rows are internal coordinates `118`, `104`, `125`,
and `111`, with residuals approximately `1315`, `1205`, `-943`, and `-943`
N. This is the expected failure mode when the root wrench is treated as the
sole equilibrium check.

The isolated contact handoff also passed one-step FP64 parity after the static
force sign was corrected, but it did not improve the release. The 64-step
`12.5 µs` run reached `81863.03125 m/s²`, `42.8818702698` maximum velocity
change, and `0.0132911428809` maximum configuration change. That failed result
is retained as evidence and is not admitted as standing or walking.

Run the local gate with:

```sh
PYTHONPATH=src python -m numilab_human.native_whole_body_audit \
  --input Docs/media/native-whole-body-force-audit-20260914/source-audit.json \
  --output /tmp/native-whole-body-force-receipt.json
```

The command is also registered as `numilab-human native-whole-body-force-audit`.
It can report a static whole-body equilibrium only after all terms are present
and the configured normalized residual closes; it never promotes that static
result to sustained standing, recovery, walking, anatomical loading, blood
transfer, material calibration, or subject calibration.
