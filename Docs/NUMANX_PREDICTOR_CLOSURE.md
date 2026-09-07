# Human candidate and support closure

The coupled candidate now includes the same muscle, gravity, and bias
acceleration as the subsequent Human step. The earlier NumanX candidate used
the accepted velocity plus Matter's correction; Stand then added the free
Human acceleration afterward. This made the physical candidate relationship
incorrect even when the package and transaction checks passed.

**Whole Numi Human remains incomplete.** This increment fixes that relationship
and support-solver defects. It does not qualify standing or walking.

## Implementation

The Metal Stand prefix computes the free velocity and source effective
tangent without advancing live q or completing a step. Matter solves its
correction around that private predictor. Stand consumes the corresponding
reaction once. The accepted velocity and rollback checkpoint remain separate.
The host borrowed pass is version 5; the pointer-free root/publication ABI and
public C/Swift bridge interfaces are unchanged.

Support contacts now use the total candidate point velocity, including body
rotation. The correction vector alone is insufficient. The restoring contact
term now has the correct positive sign in Newton's operator and includes the
normal stabilization derivative. Every Newton evaluation uses the same accepted
history, and candidate history follows the existing commit/rollback protocol.

The nonlinear stop also observes the final certificate's normalization. It
continues solving when its former initial-residual test would have stopped
too early. **Publication tolerances were not changed.**

## Exact stack and results

Tests ran through `ssh macmini` on Apple M4 Pro, 24 GiB, macOS 26.6 build 25G72.
The isolated native and Brain checkouts were clean when the
[44-artifact receipt](media/numanx-predictor-20260908/predictor-receipt.json)
was recorded.

| Owner | Revision |
| --- | --- |
| Numi Lab / native NumanX, `coupled` | `87f67836b1552ff7c110c4a6faacd32f85b0d58f` |
| NumiBrain, `main` | `45d63b51c2d98e83cebce5264472192b86095d46` |

| Check | Result |
| --- | --- |
| Native owner, root publication, candidate, adapter, attachment and full-body tests | 7/7 passed |
| Mixed MPM/FEM and monolithic multiphysics regressions | 2/2 passed |
| Brain legacy/authored publication and rejection/retry tests | 2/2 passed |
| Invalid authored-package admissions | 12 rejected as expected |
| Synthetic 160-DoF free-candidate versus final Stand q/v | zero measured difference |
| Old `v0` candidate, negative control at 1 ms | 9.89437 micrometres of root-position error |
| Support tangent versus central finite-difference force derivative | absolute error 1.90735e-5, below 3e-4 limit |
| Accepted full-body duration | 8 roots × 100 microseconds = 0.8 ms per Brain case |

The zero-reaction synthetic probe tests candidate/Stand agreement under
nonzero source acceleration. It also retains arbitrary-correction kinematics,
analytic attachment Jacobian checks, unchanged source tangent, stale-pass
rejection and buffer-alias rejection. It is separate from the 157-body,
128-DoF, 416-muscle full-body tests. The support probe uses different total and
correction velocities, a nonzero accepted impulse history, and exact rollback.

The [native log](media/numanx-predictor-20260908/predictor-native-final.log),
[Matter log](media/numanx-predictor-20260908/predictor-matter-regressions.log),
and [Brain log](media/numanx-predictor-20260908/predictor-swift-e2e.log) retain
the results. Wall times were 10.52 seconds, 0.88 seconds, and 38.688 seconds
respectively; these include setup/assertions and are not performance evidence.
The [launch recipe](media/numanx-predictor-20260908/run_predictor_swift.py)
and exact input/binary/shader hashes are retained. Raw source payloads and
the compiled Matter package remain local at the receipt's recorded paths.

## Rejected attempts and remaining scope

With only the predictor fixed, the full-body certificate rejected the
incorrect support solve. The
[physical diagnostic](media/numanx-predictor-20260908/predictor-physical-diagnostic.log)
and [intermediate support result](media/numanx-predictor-20260908/predictor-support-tests.log)
are retained. The latter isolated the early-stop/final-certificate mismatch.
An intermediate CTest selection also included an unbuilt attachment binary;
the final seven-test run includes the built, passing attachment test.

The earlier [authored-package qualification](NUMANX_AUTHORED_WORLD_QUALIFICATION.md)
continues to document asset admission and bounded transactions. It must not be
read as proof that its older native revision had this corrected candidate
relationship. The new receipt supersedes that physical integration boundary.

The 51 source joint equalities still need an exact constrained coupled
operator. A full-space projector is not an invertible substitute. Authored
anatomical tissue lowering, mass/active-force replacement, pressure contact,
causal standing control, frozen recovery/walking distributions, calibration,
systemic physiology, and measured performance remain open in the
[completion ledger](HUMAN_COMPLETION_GAP_LEDGER.md).
