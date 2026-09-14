# Native Human static-equilibrium and release audit — 2026-09-14

This audit separates the part that is solved from the part that still fails. It
uses one adult-male source package on the physical Apple M4 Pro, the canonical
`12.5 µs` clock, NHCNT2 support primitives, NHEQ1 joint equalities, and
NHTENDON3 terminal transfer. The native source revision is
`f239c6314bc3912c641db6eb909525eaa098f20a` on `human-blood-mass-20260913`.
The shared Mac mini checkout was not modified; the certificate and horizon runs
were performed from isolated source/build state.

## Static certificate

The 1024-sweep source solve is a complete static generalized-force certificate,
not a floating-root-only check. It solves the source pose, mixed 416-route
recruitment, support sharing, position-limit reactions and joint-equality
manifold together:

| Quantity | Result |
| --- | ---: |
| source body mass | `97.1319506911 kg` |
| expected weight | `952.864477038 N` |
| six active support reactions | `952.864475173 N` |
| relative weight error | `1.95707321873e-9` |
| maximum floating-root force residual | `1.86482486697e-6 N` |
| maximum floating-root acceleration residual | `2.25993978541e-5` |
| internal normalized residual RMS | `1.53764252664e-5` |
| active position limits | `30` |
| physical position-limit KKT residual | `7.14934874559e-13` |
| activation routes / nonzero / at upper bound | `416 / 262 / 47` |
| deterministic replay | `bitwise` |

The six loaded witnesses are retained with their individual normal reactions in
the machine-readable receipt. The per-DoF snapshot reconstructs the published
force components with a maximum error of `1.7053025658242404e-13 N`. Its largest
non-root internal closure ratio is `0.0003941310629124404`. One floating-root
row has a near-zero force scale, so its ratio is reported separately rather than
being used to hide the absolute reconstruction result. The raw transcript,
force snapshot and force ledger are retained under
[`Docs/media/native-stand-equilibrium-audit-20260914/`](media/native-stand-equilibrium-audit-20260914/).

This is the first evidence in this path that every published generalized row is
statically balanced at the source pose. It still does not establish that the
same state is a valid initial condition for the dynamic constraint, contact and
fibre owners.

## Dynamic release

The exact-clock native release starts from that static state with no root
assistance. The first step already creates a measurable release velocity:

| Horizon | Peak acceleration | Max velocity change | Max configuration change |
| --- | ---: | ---: | ---: |
| 1 × `12.5 µs` | `252.783828735 m/s²` | `0.00315979775041` | `3.94974719597e-8` |
| 512 × `12.5 µs` (`6.4 ms`) | `114347.445312 m/s²` | `1.40579736233` | `0.00478942599148` |

The 512-step transaction completes and deterministic replay is bitwise, but the
state is temporally divergent. Penetration stays bounded at
`1.78268038553e-7 m`; bounded penetration does not compensate for the exploding
velocity. The runtime sees ten witnesses inside its activation slop while the
static certificate has six loaded reactions. That ownership transition remains
an unresolved dynamic handoff.

Two isolated probes were rejected and are retained as failed evidence:

* Withholding the six static source support loads and giving contact sole
  ownership raises the first-step peak acceleration to `1246.31665039 m/s²`.
* Deadbanding equality target velocity below `1e-7 m` changes the first-step
  result only to `252.78237915 m/s²`; it does not remove the release impulse.

Neither experiment is promoted to the native branch.

## Acceptance boundary and next owner

This audit closes the static source equilibrium and its per-DoF accounting. It
keeps dynamic force convergence, sustained assistance-free standing,
perturbation recovery, walking, anatomical colliders and calibrated contact
materials open. Activation has a source recruitment candidate but no held-out
subject calibration. Blood mass transfer has source and synthetic conservative
owners but no anatomically registered vessel/tissue exchange. Material and
density data and subject calibration remain unresolved.

The next implementation owner is a monolithic dynamic equilibrium handoff that
transports the accepted fibre/tendon state and solves contact, equality,
position-limit and muscle/tendon constraints in the same generalized system.
Until that owner passes a common-duration refinement and a release test, the
512-step run is retained as a failure receipt rather than a standing result.
