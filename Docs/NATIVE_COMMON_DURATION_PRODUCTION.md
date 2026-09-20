# Physical Human common-duration production diagnostic

Status: `diagnostic_complete`

## Decision

The physical trace-basis subgate is complete for the captured NHCNT1 Human
workload.
All four clean-production cases ran for the same 6.4 ms simulated duration on
one physical M4 Pro Mac mini, used one native tree, binary, input commit, and
payload set, and reproduced their terminal state bitwise. This qualifies the
captured comparison surface. It does **not** qualify state or force convergence,
performance, sustained standing, recovery, walking, or whole-Human behavior.

The payload-bound [prepared support-history contract](NATIVE_PREPARED_SUPPORT_HISTORY.md)
now exists and passes source authoring, byte-exact round-trip, provenance
rejection, and the analytic physical-M4 Matter transaction. This diagnostic's
native revision and NHCNT1 payload differ from the NHINIT3 evidence revision and
NHCNT2 payload, and the production runner does not yet admit NHINIT3. The matrix
therefore remains historical trace-basis evidence, not the direct seeded
comparator.

## Exact evidence identity

- Native commit: `4e0e817a72222bccecafc51a579fe951f4649b66`
- Native tree: `112fea28802ba1e7dd54b0eb1c3c84b3f89ce552`
- Input commit: `f53c2e1053bc90e7ec44fd69a5b49f96b71b71ab`
- Binary SHA-256:
  `207226d814968e6c673243de4a22efaaf9c7f91c0736c9582e0653b7c5603687`
- Support payload: NHCNT1, 10 records, SHA-256
  `4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907`
- Physical machine receipt: Mac mini `Mac16,11`, Apple M4 Pro, 24 GB,
  identity SHA-256
  `dedf8c6a84734c3ddeea7d687877352f8ab55641d7d70f17b23f051e60dd9c5a`
- Runtime: production cap 8, 64 contact iterations, no diagnostic segment
  override, Metal debug disabled, persistent trace v5 enabled, deterministic
  replay enabled.

The cap-8 default and this runner live on the published
`human-native-runtime-20260915` integration branch and are absent from the
native repository's default line; no default-line receipt currently binds their
implementation and evidence to one resulting commit and tree.

## Physical result

| Grid    | Steps | Wall time (s) | Peak RSS (KiB) | Mean normal reaction (N) | Maximum published delta-v | Initial persistent force-reference residual (N) |
| ------- | ----: | ------------: | -------------: | -----------------------: | ------------------------: | ----------------------------------------------: |
| 100 us  |    64 |       147.712 |         52,976 |                 953.7471 |                0.00294141 |                                     0.372853463 |
| 50 us   |   128 |       243.787 |         51,024 |                 953.2543 |                0.00147134 |                                     0.372853463 |
| 25 us   |   256 |       440.032 |         50,640 |                 953.0133 |               0.000735831 |                                     0.372853463 |
| 12.5 us |   512 |       832.541 |         51,232 |                 952.9009 |               0.000367957 |                                     0.372853463 |

Every case completed without timeout, validation error, or non-banner stderr.
Swap did not grow, memory pressure remained nominal, and `pmset` reported no
thermal or performance warning. These observations make the traces usable; a
single run per grid is not a performance qualification.

Against the 12.5 us reference, the maximum same-time scalar configuration
delta decreases from `2.91095e-5` to `1.25801e-5` to `4.20958e-6`, the maximum
generalized-velocity delta decreases from `4.61996e-3` to `1.98301e-3` to
`6.61701e-4`, and RMS normal-reaction delta decreases from `4.03534 N` to
`2.27666 N` to `0.912818 N` as the compared grid is refined. The maximum
normal-reaction delta remains `7.27890 N` even for 25 us versus 12.5 us, and
the maximum-reaction owner differs at two common timestamps for 100 us and one
for 50 us.

The contact and source-limit post-projection residual maxima approximately
halve with the timestep, while the tendon force residual remains near
`8.665e-5 N` on the three finer grids. The reported maximum initial persistent
force-reference residual is exactly `0.372853462949 N` at every grid because
it is the same initial-state diagnostic. It identifies an unchanged launch
defect, but cannot by itself test trajectory-wide force convergence. The
four-grid evidence therefore does not close force convergence.

The current constraint-work totals also cannot be promoted as physical-energy
closure: signed and absolute equality work scale approximately with the
timestep across this fixed-duration launch, and the trace does not own the
final exact-coordinate overwrite or target-relative dissipation. No acceptance
tolerance is inferred from these four observations.

## Next evidence-producing action

The exact native-v8 construction and inbound request-v3 subgates are now passed
and retained in
[NumiBrain evidence commit `c64449c`](https://github.com/Numi2/numi-brain/tree/c64449c50793da1ba11ab716dcc9c550e53a2eed/evidence/numanx-exact-request-v3-contract-v0.1).
The physical-M4 probe validates the exact request, rejects mixed v1 records, and
stops at stage 900 before resource import or GPU submission. It does not yet
cross HumanMatter close or the accepted publication/ACK boundary.

1. Extend the now-qualified exact request-v3 motor boundary with exact
   HumanMatter-close, outbound-sensor, accepted-publication, snapshot,
   persistent-state, witness, and ACK owners. Keep stage 900 fail-closed until
   the complete family exists, then qualify the genuine accepted-root sequence.
   Do not add decoder-only support to the legacy standalone stand shader: it has
   no accepted support-history arena and would discard the defining NHINIT3
   continuation state.
2. Add permanent hashed NHTENDON runtime input and a quiescent copied
   post-publication snapshot of Matter's raw
   accepted support histories. Require real witness, preflight, ACK, generation
   latch, rollback and replay before exposing the path as production stand.
3. Admit the full v8 chain through one fail-closed manifest/receipt, then freeze
   the common native commit/tree, runner, runtime library, metallib,
   rigid, muscle, NHEQ2, NHLIM1, and NHCNT2 identities. For each
   100/50/25/12.5 us grid, author and qualify an exact-clock tuple of
   Matter-package SHA, world fingerprint, NHINIT3 SHA, and initial-state
   fingerprint.
4. Through `ssh macmini`, after confirming the host is uncontended, run matched
   cold-versus-seeded cases on that identical stack from isolated worktrees.
   Retain all results and compare
   same-time q/v, first-interval and trajectory support impulses,
   reaction owners, equality/limit impulses, tendon residuals, dynamic force
   residual, signed/absolute work, rollback, and replay. The contract passes
   as state continuity only if the exact authored history reaches the accepted
   Matter owner with unchanged hard mechanics invariants. Force convergence
   remains a separate uncertainty-backed gate.
5. If the seeded launch leaves the defect unchanged or worsens it, retain that
   causal negative and capture the paired source-limit/equality event plus a
   selected-step full owner snapshot next. Do not tune the seed to the output.

This order is dependency-driven, not calendar-driven. Data admission,
anatomical plantar-contact authoring, controller candidates, and conservative
physiology interfaces can proceed concurrently behind their existing permanent
contracts, but none can bypass the mechanics evidence gate.

## Immutable artifacts

- [Run manifest](media/native-common-duration-production-v1/run-manifest.json)
- [Comparator v3 report](media/native-common-duration-production-v1/native-trace-refinement-v3.json)
- [SHA-256 manifest](media/native-common-duration-production-v1/SHA256SUMS)
- Per-grid case summaries, raw stdout/stderr, and retained frame packages under
  `Docs/media/native-common-duration-production-v1/`.

The comparator report SHA-256 is
`33c4e978253e4f3ba7dee5dc15f29f165fdf9f740de1487eb9f6ccedff1ce28e`.
