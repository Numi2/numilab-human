# Physical Human common-duration production diagnostic

Status: `diagnostic_complete`

## Decision

The physical trace-basis subgate is complete for the current Human workload.
All four clean-production cases ran for the same 6.4 ms simulated duration on
one physical M4 Pro Mac mini, used one native tree, binary, input commit, and
payload set, and reproduced their terminal state bitwise. This qualifies the
captured comparison surface. It does **not** qualify state or force convergence,
performance, sustained standing, recovery, walking, or whole-Human behavior.

The next causal mechanics slice is prepared support-history continuity. The
static solve already owns support impulses, but the dynamic Matter transaction
starts its accepted support histories at zero. Preserve that state through a
payload-bound initial-state contract, then compare cold and seeded launches on
the same four-grid protocol. A negative result is useful: it retires the
state-loss hypothesis and promotes the paired source-limit/equality event
snapshot as the next diagnostic.

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
`human-native-runtime-20260915` integration branch. They are not yet
reconciled onto the native repository's default line.

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

1. Add an `NHINIT3` prepared-support-history record bound to the exact NHCNT
   byte image, while preserving NHINIT1/2 byte compatibility.
2. Initialize Matter's accepted support-history owner directly; candidate and
   checkpoint state must inherit it through the existing transaction. Keep the
   standalone stand-contact reserved word zero.
3. Prove malformed identity/count/row-order rejection, failure atomicity,
   rollback, commit, and bitwise replay in CPU and native probes.
4. On an uncontended Mac mini, run matched cold-versus-seeded
   100/50/25/12.5 us cases from isolated worktrees. Retain all results.
5. Compare same-time q/v, first-interval and trajectory support impulses,
   reaction owners, equality/limit impulses, tendon residuals, dynamic force
   residual, signed/absolute work, rollback, and replay. The contract passes
   as state continuity only if the exact authored history reaches the accepted
   Matter owner with unchanged hard mechanics invariants. Force convergence
   remains a separate uncertainty-backed gate.
6. If the seeded launch leaves the defect unchanged or worsens it, retain that
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
