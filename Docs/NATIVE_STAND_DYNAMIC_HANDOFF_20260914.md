# Native Human dynamic-handoff requalification — 2026-09-14

This receipt covers the isolated native source branch
numi-human-equilibrium-20260914 at 72be18e9, based on
f239c6314bc3912c641db6eb909525eaa098f20a. The source tree was built in
/private/tmp/numi-human-native-fiber-f239-build and executed on the physical
Apple M4 Pro. The shared Mac mini production checkouts were not modified.

The patch has four evidence-bearing effects:

- runtime support activation uses the same 2 mm near-plane band as the authored
  support owner, while geometry admission keeps the strict 1 µm penetration
  tolerance;
- the initial GPU MyoSim state is equilibrated at the prepared fixed pose until
  fibre-length change is at most 1e-7 m and fibre speed is at most 1e-3 m/s;
- the whole-body support certificate receives the actual requested timestep,
  instead of silently using its 100 µs default;
- the persistent path records a source/static versus runtime generalized
  support force audit.

## Inputs and execution

| Input | SHA-256 |
| --- | --- |
| rigid NHRIGID2 | 6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44 |
| muscle NHMYO2 | 9a988f19a6fd8e5335fb0cf2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05 |
| support NHCNT2 | c7712daf79cd8a589a6d23942a4df84a7da928e5911455ce19078f9b24daaaf4 |
| equalities NHEQ1 | b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a |
| tendon NHTENDON3 | a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555 |

The qualification used --muscle-step-seconds 0.0000125, the one-adult male
157-body package, 416 source routes, 128 velocity DoF, 51 joint equalities,
NHCNT2 supports, NHTENDON3 transfers, and Metal rendering on the same
command path. The exact commands and raw logs are retained under
[media/native-human-dynamic-handoff-20260914/](media/native-human-dynamic-handoff-20260914/).

## Static and ownership results

The timestep-aligned 1024-sweep whole-body certificate passes:

| Quantity | Result |
| --- | ---: |
| expected body weight | 952.864477038 N |
| total support | 952.864475233 N |
| relative weight error | 1.89456448354e-9 |
| active support contacts | 9 / 18 |
| maximum root-force residual | 1.8052625137e-6 N |
| maximum root-acceleration residual | 0.037669472976 m/s² |
| internal normalized residual RMS | 0.0250606490272 |
| active position limits | 30 |
| internal balanced flag | true |
| certificate replay | bitwise |

The fixed-pose GPU fibre handoff converges in two iterations with zero measured
length change on the accepted iteration and maximum fibre speed
1.26010490931e-4 m/s. The source/static versus runtime support generalized
force difference is exactly 0 N; the CPU/Metal MyoSim generalized-force
difference is 0.00760018900592 N.

These results close the static support ownership and initial fibre-state
handoff checks. They do not establish dynamic standing.

## Dynamic release

With the compiled recruitment cap at 1.0, root assistance enabled only for
the historical assisted/removal transaction, and the final no-assistance
state published:

| Horizon | Peak acceleration | Maximum velocity change | Maximum configuration change |
| --- | ---: | ---: | ---: |
| first 12.5 µs release | 214.411651611 m/s² | 0.00536026852205 at DoF 105 | 1.00505175737e-7 at q 106 |
| 512 × 12.5 µs, deterministic replay (two 512-step passes) | 149823.15625 m/s² | 1.77973592281 at DoF 105 | 0.0136960484087 at q 106 |

The 512-step run reports persistent_completed_steps=1024,
persistent_max_penetration_m=1.8062577567e-7, and
stand_deterministic_replay=bitwise. The bounded penetration and replay do not
make the horizon a standing result: the velocity and configuration release
are divergent.

A cap-0.5 negative control is retained. It produces an internal residual RMS
of 31.7160749969 and a first-step peak of 539.62097168 m/s², so reducing
activation does not repair the handoff. The remaining failure is the
monolithic coupling of muscle/tendon force, contact, equality, and position
limit rows after the static solve; no tolerance or timestep claim is promoted
over this evidence.

## Completion boundary

This receipt proves:

- the canonical 12.5 µs clock reaches the native source owner;
- support loading carries body weight and has exact source/runtime generalized
  force parity;
- the complete static per-DoF source solve is internally balanced;
- the accepted fibre/tendon state is equilibrated before the first release.

It leaves force convergence, assistance-free sustained standing, perturbation
recovery, walking, 420-trial behavioral qualification, calibrated activation,
anatomical tissue loading, blood-to-tissue transfer, unresolved material and
density data, and subject calibration open. The next implementation owner is a
single sparse or matrix-free KKT/Schur solve that admits the active contact,
equality, position-limit, and muscle/tendon rows together; the serial projection
experiments are retained as failed diagnostics and are not part of 72be18e9.

### Raw artifacts

- [one-step-cap1.log](media/native-human-dynamic-handoff-20260914/one-step-cap1.log) — SHA-256 011477f06f3fc32f9ba18b69a47ee963572136f388cd98dbd4e087a7886df763
- [support-certificate.log](media/native-human-dynamic-handoff-20260914/support-certificate.log) — SHA-256 f5bf1cdc5e61c9b46f14da535d47ce95bbd4389c753671d9eb26ce70f443424c
- [horizon-512-replay.log](media/native-human-dynamic-handoff-20260914/horizon-512-replay.log) — SHA-256 05750ecd15d57e194f14da535d47ce95bbd4389c753671d9eb26ce70f443424c
