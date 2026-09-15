# Native dynamic force-component audit — 15 September 2026

The physical Mac mini M4 Pro reran the repaired one-adult-male Human source at the canonical `12.5 us` clock for 64 persistent Metal steps (`0.8 ms`), with the accepted stationary MyoSim fibre/tendon root, ten source support witnesses, 40 source passive coordinate couplings, and no root assistance. The native source is `0967c4561392a04be6c47d158ec9bd1e4daa2009`; the audited binary SHA-256 is `74aac5cc43e0e2a19016ab5e6d5e61f042f013d457ae32b494398a85426f886b`.

The new `persistent_dynamic_force_audit` record exposes all 128 velocity coordinates. Each row contains the accepted Metal muscle wrench, source support wrench, joint-equality reaction, position-limit reaction, passive-coordinate reaction, gravity target, and the reconstructed initial dynamic residual. The largest residual is `0.03216604835060366 N` at DOF 118 (joint 146, child body 147); the next largest are DOF 104 at `0.027029216809921763 N` and DOF 103 at `0.018328018158972448 N`. The maximum source CPU/Metal muscle-force parity delta is `0.0321654636734 N`, so the residual is explained by the source-to-device muscle-force handoff rather than by a missing support or gravity term.

The same run reports `persistent_max_acceleration=0.115904301405 m/s²`, zero penetration, six active support contacts, `952.864475301 N` compiled support against `952.864477038 N` expected weight, `6.42342632457e-6` static normalized residual RMS, and bitwise deterministic replay. The audit is therefore a diagnostic closure of the dynamic component decomposition, not a standing result.

The attempted frozen static-to-dynamic handoff correction is retained as [`rejected-handoff-experiment.json`](media/native-dynamic-force-audit-20260915/rejected-handoff-experiment.json). It applied the CPU-minus-Metal initial muscle-force difference as a preload and worsened the same 12.5 µs 512-step peak from the repaired baseline `4.82679176331 m/s²` to `8.46329212189 m/s²`; the source patch was reverted and is not part of the accepted branch.

The full machine-readable result is [`receipt-v1.json`](media/native-dynamic-force-audit-20260915/receipt-v1.json), the sorted 128-row component table is [`dynamic-force-audit.json`](media/native-dynamic-force-audit-20260915/dynamic-force-audit.json), the exact native source commit is preserved as [`native-source.patch`](media/native-dynamic-force-audit-20260915/native-source.patch), and the raw native output is [`native.stdout.log`](media/native-dynamic-force-audit-20260915/native.stdout.log). Force convergence, sustained standing/recovery/walking, anatomical support/loading, activation calibration, blood mass transfer, calibrated materials, fat ownership, and subject calibration remain open.

Top residual rows:

| DOF | Joint | Child body | Metal muscle (N) | Support (N) | Equality (N) | Gravity target (N) | Residual (N) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 118 | 146 | 147 | 903.04150390625 | 459.392285096379 | -1313.88912096012 | 48.5768340908608 | -0.0321660483506037 |
| 104 | 132 | 133 | 610.195556640625 | 483.621837494936 | -1045.26759349819 | 48.5768298541777 | -0.0270292168099218 |
| 103 | 131 | 132 | 396.037261962891 | 60.972800775569 | -450.904049092583 | 6.12434166403518 | -0.0183280181589724 |
| 10 | 5 | 6 | -34.0355606079102 | 0 | 34.1183887657373 | 0.0981000020034611 | -0.0152718441763381 |
| 117 | 145 | 146 | 426.277557373047 | 57.9180510259649 | -478.086107264314 | 6.12434219474116 | -0.0148410600433762 |
| 6 | 1 | 2 | 3.84993481636047 | 0 | -3.86123827113562 | -0.00248997272568284 | -0.00881348204946344 |
| 9 | 4 | 5 | -180.83039855957 | 0 | 180.837638640621 | -8.03082206122077e-34 | 0.00724008105103735 |
| 12 | 7 | 8 | 13.8599987030029 | 0 | -19.2545054261408 | -5.38732097423098 | -0.00718574890691759 |

