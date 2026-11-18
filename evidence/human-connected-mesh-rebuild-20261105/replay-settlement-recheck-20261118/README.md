# Full-body reference replay and settling probe

Verified on SSH Numi (Apple M6) on 18 November 2026. The full Human remains unqualified.

| Check | Result |
| --- | --- |
| Matched mechanics | Geometry 495 accepted four 15.625 µs steps. All 776 captured force metrics, 282 strain metrics and 77 coupled trace channels matched the retained 565 control exactly. Candidate 645 remains rejected with 102 force and 139 strain failures against that reference. |
| Performance | 62.5 µs of simulated time in 73.14 s of profiled body runtime, or 8.55e-7×. On root 1, linear solves consumed 21.30 s of the 25.58 s covered by GPU stage timers. The 544 Krylov columns on that root used the full restart width. This release transient gives neither sustained throughput nor an optimization gain. |
| Supported motion | The FEM bed observer recorded eight accepted roots and the settlement gate failed. At 125 µs the bed reaction was 0.000994 N against 952.86 N of weight, with max FEM speed 3.385 m/s. The matched control first-step ankle velocity changes imply −581 and −572 rad/s² before the last velocity projection. |
| Coupled physiology owner | The reduced component preserved circulation, gases and Brain history through a forced rejection and replayed acceptance bitwise. In 12 s it completed two breaths and 14 ventricular filling/ejection cycles. A 0.6 drive multiplier from 4–8 s reduced inspired volume by 26.9%, raised arterial CO₂ and lowered arterial O₂. The 4 s follow-up did not establish recovery. |

The settlement patch reads the accepted root impulse on the existing fixed Matter bed, validates that the articulated support rows remain absent, and measures gaps from physical outer skin nodes. It changes the observer and admission to the diagnostic. Geometry, loads, stiffness, contact, timestep and settlement thresholds stayed fixed. The tested native build came from the preserved dirty checkpoint. The guarded patch passes resting-target syntax and retains the baseline non-resting build errors of that checkout.

The full-body run used the restored exact articulated Metal library and 17 SHA-verified inputs from the [checkpoint archive](https://github.com/Numi2/numilab-human/releases/tag/numi-human-20261105-intermediate-recovery). Replays pin the mass receipt dependencies as well as the direct inputs. Plans and raw receipts are under `/Users/numi/Work/human-integrated-gaps-20261104` on SSH Numi in `resume-fidelity-and-profile-02`, `resume-fem-bed-settlement-probe-03` and `resume-physiology-transaction`.

At the unchanged timestep, 0.1× allows 156.25 µs of wall time per root. A model and solver that reduce the cost of active linear work and provide a physically equilibrated starting state are still needed. The native viewer, skin/fat through sustained motion, anatomical ejection through repeated beats, and the integrated baseline/intervention remain open.
