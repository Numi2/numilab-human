# Preparing bonded skin and fat for supported motion

The fitted start minimizes the existing material energy under gravity over connected stars containing the measured bed and strain witnesses. Only the initial coordinates of free interior and bonded interface nodes can move. Both exterior boundaries and fourteen elastic attachment locations retain their coordinates. The reference mesh, tissue masses and material programs are identical. Regular forward mechanics and normal contact resume at release.

Across 3,417 affected cells, the fit moved 208 physical nodes by up to 0.105451 mm. The free-node force norm fell from 1126.902 N to 1064.358 N. Peak stress in the affected stars changed from 395878.9 Pa to 415888.7 Pa. All 1,448,799 cells passed preparation checks. The three physical surfaces had zero inter- and self-crossings at initialization.

First moments in the deformed material were independently recomputed for 3,072 affected source groups. The maximum change of an advected group moment was 2.173e-07 kg m and the largest transport-accounting error was 1.153e-18 kg m. The eleven anatomical donor/material sums changed by at most 7.995e-11 kg m.

| Full native release | Accepted steps | Accepted time, µs | Wall seconds, including preparation | Rejection diagnostics |
|---|---:|---:|---:|---|
| Previous starting state | 2 | 31.25000 | 254.73 | [0.4965241253376007, 0.4965241253376007, 1035269, 2] |
| Fitted start | 0 | 0.00000 | 187.99 | [9.926970356843867e-11, 9.536742923144104e-12, 5.7366869441466406e-05, -912] |
| Fitted start, contact-aware Newton stopping | 0 | 0.00000 | 186.47 | [8.949104994826484e-06, 0.07546594738960266, 1, 0] |
| Fitted start, matched preflight and Newton stopping | 2 | 31.25000 | 255.47 | [0.4965241849422455, 0.4965241849422455, 1035269, 2] |

The first fitted release satisfied the mechanical residual criterion while a compressive impulse remained on an out-of-tolerance separated bed row. The fresh certificate rejected the step. The revised GPU solver checks that same gap and friction predicate while deciding Newton convergence. The independent final contact and loaded-volume certificates remain active. The support gate shares the existing single-environment reduction and certificate scratch word, with explicit ordering. The first solver change exposed a discrepancy between the preconditioner preflight and the actual solve. The final solver applies the identical condition at both points. Its trace samples root zero, while the earlier trials sampled root two.

The accepted-2 endpoint reached 3.12500015e-05 s at 1.43189e-07× end to end. The bed exerted 2161.577 N, and the upward tissue load on the skeleton was 17.870 N. The exterior had 10,895 skin/tissue triangle crossings and 0 skin self-crossings. The skin/fat layer count was 0. Minimum current volume relative to the previous and fitted release was 0.4561823 and 0.4561823, respectively.

The matched two-step release reduced the bed reaction from 2,330.616 N to 2,161.577 N. Upward tissue force on the skeleton changed from 17.857 N to 17.870 N. The fit retained 10,895 exterior skin/tissue crossings. Thirty-seven skin and twenty-two fat cells exceeded the preparation stretch envelope in motion, including one severe skin stretch. Its third attempted step reached the original fat cell again. All four corners of that cell were excluded from preparation by the earlier cavity protection. The smallest material-reference altitude is 21.45 µm. The fitted geometry remains an experimental start.

The reference bed weight is 952.864 N and the skeleton weighs 811.448 N. A complete supported release requires sustained force and motion balance with physiological loading, clear anatomy, and integrated timing. The 16-second, longer intervention and 1× real-time goals remain open. See decision.md for the prediction and matched-time criteria. All measurements used SSH Numi’s Apple M6. The recovery manifest identifies the independently downloaded binaries, physical coordinates, launch commands and accepted-state evidence.
