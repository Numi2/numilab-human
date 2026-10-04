# Dual-quaternion skin deformation screen — 2026-10-04

I tested normalized dual-quaternion blending (DQ) as a replacement deformation
formulation for the retained ABI 5 source-weight linear-blend-skinned (LBS)
visual shell. The screen used the exact same 86-body weights, source vertices,
triangle indices and body transforms at bilateral knee angles 0.4, 0.8 and
1.2 rad. It was preregistered and run on CPU only. Before scoring each DQ
surface, independently reconstructed LBS positions matched the corresponding
retained native M4 pack within 0.00143 mm, below the 0.002 mm parity gate.

| Bilateral knee pose | Native LBS pairs | DQ pairs | New DQ pairs |
| --- | ---: | ---: | ---: |
| 0.4 rad | 0 | 0 | 0 |
| 0.8 rad | 0 | 14 | 14 |
| 1.2 rad | 75 | 116 | 116 |

At 1.2 rad, DQ removed the 75 LBS pair identities and introduced 116 other
pairs. The registered prediction of no new exact intersections therefore
failed at both 0.8 and 1.2 rad. DQ is not adopted, and the native renderer is
unchanged. This rejects DQ as a direct substitute for this source-weight
shell; it does not establish which tissue mechanics or anatomy should replace
it.

The source outer shell remains open with 171 boundary edges and two
vertex-link defects. The result is limited to three kinematic snapshots and
does not qualify continuous motion, skin contact, physical material behavior,
clinical anatomy or whole-Human capability. The preregistration, row receipts,
source hashes, exact-pair sets and code/test identity are retained in
[`skin-dual-quaternion-20261004`](media/skin-dual-quaternion-20261004/).
