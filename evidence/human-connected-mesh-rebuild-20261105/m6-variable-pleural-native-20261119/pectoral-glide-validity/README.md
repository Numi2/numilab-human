# Pectoral fascia and registered support edge

Native trials ran on SSH Numi (Apple M6, Mac18,5, 16 GiB) at the original 2 ms control step with live variable pleural pressure, NumiBrain and transactional rollback. The trial retained 26 fascia regions, 495 gliding samples, the registered ligaments and the existing bed reaction. The launch used the reduced 32-section bed owner and regional deformable contact. Whole-body skin/fat and organ FEM contact were outside its active configuration.

## Pectoral interface

The original registration applied the gliding foundation to the exterior rather than the muscle-facing side of the pectoral sheet. The trial placed the muscle-facing side at the source envelope and built the exterior one authored thickness outward. It measured load-path depth in the registered donor mesh using the contact normal. The existing foundation modulus was reused with 119 supported depth samples and the existing thickness input at 10 unsupported samples. This is an inferred material registration trial. Source identities and the whole-human mass partition are preserved.

| Configuration | Accepted time | Native failure or outcome |
| --- | ---: | --- |
| Pinned regional-contact control | 4 ms | Pectoral cell 102 at microstep 90, determinant at the 0.35 material floor |
| Re-registered interface | 4 ms | Same cell at microstep 132 |
| Interface and final material predicate during trial selection | 4 ms | Material-floor failure replaced by final coupled equilibrium rejection |
| Add measured donor depth, 16 Newton iterations | 6 ms | Next coupled residual was 0.00102582 |
| Same model and acceptance tolerances, 24 Newton iterations | 8 ms | Next coupled residual was 0.00346204 at root 4, microtick 12 |

The 24-iteration run took 141.24 s from launch through failed-step handling, or 0.0000566×. The accepted GPU command durations totaled 101.81 s. Variable pleural pressure in those four accepted steps ranged from −492.15 to −490.64 Pa. The failed root rolled back while the accepted controller time remained at 8 ms. The regular native compiler FEM reference test passed.

A capture immediately after the final equilibrium certificate recovered the shared physical residual and a minimum element determinant of 0.47931 across its object reports. The trace before that certificate showed several deformation corrections restricted first by FEM validity and then further by deformable collision. For the first three captured iterations the fractions changed from 0.25, 0.0625 and 0.0625 to 0.15365, 0.04724 and 0.00850. The circulation and per-object copies of the fraction agreed. A single-writer shader counterfactual also reproduced the 6 ms rollback. The equilibrium certificate diagnostics and the fraction-mismatch diagnostics use the same status code and have overlapping layouts. The recovered certificate identifies the actual rejection stage.

The matched tiled contact limiter kept the accepted physiology trace byte-identical and rejected at the same residual. Accepted GPU command time increased from 101.81 s to 102.08 s. The observed end-to-end time changed from 141.24 s to 140.18 s, a single-run difference that does not establish a throughput improvement. The tiled witness identifies active contact slots 381, 377 and 376 on runtime objects 13 and 11 as the leading limits in the first four iterations.

## Crossings

The earlier 16-second matched capture retained 14,207 exact skin-to-tissue triangle pairs in 217 connected curves, and 45 skin self-intersection pairs in 13 curves. Its terminal accepted pack SHA-256 is `aa013b6d4c05a4504c08b03ef5ed91651fec68dff5ec2bbdad31d0e0445185d6`. The current native trial never reached that pose.

The remaining protected witness ties skin face 14040, bed edge 6889–7240, and tissue bank 51005/121 triangle 1476. The edge is 9.58 mm long at the late pose and both endpoints are active source support anchors. A muscle-only repair that removed the original triangle pair relocated the exact cut to adjacent triangles. Bounded segment–triangle fits either left a cut or failed exact finite-shape checks. The skin source, actual support anchors and crossing evidence remain pinned.

Compact traces, launch pins, source overlays and rejected geometry proposals are included here. The artifacts record preparation, geometry screens and short native attempts. They do not qualify a production model, full contact coverage, zero crossings, or a 16-second run at 0.1×.
