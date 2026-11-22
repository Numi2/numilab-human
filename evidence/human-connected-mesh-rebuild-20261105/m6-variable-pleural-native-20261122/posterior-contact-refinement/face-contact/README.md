# Registered bed contact on Apple M6

The current dense scene accumulates bed load at outer material nodes. Splitting the same posterior triangles and attached tetrahedra reduced its initial reaction from 953.18 N to 461.24 N at identical body placement. A balance obtained with that discrete rule depends strongly on triangulation.

The new Metal kernel integrates Matter's stationary plane barrier on the part of each registered skin triangle within contact range. It distributes the normal reaction to the three physical material nodes using the same shape functions as the force derivative. An active face uses one SIMD group and emits a symmetric local stiffness matrix for the solver. The material-area coefficient is inherited from the original source boundary. The geometry may be refined without changing that physical coefficient. The measurements cover a fixed plane, normal contact and stationary barrier stiffness.

The M6 experiment used the archived registered body, the captured two-step geometry and four conforming skin/fat meshes with 531 through 24,023 local surface faces. Matched normal-load calculations checked every omitted outer face stayed outside the active band throughout the trial displacement.

| Measurement at identical geometry | Original surface | 11,746 added independent nodes |
| --- | ---: | ---: |
| Initial force, area integration | 311.33196 N | 311.33187 N |
| Accepted-coordinate force, area integration | 746.56610 N | 746.56599 N |
| Initial force, current nodal rule | 953.17724 N | 461.24 N |
| Accepted-coordinate force, current nodal rule | 2,245.76072 N | 1,152.49 N |

At the independently accounted weight of 952.86444 N, the coarse-to-finest initial center of pressure moved 0.000027 mm. The source-node reaction vector changed by 0.000080%. At the captured coordinates the changes were 0.0000021 mm and 0.000044%. The largest relative error between the M6 and double-precision force or stiffness action across all sixteen cases was 7.36e−7. The analytic directional stiffness was also checked against force differences. Four unit tests cover partial contact, contact-band transitions, the regularized floor, subdivision, force and moment, and tangent symmetry. A second complete device replay reproduced all physical outputs.

The current dense body's starting pose yields only 311.33 N under this surface law against 952.86 N of body weight. A vertical rigid translation of −0.592 µm balances the resultant on the finest mesh at the initial pose. That scalar balance leaves skeletal forces, moments and internal tissue equilibrium unresolved. The captured positions and rigid translations are prescribed configuration tests. The full motion scene still uses its original contact owner. Integrating this operator requires a single normal-force owner across residuals, tangents, reactions and acceptance, together with the corresponding loaded tissue/skeleton pose. A second support force would double-count the bed.

To reproduce in the archived M6 work directory, build with the stored compiler options and run the registered refinement qualification:

```sh
python bed-face/build_face_operator.py velocity-consistent/identities.json bed-face
python bed-face/verify_face_bed_refinement.py . ../conserved-two ../conserved-quadrature.json bed-face
```

[Device and material measurements](qualified.json) · [Exact build](build.json) · [Unit tests](test.log) · [Independently restored executable, Metal library and source](recovery.json)
