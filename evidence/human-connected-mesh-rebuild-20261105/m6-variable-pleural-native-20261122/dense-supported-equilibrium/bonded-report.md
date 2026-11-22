# Why the dense human still fails supported rest

The registered dermis and fat now clear the native cell-shape and orientation checks at the independently captured two-step endpoint. The full scene still rejects its third step. The attempted repair improved the nonlinear residual by 26.1% without gaining a physical step. Keep the dense model experimental.

| Same M6 release, with variable pleural coupling enabled | Paired dermal reference | Bonded dermis/fat fit | Source-moment-conserving fit |
|---|---:|---:|---:|
| Accepted time before the next rejection | 31.25 µs | 31.25 µs | 31.25 µs |
| Residual on rejected step (gate 0.00001) | 0.049171 | 0.040739 | 0.036325 |
| Full-correction material tiles reporting a violation | 5 | 4 | 2 |
| Dermis / fat cells below the native shape floor at the accepted endpoint | 0 / 1 | 0 / 0 | 0 / 0 |
| Nonocular skin/anatomy triangle pairs | 10,895 | 10,895 | 10,895 |

The final fit freed 68 coordinates in a connected neighborhood of 1,078 cells. The largest move was 35.2 µm. The registered outer skin, deep-fat boundary, fourteen attachments, material identities and source-mass routes were preserved. The largest regional first-moment change was 7.34e−11 kg m. The six prior cavity replacements passed the native mass and first-moment checks after this move. All 1,448,799 cells passed initial geometry, stretch and stress admission. The replay used the existing nonlinear, contact and vascular certificates. The rejected proposal left accepted common coordinates and respiratory status unchanged.

At the accepted endpoint, no inverted cells, skin self-intersections, crossings between skin/fat layers, or skin/bone pairs were found. The full audit checked 878 anatomical banks. Thirty-seven dermis cells and sixteen fat cells left the registration stretch envelope. Three dermis cells exceeded the threefold distortion criterion. The final rejected line-search trial reached the fat compression bound at a volume ratio of 0.35. Its retained source cell is 1032163. Circumventing that bound would admit a different mechanical state.

## Where the bed load goes

The initial bed reaction was 953.18 N against 952.86 N of body weight. After two steps the native reaction was 2,252.30 N. The articulated owner received 17.85 N vertically from FEM against 811.45 N of articulated weight. These forces describe a transient. They do not establish that the body is settled or that the skeleton would receive this same force in steady state.

Four outer skin coordinates carried 66.5% of the independently replayed endpoint elastic reaction. At torso vertex 23186 the calculated reaction went from 12.39 to 849.62 N. Its six incident triangles have an edge as long as 109.25 mm. Of the 58.06 cm² of projected triangle area inside the plane's numerical contact band, 40.15 cm² is incident on that one vertex. The largest face contributes 21.41 cm². Area is summed on the actual piecewise-linear surface and can include overlapping projections.

An offline matched sensitivity test redistributed the existing barrier coefficients over the reference skin triangles and integrated the same gap law on their clipped areas. The installed vertex rule gave 953.18 N at initialization and 2,245.76 N at the endpoint. Face integration gave 311.33 N and 746.57 N respectively. Increasing the quadrature order from four to eight changed individual nodal forces by less than 0.000001 N. The test reproduces uniform-gap forces and conserves the resultant and its moment under subdivision. Both measurements use the same material coordinates, density and plane. They show dependence on contact discretization. They do not establish which distribution a physical bed would produce. Face integration was an offline test and was never applied as an additional force.

The earlier captured-displacement experiment also provides a limit on interior-only repairs. With the same boundary motion, refining an implicated three-cell fat pocket from 3 to 192 to 1,536 tetrahedra raised its peak volume ratio and changed the last two boundary reactions by 12.96%. That offline unconstrained equilibrium exceeded the source material's admissible domain. It cannot justify promotion of that refinement.

## Decision

Retain the connected material and source-moment fixes. Stop selecting local coordinate changes by how far they advance the third step. The next mechanical release should solve the loaded posterior torso and pelvis together with the articulated body under gravity and actual bed contact. It needs to reproduce the force and moment carried by the same registered patches when the contact surface is refined, and show the tissue-to-skeleton reaction and acceleration balance. A changed bed rule would require its matching consistent tangent and a new loaded equilibrium. The current initial coordinates were prepared for the vertex rule. Swapping the force rule under that pose would remove approximately 642 N of initial support. The long-run and throughput gates apply after this equilibrium advances under the existing acceptance checks.

The latest dense horizon took 54.54 s for 31.25 µs (5.73e−7×). Cold execution took 203.08 s (1.54e−7×). Pleural pressure was −490.33 Pa at both retained times. The separate compact 16 s run completed three breaths and nineteen cardiac cycles. Its 0.09678× cold throughput and 13,913 nonocular skin/anatomy pairs belong to that compact configuration. Dense motion for 16 seconds, zero whole-body crossings, shoulder ligament loading through motion and end-to-end 0.1× remain unqualified.

The evidence folder includes the matched response, input identities, tests and replay command. The restoration manifest identifies the exact native executables, material maps, GPU receipts, accepted frames, rejected proposals, offline alternatives and earlier source archive.
