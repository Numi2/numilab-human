# Held-out motion of the same registered native body

The coupled run used the same binary, input hashes, physical timestep, controller and registration as the matched eighth-step run. The horizon was doubled to sixteen physical steps. It advanced 0.250 ms in 214.751 s of body runtime and 362.552 s launch to exit. Body throughput was 1.16414e-06× and process throughput was 6.89556e-07× on the Apple M6. Of 77 trace columns, 0 changed over the matched first eight accepted steps. Physical and physiological clocks matched. The endpoint had 0 completed breaths and 0 completed filling/ejection cycles.

| Tissue | Tetrahedra | Inverted | Volume at or below 1e−18 m³ | Normalized shape below 1e−5 |
| --- | ---: | ---: | ---: | ---: |
| Skin | 705,091 | 0 | 0 | 0 |
| Fat | 743,713 | 0 | 1 | 5 |

The three material boundaries produced 29 intersecting triangle pairs, with 29 within deep fat. See the layer matrix and exact facet witnesses. Release and step eight had zero crossings or failed material cells on this registration. These are sampled endpoints. The external audit counted 11,661 nonocular skin-to-structure pairs, 350 source ocular pairs, 274 display-only pairs and 0 skin-to-bone pairs. The terminal normal bed reaction was 0.000 N. A 16-second supported native qualification remains open.

The comparison below uses indexed triangles from the source muscle asset and the actual accepted eighth-step frame. A source seam is identified only when the coordinates also coincide in the native pose.

| Arm surface | Source self-intersections | Posed self-intersections | Skin intersections |
| --- | ---: | ---: | ---: |
| right brachioradialis | 705 | 0 | 617 |
| left brachioradialis | 0 | 0 | 746 |
| right extensor digitorum | 0 | 359 | 1794 |
| left extensor digitorum | 0 | 358 | 592 |
