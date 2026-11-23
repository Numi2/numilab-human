# Digital extensor registration in the supported reference adult

The NHTENDON3 compiler can register the eight bilateral index, middle, ring and little finger extensor insertions to their individual BodyParts3D distal phalanges. Each site moves to its closest admissible cortical triangle. Four traction nodes carry the reconstructed force and moment. These are inferred insertions in a mixed-source reference assembly. The intermediate guides and thumb keep their authored positions. Use the existing compiler command `numi-human-tendon-envelope-payload --register-digital-extensor-insertions`. The option is independent of foot registration.

On the installed M6, `~/Work/run-numi-human-rest --registered-finger-motion [new-output-directory]` reproduces the 8,000-step supported trajectory. This option is experimental. The current default is preferable for further baseline comparisons.

| Native measurement | Original supported registration | Registered finger insertions |
| --- | ---: | ---: |
| Simulated time | 16.000 s | 16.000 s |
| Total process wall time | 172.37 s | 172.96 s |
| Cold real-time factor | 0.09282× | 0.09251× |
| Warm factor over the instrumented body window | 0.10012× | 0.09970× |
| Nonocular skin-to-tissue triangle pairs at 7.998 s | 11,502 | 11,508 |
| Nonocular pairs at 16.000 s | 11,543 | 11,548 |
| Complete breaths / filling–ejection cycles | 3 / 19 | 3 / 19 |

The registered run had pleural pressure from −770.01 to −470.34 Pa, a 528.82 mL tidal volume and a 69.29 mL final left-ventricular stroke volume. Over the last second the mean bed reaction was 952.49 N against 952.86 N of weight and peak center-of-mass speed was 0.0463 mm/s. The reduced model uses local skin/fat bed sections, articulated force transfer and variable external thoracic pressure. Certification of its local tissue strains and native presentation cadence remains open.

In the full-volume scene the same route payload accepted 16 steps at 15.625 µs. The 0.250 ms release took 324.54 s end to end. Its last bed reaction was 0.00184 N. The endpoint had 10,764 skin-to-tissue pairs and 29 crossings between material boundary triangles. No tetrahedron inverted in that run. One fat cell failed the volume floor, five failed the shape floor and three exceeded the stretch envelope. Some criteria overlap. These two native configurations exercise different skin owners.

## Source of the hand error

In the supported configuration both index and little finger insertions were outside the skin before step one by 3.36 to 10.13 mm. The same skin material triangles moved at most 0.67 mm relative to their phalanges over 16 seconds. Two 32-step controls repeated the opening measurement after matching the arm angles and then also matching the authored NHSKIN payload used by the full-volume scene. All four gaps persisted. At the full-volume endpoint all eight registered terminal sites were inside the physical outer boundary and within 0.070 µm of the recorded bone surface. Other extensor guides still protruded.

Naively transporting the complete registered material through the supported skeleton with its single anatomical donor per node enclosed all eight insertions at release but inverted 2,089 tetrahedra. At the terminal pose it inverted 23,314. This kinematic experiment has no force claim. The supported model needs a common load-bearing boundary with continuous material transfer across donor regions. The existing physical bed remains the reference support. Skin-only offsets, rigid donor transport and changes to tendon forces are inadequate substitutes for this coupling.

## Verification and limits

All eight inferred attachment positions and traction maps independently recompiled identically from the regional cortical surfaces. The comparison source had one face renumbered with identical vertex coordinates. The applied payload replaced those eight records in the exact loaded parent. The remaining records matched the parent. For 128 unit-force directions per insertion the maximum force residual was 3.26e−8 N per N and moment residual 4.08e−10 Nm per N. The 0°, 5°, 15° and 30° hinge checks retained the extension direction. A moment-arm change up to 0.423 mm is an actual change in mechanics.

The opt-in compiler regression also compiled the reference without the foot migration option, verifying all eight targets and equality of every unaffected endpoint. Nine existing source checks passed. Two historical repaired-knee fixtures require a different paired source. On the available predecessor bone artifact their legacy-failure expectation fails on the unmodified parent and the candidate alike. Both traces are archived. The original full-body source chain includes offloaded rib dependencies. Full historical rebuild equivalence is proven here only for the eight changed records and their cortical faces.

The native launch files pin input artifacts and the binaries. Independent GitHub downloads verified the archives member by member. The compact replay receipts cover release, 7.998 and 16 s. Crossing counts identify intersecting triangle pairs at those instants. They establish an anatomical failure even with uninterrupted numerical acceptance.

[Native measurements](digital-extensor-results.json) · [Recovery receipt](digital-extensor-recovery.json)
