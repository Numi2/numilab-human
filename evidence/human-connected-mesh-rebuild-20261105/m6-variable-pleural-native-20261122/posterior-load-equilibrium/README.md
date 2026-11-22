# Posterior bed-load initialization on Apple M6

The reference is the source-identical pelvic-contact scene at c8c8c689. The experimental initialization reduced the first accepted bed reaction from 4,691.21 N to 1,588.94 N for a 952.86 N body. It missed the second-step prediction. The original scene remains the qualified comparison input.

The authoring solve uses the compiled dermal and adipose strain-energy expressions, reference tetrahedral volumes, transferred P1 masses and gravity. It fits initial coordinates; the native Metal runtime advances every physical step. The outer surface, 143 initial bed gaps, 14 skeletal insertion coordinates, rest mesh and mass-transfer bytes are identical. There are no additional runtime forces. Geometrically independent vertex groups can take different descent fractions. A global-fraction control reduced the measured posterior force norm by 0.42%. The locally constrained fit reduced it from 2,593.32 to 1,247.93 N. The finite-difference gradient and tangent checks agreed. Five witnessed deep-interface vertices were held at their original coordinates during initialization to eliminate two introduced starting folds.

| Measurement | Original scene | Fitted initialization |
| --- | ---: | ---: |
| Starting stationary bed reaction | 952.86 N | 952.86 N |
| First accepted native bed reaction | 4,691.21 N | 1,588.94 N |
| First accepted time | 15.625 µs | 15.625 µs |
| Bed support points at that endpoint | 130 | 123 |
| Skin/fat boundary crossings | 0 | 0 |
| Nonocular outer-skin triangle pairs across 878 banks | 11,793 | 11,793 |
| Inverted skin/fat cells at that endpoint | 0 | 0 |
| Cells below normalized shape 0.00001 | 0 | 4 |
| Fresh residual on rejected next step, tolerance 0.00001 | 0.0661842 | 0.0426593 |

The whole-volume launch checked all 705,091 skin and 743,713 fat cells. The first completed root took 31.84 s including presentation, or 175.73 s from process launch to exit. The two-root attempt took 212.12 s, accepted one root and rolled back the other. Variable pleural pressure was enabled. This interval is too short to establish breathing or gas regulation. The first launch attempt stopped at its existing disk reserve; closed outputs were losslessly compressed and the same configuration reran.

The limiting mixed-interface fat cell 1064386 reached a smaller determinant ratio after the fit, 0.36437 versus 0.46010 at the original first endpoint. Five of the six prior terminal determinant witnesses have all four vertices on the same anatomical layer. Two of the four new shape failures lie wholly outside the fitted coordinates. The attached source-row evidence includes actual endpoints and source quality. A local cavity screen examined all ten witnesses. Six replacement patterns passed geometry at the sampled positions. None passed the combination of local mass and first-moment feasibility and the 5% local strain-energy comparison. This screen does not exclude a larger connected reconstruction or other conservative mass assignments.

Reproduce with the pinned parent registration and launch from the pelvic-contact archive, using the fit and native binder in the registered material tools. The fit, delta, commands, binary and material hashes, native admission and rollback traces, complete endpoint cell counts, both layer audits and independently recovered payloads accompany this report. Failed proposals remain available. The 60 s reduced native control at fe98cb96 used 32 physical bed sections and a different tissue execution model.
