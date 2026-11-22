# First-step tissue quality and support distribution

The matched full-volume interpolation screen selected one quarter of the registered posterior displacement. A fresh native M6 run confirmed the first-step prediction. All 1,448,804 skin and fat cells met the existing shape and volume thresholds, with zero inversions. All three registered layer boundaries and head walls were free of unexplained intersections at the emitted start and first accepted step. The 878-bank native envelope still had 11,793 nonocular skin-to-tissue triangle pairs and zero bone pairs.

| First accepted 15.625 µs | Published control | Quarter displacement | Full displacement |
| --- | ---: | ---: | ---: |
| Dynamic bed reaction | 4,691.21 N | 3,640.81 N | 1,588.94 N |
| Excess reaction decrease relative to control, body weight 952.86 N | — | 28.10% | 82.99% |
| Cells below shape threshold | 0 | 0 | 4 |
| Fresh residual on the next rejected step | 0.0661842 | 0.0537829 | 0.0426593 |

The quarter-displacement two-step run took 218.14 s, accepted one step and rolled back on the second. Brain and respiration remained at accepted step one. For the successful one-step capture the native horizon took 33.93 s and the entire process took approximately 180 s. This is the preferable full-material starting state among the fits tested for first-step load and cell validity. It is not a settled checkpoint.

## Mechanical decision

The developed skin and fat can carry real bed forces. The current footprint concentrates 763 N of the initial 953 N on articulated donor 128. The separate settled 60 s native control averaged 372 N on that donor. Both scenes use the same rigid asset and source body mass. They have different tissue representations and sampled postures. The 60 s model used 32 normal-compression sections.

Summing the actual 58–60 s impulses shows 141.57 N on donors with no initial dense contact. Four of their outer surfaces start 31.7–41.8 mm above the bed. A nonnegative linear-program bound allows arbitrary normal loads on all 143 available dense support points while preserving total weight and zero roll/pitch moment about the accounted center of mass. Even this optimistic bound limits donor 20 to 271.24 N; its settled 32-section mean is 344.42 N. The minimum sum of absolute per-donor force differences is 451.42 N. Matching total force alone cannot reproduce this distribution.

A separate pose counterfactual applied the actual 60 s accepted rigid transforms to each registered donor's installed tissue coordinates. It produced 18,387 inverted skin cells, 15,665 inverted fat cells and 1,812 outer nodes below the bed. No such pose was installed. Subsequent construction should solve supported articulation and connected volumetric deformation together. The fine mesh provides force, strain, thickness and clearance checks for a reduced runtime. The archived 60 s control remains the integrated reference for ventilation, circulation and external support.

The storage index lists the exact original paths, GitHub release and asset names, compressed SHA-256 and decoded SHA-256 for independently recovered closed historical runs. Download a listed asset with GitHub CLI release download and restore the original filename. The separate APFS receipts describe independent copy-on-write files with matching byte streams and unchanged paths.
