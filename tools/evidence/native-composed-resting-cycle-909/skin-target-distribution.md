# Completed 909 skin-crossing distribution

This report summarizes retained exact Float32-lattice triangle witnesses for the accepted 909 geometry. Each count is a pose-specific triangle pair; counts are not deduplicated across poses or face pairs. Complete pair coverage passed for all 859 target surfaces at all eight poses, with zero degenerate faces. The audit found intersections, so it is not an intersection-free qualification. No physical or clinical qualification is implied.

Inputs are hash-bound in target-distribution-v2.json; exact per-pair triangle coordinates and intersection points remain in each step-*.crossing-witnesses.jsonl. The current NHA source face-count and index-order validator passed all 524 surfaces at all poses.

| Accepted step | Exact triangle pairs |
|---:|---:|
| 0 | 0 |
| 4991 | 6,144 |
| 5375 | 4,972 |
| 5759 | 4,377 |
| 6111 | 5,353 |
| 6495 | 6,092 |
| 7743 | 6,138 |
| 9999 | 6,081 |

Total pose-specific pairs: **39,157**.

| Target category | Pair count summed over poses |
|---|---:|
| NHTISS/muscle | 28,371 |
| NHA anatomy | 8,669 |
| NHBONES | 1,264 |
| tendon | 853 |

## Most affected target surfaces

| Surface | Source | Poses with hits | Pairs by step [0, 4991, 5375, 5759, 6111, 6495, 7743, 9999] | Total |
|---|---|---:|---|---:|
| 51005:68 | left external oblique | 7/8 | 0, 1,016, 540, 238, 697, 1,000, 1,025, 989 | 5,505 |
| 51005:67 | right external oblique | 7/8 | 0, 938, 429, 139, 609, 921, 943, 921 | 4,900 |
| 51005:36 | left gracilis | 7/8 | 0, 386, 390, 413, 400, 400, 390, 391 | 2,770 |
| 51005:35 | right gracilis | 7/8 | 0, 353, 349, 350, 350, 356, 377, 349 | 2,484 |
| 51005:64 | left vastus lateralis | 7/8 | 0, 280, 274, 277, 270, 282, 265, 278 | 1,926 |
| 51011:518 | left posterior tibial vein / FJ2118 | 7/8 | 0, 258, 257, 255, 256, 257, 258, 258 | 1,799 |
| 51005:63 | right vastus lateralis | 7/8 | 0, 246, 235, 239, 240, 237, 250, 242 | 1,689 |
| 51011:517 | right posterior tibial vein / FJ2173 | 7/8 | 0, 238, 236, 235, 236, 236, 233, 243 | 1,657 |
| 51005:3 | medial head of right gastrocnemius | 7/8 | 0, 167, 167, 163, 163, 162, 166, 164 | 1,152 |
| 51005:4 | medial head of left gastrocnemius | 7/8 | 0, 147, 149, 149, 149, 150, 148, 146 | 1,038 |
| 51011:487 | left ulnar artery / FJ2258 | 7/8 | 0, 146, 146, 146, 146, 144, 144, 147 | 1,019 |
| 51005:22 | left extensor digitorum longus | 7/8 | 0, 152, 147, 145, 138, 138, 138, 134 | 992 |
| 51005:57 | right tibialis anterior | 7/8 | 0, 117, 119, 120, 120, 120, 119, 116 | 831 |
| 51011:504 | left femoral vein / FJ2102 | 7/8 | 0, 98, 99, 100, 100, 98, 96, 90 | 681 |
| 51011:486 | right ulnar artery / FJ2310 | 7/8 | 0, 110, 105, 104, 102, 91, 71, 95 | 678 |
| 51005:14 | left adductor magnus | 7/8 | 0, 95, 95, 99, 99, 99, 91, 61 | 639 |
| 51004:14 | ulna_r / FJ3391 | 7/8 | 0, 77, 77, 77, 77, 77, 77, 77 | 539 |
| 51005:13 | right adductor magnus | 7/8 | 0, 75, 77, 79, 76, 76, 75, 73 | 531 |
| 51005:27 | right flexor hallucis longus | 7/8 | 0, 75, 75, 75, 75, 75, 75, 75 | 525 |
| 51005:28 | left flexor hallucis longus | 7/8 | 0, 72, 72, 72, 72, 72, 72, 72 | 504 |

Across poses, 41 target surfaces had at least one crossing. The witnesses involve 1,708 unique skin source face rows and 1,438 unique captured skin pack vertex IDs, and 7,817 unique target face rows. These unions identify affected captured regions; they do not classify the overlap as intended or as a material defect.
