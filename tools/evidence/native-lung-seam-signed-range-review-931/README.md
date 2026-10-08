# Native lung seam signed-range review (927 inputs, 925 captures)

This report retains the canonical per-pose/per-face-pair intersections from `native-lung-seam-structural-audit-927/targeted-native-report-v2.json` and measures signed normal ranges from the original 922/924 source triangles and the corresponding captured 925 triangles. It does not alter the source NHA, capture, predicate, or candidate assets.

The event count is 93: 81 pairs for 305/308, 6 for 306/307, and 6 for 307/309. Per-pose totals remain `[19,16,18,7,9,10,8,6]`. The earlier `79+6+6=91` extraction is superseded: no canonical 927 row was intentionally excluded here. The only duplicate exact point-set found is the step-0 305/308 pair `(24552,26583)` and `(24554,26582)`; these are distinct face pairs and remain separate events. Merging only that point set would still leave 92, so the earlier missing second event cannot be reconstructed from a retained event list.

For orientation, source-shared pairs use owner A's source-face winding. Source-disjoint pairs use the source-space closest-point direction from A to B. The native direction is that source-derived direction rotated by accepted body-index-20 quaternion from the matching receipt; no native outcome selects its sign. `report.json` records each event's source and captured signed gap range, exact captured intersection span, local half-ULP coordinate bound, and rigid-transform vertex residuals.

| Owner pair | Events | Source signed-range envelope (nm) | Captured signed-range envelope (nm) | Max tangential span (µm) | Max inward excursion below source minimum (nm) | Max pair half-ULP bound (nm) | Max source closest gap (nm) |
|---|---:|---:|---:|---:|---:|---:|
| 305/308 | 81 | 0 to 9,547.121 | −10.790 to 11,397.322 | 53.052 | 15.639 | 32.279 | 9.931 |
| 306/307 | 6 | 0 to 243.949 | −7.100 to 254.139 | 101.877 | 17.985 | 33.115 | 12.445 |
| 307/309 | 6 | −152.949 to 340.268 | −148.827 to 334.720 | 61.076 | 11.534 | 36.178 | 3.815 |

The maximum measured inward range change is 17.985 nm (0.543 of that event's pair half-ULP bound). The largest absolute inward range change is 17.985 nm (0.543 of that event’s pair half-ULP bound); the largest eventwise excursion/bound ratio is 0.633 on a smaller 305/308 event. This is a coordinate-representation comparison only. It is not a tolerance, a zero-intersection result, or proof that every event is intended anatomical apposition. Exact native hits remain in the report. The positive-side range can extend substantially at inspiration; this is reported separately as changed separation, not penetration depth.

The simple accepted body-20 rigid transform reproduces witness coordinates to at most 18.8 nm at steps 0 and 4991, but the full-cycle maximum is 5.224 mm at step 5759, with a maximum normal-component residual of 1.669 mm. Those large residuals show that respiration makes the capture non-rigid; they must not be described as float rounding. The local normal signed-range comparison is retained beside, not substituted for, those full-vector residuals.

Across 33 unique face pairs, the pinned source/native comparison classifies 11 as exact source vertex/edge intersections (all allowed vertex/edge contacts) and 22 as source-disjoint. Maximum source closest gaps are 9.931 nm (305/308), 12.445 nm (306/307), and 3.815 nm (307/309). Source reciprocal patch counts are 8,884 faces for 305/308, 8,727 for 306/307, and 9,605 for 307/309. This is evidence of mapped adjacency, not an intended-interface classification. The report does not waive exact crossings or globalize its interpretation beyond these three families and the eight retained poses.

The 928 v4 and 930 v5 repair alternatives remain failed/unqualified and are not used as inputs. In particular, 928 v4 has a separate source-disjoint 305/308 witness `(40219,46134)` with nearest source vertices about 164.234 µm apart; that is outside the canonical 927 three-family event list and is not explained by the nanometre-scale signed-range result above.

Reproduce on Mini with the exact command pinned in `report.json`:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/n/numi-human-prep-venv-20261005/bin/python /Users/n/numi-human-resting-evidence-20261005/native-lung-seam-signed-range-review-931/analyze.py
```
