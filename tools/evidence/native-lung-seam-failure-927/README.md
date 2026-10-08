# Native 925 lung seam structural audit

This repository bundle contains the compact summary and external evidence pins. The complete reports, audit source and pose captures are retained at /Users/n/numi-human-resting-evidence-20261005/native-lung-seam-structural-audit-927/ on the Mac mini. Filenames discussed below refer to that directory; external-SHA256SUMS.txt is its checksum inventory.

This is a bounded exact audit of the current lung/pleura/diaphragm composition. It is not a full 305-311 all-pairs native collision scan.

The audit pins composed NHA 924 (SHA 3c444be7736c066a992988cc32b687917e1c4c5c3968a16b4d5f0106d5b5024e) which has byte-identical 305-311 rows to candidate 922, and all eight accepted native 925 MRVPACK/receipt pairs. It checks exact source self-intersections and topology for rows 305-311, then validates all eight declared reciprocal source patches (all matched faces are opposite-wound with no duplicate keys). Candidate 922 includes the registered 307/309 shared-sliver collapse as well as later synchronized edge collapses.

The native exact scan is limited to the 21 source-mapped witness neighborhoods retained from 916 and 10 collapse-operation changed stars. All 31 target groups were complete in all eight captures. Deduplicated unallowed native face-pair counts by step are 0:19, 4991:16, 5375:18, 5759:7, 6111:9, 6495:10, 7743:8, 9999:6. Hits occur only at the three scanned lobe interfaces: 305/308 (left inferior/superior), 306/307 (right inferior/middle), and 307/309 (right middle/superior). Largest exact intersection extent is 101.9 micrometers.

A source-side exact recheck of the 33 distinct face pairs that intersected in any native pose found 11 pairs touching only at an allowed source shared vertex/edge and 22 source-disjoint pairs; no pair has a source intersection beyond that adjacency. The native accepted Float32 geometry nevertheless contains unallowed intersections. These are retained as a clearance failure, not treated as harmless rounding. This scoped scan does not establish complete native cross-surface clearance for every triangle in rows 305-311.

See report.json for exact pair IDs/coordinates, per-pose counts, source/native comparisons, all pack/receipt hashes, source topology, and the audit boundary. targeted-native-report-v2.json preserves the detailed scan before source-coordinate comparison fields were added. source-native-hit-pair-comparison.json pins to that retained native report and exact source payload.

All 21 retained witness groups were scanned at all eight poses. The report contains a per-witness, per-step matrix; witness neighborhoods overlap, so raw hit counts by witness must not be summed as unique triangle pairs. The deduplicated unique face-pair lists are keyed per pose and preserve the labels that captured each pair.
