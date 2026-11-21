# Whole skin in the mechanically loaded native human

The initial native pose already has 8,868 nonocular skin-to-tissue triangle pairs and 21 nonadjacent skin self-intersections. All 8,889 witnesses cross transversely at both 0 and 2 ms. Native step 415 has 10,127 nonocular pairs and 302 self-intersections. That endpoint passes the separate six-bank pelvic tests while failing whole-skin clearance.

| Actual native pose | Bone | Muscle | Vessel | Total nonocular | Skin self |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supported initialization | 1,840 | 6,454 | 574 | 8,868 | 21 |
| First accepted step, 2 ms | 1,840 | 6,454 | 574 | 8,868 | 21 |
| Accepted step 415, 830 ms | 2,135 | 7,432 | 560 | 10,127 | 302 |

The inspection used each receipt's emitted Float32 triangles across 878 typed primitive targets. The early and late receipts agree on the native base pack. Of the nonocular pairs 3,940 persist, 4,928 clear, and 6,187 appear between the two motion samples. This measures intersection pairs, including multiple triangles in one region. Intermediate poses, containment and contacts between two internal organs require their own checks. The original finite/surface native receipt does not measure this clearance.

Use the common loaded body reference to repair the outer skin, underlying material and internal surface bindings together. The current 32 local bed sections transfer real load. Whole-body skin/fat FEM is disabled in this replay. Promoting the displayed skin as a physically registered boundary requires repeatable reference correspondence, layer and mass transfer, and accepted-motion clearance. Solving additional pelvic iterations alone cannot correct crossing witnesses already present at startup.

The native driver optionally accepts a comma-separated list of extra capture steps, the original exact audit entrypoint and its matching Float32 primitive implementation as arguments 8 through 10. The original seven-argument invocation is preserved. The dual-sided exact plane-straddle test classifies proper crossings among previously certified intersection witnesses. It keeps boundary-only cases separate. Use the existing human tools and compare_native_skin_endpoints.py to match source-stable faces across captured endpoints. SHA256SUMS and the compressed source receipts pin every listed pose.
