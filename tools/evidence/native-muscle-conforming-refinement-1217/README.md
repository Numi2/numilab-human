# Native muscle conforming refinement 1217

## Result

A source-derived conforming edge refinement was applied to stable ID 64 (left vastus lateralis) in the accepted 039 passive NHTISS package. The producer sampled the existing femur/patella/tibia route-weight field at the midpoint of source edge (2597, 3054). The final surface has one additional vertex and two additional faces; the two incident source faces were split conformingly. Existing vertex records and the three route-binding records for stable 64 are byte-identical; the other 149 surface rows are preserved. The composer reports no physical route, mass, or force-state changes.

The native run used the flat support configuration, unchanged physical body, respiration, MyoSim route, and contact inputs; only the declared passive surface asset changed. It used 64 contact iterations and completed 20,000 accepted steps (dt=0.002 s, 40.0000019 s). The coupled state reported seven complete breaths and 47 complete filling/ejection cycles during the recorded run; reported RTF was 0.11455. The coupled physiology, COM momentum, and COM support-impulse CSVs are byte-identical to the flat 1201 control. This is a mesh-refinement/native replay comparison, not a five-minute anatomy pass or a clinical result.

## Geometry checks

Exact stable-63/right-VL and stable-64/left-VL self/topology audits passed at accepted steps 0, 9983, and 20000: zero self-intersections, no degenerate faces, and closed/oriented/unused-free components. Stable 64 retained all 6,690 faces; the separate outer-skin enclosure eligibility supplement passed at those same three poses and removed no faces.

The full exact 859-target skin audit is complete at those three captures. At 0 s and 19.966 s it found zero skin-target crossings. At 40.0000019 s it found **3,112 nonocular crossing pairs** involving 934 skin face rows, 2,025 target face rows, and 26 target surfaces. It found zero ocular crossings, zero skin self-crossings, and zero invalid or degenerate triangles. No contacts were exempted. The five-minute anatomy gate remains open.

## Recording review

The original recording review read 632 compressed-track sample buffers, of which 627 carried non-empty image samples, and decoded ten representative snapshots. The movie duration was 351.32 wall seconds, with the last surface row at 40.0000019 simulated seconds and a maximum image-sample gap of 2.0683 seconds. A later sequential BGRA decode produced 628 frames in increasing PTS order: the 627 non-empty compressed image PTS values plus an additional decompressed PTS 0.0. There are 627 surface-audit rows. The last sequential frame (PTS 350.7583 s) matches the existing terminal extraction pixel-for-pixel; the earlier apparent blackout was not substantiated. This is sample/timing/render evidence only, not geometry or long-run qualification. The movie and images are retained externally and pinned in external-artifacts.json.

## Attempts and reproducibility

- Producer attempt 001 failed because its source snapshot lacked the canonical MyoSim source overlay. The failure and execution record are retained; no output from it was accepted.
- Producer attempt 002 succeeded from the complete source snapshot, emitted the stable-64 subset, and passed the accepted-039 row/binding/route verification.
- Native attempt 001 stopped before a capture because accepted step 10,000 was not on the accepted-step cadence. Native attempt 002 corrected the capture schedule to steps 0, 9,983, and 20,000 and completed. The failed attempt remains separate.

Exact source snapshots, producer/composer scripts and reports, run declarations, selected comparison/audit scripts and reports, and recording-review scripts are included. Large NHTISS/NHA/NHSKIN/MRVPack/movie/witness-ledger assets are not duplicated; their original paths, byte sizes, and rechecked SHA-256 values are in external-artifacts.json. SHA256SUMS binds every file in this evidence bundle except itself.

## Claim boundary

This is a one-edge, source-derived mesh-refinement experiment with a 40-second native run and three-capture exact anatomy checks. The terminal skin-target crossing result is nonzero. It does not qualify the complete anatomy over five minutes, and makes no clinical claim.

## Focused code tests

The included composer test file was run with the pinned preparation Python 3.13 environment: 12 tests passed. The exact command and copied test-file hash are recorded in validation/focused-tests.txt. This unit suite does not change the capture-level anatomy limits above.
