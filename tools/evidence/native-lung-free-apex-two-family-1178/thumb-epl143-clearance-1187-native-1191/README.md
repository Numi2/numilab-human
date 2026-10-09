# EPL143 right-thumb reference-skin candidate on the 1178 lung scene

This bundle pins an inferred 0.5 mm right-thumb skin-reference correction, CPU owner-composition preview 010, and actual native viewer run 1191. It is a short-run geometry/evidence increment. The source-level 16-pose screen used retained baseline and treatment poses from the earlier 1159 scene; the 1191 native run uses the 1178 lung scene. These are separate evidence scopes.

The correction changes 33 source positions and 90 incident skin faces, then recomputes 207 shading normals. The full 86-weight matrix, 86 binding records, triangle indices/order, source archive identity and static NHCNT support payload are unchanged. It is a common-atlas inferred reference-skin adjustment, not measured participant anatomy or measured skin thickness. The candidate lies inside active right distal-thumb NHCNT seed region 15. The 1191 native run used the ordinary skin-support selection and force-Jacobian path with this candidate loaded. Its 20-second support/contact trace and the other three native CSVs are byte-identical to the 1178 parent. This establishes no changed sampled trace in that run; there is no separate selected-point/Jacobian telemetry or long-run candidate contact qualification.

## Evidence

- **Prior pose screening:** source/attempt-006-report.json is the eight-pose baseline screen; source/attempt-007-treatment-report.json applies the exact same fixed candidate to eight treatment poses. Both report 0.5 mm as the first tested baseline-clear amplitude. The screening NHA hash is c10dce4609be99fdc569801c2705e23c46120b610c8aa86c631165d38baf4713; it is not the 1178 NHA. The copied package manifest retains all 16 accepted pose pack/receipt/result/witness hashes without copying the multi-gigabyte packs.
- **Composition admission:** source/admission-010-preview.json, source/admission-010-native-owner-command.json, and the candidate registration/scene manifests bind owner composition attempt 010 to the 1178 scene. The admission preview did not launch native code. The later 1191 run exercised the normal runtime path, but the preview itself did not establish per-step selector/Jacobian invariance.
- **Actual native 1191:** the invocation and run metadata pin the viewer 018 executable, physical runtime 014 libraries, all input assets/configuration and the 20-second accepted horizon. The run accepted 10,000 roots (20.000000949949026 s), exited 0, verified the loaded runtime, and recorded no source-file changes during the run. native-1191/non-skin-identity-v2.json is the corrected identity report; its predecessor with a literal trailing backslash-n is preserved in failures/ and identified by hash in the manifest.
- **Differential mesh audit:** native-1191/differential-summary.json records accepted capture steps 0, 4991, 5375, 5759, 6111, 6495, 7743, 10000. The full 109,211-face skin self-mesh was audited at every pose. Every one of the 90 changed skin faces was tested against all 859 registered target surfaces at every pose, with zero unallowed skin-self pairs and zero changed-face/target intersections. Untouched skin-target results were transferred from the pinned complete 937 audit only after exact unchanged input/geometry/state checks; this bundle does not claim that all 109,211 faces were freshly compared against all 859 targets. This is discrete captured-state coverage, not continuous-time proof.
- **Recording:** the copied 1191 recording review reports 315 movie samples; its paired visual review covers 10 selected decoded frames. The review notes include footer clipping in some views and do not imply anatomical qualification.

## Preserved failures and limits

The copied attempt-history and compact failure records retain failed candidate-generation/admission attempts and the interrupted earlier scan attempt; they are not counted as passes. The corrected identity JSON supersedes, but does not delete, the malformed predecessor. Candidate 1191 ran for 20 seconds only. The treatment-pose results are offline saved-state checks from the prior scene. No continuous-time clearance, long-run candidate geometry/contact behavior, independent selected-point/Jacobian telemetry, clinical interpretation, or full-body acceptance is established.

Large skin payloads, anatomy receipts, capture MRVPacks, native logs/movie and raw CSVs are referenced by absolute retained paths and SHA-256 in publication-manifest.json; they are not duplicated here. This keeps the bundle compact while preserving source identities.

## Verification

From the repository root, verify copied files with:

    cd tools/evidence/native-lung-free-apex-two-family-1178/thumb-epl143-clearance-1187-native-1191 && shasum -a 256 -c SHA256SUMS

publication-manifest.json contains the external path/hash inventory and hashes for every local file. The 16 prior saved-pose packs are pinned by the copied package manifest; the eight native packs and receipts are pinned by the copied final differential summary and external-artifact index. This publication step runs no native or GPU work.
