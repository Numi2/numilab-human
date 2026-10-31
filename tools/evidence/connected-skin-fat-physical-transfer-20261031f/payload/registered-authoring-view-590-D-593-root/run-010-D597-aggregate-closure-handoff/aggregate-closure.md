# D597 source-authoring closure handoff

Frozen 2026-10-31. Source-metadata handoff only; D597 is not qualified.

No physical candidate arrays were edited. No tests, geometry jobs, or actual230 validation against D597 were run in this handoff. No job owned by this agent remains active; this is not a system-wide process inventory.

## Current pins

- D597: /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/fat-current-D-546-597-root/run009-typed-bind-D597-current-D558-v1/candidate-D-typed-current.npz (SHA 97eb4720ebb09d62659aa03f6c4f86482f9cf0e949101f636f6d1fff0e1daa89), 120,648 points and 215,031 faces.
- D128 typed registry candidate: /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/fat-inferred-boundary-D-refinement-128-v2/candidate-D-inferred-boundary-refined-typed.npz (SHA ff73291bce77dda81eff985fecdbba1b198f24e95a7f8f9af1d6058172d4da45). Raw registry and original primary/CSR arrays are pinned in aggregate-closure.json.
- v9 adapter: /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/registered-authoring-view-590-D-593-root/prepare_registered_deep_authoring_view_593_v9.py (SHA 031c11d515f64bac6c5f9d084aeb14f2c0c3fd7e20002ded7b1fec7b63bbbedd).

## Aggregate result

Census: 470 active IDs, 353 active derived-D IDs, all with parent lists. Eight D128-only IDs (704001247-704001254) are absent from D597 and its parent closure. The authenticated reachable parent union has 153 IDs and no missing registry IDs. This closes ID presence, not role or asset qualification.

Resolution: 119 IDs from typed primary/CSR rows; 29 BONE parents at layer 3; four 62000202 crosswalk parents at layer 2; one 62000200 source parent at layer 2.

62000200 exact evidence verifies 245 F32 vertex matches and 190 directed faces, all producer rows layer 2. Its 1,048 CSR entries resolve 30 parents with layer union {2,3}. Keep BONE layer 3. Registry face_count is 96 while source and producer each contain 190.

62000202 source, parent, report, proof, crosswalk, true IDs, and layers are pinned in aggregate-closure.json. Keep the mixed producer namespace separately. 62000204 retains face_source_face_id=-1 with identity0 rows 0-14.

## Open status

v9 is not compiled, tested, or run against D597. It binds inherited-layer evidence to D202 only and does not overlay D200. Historical D128/run005 actual230 validation failed on 100000185 (wrong layer role) and 100000186 (no pinned parent asset/source hash). Candidate and failure receipt remain pinned in JSON; they do not qualify D597.

Next: author a versioned v10 for D597, add authenticated D200 layer-2 evidence without rewriting raw null layer or BONE layer-3 ancestry, resolve or leave 100000185/100000186 NOT_QUALIFIED, and preserve the 96-versus-190 discrepancy. Existing tests are pinned but were not run on v9.

The intended serial command is stored verbatim in aggregate-closure.json. Do not execute it until v10 exists and is pinned.