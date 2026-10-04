# Attempt 002 — direct source-mask tetrahedralization

## Outcome

All 12 pinned scan-001 labels produced one tetrahedral volume candidate each: 3,365,578 source voxels and 20,193,468 tetrahedra. The original source NIfTI was streamed from the licensed archive, its decompressed SHA-256 matched the intake receipt, and every label count matched both the plan and intake. The compiler also found exact triangle-multiset agreement between the registered source surfaces and direct-mask boundaries.

The [compiler receipt](meshes/receipt.json) records all twelve outputs. The separate [independent verification](independent-verification.json) rereads the original scan and all VTK files; it reports positive tetrahedra, valid face incidence, exact source-mask boundary triangles, and volume error from `1.54e-16` to `3.38e-15` relative. The verification JSON SHA-256 is `7c77a04cf77950a4fdb3d7ed59680959732c1f286aae2a36bd069c4f3d6d4ced`.

## Preserved attempts and provenance

- [`owner-cli-runner-failure.json`](owner-cli-runner-failure.json) preserves the `numi` Python 3.14 runner stopping before source ingestion because NumPy was unavailable. The same owner CLI module ran under the local Python 3.10 NumPy environment.
- [`compiler-run-closeout.json`](compiler-run-closeout.json) preserves the summary-key error raised after the compiler had written the complete receipt and all meshes. The exact compiler source used by the plan is archived as [`healthy_total_body_ct_organ_voxel_tets_at_build.py`](healthy_total_body_ct_organ_voxel_tets_at_build.py); the current module now reports the updated qualification field.
- [`independent-verification-attempt-001-failure.json`](independent-verification-attempt-001-failure.json) preserves the first audit failure, caused by the verifier's copied x/y face-corner convention. The verifier source was corrected and the full 12-label audit then passed.
- [`plan.json`](plan.json) binds the source archive, intake, surface receipt, labels, and compiler sources. The compiled receipt binds its exact plan hash.

The mask source is TCIA Healthy-Total-Body-CTs v3 (CC BY 4.0), automatically segmented with MOOSE at the 90-minute timepoint. This result is scan-specific geometry, not an expert-reviewed or Numi-subject anatomy. No materials, mass owners, mechanics, vascular lumen, perfusion, or physiology are admitted. The verifier checks tetrahedron face incidence, not full vertex-link manifoldness or mechanical suitability.
