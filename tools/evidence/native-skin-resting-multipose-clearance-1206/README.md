# Resting skin multi-pose clearance fit 1206

## Result

Both source-position fit attempts failed. Attempt 001 stopped immediately because the required shared source direction projected only 0.446271 onto an active face, below the existing 0.5 bound. The published direction-conditioning helper later addressed that selector constraint. Attempt 002 then completed eight correction iterations, but the shared solver stopped with 309 exact nonocular skin-to-target pairs remaining at the 40-second step-20000 pose (zero at fit steps 0 and 10000). It did not emit an admitted full NHSKIN candidate.

No fit-derived candidate was composed, admitted, or run natively. No held-out six-pose audit or composer was executed. The only native evidence here is the already-existing baseline used as fit input; it is not a run of either candidate. Native physiology/integration remains unqualified at five minutes.

## Baseline and source identity

The zero-delta preflight pins the selected 1187 source NHSKIN (SHA-256 b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b), 86 skin bindings, 54,949 stored vertices, and 109,211 faces. It reports exact zero-delta reproduction at all nine accepted states. The three fit-pose baseline counts are [0, 0, 3,112] for steps [0, 10,000, 20,000]; the last is the 40-second baseline state. Ocular and skin self-pair counts are zero at these fit poses. The 40-second 3,112 result is a baseline skin-to-target finding, not evidence that a fitted candidate passed.

The active-face selector correction is published in Human commit f5f441e286880e1366360998557185f3f7cb09bc. The bundle records the exact source and focused-test file hashes. Root reported 31 focused tests passing for that commit; tests were not rerun during this evidence packaging.

## Fit attempt details

Attempt 001 failed after 4.861 seconds at fit pose step 10,000: active face index 446 (source face row 60435), corner 2/source vertex 30182 had projection 0.446271, below the solver's 0.5 direction bound. The retained read-only diagnosis shows the current owner's conditioned direction reaches a minimum projection of 0.500001 over the incident active-face constraints. That fixes the selector condition only; it is not a skin-clearance result.

Attempt 002 used the conditioned runner and preserved each trial's compact source-position array and exact target-audit receipt externally. Across eight solver iterations, the remaining nonocular pairs at the three fit poses changed from [0, 0, 3,112] to [0, 0, 309]. The solver raised after iteration eight rather than returning a final candidate. The last trial's maximum source displacement was 27.3936 mm; its separate shape diagnostic reports minimum face-area ratio 0.7733, maximum ratio 2.0531, and source-edge stretch 0.8973–1.6831. These are diagnostics for the failed last iterate, not anatomical acceptance.

The solver's per-trial NPYS contain compact referenced-vertex coordinates (54,663 by 3); the initial source arrays contain all 54,949 stored vertices. Large arrays, MRVPACKs, and exact witness/audit files remain at their retained external paths. external-artifact-references.json records each path, declared hash, observed hash, and size. failed-fit-input-postcheck.json rehashes every input declared by both failed-fit preflights.

## Exact-predicate and owner evidence

prepared-index-predicate-comparison.json records that existing exact narrow-phase predicate bodies were unchanged; its focused broadphase test record is six tests passing. Baseline audits cover all 859 target surfaces at the retained accepted states. This bundle does not claim a held-out candidate audit or a native admission result.

## Reproduction and scope

The fit runner and zero-delta preflight scripts are copied under source/ with hashes in SHA256SUMS.txt. Inputs and outcomes are retained separately under preflight/, attempts/, direction/, and predicate/. The external failed-fit directory and larger referenced artifacts are not altered by this bundle.

This is failed source-fit evidence only. It does not establish continuous-time clearance, skin thickness, physical collision admission, or physiological integration.
