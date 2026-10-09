# Rejected source-surface Dirichlet skin binding experiment (1202)

Status: **rejected and not adopted**. This is CPU-only source construction, a source-payload preflight, and a one-pose counterfactual screen. It is not a native candidate run or a full-cycle anatomy/physics qualification. No production/runtime source change is included.

The default screened source reconstruction reproduced the complete 86-column Float32 weight matrix exactly (SHA e43f72b3...6aca52b4) while its raw source graph had 109,183 triangles versus 109,211 in the current 1187 skin. The source/current topology mismatch made the reconstruction command return exit 2 even though the weight comparison passed. The former compressed NPZ container was unavailable (SHA 99c91381...a5a281fc); the regenerated NPZ container has different bytes, so only decoded arrays and weight bytes were compared.

The optional Dirichlet source inference kept the same admitted inverse-gap projection seed shares fixed and harmonically interpolated unseeded source-graph vertices. Its independent raw-source audit passed with 86 bindings, 12,025 seed candidates, 5,039 seeded vertices, 54,663 quotient-graph vertices, 163,860 edges, maximum equation residual 2.36e-15, and fixed-seed error 0. The candidate changes all 54,949 ABI-5 weight rows; it is a new inferred binding field, not a reconstruction of the existing field.

Raw-source preflight attempt 1 stopped before payload audit because the supplied v5 manifest had a stale MyoSim runtime reference. Attempt 2 rebuilt the v5 manifest and passed independent source provenance, weights, seams, and rest reconstruction checks. These are source-payload checks only.

The exact Float32 counterfactual screen evaluated all 859 target surfaces at accepted step 20,000 (40.0000019 s), holding the captured common-field/respiratory residual and target geometry fixed while applying the candidate LBS delta. It found **8,857 target crossings and 197 skin self-crossing pairs**, with zero degenerate faces and zero ocular crossings. The flat baseline at that same accepted state had **3,112 target crossings and zero skin self-crossings**. The candidate worsened the exact screen and was rejected. It was never captured by the native runtime, and its changed weights would also change the contact Jacobian.

The rejected source snapshot preserves the exact five source/test files and the unadopted patch. It also records the single-column helper edge case found during review: SciPy returns a one-dimensional spsolve result for an (n,1) RHS; the screened branch then fails on sum(axis=1), while the Dirichlet branch fails assignment to (n_free,1). The intended 86-column case is unaffected. This code was archived, not merged or left active.

The reproduction scripts, declarations, command records, logs, reports, and exact-scan result are included here. Large NHSKIN/NPZ payloads, source archives, native pack, and witness JSONL remain external; see external-artifacts.json for path, size, and SHA-256. No 1202 native/GPU run was performed; the counterfactual screen used an already accepted native pack and offline CPU predicates.

The focused review selection passed 8 tests (test_skin_binding_constraints.py and selected Dirichlet/source-oracle tests). It did not cover single-column RHS handling.
