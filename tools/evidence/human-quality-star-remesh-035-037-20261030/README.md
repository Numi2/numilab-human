# H706 quality-star remesh source evidence (035–037)

This evidence bundle records the bounded source-conditioned O/I/D surface proposals and a failed volume feasibility attempt. It is **source preparation only**. No physical mesh, mass transfer, or native runtime is admitted by these results.

## What the records show

- The 035 batch retained its raw operation records and typed source metadata. Its reports and tests describe only the saved source-proposal scope.
- The 036 and 037 batches retain their original reports, operation JSONs, raw proposal arrays, typed metadata, point/face maps, and candidate-specific rim/core inputs.
- The 037 independent fixed-coordinate verifier reports zero exact intersections for its generated-face delta against the complete current role triangles and all 185 bone records in that verifier’s scope. It is a local delta check with pinned baseline assumptions, not a complete physical-admission result.
- The 008 TetGen feasibility probe failed for both Skin and Fat (return code 2). Skin reached mesh improvement and hit the reported TetGen failure; Fat reported 11 warnings and skipped four segments due to intersections. The probe report records `physical_admission: false`, runtime 0, and accepted steps 0.
- The 036/037 operation hash correction reconstructs the report-pinned bytes for all 110 operations after removing only the two producer-added fields `new_output_face_rows` and `patches`. It validates report row maps, output triangles through point maps, and retired-face domains. The enriched operation files and original reports were not edited.
- A nested `.gitattributes` file disables only `blank-at-eof` for three exact frozen 035 source files; `whitespace-exceptions.json` records the paths, hashes, and findings. No global whitespace checks are waived.

## Limits

No positive TetGen volume, canonical volume-node ownership, conservative mass transfer, install, motion, or accepted-state stepping is demonstrated here. The failed volume inputs and logs are preserved to make that failure reproducible. Earlier 004/005 records are explicitly failure context only.

See `manifest.json` for the exact source path, byte count, and SHA-256 of each included file.
