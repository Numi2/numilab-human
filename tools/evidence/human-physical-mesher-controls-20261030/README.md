# Human physical mesher controls — 2026-10-30

This bundle records the bounded TetGen source change and source-preparation controls for Human. It is **source and fixture-test evidence only**. It does not qualify a physical mesh, mass transport, native installation, runtime, or accepted state.

## What is tested

- The frozen 054 TetGen variant preserves stock flip/iteration defaults. When both improvement counts are explicitly zero, it skips queue-based mesh repair while still running the all-cell `get_tetqual` degeneracy check and retaining the `-c` exterior skip. The compile receipt pins the full upstream source/header, predicates source/object, license, parent source, and executable.
- Four one-tet controls compare stock default, patched default, explicit zero-improvement, and the trailing-`Q` parser case. Their mesh arrays are bitwise equal in the fixture; this is a parser/control result, not a full PLC result.
- The 054 capacity gate has **6 passing unit tests**. The 217 role adapter has **19 passing tests**. The 056 region-seed helper has **8 passing tests**. All 33 tests were independently rerun on `numi` for this bundle; see `test-rerun-receipt.json`. The region-seed retry is disabled by default and is opt-in with `NUMI_TETGEN_REGION_SEED_COMPLETION=1`, limited to one retry per material. It prepares verified interior region seeds and does not change cell labels, geometry, or acceptance rules.
- The 058 helper prepares verified PLC inputs without invoking TetGen. The adjacent 047 helper is included as its source comparator; the exact source hashes are listed in the manifest.

## Retained counterexample and limits

The historical 037 skin run exited with code 0 but the existing owner rejected its material attribute. The 055 diagnostics report 31 cells with attribute 0; centroid-containment diagnostics place those centroids inside the shell but do not establish cell ancestry. The same report records 20 zero and 20 negative F32 volumes in attribute 15. These are failure witnesses, not evidence that the adapter fixes the candidate. The raw 037 PLC and `.1.node/.1.ele/.1.face/.1.edge` outputs remain at their original local paths and are represented by SHA-256 pins in the manifest rather than copied here.

The included fixtures and unit tests do not run a candidate full-body PLC through the adapter. The adapter preserves the source004 owner's strict final F32 positivity, material, and capacity gates. Region-seed completion is preparation-only unless explicitly enabled and still cannot waive those gates. Every receipt in this bundle records `physical_admission: false`, `runtime: 0`, and `accepted_steps: 0` where those fields apply.

## Reproducibility

`manifest.json` pins each included file by destination, original source path, byte length, mode, and SHA-256. `.gitattributes` contains only file-specific whitespace exceptions needed to preserve exact vendored source and generated fixture bytes; it does not disable whitespace checks globally. It also records the omitted local bulky artifacts by exact path, size, and SHA-256. The Git tree/commit supplies the pin for `manifest.json` itself. The referenced native-domain-217 source004 owner is already present in the repository base at `docs/evidence/human-h706-source-volume-20261030/tools/native-domain-217-source-004/build_native_domain_217.py` with SHA-256 `46799a2d63fd20bc8a191f319f1f56244610a5e719983a7ce44cc5222659b6bd`.
