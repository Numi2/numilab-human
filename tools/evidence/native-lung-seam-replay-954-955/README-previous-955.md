# Captured/ideal lobe-seam replay bundle (955)

This compact bundle packages the reusable offline replay and its verified 952 terminal fixture. It does not copy or change source anatomy, accepted native packs, runtime binaries, or scan inputs. Those inputs remain at the absolute paths recorded in known-952-witnesses.json and retained-artifacts.json; every referenced file was checked against its SHA-256 when this bundle was prepared.

The runner accepts an explicit case containing the exact NHA payload, anatomy manifest, accepted MRVPACK and receipt, native anatomy-parameter blob, complete full-scan report, classification report, row-310 parent-face lineage, pinned parser/predicate/pack-reader modules, and explicit lobe_pair or pleura_parent_pair witnesses. It checks path/hash identity, accepted capture step, source face ordering, semantic/stable row identity, body-index ownership, witness provenance in the pinned full-scan classification, and pleura-parent coordinate lineage. Classification input labels are resolved by unique exact path and digest rather than by historical NHA/capture labels.

The terminal validation reproduces 25 selected pleura-parent witnesses: source and binary64 replay each classify 17 as disjoint and 8 as single-point contacts, while all 25 captured Float32 pairs remain exact multi-point hits and unwaived. The largest upward-rounded pair L1 displacement envelope is 48.915 nm. A separately exercised row-305/308 lobe-pair witness is disjoint in source and replay but has a native exact two-point hit; its measured replay gap is contained by its pair L1 envelope. These selected checks do not prove global injectivity, a formal shader rounding bound, or clearance between captures, and do not waive exact native events.

## Reproduce

Use macmini with the pinned external inputs available:

    /Users/n/numi-human-prep-venv-20261005/bin/python /Users/n/numi-human-resting-evidence-20261005/native-lung-seam-replay-publication-bundle-955/replay_lobe_seam_witnesses.py --case /Users/n/numi-human-resting-evidence-20261005/native-lung-seam-replay-publication-bundle-955/known-952-witnesses.json --out /Users/n/numi-human-resting-evidence-20261005/native-lung-seam-replay-publication-bundle-955/replay-952-parameterized.json

    /Users/n/numi-human-prep-venv-20261005/bin/python /Users/n/numi-human-resting-evidence-20261005/native-lung-seam-replay-publication-bundle-955/verify_replay_954.py

The runner refuses to overwrite an existing output. The verifier checks the external input path/hash/size index, creates small identity-test cases, and writes verification-results.json in this bundle. The previous failed attempts remain untouched in evidence directory 954 and are referenced with path/hash in retained-artifacts.json.
