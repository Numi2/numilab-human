# Registered late-pose geometry audit adapter (1171)

This read-only successor accepts a registered science v2 trial directory. It validates the sealed registration -> started -> process -> exit receipt chain, the arm-specific P18 launch template and native invocation, loaded Metal runtime proof, all output and scene-asset hashes, and exactly eight accepted geometry receipts ending at step 155000.

It deliberately does not look for a scene run-metadata.json. The registered exit receipt is the authority for a run-native science trial. The owner run-metadata.json used for template derivation belongs only to the completed 10,000-root parent preflight. This distinction is recorded in the output declaration. Failed or incomplete receipts are rejected before any geometry scan.

## Invocation

Use a fresh output directory and pass the registered trial directory to --run:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python3.13 \
  /Users/n/numi-human-resting-evidence-20261005/native-lung-late-pose-audit-runner-1171/audit_lung_cycle_1159.py \
  --run /absolute/path/to/study/trials/resting-baseline \
  --registered-arm control \
  --out /Users/n/numi-human-resting-evidence-20261005/native-lung-late-pose-audit-runner-1171/attempt-001 \
  --nha /absolute/path/to/registered-current.nhanatomy \
  --nha-sha256 EXPECTED_SHA256 \
  --d-map-composition-report /absolute/path/to/current-composition-report.json \
  --lobe-lineage-report /absolute/path/to/current-reciprocal-map-report-v2.json \
  --workers 1
```

For treatment, use --registered-arm treatment and that arm's exact registered trial directory. Do not pass output/scene to --run. Do not run until the registered receipt is sealed successfully. The step-0 regression is separate and does not accept --registered-arm.

## Source pins

- Retained 1161 exact geometry and predicate owners remain pinned in audit_lung_cycle_1159.py.
- Registered receipt adapter: registered_receipt_adapter_1171.py; its SHA-256 is pinned by the runner.
- P18 analyzer: /Users/n/numi-human-resting-evidence-20261005/final-integrated-study-readiness-1170/package-v018/analyze_final_pair.py, SHA-256 defe06f8ded56d66fe5ba903445bb8dd81ef96ad57fa59e3be4c615c5892cb5c.
- P18 plan owner: /Users/n/numi-human-resting-evidence-20261005/final-integrated-study-readiness-1170/package-v018/prepare_final_plan.py, SHA-256 f5d57e46afce90ea756e8eb8e23bfd1f4a6a6e9fcb971659f946ce29768531be.


- The exact published V8 current-D-map reader from the 1096 provenance package is copied byte-for-byte into , SHA-256 . The successor resolves and hash-checks that source.
- The earlier 1095 compatibility copy is retained only as a historical intermediate; no live adapter or test imports it. The exact 1096 reader was exercised against the pinned V8 composition and returned 47,343 registered D-map faces with lobe counts 19,743 / 21,600 / 5,514 / 486 / 0.

- The exact published V8 current-D-map reader from the 1096 provenance package is copied byte-for-byte into v8_dmap_adapter_1096.py, SHA-256 9a55338d874ecc4695e1545829c21266b9b5734d8ca8d64acab817906091c7bd. The successor resolves and hash-checks that source.
- The earlier 1095 compatibility copy is retained only as a historical intermediate; no live adapter or test imports it. The exact 1096 reader was exercised against the pinned V8 composition and returned 47,343 registered D-map faces with lobe counts 19,743 / 21,600 / 5,514 / 486 / 0.

## Verification and limits

The focused suite covers the eight-step schedule, current D-map rebinding, rejection of an incorrect selected-NHA hash, current respiration-area binding, adapter pins, and synthetic registered receipts: valid output without scene metadata, failed exit, missing receipt, hash mismatch, receipt-chain mismatch, and terminal N-1 rejection. It does not run the geometry scan. No successful registered arm was available in the evidence tree during this preparation, so 1171 has not yet been exercised against a completed run-native study.
