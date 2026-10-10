# Active-face direction-conditioning evidence (1236)

This bundle preserves the narrow optimizer-status change, its tests, and the evidence that motivated and checked it. It does not include a geometry proposal and does not claim skin clearance or native admission.

The retained one-iteration resume of d319 stopped at compact vertex 30204 with SLSQP status 8 (`Positive directional derivative for linesearch`). That optimizer status alone did not establish infeasibility. The small captured constraint fixture reproduced a nonzero result whose normalized direction met every unchanged active-face alignment constraint: minimum alignment `0.5000009999541253` against the existing `0.5` gate. An independent max-margin probe on three initial seeds found a feasible direction with minimum alignment `0.856128180643463`. Those results show that a finite, nonzero, independently gate-passing direction can be useful even when SLSQP does not report convergence; they do not certify the SLSQP result as the closest-point optimum.

The owner change accepts such a result only after rechecking all unchanged per-pose active-face constraints. It still rejects zero, nonfinite, or below-gate results and records optimizer status, message, iteration count, pre-normalization norm, and minimum alignment. The full-active conditioner ran against the pinned d319 candidate and all 17 retained poses: 168-face active union, two conditioned vertices, minimum active projection moved from `0.4120001364` to `0.50000099995`. It reused the pinned forward states/tables; it did not rerun target or self-intersection scans.

The earlier d319 owner resume remains a failure: one owner iteration, 1,595.11 seconds, 2,521 nonocular target pairs versus 3,131 at accepted attempt 4, then stopped at vertex 30204 on the optimizer status. Its 752-input postcheck reports unchanged. The source/test update and conditioner evidence do not rerun that fit. No fit was rerun after this source change, no native run was launched, no candidate was composed, and no geometry was admitted.

## Validation

The focused test command completed with **51 passed**:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/n/numi-human-prep-venv-20261005/bin/python3.13 -m pytest -q tests/test_common_atlas_skin_clearance.py
```

The exact changed owner and test files are included under `source/`; the full-active runner and its input-bound report are under `conditioning/`. The single-vertex fixture, diagnosis, and independent max-margin script/results are retained alongside them. The failed resume log, execution receipt, preflight, launch, wrapper, and unchanged-input postcheck are retained under `failed-fit/`. Large native captures and the d319 geometry arrays are not copied; their paths and hashes remain pinned in the included execution/conditioning reports.

`COPY-MANIFEST.json` records bytes and SHA-256 for every staged file. `SHA256SUMS` covers the manifest and all staged files.
