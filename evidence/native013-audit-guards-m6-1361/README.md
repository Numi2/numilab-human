# Native013 audit guard review

This evidence increment adds fail-closed source-identity checks to the two native013 audit successors and tests those guards with small synthetic arrays.

The original audit scripts and D13 preparation inputs remain unchanged. The successors now reject duplicate compact IDs, malformed or out-of-range local faces, a mismatched global face mapping, incomplete face coverage of compact vertices, and any shared-ID count other than the bundle-pinned 2,070. The main audit also explicitly validates the 158 unique source-row inventory before checking 156 untouched rows. The external census checks bitwise agreement of every shared vertex in its native capture.

The regression test accepts valid synthetic mappings, rejects duplicate IDs, mapping mismatch, out-of-range faces, an empty shared intersection, and wrong shared cardinality. It also validates the real candidate150 NPZ mappings and their 2,070 shared global IDs against the preparation receipt and geometry hashes.

Static review found the main audit compares all 866 non-target surfaces at each of its eight native captures, checks 157 accepted body poses and the 1,000-row physiology trace against native012, and requires exact self-pair/degen/topology/component-sign checks for each target row and their joined union. The external census is intentionally only a single-step boundary-crossing census and makes no anatomical admission claim.

Validation: the synthetic regression and read-only candidate bundle preflight passed; Ruff F821/F823 and Python compilation passed. No native audit, GPU launch, or anatomy admission was performed. The earlier CPU Float32 union failures remain separate evidence and are not overridden here.

Reproduce the regression with:

    /Users/n/numi-human-prep-venv-20261005/bin/python /Users/n/numi-human-left-pq-publication-1344/evidence/native013-audit-guards-m6-1361/test_native013_audit_guards.py
