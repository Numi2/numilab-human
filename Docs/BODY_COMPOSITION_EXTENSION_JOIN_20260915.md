# Body-composition extension join — 2026-09-15

The v14 body-composition receipt already joins the organ, blood, muscle,
skin, activation and compiled rigid-body source layers. This extension
hash-binds the new foot-proxy and muscle route-volume receipts to that exact
base graph.

The immutable receipt is
[`Docs/media/body-composition-extension-join-20260915/receipt-v1.json`](media/body-composition-extension-join-20260915/receipt-v1.json).
It records 30 registered foot proxy enclosures and the complete 416-route
muscle identity table, including 60 closed-geometry candidates and 82 shared
route incidences across 78 routes. The base and extension source hashes are
cross-checked, and the physical-owner count remains zero.

This join does not promote foot proxies to anatomical colliders, does not
partition shared muscle surfaces into tissue mass, and does not assign
density, material, active force, activation calibration, fat, subject,
standing, recovery or walking authority.

Validation:

- `tests/test_body_composition_extension_join.py`: 3 tests pass.
- The `human body-composition-extension-join` CLI writes an immutable
  canonical receipt and rejects base-owner promotion.
