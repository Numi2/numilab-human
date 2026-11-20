# Native muscle registration and clearance checks

The shared-tissue Apple M6 run pinned in commit `b302ab5b8` accepted 16
roots through 0.250 ms. Its terminal audit found zero skin self-pairs,
zero skin-to-bone pairs and 14,179 nonocular skin-to-structure pairs.
Muscles contributed 13,037. This project tests geometric repairs without
changing collision constraints, material state, or physical tolerances.

## Source and paired geometry

The selected source is the actual ABI 5 NHTISS4 payload from the native
launch receipt. Its SHA-256 is
`8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042`.
The shared-tissue root 16 capture is the primary reference. The earlier
regional variable-pleural root 8000 is an independent geometry screen. Its
mechanical shell differs from the full registered skin/fat model. These
isolated frames cannot certify continuous co-motion.

All 91 source banks represented among the muscle witnesses reproduce the
shared model's posed mesh within 0.12 mm. Eighty-four also satisfy that
threshold in the regional capture. The maximum error on the largest
forearm extensor and external oblique meshes is 0.259 micrometres across
both. The extensor blend determinant reaches 0.225. The obliques stay near
1.023, so small rotational determinant correction cannot explain their
shared-model-only intersections. This is a geometric comparison and the
blend determinant is not a tissue constitutive Jacobian.

The rigid dual-quaternion test admitted two of ten inspected banks
under its same-scale and registration rules. One failed orientation
and the other had unchanged crossing counts. The dominant extensor
banks have calibrated per-binding scales and were excluded from this
particular rotation test.

## Results and disposition

| Method and bank | Shared root 16 | Regional root 8000 | Disposition |
| --- | ---: | ---: | --- |
| Source fit, right external oblique 67 | 722 → 659 | 0 → 0 | No new self or surrounding pairs in either static capture |
| Source fit, left external oblique 68 | 885 → 833 | 0 → 0 | Three new surrounding pairs in each pose |
| Source fit, left brachioradialis 110 | 744 → 736 | 404 → 404 | Added 65 and 90 surrounding pairs |
| Source fit, temporalis 164 | 466 → 450 | 0 → 0 | No new self or surrounding pairs in either static capture |
| Area-constrained right extensor 137 | 1,794 → 1,689 | 475 → 472 | Added posed self and surrounding pairs |
| Area-constrained left extensor 138 | 592 → 574 | 782 → 773 | Added posed self and surrounding pairs |
| Self-contact-constrained right extensor 137 | 1,794 → 1,733 | 475 → 467 | Zero added self pairs, added 14 and 16 surrounding pairs |

Values are raw integer triangle-pair witnesses at the captured Float32
coordinates with modelled source deltas. Lower total counts can include
new pairs in an already affected skin ring. The two source-fit banks
that passed the surrounding check still cross skin and need native
admission, material attachment checks and motion qualification.
The self-contact-constrained left extensor and the subsequent seed of
right-extensor neighboring barriers yielded no candidate passing even
the preceding skin/self screen. The live tissue payload was not modified.

The right extensor and its neighboring bones
and muscles now have exact blocking witnesses. A smaller aggregate skin
count cannot substitute for preserving those interfaces. A coupled
registration repair should protect the bone envelope before fitting the
muscle, preserve the authored route attachment sites, and rebuild and
certify the fascia bindings via the native owner.

## Reproduce on SSH Numi

Place these scripts under the retained native case directory as
`muscle-shell-registration-018` beside the dependency case directories.
Verify the pinned helper hashes. The native capture receipts and raw pack
hashes are verified during extraction. The paired NumPy archive contains
source and captured anatomical geometry and stays on SSH Numi.

Run the following with the retained preparation Python environment and
`OPENBLAS_NUM_THREADS=1`. Each generated solver is the exact version used
in the corresponding experiment.

```sh
python3 prepare_geometry.py
python3 check_rotation_transport.py
python3 fit_surface_clearance.py targeted-first 137 138 110 67 68 73 74 76 120 164
python3 check_surrounding.py targeted-first
patch -s -o fit_orientation_constrained.py fit_surface_clearance.py < orientation-constrained.diff
python3 fit_orientation_constrained.py area-extensors 137 138
python3 check_surrounding.py area-extensors
patch -s -o fit_contact_constrained.py fit_orientation_constrained.py < self-contact-constrained.diff
python3 fit_contact_constrained.py contact-extensors 137 138
python3 check_surrounding.py contact-extensors --emit-neighbor-witnesses
patch -s -o fit_neighbor_constrained.py fit_contact_constrained.py < neighbor-constrained.diff
python3 fit_neighbor_constrained.py --neighbor-witnesses contact-extensors-neighbor-witnesses.npz neighbor-extensor 137
```

The area gradient matched centered differences to 1.01e-11 on the
deterministic projected test. The separation constraint passed synthetic
crossing, separated and disjoint coplanar checks. Its fixed-neighbor
specialization gave identical coefficients to the same system with
the second surface pinned. All three experiment diffs reproduced their
Python sources byte for byte.

## Full-tissue GPU profile with matched physics

An instrumented headless replay of three roots accepted all three on SSH
Numi. Each of 77 fields in each accepted trace row matched the retained
native run exactly. The variable-pleural path remained active and the
three steps were too short to resolve a pressure change. No presentation
was included. The observed native throughput window from the first to
the third accepted state was 0.000001339×. The complete invocation took
187.393 s and covered 0.046875 ms of simulation.

Root 1 device-counter stage totals were:

| Stage | GPU time |
| --- | ---: |
| Linear solver | 9,901.998 ms |
| Newton assembly | 1,515.167 ms |
| Globalization | 576.980 ms |
| Fresh final certificate | 273.741 ms |

Linear work accounted for 80.7% of these four stage totals.
This root used 240 Krylov columns in 15 active restart cycles, with
four solves exhausting a budget. The counter profile splits compute
encoders and is a one-root observation. It provides no steady-state
speedup claim, and it does not resolve the split among operator,
preconditioner and orthogonalization costs inside the linear stage.

```sh
python3 profile_matched_roots.py profile-root-1
python3 check_profile_matched_roots.py profile-root-1
```
