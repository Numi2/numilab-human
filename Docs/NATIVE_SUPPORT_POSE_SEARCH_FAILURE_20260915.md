# Native support pose search failure — 15 September 2026

The isolated `human-completion-static-preload` build on the physical Mac mini
ran the source support geometry and pose search at the canonical 12.5 µs clock.
The geometry phase admitted its tolerance (`1.06834399844e-7 m` minimum gap)
and retained eight separated witnesses, but rejected 88 pose candidates. The
native run then failed closed with `whole-body unilateral support wrench did not
close`.

This is an anatomical active-set blocker. It does not establish loaded foot
contact or body-weight support, and it cannot be promoted to standing,
recovery, walking, force convergence, activation calibration, or material
calibration. The raw stdout/stderr and immutable receipt retain the exact
source commit, binary hash and payload identities for the next contact solve.

Receipt: `Docs/media/native-support-pose-search-20260915/receipt-v1.json`.
