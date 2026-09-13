# Native vessel source-registration admission — 13 September 2026

The physical Mac mini native Matter owner now consumes the same six-vessel
registration fixture emitted by the Human source compiler.  The focused native
check validates the schema/compiler identity, all six exact BodyParts3D member
and region identities, source membership hashes, positive source/world moment
values, and the pinned source-to-MyoSim world transform.

Native evidence:

- Host: `ssh macmini`, Apple M4 Pro, Matter ABI38.
- Branch and revision: `human-blood-mass-20260913`, `c8222041ce04ee84d65cb4d1a52a6b39dd489427`.
- Fixture: `matter/tools/fixtures/human_organ_vessel_registration_20260913.json`.
- Binary: `/Users/n/MetalRobo-blood-mass-build-20260913/matter/numi-matter-vessel-registration-check`.
- Focused selection: 8/8 CTest targets passed, including the fullbody bridge,
  fullbody vascular admission, compiler, Metal, Human binding, and cavity
  checks.
- Native stdout: `numi_matter_vessel_registration=pass vessels=6
  source_membership=pass source_moments=pass source_to_world=pass
  mechanics=unqualified calibration=unqualified`.

The check is deliberately bounded.  It admits source-to-world frame
bookkeeping only.  It does not infer a vessel tube, lumen or centreline,
cross-sectional area, wall material, density, blood mass owner, pressure
gradient, body-link mechanics, tissue exchange, or subject calibration.  The
Human organ/blood mechanics and loading rows therefore remain open.

The follow-up native checker at `human-blood-mass-20260913` commit
`f239c631` also consumes the exact body-link receipt and passes
`matter.compiler.vessel_registration` plus
`matter.compiler.vessel_body_links` (`2/2`). This validates named source/core
body-frame bookkeeping only; it does not promote body-link mechanics or a
vessel physical owner.

The retained logs and patch are under
`Docs/media/native-vessel-registration-20260913/`; the immutable receipt is
`Docs/media/native-vessel-registration-20260913/receipt.json`.
