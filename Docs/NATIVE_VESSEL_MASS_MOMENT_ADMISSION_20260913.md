# Native vessel mass-moment admission — 13 September 2026

The physical Mac mini now has a native Matter checker for the corrected
six-vessel source-surface mass-moment candidate. The checker consumes the
candidate, corrected source/world registration, and exact vessel body-link
receipts. It recomputes each row's density-scaled mass, first and central/raw
second moments, initial-velocity momentum, aggregate totals, source-member
hashes, and single-owner count. It also requires the unresolved lumen,
pressure-transfer, tissue-exchange, material, calibration, and behavior fields
to remain fail-closed.

Evidence:

- Host: `ssh macmini`, Apple M4 Pro.
- Isolated native branch: `human-blood-moment-owner-20260913` at
  `c875a0b8da4cf050a2e0132e03681fd78ab14d54`, clean after publication.
- Native binary:
  `/Users/n/MetalRobo-blood-moment-owner-build-20260913/matter/numi-matter-vessel-mass-moment-check`.
- Input candidate SHA-256:
  `2ac61a232c5ba2e91311428067ea6c57644f098a45290ef6570e85d62faa0c40`.
- Result:
  `numi_matter_vessel_mass_moment=pass vessels=6 source_identity=pass body_link_identity=pass moments=pass ownership=single surface_proxy=pass mechanics=unqualified calibration=unqualified`.
- The unchanged native vessel registration/body-link CTest selection passes `2/2`
  on the same isolated build.
- A tampered first-row mass is rejected with `exit=1` and
  `numi_matter_vessel_mass_moment=fail error="mass disagrees"`.

The retained output, CTest result, tamper result, input hashes, and manifest are under
[`media/native-vessel-mass-moment-corrected-20260913/`](media/native-vessel-mass-moment-corrected-20260913/).

This closes native admission of the deterministic source-surface candidate.
It does not create an anatomical lumen or tube, wall material, physical blood
owner, pressure-driven transfer, tissue exchange, or subject calibration.
