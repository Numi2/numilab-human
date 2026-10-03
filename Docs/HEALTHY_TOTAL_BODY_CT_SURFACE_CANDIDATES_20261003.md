# Healthy Total Body CT source surfaces — 3 October 2026

Numi Human can now emit lossless voxel-boundary PLY surfaces from the pinned
automatic TCIA segmentation masks. These are scan-specific reference geometry
candidates in each source affine's RAS+ frame. The exporter performs no
smoothing, interpolation, decimation or topology repair.

The compiler-bound scan-001 runs now cover all 36 labels observed in that
scan: 12 whole-organ and major-vessel labels, 20 skeletal labels, and four
soft-tissue labels. For every label, the source archive, decompressed NIfTI,
affine, exact intake voxel count and signed mesh occupancy volume were checked.
Independent audits re-read every compressed PLY and checked binary payload
lengths, finite coordinates, index ranges, source envelopes, edge incidences
and occupancy-volume arithmetic. The maximum relative signed-volume
discrepancy is `1.7384093126870536e-14` (Fingers), below the `1e-9`
preregistered arithmetic tolerance.

Fourteen of 36 masks are closed two-manifolds in this discrete mesh
representation: 8/12 organ/vessel, 5/20 skeletal, and 1/4 soft-tissue labels.
The other 22 retain nonmanifold voxel contacts: Adrenal-glands, Kidneys, Liver
and Lung; Carpal, Femur, Metacarpal, Metatarsal, Pelvis, Fingers, Radius,
Ribcage, Scapula, Skull, Spine, Sternum, Tarsal, Tibia and Toes; plus
Skeletal-muscle, Subcutaneous-fat and Torso-fat. Each receipt records the exact
edge-incidence and vertex-link defects. These are source-topology findings;
they remain visible and have not been smoothed away. A closed two-manifold
result describes mesh topology only.

The scan-001 Patella mask has a closed boundary surface (17,860 triangles).
A preregistered scan-local position check compared its anterior-facing surface
with the distal Femur wherever their projected X–Z voxel columns coincide.
Across 1,730 shared columns, Patella was anterior in every column, with a
median lead of `23.4375 mm` (5th–95th percentile `15.625–31.25 mm`). The Tibia
comparison had only 18 shared columns and is descriptive. The
[source-frame plot](media/healthy-total-body-ct-surface-20261003/patella-scan001-source-reference.png)
shows the label surfaces with RAS +Y to the anterior. This checks one external
automatic-segmentation reference only; it does not register, validate or repair
the separate Numi knee model.

The frozen organ-set plan, compiler receipt and independent audit have SHA-256
values `9bfa6e7f459e430dcc504a9e7b81ecb5162cbd5647321afd10d14bfdbb8ae013`,
`9bd2839f1efd08ff2c14cad7ee938b64d68e2656c5841745738ea3b30ec51b0d`, and
`1432a3bad82fc858215b4e8b457df33670569a15a194ad8dc99c6525f1667300`.
The skeletal-set plan, compiler receipt and independent audit have SHA-256
values `b5a3f4a558c47c974d43eae04b5e50620166dd0f9bb7507e581c2faefead9ec4`,
`2ceab1a5d9be07f35e3a29628065f3f443acaba79820089ff47970fb7a6b759c`, and
`108f12d8dc4127c45b64907231b30eee43d2336bbf6706be2326929f83c17268`. The
patella anteriority plan and result have SHA-256 values
`0822a1c405d33d943bdc8574ed79e908f7b32d56997e67165b5b3fe6f45482d8` and
`ff3f27f33d33345103adb3f16c8bf8f92cd30da96851c2aa68ae367d190d705a`.
Compiler source hashes and Python/NumPy versions are bound in each plan and
receipt. The first three-label run is retained alongside a compiler-bound
byte-identical reproduction. The soft-tissue plan, compiler receipt and
independent audit have SHA-256 values
`879a8b544c23c93b267fce4f5edd692e6db8ec62a837ef3d791046f89e9a07d3`,
`4230e6482b3cb506fce07e83a75ed8741e796e3ab1428b767e64e9e299a181fb`, and
`ead1647bf289199afd9e8c7fe70a83c2b2076d5465c1e8c426a56d20166f7a84`.
The consolidated all-label audit is
[`scan-001-all-source-labels-audit.json`](media/healthy-total-body-ct-surface-20261003/scan-001-all-source-labels-audit.json)
with SHA-256 `2cccac715e63c614cfe3b2257f93c6c4c7eb9c3929c6b7a8b244dca69e1f05d1`.

The 12 whole-organ and major-vessel surfaces were then regenerated from scan
001 and extracted from scan 002 using the same compiler revision, intake,
archive, labels, and Python/NumPy/SciPy environment. A new
[independent source-surface audit](media/healthy-total-body-ct-surface-20261003/independent-organ-scans-001-002-audit-v1.json)
re-hashed the registered archive and each decompressed NIfTI member, reparsed
all 24 PLY files, recomputed edge incidence and vertex links, checked every
vertex against its scan-local voxel envelope, and recalculated signed
occupancy volumes. All 24 files passed those integrity and geometry checks;
the largest relative volume discrepancy was 1.0251171986964199e-15. The
independent audit receipt SHA-256 is
98c9f2145a7d5cbdb0939f3aad75e8d705beee8aa5c93e58dfe9b55c093c42c5.

The raw, unsplit topology varies by scan: 8/12 scan-001 surfaces and 4/12
scan-002 surfaces are closed two-manifolds. Scan 001 has nonmanifold
Adrenal-glands, Kidneys, Liver and Lung meshes; scan 002 has Adrenal-glands,
Bladder, Brain, Heart, Kidneys, Liver, VCI and Lung. Only Aorta, Pancreas,
Spleen and Thyroid are closed in both scans. The independent audit reproduces
the raw defects from serialized mesh bytes.

A preregistered scan-001 topology-split candidate now closes all 12
organ/vessel surfaces, including the four defects in the raw controls. The
Liver and Lung each had one four-face saddle where the source-voxel pairing
left a nonmanifold edge; the compiler selected the sole alternate face pairing
that yields a globally closed two-manifold for each mesh. Adrenal-glands and
Kidneys close under the source-owner pairing. The controlled comparison
confirms all triangle counts and triangle-coordinate sequences are unchanged,
surface-area deltas are zero, and the maximum absolute signed-volume delta is
`4.656612873077393e-10 mm3`. The independent occupancy-volume error is at most
`4.975140063085482e-16`. This is vertex-index topology repair only; it does
not test geometric self-intersection or embeddedness, and it does not correct
or qualify the underlying automatic segmentations.

The v4 plan, compiler receipt, independent audit and raw-control comparison
are retained as
[`preregistered-contact-split-scan-001-manifold-plan-v4.json`](media/healthy-total-body-ct-surface-20261003/preregistered-contact-split-scan-001-manifold-plan-v4.json),
[`contact-split-scan-001-manifold-v4/receipt.json`](media/healthy-total-body-ct-surface-20261003/contact-split-scan-001-manifold-v4/receipt.json),
[`independent-contact-split-scan-001-audit-v4.json`](media/healthy-total-body-ct-surface-20261003/independent-contact-split-scan-001-audit-v4.json),
and
[`contact-split-controlled-comparison-scan-001-manifold-v4.json`](media/healthy-total-body-ct-surface-20261003/contact-split-controlled-comparison-scan-001-manifold-v4.json).
The comparison source is
[`contact_split_control_compare_v3.py`](media/healthy-total-body-ct-surface-20261003/contact_split_control_compare_v3.py).
The candidate receipt, independent audit and comparison report SHA-256 values
are `8bc62fbf87a018bee0f5722e428bf7d9a0f89b5ad1981a448493f09df87b0dcd`,
`727882bd1beaf9210fc5a6f6e16d2d4a0147800052a4113c3f70877ab000c2e0`, and
`d33da8345d15ac82deb84885ff37e34cbf1119ab748456f782ba8c1e530ce5d4`.

## Scan-001 lower-limb contact-split candidate

A separate preregistered run applies the same coordinate-preserving contact
split to the scan-001 Femur, Fibula, Patella, and Tibia masks. The raw Femur and
Tibia each had four-face contact edges and nonmanifold vertex links; Fibula and
Patella were already closed. All four candidate meshes are now closed
two-manifolds. The independent source/PLY audit passes all four, and the
controlled comparison against the retained raw meshes confirms unchanged face
counts, identical ordered triangle coordinates, and zero surface-area delta.
Maximum absolute signed-volume delta is `1.0477378964424133e-9 mm3`; maximum
relative candidate occupancy-volume error is `1.1019419408453997e-15`.

The [preregistered plan](media/healthy-total-body-ct-surface-20261003/preregistered-lower-limb-contact-split-scan-001-plan-v1.json),
[candidate receipt and PLYs](media/healthy-total-body-ct-surface-20261003/lower-limb-contact-split-scan-001-v1/receipt.json),
[independent audit](media/healthy-total-body-ct-surface-20261003/independent-lower-limb-contact-split-scan-001-audit-v1.json),
and [raw-control comparison](media/healthy-total-body-ct-surface-20261003/lower-limb-contact-split-controlled-comparison-v1.json)
bind this result. The plan, candidate receipt, audit, comparison, and comparison
script SHA-256 values are `64b6d640886666bd0a8e5f18c78b086faa44d683d67f8c3f16008b515a0224f57`,
`d3b57ab6ceebcfbef76671a27a90c9d811cb7cf19ed3211ada974fdc20fe705d`,
`be32dc937fcce817ac384d64a7b3517f2b659017b65d1f44514916063b817da8`,
`2f3c447199f6158e0c9d6cd8836cd20ff452be4e003b4f24e7a92b671e656001`, and
`3a27850eef94c7f8d3107a56a34e33cd1e5ce4ca36ed2364d0dad4414331bc66`.

The split only duplicates vertex indices at voxel-contact fans. These remain
scan-specific candidates from an automatic segmentation; neither topology nor
anterior surface ordering establishes expert segmentation accuracy, subject
binding, clinical anatomy, patellar tracking, physical tissue ownership,
mechanics, or physiology.

These are topology-only candidates for external automatic segmentations in
each scan's own coordinate frame. The scans are not registered to Numi's
mechanical subject; expert segmentation accuracy, tissue ownership, mechanics,
physiology and clinical anatomy remain unqualified.

The reusable verifier is available as the numi human
healthy-total-body-ct-surface-audit command. The immutable scan plans, compiler
receipts, meshes, checksums and independent audit are in
[media/healthy-total-body-ct-surface-20261003](media/healthy-total-body-ct-surface-20261003/).
Neither scan is registered to Numi's mechanical subject, and the automatic
segmentations remain unreviewed; no tissue, mechanical, physiological or
clinical qualification follows.

The artifacts are under
[`media/healthy-total-body-ct-surface-20261003/`](media/healthy-total-body-ct-surface-20261003/).
Install the CPU mesh-audit dependencies with `pip install -e '.[volume-surface]'`.
The export command is available through `numi human
healthy-total-body-ct-surface`. The 30-scan CT cohort is separate from Numi's
mechanical subject, and its MOOSE segmentations are automatic rather than
expert reviewed. They do not qualify common participant coordinates, tissue
accuracy, physical tissue volumes, mass, subject binding, mechanics, physiology
or clinical use. Subcutaneous fat does not provide a skin layer; the one
Skeletal-muscle label does not identify individual muscles, tendons or fascia;
the vessel masks do not distinguish lumen from wall or provide flow
connectivity; the Heart label has no electrical activity or chamber mechanics.
