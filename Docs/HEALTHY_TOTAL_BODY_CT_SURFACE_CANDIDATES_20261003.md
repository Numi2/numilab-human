# Healthy Total Body CT source surfaces — 3 October 2026

Numi Human can now emit lossless voxel-boundary PLY surfaces from the pinned
automatic TCIA segmentation masks. These are scan-specific reference geometry
candidates in each source affine's RAS+ frame. The exporter performs no
smoothing, interpolation, decimation or topology repair.

The compiler-bound scan-001 runs cover 12 whole-organ and major-vessel labels
and all 20 source skeletal labels. For every selected label, source archive,
decompressed NIfTI, affine, exact intake voxel count and signed mesh occupancy
volume were checked. Independent audits re-read every compressed PLY, checked
binary payload lengths, finite coordinates, index ranges, source envelopes,
edge incidences and occupancy-volume arithmetic. All selected signed volumes
close to the affine-scaled source voxel counts within `1.1e-15` relative error
or better.

Eight of 12 organ/vessel masks and five of 20 bone masks are closed
two-manifolds in this discrete mesh representation. The other masks retain
nonmanifold voxel contacts: adrenal glands, kidneys, liver and lung among the
organ/vessel labels, and 15 skeletal labels. Each receipt records the exact
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
byte-identical reproduction.

The artifacts are under
[`media/healthy-total-body-ct-surface-20261003/`](media/healthy-total-body-ct-surface-20261003/).
The export command is available through `numi human
healthy-total-body-ct-surface`. The 30-scan CT cohort is separate from Numi's
mechanical subject, and its MOOSE segmentations are automatic rather than
expert reviewed. They do not qualify common participant coordinates, tissue
accuracy, physical tissue volumes, mass, subject binding, mechanics, physiology
or clinical use. Subcutaneous fat does not provide a skin layer; the one
Skeletal-muscle label does not identify individual muscles, tendons or fascia;
the vessel masks do not distinguish lumen from wall or provide flow
connectivity; the Heart label has no electrical activity or chamber mechanics.
