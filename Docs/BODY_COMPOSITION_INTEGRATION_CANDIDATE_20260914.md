# Body composition integration candidate

The `HumanPack.body-composition-integration-candidate.v1` compiler joins the
current one-adult-male source package at the boundary between organ identity,
blood transport, oxygen exchange, cardiac blood budgeting, muscle activation,
and surface identity. It is available through:

```sh
numi human body-composition-integration --output /private/tmp/body-composition-integration.json
```

The candidate is source-bound and hash records the exact upstream files. It
checks that the current graph contains:

- 378 unique organ members across the 18-region mass inventory, with 342
  single-closed candidates and eight shared members retained as unresolved;
- 329 unique members assigned to the seven exact-clock blood-transport beds,
  with 511 accepted and one rejected `12,500 ns` step and neutral rollback;
- 150 distinct muscle/tendon surface identities covering the current surface
  rows and all 416 activation route identities, while retaining 238 routes
  without a surface binding;
- the hash-locked BodyParts3D source-geometry audit for those 150 surfaces,
  including 3.2682904751 m² of recomputed surface area, 60 single-closed
  algebraic volume candidates, six closed multi-component surfaces, and 84
  topology-defective surfaces. None is promoted to physical volume, mass,
  material, or active-force ownership;
- the exact BodyParts3D `FJ2810` full-skin source and its registered visual
  outer shell (54,949 retained vertices, 109,183 triangles, 86 body
  influences, and sub-micrometre rest-pose reconstruction), plus the native
  Apple M4 Pro four-view visual admission (29,661 / 25,803 / 18,454 / 30,666
  positive shell pixels);
- the exact bilateral foot contact handoff: 60 source mesh rows, 30 reviewed
  source members, seven-pose continuity, 18 support witnesses, and six
  active static witnesses, while retaining collider and dynamic-loading gates;
- the same seven beds and clock in the bidirectional oxygen-exchange candidate;
- the exact CVSim21 21-compartment aggregate blood owner, which conserves
  `5.459 kg` and `5.150 L` through 511 accepted and one rejected steps;
- six corrected source/world vessel registrations whose source surface-volume
  candidates are inside the organ inventory and disjoint from the regional
  blood-bed member partition;
- the Rodero case-18 cardiac-wall manifest and source config, binding 24
  positive-oriented region identities across the 1,470,083-cell source mesh
  while retaining imported boundary defects, closure faces, unloaded-reference,
  density, pressure-port, native-mechanics, and subject-calibration gaps;
- the pinned Open Knee tissue-calibration candidate for one `oks003` PTC plug,
  with six training and three same-plug held-out observations, while retaining
  native solver validation, stress-free reference, population, and whole-human
  material gates as false;
- both cardiac geometry conventions bound to the common `0.39974235600733804
  kg` hydraulic blood-mass budget without selecting a physical owner.

The resulting receipt is a graph and ownership certificate. It does not add
candidate masses to rigid dynamics, create anatomical blood or lumen volume,
assign skeletal-muscle, fat, skin, tendon, or organ mechanical volume, or
claim material or subject calibration. The skin shell is a visual registration
input only; its thickness, material, collision, self-contact, and deformation
remain open. The source receipt hashes, counts, and owner-null assertions make
this join fail closed if an upstream identity, route, bed, clock, or ownership
boundary changes. It also emits stable digests for the organ, blood-member,
muscle/tendon surface, and skin-shell identity sets, along with the
subset/disjointness checks used to construct the join.

The CVSim21 mass owner is included as a conserved aggregate source owner only;
its explicit density remains an engineering candidate and it is not promoted
to an anatomical vessel/lumen, tissue, or rigid-body mechanical owner.
The six vessel rows remain registered source surfaces: they have no lumen area,
tubular field, calibrated density, blood-mass owner, or tissue-exchange owner.

The cardiac-wall rows are source labels and geometric-volume candidates only.
The manifest and source config are required to carry the same SHA-256, and the
source mesh is recorded as 24 labels, 300,965 points, 1,470,083 tetrahedra,
and zero negative orientations. The imported wall does not acquire a physical
volume or mechanical owner from this join.

The tissue calibration row is a provenance link, not a material assignment:
its `qualified` flag remains false, its native material is not solver
validated, and its held-out repeat is a test-day split within the same plug.

The candidate therefore advances the organ, systemic-physiology, and muscle
rows from disconnected source evidence to one reproducible integration
boundary while preserving the completion gates for calibrated mechanics,
anatomical blood/tissue transfer, standing, recovery, and walking.

The immutable `v7` receipt additionally binds the native-compatible full-skin
candidate and its four-view M4 Pro visual admission. It records the one-plug
material fit and held-out observation counts and keeps `material_calibration`
false; it also binds the muscle-surface geometry audit. No fitted coefficient
or algebraic geometry candidate is assigned to the Human body or native Matter
runtime.
