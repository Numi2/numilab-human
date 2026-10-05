# Respiratory source connectivity repair

The native integrated scene stopped at accepted step 30335 (60.6700028817 s)
when lobe 305 face 403 became exactly collinear in its submitted Float32
positions. Replaying the same frozen runtime and inputs reproduced the failure
and the coupled/surface CSVs exactly. Its source altitude was 2.68 nm; this was
an actual geometric collapse, not merely cancellation in a Float32 cross
product. The failed run and capture remain on the SSH Mac mini.

`resting_respiratory_mesh_quality.improve_sliver_faces` performs optional local
diagonal flips after conforming-cell preparation and the shared short-edge
quotient. A flip must stay inside one affine respiratory cell, preserve face
orientation and anatomical interface owners, and occur in every reciprocal
copy. It moves no source vertex. Both replacement faces retain both parent
identities; patch remapping now removes duplicate child IDs.

The retained six-surface preparation performs 38 flips. All six surfaces remain
closed and oriented with the same vertex/face counts. The largest absolute
signed-volume difference is 4.14e-14 m³. The changed faces have no forbidden
exact self or cross-surface intersections. Reindexing the actual failed GPU
frame with the new connectivity removes every zero-area face in the six
respiratory surfaces. This diagnostic is distinct from a new native run.

Thirteen focused tests passed on the SSH Mac mini: eight conforming/refinement
tests and five mesh-quality tests, including the retained native failure.
The first full builder attempt failed because multi-parent lineage duplicated
patch face IDs; `remap_ids` now deduplicates them. The first native launch request
was rejected before simulation because capture 2999 did not match its submission
cadence; the corrected request uses 3007. Both failures are retained.

The source conditioning leaves some faces below the 0.25 µm altitude target.
This increment does not qualify every respiratory pose, anatomical clearance,
five-minute endurance, clinical plausibility, or real-time performance. Native
validation is recorded separately below when complete.

All builds, tests, source calculations, audits, and native execution ran through
`ssh macmini` on the available M4 Pro. The Air was used only for source editing,
git, and transfer of compact evidence.

The large assets, accepted MRV captures, movies, complete traces, and preparation
lineage maps remain under `/Users/n/numi-human-resting-evidence-20261005`:

- `integrated-stability-011` and `integrated-stability-failure-replay-011`:
  original failure and exact replay.
- `respiratory-mesh-quality-001`: source/failed-frame diagnostics and exact audits.
- `thorax-conforming-field-010`: failed builder with frozen failing source.
- `thorax-conforming-field-011`: complete six-surface conditioned preparation.
- `respiratory-quality-composed-native-inputs-001`: conditioned respiratory rows
  and regenerated pleura in the current integrated scene; all other surface
  content remains byte-identical to the retained passive-neighbor composition.
- `respiratory-quality-native-001`: rejected capture-cadence request.
- `respiratory-quality-native-002`: corrected native regression run.

The compact files here retain exact input/output identities and the failure
fixture. Source rights and mixed-source reference attribution remain in the
existing anatomy receipts; the connectivity repair is not a new measurement.
