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
five-minute endurance, clinical plausibility, or real-time performance.

The corrected native run completed 31,000 accepted 2 ms steps (62.000002945 s)
with 11 complete breaths and 72 complete filling/ejection cycles. All 948
coupled samples from the failed replay match exactly. At the former failure
frame, all six respiratory surfaces have byte-identical world positions to the
failed run; the repaired connectivity has no zero cross products. The reciprocal
diaphragm copy of the old collapsed lung triangle is repaired too.

All 970 surface samples have functional-geometry status zero; maximum relative
functional-volume error is 1.703e-6. Maximum reported blood-volume error is
0.018627 mL, and root assistance remains zero. These are numerical checks, not
complete anatomical intersection checks. The known costal, cardiac-wall, and
liver defects remain outside this increment.

Native simulation wall time was 601.322744375 s, giving a real-time factor of
0.103106033. Wrapper wall time was 604.447383124 s. No input changed during the
run. The continuous native movie contains 970 image frames, is 603.03 wall
seconds long, and has a maximum interframe interval of 4.035 wall seconds; it
was not retimed. Its hash and the exact launch are retained in `native/`.

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
