# Subject-scaled native runtime input

The subject mass candidate can now be lowered into the published `NHRIGID2`
binary format for native testing. The compiler verifies the exact 157-engine-
body layout, the 103 source-to-core map, and every source mass/inertia row
before changing only mass, inverse mass, inertia, and inverse inertia.

Generate the immutable binary and receipt with:

```sh
.numi/commands/human-subject-scaled-runtime-input \
  --output Docs/media/subject-scaled-runtime-input-20260915/myosim-fullbody-core-reference.nhrigid \
  --receipt Docs/media/subject-scaled-runtime-input-20260915/receipt-v1.json
```

The source identity, body topology, poses, joints, and source map remain
unchanged. The resulting binary closes the 65.5 kg scalar mass target under the
explicit uniform fixed-geometry assumption. It is a native test input, not a
qualified subject model: segment composition, geometry, inertias, activation,
organ/blood/tissue/fat/muscle ownership, standing, recovery, and walking remain
open until the Mac mini runtime consumes it and held-out comparisons pass.

The subsequent [native requalification](SUBJECT_SCALED_NATIVE_REQUALIFICATION_20260915.md)
consumes this exact binary on the physical Apple M4 Pro. That result is a
bounded replay record and keeps the calibration and sustained-behavior gates
closed.
