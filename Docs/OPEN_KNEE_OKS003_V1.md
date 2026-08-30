# Open Knee(s) oks003 bilateral payload v1

NumiLab Human now has an exact-source left-knee anatomy payload and an
explicitly labelled mirrored right-knee payload compiled from Open Knee(s)
subject `oks003`. They replace visual guesses at both knees with one coherent
specimen topology containing bones, articular cartilage, menisci,
cruciate/collateral ligaments, patellar tendon and quadriceps tendon.

## Source and scope

- Source: Open Knee(s) oks003, DOI `10.18735/b0zv-n395`, CC BY 4.0.
- Subject metadata: left knee, female, 25 years, 1.73 m, 68 kg.
- Exact topology: 16 regions, 248,236 nodes, 844,287 tetrahedra, 88 named
  surfaces, 729,068 surface faces, 42 node sets and 19 surface pairs.
- Left rigid attachment sets bind 11,350 nodes to femur body 145, 12,161 nodes
  to tibia body 150 and 5,394 nodes to patella body 156.  The right payload
  binds the same exact node sets to bodies 131, 136 and 142.
- The compiler preserves exact region and contact names and writes them to the
  `NHKNEE1` ABI. It does not replace them with generated primitives.

The hashes and full receipts are retained in the
[`left manifest`](media/open-knee-oks003-v1/open-knee-oks003-left.manifest.json)
and
[`mirrored-right manifest`](media/open-knee-oks003-v1/open-knee-oks003-right-mirrored.manifest.json).

## Anatomical registration

The Open Knee femoral frame is mapped to the live MyoSim left-knee frame using:

1. `Xf` to the flexion axis, `Zf` to the proximal axis, and the resulting
   proper anatomical basis for rotation;
2. one uniform condylar-width scale (`0.9842157677`);
3. `FMO` to the live knee origin plus a bounded translation-only distal-surface
   refinement (`12.863 mm`, below the `35 mm` ceiling).

No reflection, anisotropic warp, or extra joint is admitted in the left
registration. Proper-rotation
determinant is `1.0000000000000002`. Held-out distal-femur surface distance was
mean `9.151 mm`, median `6.786 mm`, and p90 `18.951 mm` against the `20 mm`
gate. The `42.391 mm` maximum is retained in the manifest: this is an exact
specimen registered to a different MyoSim specimen, not a subject-matched or
clinical reconstruction.

The right payload mirrors the qualified left world registration once across
the measured bilateral femur midplane (`x = -0.0250448 m`) and then converts
the result into the live right femur/tibia/patella frames.  Bilateral frame
symmetry error is 0.124 mm.  This is an inferred right-side counterpart, not
an independently segmented right subject.

Payload SHA-256 values are:

- left: `1078198d02b9f0902a528799bbe2dd08ed1faeb701d3ab537b0dc59f065f8e2a`
- mirrored right: `ba4a86da89dbad228bf4f16dfb114c461532fd40b9180e6c844764af5b2ec077`

## Build

```sh
numilab-human open-knee-oks003-payload \
  --sources Sources \
  --open-knee Sources/open-knee-oks003 \
  --registration Build/coherent-body-v4.registration.json \
  --output Build/open-knee-bilateral-v1 \
  --side left

numilab-human open-knee-oks003-payload \
  --sources Sources \
  --open-knee Sources/open-knee-oks003 \
  --registration Build/coherent-body-v4.registration.json \
  --output Build/open-knee-bilateral-v1 \
  --side right
```

The command fails closed on source identity, region counts, attachment/contact
topology, anatomical axes, live body indices, proper rotation, scale and
held-out p90 placement.

## Evidence boundary and next mechanics gate

The M4 Pro neutral review covered front, oblique, side and rear views.  A
right-knee flexion check at q106 = 0.75 rad confirms that the tibia moves
posteriorly under the femur rather than bending in the reversed direction.
The same check also makes the remaining mechanics gap visible: ligament and
tendon surfaces currently follow one owning rigid body and detach in flexion.
They are therefore not showcase or deformation evidence.

This is exact geometry/topology/contact metadata and bounded placement. It is
not yet a coarsened Apple FEM solve, loaded cartilage contact, or deformable
ligament validation. The next knee mechanics gate is multi-body attachment and
deformation of spanning tissues while preserving the 42 attachment sets and
19 contact pairs; the exact source payload remains the reference used to
measure coarsening and contact error.
