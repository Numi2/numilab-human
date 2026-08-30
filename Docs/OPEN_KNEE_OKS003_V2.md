# Open Knee(s) oks003 bilateral anatomical payload v2

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
[`left manifest`](media/open-knee-oks003-v2/open-knee-oks003-left.manifest.json)
and
[`mirrored-right manifest`](media/open-knee-oks003-v2/open-knee-oks003-right-mirrored.manifest.json).

## Anatomical registration

The Open Knee femoral frame is mapped to the live MyoSim left-knee frame using:

1. `Xf` to the flexion-axis line, `Zf` to the proximal axis, then resolve the
   otherwise ambiguous flexion-axis sign by requiring source `Yf` to align
   with Human anterior (`-world Y`);
2. one uniform condylar-width scale (`0.9842157677`);
3. `FMO` to the live knee origin plus a bounded translation-only distal-surface
   refinement (`6.496 mm`, below the `35 mm` ceiling).

No reflection, anisotropic warp, or extra joint is admitted in the left
registration. Proper-rotation
determinant is `1.0000000000000002`. Held-out distal-femur surface distance was
mean `5.695 mm`, median `4.879 mm`, and p90 `9.243 mm` against the `20 mm`
gate. The `25.476 mm` maximum is retained in the manifest: this is an exact
specimen registered to a different MyoSim specimen, not a subject-matched or
clinical reconstruction.

The old proper rotation used the opposite sign of the same flexion-axis line.
Because the distal femur is approximately symmetric, its surface gate still
passed even though the patella was posterior and the fibula medial. That
payload and its receipts are retained only as
[`rejected axial-sign evidence`](media/rejected/open-knee-oks003-v1-axial-sign/).
V2 fails closed unless anterior alignment is at least `0.999`, patellar-bone
centroid is at least `25 mm` anterior, and fibular-bone centroid is at least
`20 mm` lateral. Both sides measure `0.999999366`, `46.934 mm`, and
`28.565 mm` respectively.

The right payload mirrors the qualified left world registration once across
the measured bilateral femur midplane (`x = -0.0250448 m`) and then converts
the result into the live right femur/tibia/patella frames.  Bilateral frame
symmetry error is 0.124 mm.  This is an inferred right-side counterpart, not
an independently segmented right subject.

Payload SHA-256 values are:

- left: `035062d9dd4ad0e283c86181fe9edbcfee63fe79e5cda12c9769866d6de6752b`
- mirrored right: `8bfc6949e20c59b50590bb36f5a43ff8598f2b53fa7fe90be8de7310d9a5c0bf`

The mirrored payload reverses the connectivity parity of each `tet4` and
`tri3` after reflecting its node positions. This preserves positive volume and
surface orientation while leaving source topology and attachment membership
unchanged. The original right payload omitted that parity correction and is
retired because its reflected tetrahedra were inverted.

## Build

```sh
numilab-human open-knee-oks003-payload \
  --sources Sources \
  --open-knee Sources/open-knee-oks003 \
  --registration Build/fullbody-articular-v3.registration.json \
  --output Build/open-knee-anatomical-v2-left \
  --side left

numilab-human open-knee-oks003-payload \
  --sources Sources \
  --open-knee Sources/open-knee-oks003 \
  --registration Build/fullbody-articular-v3.registration.json \
  --output Build/open-knee-anatomical-v2-right \
  --side right
```

The command fails closed on source identity, region counts, attachment/contact
topology, anatomical axes, live body indices, proper rotation, anterior
alignment, patella/fibula placement, scale and held-out p90 placement.

## Evidence boundary and next mechanics gate

The M4 Pro neutral review covered front, oblique, side and rear views on both
sides. Front exposes the patella and extensor mechanism; rear exposes the
cruciates. Five exact tetrahedral tissues (ACL, PCL, MCL, LCL and PTL) were
requalified as one 47,439-node, 195,032-tetrahedron, three-reaction-owner
Matter transaction. Both sides retained rejected-step rollback and bitwise
replay. The accepted frames and raw transcripts are in
[`open-knee-oks003-v2`](media/open-knee-oks003-v2/).

This is exact geometry/topology/contact metadata, bounded neutral placement,
and an exact-topology isotropic-matrix FEM preflight. It is not loaded flexion,
cartilage contact qualification, source transverse-isotropic fibre mechanics,
initial prestretch, subject matching, or clinical validation. Flexed tissue
images remain rejected until accepted deformable nodes own every spanning
surface in that pose.
