# Deformable pectoralis fascia mechanics v2

## Outcome

`NHFASC2` closes two defects in the regional pectoralis-fascia path:

- Human tendon transfer, Matter load assembly, implicit FEM dynamics, and
  commit or rollback now execute in the same owning Metal command buffer for
  every Human step.
- The payload now carries the exact high-resolution anterior triangles chosen
  by the compiler. The renderer consumes those indices directly instead of
  displaying the entire closed pectoralis muscle volume as fascia or guessing
  an anterior cap at runtime.

The six named regions retain 326 FEM nodes, 471 positive tetrahedra, 68 fixed
nodes, and 68 traction nodes. The exact presentation boundary contains 6,601
referenced BodyParts3D vertices and 8,449 triangles. BodyParts3D does not
contain a separately segmented pectoral fascia, so this remains an explicit
source-derived mechanics fallback rather than an anatomical segmentation.

## Payload contract

The compiler uses the pinned BodyParts3D pectoralis-major parts and preserves
the six MyoSim actuator identities. For each region it writes:

- the thin-solid FEM envelope and nodal masses;
- medial anchor and lateral traction flags;
- the exact source-vertex indices of every selected anterior presentation
  triangle; and
- the pinned BodyParts3D archive and MyoSim manifest hashes.

The presentation records are part of the binary ABI. The native loader checks
every region and source index before simulation. This removes the former
runtime normal/centroid heuristic and prevents posterior and inferior faces of
the closed muscle volume from being mislabeled as fascia.

```bash
PYTHONPATH=src python3 -m numilab_human.cli \
  numi-human-pectoralis-fascia-payload \
  --sources sources \
  --artifact Build/myosim-fullbody \
  --output Build/pectoralis-fascia-v2
```

The generated qualification payload has SHA-256
`7f820ad06427cf3293df38471b0ee986b44e8b34d08e3234c0a30c9bd255d6ee`.

## Native transaction

The Numi Matter adapter reads the borrowed environment-major NHTENDON2/3
terminal-force buffer directly on Metal. It assembles the admitted 10% share
into FEM nodal loads, translates the Human step status, and encodes Matter
pre-dynamics and post-commit phases without committing or waiting. Consumer
rejection cancels the pending Matter transaction before the Human command
buffer is abandoned.

MyoSim `J^T` remains the only rigid generalized-force owner. The fascia path
is deliberately configured with production force ownership equal to zero;
adding the same endpoint share to both FEM and rigid dynamics is forbidden.
Two-way production ownership requires replacing the matching `J^T` share and
returning accepted anchor reactions.

## Apple M4 Pro evidence

The one-step focused smoke used the current cumulative bone and soft-tissue
registration, all 416 source routes, NHTENDON2, and NHEQ1. It reported:

- 832 tendon endpoint transfers;
- 6.156 N admitted across the six fascia regions;
- 0.0841 mm maximum FEM displacement;
- minimum determinant `J = 0.99293`;
- exact rollback after injected downstream rejection; and
- bitwise Human and FEM replay.

The explicit anterior topology reduced the worst presentation-to-mechanics
mapping distance from the invalid full-volume value above 180 mm to 48.965 mm.
All 6,601 referenced presentation vertices receive bounded displacement. This
still exposes the coarse regional mechanics envelope: it is not a dense
surface FEM or a validated subject-specific fascia model.

An eight-step selected-pectoralis stress run admitted 49.363 N, reached
10.888 mm maximum FEM displacement and minimum `J = 0.55658`, converged in 20
FGMRES iterations, and retained bitwise replay plus exact rejection rollback.
All four 1024-pixel views had nonzero fascia coverage without posterior bleed,
mesh inversion, or visible tearing. The side and oblique views also show that
this loading lifts the coarse shell visibly away from the ribs. The run is a
transaction and large-deformation stress certificate, not the visual or
physiological reference. The one-step smoke above is the bounded presentation
reference until rib/skin contact and a denser conforming envelope are added.

## Evidence boundary

The GOH constants remain a mean uniaxial fit from published human
pectoralis-major fascia data. Thickness, near-incompressibility, medial
fixation, traction band, and the 10% load share are declared assumptions.
Current evidence establishes executable Apple-native one-way regional load
transfer, accepted-state deformation, rollback, replay, and exact presentation
topology. It does not establish calibrated two-way muscle-fascia-bone force
redistribution, contact with ribs or skin, whole-body fascia, or clinical
validity.
