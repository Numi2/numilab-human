# Thoracoabdominal myofascia mechanics v4

## Outcome

`NHFASC4` extends the live same-command-buffer Human/FEM owner from the six
pectoralis and six latissimus regions to all fourteen remaining Abdomen-body
external- and internal-oblique terminals. The payload now contains 26 driven
regions, 1,202 nodes, 2,127 positive tetrahedra, 212 moving attachment nodes,
and 212 traction nodes.

Every abdominal terminal remains at its exact pinned MyoSim local coordinate.
The bilateral coordinates must mirror, and adjacent source-lattice cells must
share their y boundary exactly. The runtime replaces, rather than duplicates,
the declared 10% anchor-side `J^T` share and projects the accepted prescribed-
node reaction through the owning body Jacobian.

## Anatomy and evidence boundary

BodyParts3D supplies exact bilateral external-oblique surfaces (`FJ1452` and
`FJ1452M`). It does not supply a separately named internal-oblique surface in
the pinned English hierarchy. Z-Anatomy is derived from BodyParts3D and did not
provide a demonstrated superior accessible segmentation for this missing
layer. The mechanics compiler therefore uses the exact MyoSim Abdomen-terminal
lattices for both layers, with published population thickness proxies of
4.5 mm for external oblique and 6.0 mm for internal oblique.

The rectangular terminal-lattice cells are efficient mechanics envelopes, not
anatomical presentation geometry. They remain live in Matter but are hidden by
the anatomical renderer. Visible external oblique uses the exact BodyParts3D
surface. Internal oblique remains mechanics-only until a provenance-pinned
segmented surface is available. This avoids presenting inferred FEM cells as
medical anatomy.

Human abdominal architecture and regional fibre direction are supported by
PMCID `PMC3017737` and PMID `15698694`; population thickness is from PMCID
`PMC12276042`. Layer-specific human uniaxial medians from PMCID `PMC10604332`
anchor external-oblique and internal-oblique axial stiffness. The live Matter
law is now transversely isotropic: all 26 exact terminal routes are validated,
then six pectoral objects remain independent while right/left TLF, EO, and IO
routes share only a same-class, same-side direction, for 12 material objects.
The grouped axis is a polarity-invariant mean because fibre direction has no
mechanical sign. This is architecture-directed mechanics, not a spatial
histology field, independently calibrated matrix/fibre split, subject-specific
anisotropy, clinical segmentation, or validated physiological load sharing.
Transversus abdominis is not actuated because it is absent from the source
MyoSim actuator set.

## Deterministic payload

```bash
PYTHONPATH=src python3 -m numilab_human.cli \
  numi-human-pectoralis-fascia-payload \
  --sources sources \
  --artifact Build/myosim-fullbody \
  --output Build/thoracoabdominal-myofascia-v4
```

The qualification payload SHA-256 is
`a39916cc3207f8557f47f70dff660ab7f39d98f1279db3354993dd22f8fd3760`.
Compiler replay is byte-identical, all 14 source terminals are asserted, and
every tetrahedron has positive rest volume.

## Apple M4 Pro qualification

The directional four-step selected-oblique run used 10 microsecond coupled
substeps, all 416 MyoSim routes, NHTENDON3, NHEQ1, source foot support, and the
Numi Matter same-command-buffer adapter. Runtime base revision `a06d182`
plus the source-hashed directional increment registers
the machine-readable receipt and inspected frames in
`docs/media/numi-human-thoracoabdominal-myofascia-live-v1`. It reported:

- 3,328 tendon endpoint transfers, including 2,568 distributed envelopes and
  760 explicit point fallbacks;
- 70.606 N admitted force across the 26 regions;
- 11.044 N final and 3.663 N minimum per-step fixed-node reaction L1 across
  all four audited steps;
- 0.0629 mm maximum continuum displacement and minimum `J = 0.997936`;
- 5 total FGMRES iterations across the accepted four-step state;
- exact terminal-axis alignment above 0.999999, sheet alignment above 0.533,
  and worst grouped-axis/source-route alignment of 0.9656;
- nonzero Human-state change relative to source `J^T` ownership;
- bitwise replay and verified downstream-rejection rollback; and
- Apple M4 Pro ownership for both Metal muscle transfer and Matter FEM.

The matching one-step run reported 70.593 N admitted force, 3.790 N fixed-node
reaction L1, minimum `J = 0.982711`, two FGMRES iterations, bitwise replay,
and rollback. Existing loader and ownership
negative controls reject both the older `NHFASC3` payload and a fascia horizon
that differs from its owning Human transaction before execution.
The receipt, one/four-step transcripts, negative-control transcripts, and four
inspected frames are tracked together in the runtime evidence directory.

At 100 microseconds the coarse internal-oblique sheet inverted on the second
step. The qualified runtime boundary is therefore 10 microsecond substepping;
the larger step is rejected, not silently presented as stable. Four 1024-pixel
views were also inspected. The initial mechanics-only view exposed false
rectangular lumbar plates; the final renderer suppresses those debug cells and
shows exact bilateral BodyParts3D external-oblique anatomy instead.

<p align="center">
  <img src="media/thoracoabdominal-myofascia-v4/front.png" width="24%" alt="Thoracoabdominal myofascia front" />
  <img src="media/thoracoabdominal-myofascia-v4/oblique.png" width="24%" alt="Thoracoabdominal myofascia oblique" />
  <img src="media/thoracoabdominal-myofascia-v4/side.png" width="24%" alt="Thoracoabdominal myofascia side" />
  <img src="media/thoracoabdominal-myofascia-v4/rear.png" width="24%" alt="Thoracoabdominal myofascia rear" />
</p>
