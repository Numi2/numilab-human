# Rodero-2026 HCM1 source geometry and activation audit

The CPU-only source audit now binds the pinned HCM1 four-chamber VTK mesh to
the imported sample 53 activation-time field. It parses all 749,238 points and
3,917,596 tetrahedra, checks every connectivity index and publisher element
tag, then counts active activation nodes in each tetrahedron. The input mesh
SHA-256 is
`44e350b1df2eeff0633e2ab9683d1781490997b472bdb40c26f1bc01a0ec3fc1`; the
little-endian activation sidecar SHA-256 is
`e0f42c5f5b0a64c6ca0c164ff5938982be8b9f9c0c03cadf8ae5b821f2a9da35`.

All tetrahedra have positive signed orientation and none has zero volume.
The source-unit tetrahedral-volume candidate sums to 468.6839189 ml. This is
only a sum over this source mesh; it is not a calibrated chamber volume, a
subject-matched anatomical measurement, or a pressure-volume result. The
activation map covers 680,978 points from 0 to 165.026314 ms and retains the
publisher's 68,260 inactive `-1` sentinels. Direct point order is established
by the published visualization code, which assigns the source vector to the
matching mesh without reordering.

The audit found 24 element tags in this mesh (IDs 1–24). The publisher lists
the additional conduction identities 25–29 on a different
`myocardium_AV_FEC_BB_lvrv.vtk` mesh. That electrical-geometry source is still
missing from this Numi intake, so the imported activation map is not yet
validated against its fast-conduction tissues. The base source mesh describes
HCM1's mid-to-apical hypertrophic-cardiomyopathy phenotype; it is not the
Healthy case18 reference. See the publisher's [mesh record](https://zenodo.org/records/21282274)
for the mesh units, regional tags, and separate electromechanics mesh name.

## Reproduce

```sh
PYTHONPATH=src ./.venv/bin/python -m numilab_human.cli \
  rodero-hcm1-source-geometry-audit \
  --mesh-vtk Build/cardiac-hcm-ep-validation-20261002/HCM1.vtk.part \
  --activation-field Build/cardiac-rodero26-hcm1-activation/hcm1-sample53-activation-time-ms.f64le \
  --activation-receipt Build/cardiac-rodero26-hcm1-activation/source-activation-import.json \
  --output Docs/media/hcm1-source-geometry-audit-20261003/receipt.json
```

The command checks the activation import's report digest, pinned archive and
member identities, sidecar digest, and exact point-order contract before it
processes the mesh. It writes an immutable [receipt](media/hcm1-source-geometry-audit-20261003/receipt.json)
with SHA-256
`9845458445a7ac5bec9ce485a2d797368905428ec2f56aee9ac56ad2be0c45da`.

## Evidence boundary

This is a source-geometry and activation-coverage audit of one patient-specific
mesh. It does not register HCM1 to Numi's current healthy subject, add missing
conduction tissues, run Numi electrophysiology, import voltage or ionic state,
couple electrical activation to mechanics, or qualify a native heartbeat or
clinical prediction. The separate electromechanics mesh and its matching
electrical/mechanical source contract are next requirements for cardiac
integration.
