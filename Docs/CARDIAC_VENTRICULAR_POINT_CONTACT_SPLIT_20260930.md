# Ventricular point-contact mechanics correction — 30 September 2026

The prior full-source ventricular FEM cook merged three LV/RV source nodes
that touch **only at a point**. The [independent source activation gate](CARDIAC_SOURCE_ACTIVATION_20260930.md)
finds 4,494 complete shared LV/RV faces, 2,628 face-connected shared nodes,
and three point-only source nodes: `17565`, `170947`, and `235754`. The current
Numi Lab `coupled` cook now gives the RV side of those three contacts separate
mechanical states, while continuing to share nodes across the complete faces.
The cook produces **218,080 nodes** over all **1,097,534** ventricular
tetrahedra. A separate package reader compared every cooked tetrahedron and
node position with the exact Rodero case18 source mesh.

With the same source-derived 100 ms tension field, synthetic 1050 kg/m³
inertia, three fixed nodes, and 1 µs timestep as the prior fixture, the Apple
M4 accepted a native step. Two active runs produced byte-identical complete
accepted states; a zero-tension control also accepted. Independent Float64
source-volume assembly checked every nodal mass to a maximum relative
difference of `1.77e-5` from the native Float32 assembly, and total mass stayed
within `1e-12 kg` of the former merged-node package. Each former shared mass
now divides between its LV and RV state. The pairs moved apart by
`3.25e-8`, `2.35e-7`, and `5.48e-8 m`, respectively. Six incumbent node
positions changed relative to the old active result, with a maximum difference
of `2.02e-8 m`; the matched zero-tension baseline did not change. This is a
measured mechanical effect of correcting the contact topology, not evidence of
calibrated cardiac motion.

The [retained evidence](media/cardiac-point-contact-split-20260930/) contains
the full active, replay and zero accepted node buffers, cooked source-node map,
source-ordered tension, native receipts, exact hashes and the
[Human-side auditor](../tools/audit_cardiac_point_contact_split.py). The
219 MB native package remains in the local ignored Build directory, pinned by
SHA-256 in the receipt. The native cook and package checker are in Numi Lab
`coupled` revision `2de0a62`.

This repairs a false *mechanical* LV/RV connection in the bounded ventricular
fixture. It does not resolve source-mesh valve/vein closures or the heart's
nonmanifold outer boundary; those still require anatomical adjudication. The
other 372,549 source tetrahedra remain outside this accepted step. The fixture
has no anatomical supports, unloaded reference, chamber loading, native
electrical-to-mechanical coupling, calibrated pressure/flow response, or
heartbeat qualification.

To verify the retained state against the pinned source and native binaries:

```sh
PYTHONPATH=src:. .venv-mujoco312/bin/python \
  tools/audit_cardiac_point_contact_split.py \
  --receipt Docs/media/cardiac-point-contact-split-20260930/receipt.json
```

Add `--execute` to recook the full source package and rerun active, replay,
and zero native transactions against the exact retained hashes.
