# Conservative cardiac cavity partition — 12 September 2026

`numi human-circulation-partition` now constructs two explicit ownership candidates for the overlapping BodyParts3D right atrial and ventricular cavities. Both emitted candidates have disjoint interiors, a declared shared boundary, and no intersections with either left cavity. The duplicated source volume is **0.07275373956738762 mL**, distributed across two connected regions.

This closes geometric overlap construction and verification. It does not select a biological atrioventricular boundary, register the atlas to a subject or cardiac phase, or assign mechanical blood mass. CVSim parameters, native payloads and physical state remain unchanged.

## Ownership and source preservation

Let A be the source right atrium and B the source right ventricle. The two conventions are:

| Convention | Right atrium | Right ventricle | RA volume (mL) | RV volume (mL) |
| --- | --- | --- | ---: | ---: |
| Atrium priority | A | closure(B minus A) | 84.551532018 | 116.899602671 |
| Ventricle priority | closure(A minus B) | B | 84.478778278 | 116.972356410 |

Only the duplicated region changes ownership. Each convention preserves the source union and both source-exclusive regions exactly in the rational construction. The original archive, source meshes and historical overlap receipts remain unchanged. Neither candidate is automatically selected for anatomical use.

The [source leaflet audit](media/cardiac-partition-20260912/independent/valve-source-audit.json) records the atlas tricuspid members and relationship-table matches. Those leaflet meshes do not supply a source-defined partition surface. A fitted plane would require a separate anatomical convention and could cut source-exclusive chamber volume. The current interfaces instead follow the existing source faces inside the overlap; they are explicit geometric alternatives, not inferred valve anatomy.

## Exact construction and independent checks

The [construction owner](../src/numilab_human/cardiac_cavity_partition.py) interprets the published quotient Float64 metre coordinates as exact rational values. All 42 intersecting source triangle pairs are split into a common arrangement. The 4,162 source faces become 4,330 source-parented triangles with 42 added intersection vertices. Original vertex coordinates remain unchanged.

The [face arrangement owner](../src/numilab_human/cardiac_face_arrangement.py) preserves winding, oriented coverage, cut constraints and all shared boundary vertices. It rejects unsupported coplanar or tangent intersections, holes, dangling cuts and disconnected planar loops. It does not apply tolerance welding or fabricate closing planes.

The separate [certificate owner](../src/numilab_human/cardiac_partition_certificate.py) verifies the pinned source identity, each child's source plane and positive face coverage, absence of uncut intersections through child interiors, and independently computed inside/outside classifications. It then verifies closed oriented boundary chains, shared-interface cancellation and exact additive volume, first moment and second moment identities. In particular, the partition total equals A plus B minus their intersection for all three integrals. These are raw additive moments; central moments are not added without the required frame terms.

The overlap has 53 vertices, 98 triangles and two closed components. Atrium priority uses 42 shared interface faces; ventricle priority uses 56. Both final chamber boundaries are connected and embedded.

## Float64 emission and rounding

Each unique rational point receives one deterministic Float64 conversion, shared by both regions. The emitted geometry undergoes a separate exact audit of its actual binary coordinates. That audit checks topology, self-intersections, opposite interface winding, every cross-domain contact, every non-interface face's location outside the other region, and strict interior witnesses. Contacts through face interiors or outside the declared interface are rejected. The unchanged left cavities also pass strict no-contact and containment checks against both emitted right regions.

The maximum coordinate rounding error is 1.103037094108598e-16 m. The emitted union's volume differs from the rational union by 7.344254098976328e-22 m³ for either convention. Each region's full signed difference in volume and first and second moments is retained in the artifact. Exact source-set conservation is a property of the rational construction; it is not asserted as an exact identity for rounded emitted coordinates.

An independent Manifold 3.5.3 calculation on the Mac mini gives an overlap of 7.275373956738772e-8 m³, versus the exact certificate's 7.275373956738762e-8 m³ rounded for display. The [reference output](media/cardiac-partition-20260912/independent/manifold-first.json), isolated dependency hashes and exploratory strict-contact audits are retained. A [separate comparison](media/cardiac-partition-20260912/independent/compare-manifold-reference.json) verifies that its oriented source triangle multisets match the admitted source geometry and measures a 9.686680209420784e-23 m³ difference. Manifold is an independent numerical geometry reference, not a runtime dependency or the admission authority. Its [official implementation](https://github.com/elalish/manifold) documents its mesh Boolean operations.

## Reproduction and evidence boundary

```sh
numi human-circulation-partition --output Build/cardiac-partition/candidates.json
python3 tools/verify_cardiac_partition.py
python3 tools/verify_cardiac_partition.py --recompute
```

The command verifies the pinned archive, anatomical tables and cavity members. It emits one immutable artifact containing original geometry, exact construction, independent certificate, both candidate meshes, rounding differences and explicit unqualified physical fields. Rational scalars use hexadecimal numerator/denominator pairs so large exact moment denominators do not require disabling Python's decimal conversion limit.

Qualification runs on the physical M4 Pro through `ssh macmini`, in an isolated source copy. The [execution record](media/cardiac-partition-20260912/execution.json) binds all consumed source and owner hashes before and after execution, compiler output, regression logs and Python/host identity. This run performs offline geometry and tests; it does not step hydraulic or mechanical physics. Previous native CVSim and cavity identity qualification remains separate and unchanged.

All **58 regression tests pass**, including 34 new arrangement, partition, certificate and output-preservation tests. Eight additional evidence controls reject missing owners, failed or changed regression commands, promoted biological or mass claims, missing candidates and changed geometry identities. The final authoring run took 73.52 seconds; regressions took 9.06 seconds including process startup. These are observed qualification durations, not performance claims.

The Mac mini's Python 3.13 and local Python 3.14 runs emitted the same 3.6 MiB artifact, SHA256 `fc0241b6b25319c0565c226266dd3a431941aac4f948d519f42c4ee8c1d0572f`. All 21 captured inputs matched before and after execution. The first remote attempt compiled the same artifact but failed one regression because the isolated copy omitted the historical intersection fixture; its [failed execution and logs](media/cardiac-partition-20260912/attempts/001/execution.json) are retained. The fixture is now an explicit captured input. All 19 owners in the earlier cavity receipt still match their original hashes.

The next physical ownership gates are selecting or bounding the anatomical interface using matched evidence, cardiac phase and body-frame registration, sourced density and donor blood inclusion, spatial receiver quadrature, and conservative mass/momentum transfer inside Matter's existing accepted transaction. The remaining 17 aggregate circulation regions, organ-specific perfusion and tissue exchange, physiological calibration, standing and walking remain open.
