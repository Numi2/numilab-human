# Rodero case18 source boundary audit

**The left and right atrial components are embedded, mutually disjoint, closed source-conforming cavity-reference boundaries.** The ventricular candidates and outer surface have nonmanifold edges. No vertices, faces, caps, labels or cell connectivity were edited by these audits.

The audited asset is `/Users/n/human-cardiac-wall-anatomy-20260912/asset`. Its manifest SHA256 is `7eb93af573b328c77f3960f3f9fb40e337356d64fa6fa3ff59e258d1959b0dee`, configuration SHA256 `23c931fef53edced85e0e0a36c73d8490dddb87db6bd988482a2cdfc5a1442cc`, source archive SHA256 `b50919a711dc3914cc0a4dd9cabc19679a9f36be9ac6eb26a23ca6799f501720`. Only the necessary binary arrays were read on the Mac mini; the full asset was not copied locally or reimported.

## Boundary topology

All 230,364 boundary faces match an oriented face of their recorded source tetrahedron. Component edge connectivity was independently recomputed. There are no shared vertices between the six components. Every problematic edge has **four incident faces**, two in each direction; cancelling oriented edge chains therefore does not make these surfaces manifold.

| Component | Source interpretation | Faces | Four-face edges | Bad vertex links | Euler | Material-outward signed integral, mL |
|---:|---|---:|---:|---:|---:|---:|
| 0 | LV candidate, labels 1/7/9 | 31,966 | 4 | 4 | 4 | −92.890323 |
| 1 | LA candidate, labels 3/7/11–15 | 18,532 | 0 | 0 | 2 | −43.346388 |
| 2 | Outer material boundary | 114,776 | 25 | 40 | 7 | +487.198728 |
| 3 | RV inference from 8/10; tissue labels 1 and 2 | 42,506 | 2 | 3 | 3 | −119.168310 |
| 4 | RA candidate, labels 4/8/16/17 | 22,520 | 0 | 0 | 2 | −52.851852 |
| 5 | Unassigned small internal component, labels 1/2 | 64 | 0 | 0 | 2 | −0.004009 |

The three topology-clean components have one connected surface and spherical topology. Component 5 has not received an embeddedness or physiological-identity qualification. Signed integrals of defective components are diagnostics, not admitted lumen volumes. There are 47 bad vertex links in total, all incident to four-face edges; no additional isolated vertex-link pinch was found.

Concrete zero-based source-index defects:

- LV component 0: edges `(194623,222142)`, `(194623,294970)`, `(222142,288939)`, `(288939,294970)`. All incidences are myocardial label 1, so removing or reclassifying valve caps would not fix them. For edge `(194623,222142)`, source tetrahedra are `1032607,231210,1039476,679124`, boundary faces `18889,53821,184163,184164`.
- Component 3: edges `(26256,170947)` and `(164090,170947)`; bad links at those three nodes. The first edge involves source tetrahedra `1309400,695418,1309400,643008` and tissue labels 1/2.
- Outer component 2: one example is edge `(11635,126709)`, four faces from tetrahedra `194329,194329,567184,567184` in atrial labels 3/4. All 25 edges, source positions, directed face incidences, owning cells, labels and complete link graphs are in `report.json`.

The source configuration names label 8 “Tricuspid valve” with interpreted RA→RV port semantics and label 10 “Pulmonary valve” with interpreted RV→pulmonary-artery semantics. Their joint presence supports an **RV candidate inference** for component 3 despite LV-tagged septal tissue. This is documented source-label interpretation, not a supplied component-ID annotation; the audit does not relabel it as an unambiguous source fact.

## Exact atrial embeddedness and conformity

The existing Human exact integer/rational triangle predicates found:

| Component | Exact AABB candidate pairs | Permitted shared-index edge/vertex contacts | Forbidden self-intersections |
|---|---:|---:|---:|
| LA, component 1 | 114,598 | 113,716 | 0 |
| RA, component 4 | 139,126 | 138,187 | 0 |

Adjacent faces were not exempted wholesale: their intersection had to lie entirely on the shared topological edge or vertex. A deterministic subset independently matched the existing direct owner’s candidate counts and decisions. Exact integer AABB pruning excluded only provably disjoint boxes.

There are zero RA–LA triangle intersections. Reciprocal exact ray-parity witnesses, at source nodes 12 and 24 using ray `(1,1,1)`, are both outside the other surface. Thus the two closed domains are disjoint. Their positive lumen reference volumes are **43.34638810879855 mL** and **52.85185151113091 mL**. These are mesh integrals, not physiological volume calibration.

Every atrial face's opposite source-tetrahedron vertex lies on the exact material side of the oriented face. The original arrays are material-outward; reversing orientation gives the cavity-outward convention for later authoring. Source node IDs, face IDs and owning tetrahedron IDs remain unchanged. This proves local conforming adjacency, not exclusion of every remote tetrahedron from cavity interiors; global wall tetrahedral embeddedness remains unchecked.

Artificial source faces remain explicit:

| Surface | Myocardial faces | Artificial valve faces | Artificial vessel/appendage closure faces |
|---|---:|---:|---:|
| LA | 16,759 | 766, label 7 | 1,007, labels 11–15 |
| RA | 20,074 | 1,189, label 8 | 1,257, labels 16–17 |

`embeddedness.json` inventories every source boundary face by role. These caps were already in the source volume mesh; none was created by this audit. Their existence does not make them measured functioning valves. Native pressure boundary admission remains false until their port/work ownership is explicit. The existing native material-only cavity role cannot be populated by silently relabeling these source caps.

## Reproduction and remaining gates

The final topology audit completed in 3.11 s; embeddedness and pair tests in 10.29 s. Both returned 0. Full commands, input/owner/script hashes and log hashes are recorded in `execution.json`. Source-sized binary input arrays remain remote. `report.json` and `embeddedness.json` are locally retained under this directory, along with both scripts and logs.

This establishes useful **matching atrial wall/cavity geometric candidates**, unlike the independent BP3D atrial wall/cavity pairs. It does not admit the entire wall as an embedded material domain, repair the 31 source edge defects, supply an unloaded reference configuration, identify density or mass ownership, implement per-element native fibres, or qualify material/loading/circulation reproduction. Those are separate remaining owners and gates.
