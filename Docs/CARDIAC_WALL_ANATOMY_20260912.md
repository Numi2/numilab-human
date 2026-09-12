# Anatomical cardiac wall source, 12 September 2026

The Human source importer now produces a tetrahedral four-chamber heart asset
with matching material-boundary indices, regional labels, fibre and sheet fields,
and ventricular coordinates. This is source preparation for Matter, with zero
physical steps. It does not qualify an anatomical heart simulation.

## Source selection and retained failures

The existing BodyParts3D right and left atrial walls are embedded, closed tissue
solids, but they intersect their respective reference cavity surfaces in 3,000
and 1,880 triangle pairs. Exact cavity-face witnesses lie both inside and outside
the tissue. There are no shared exact wall/cavity vertices or faces. Directly
tetrahedralizing either wall cannot make the unrelated cavity its FEM boundary.
The [audit](media/cardiac-wall-anatomy-20260912/atrial-audit/REPORT.md) retains
the pinned sources, exact predicates, all pairs and witnesses. No source was
capped, moved, repaired or discarded.

The selected replacement input is case 18 of the
[Rodero et al. CT-derived cohort](https://zenodo.org/records/4590294), version
1.0.0, CC BY 4.0. The [source config](../config/cardiac-wall-rodero18.v1.json)
retains full attribution, 24 regional labels, source parameters, units and
qualification limits. The 53,661,258-byte archive has SHA256
`b50919a711dc3914cc0a4dd9cabc19679a9f36be9ac6eb26a23ca6799f501720`.
Its sole `18.vtk` member has SHA256
`f0d4f3fc21ea229fa1b334888d374ee606a2915b414fff047764aad44f1c0a41`.

## Executable import

```sh
numi human-cardiac-wall --archive /path/to/18.tar.gz --output /path/to/cardiac-wall-asset
```

Without the dispatcher, use `PYTHONPATH=src python3 -m
numilab_human.cardiac_wall_source` with the same arguments. Python performs
offline source conversion and geometry predicates; it never steps Matter.

The importer checks the entire source-config identity, archive and member bytes,
VTK counts and field layout, finite values, indices, duplicate and degenerate
tetrahedra, and exact tetrahedron orientation. Coordinates convert from mm to m
once. Any negative tetrahedron would receive a recorded local 0/1 permutation;
case 18 needs none. Fields remain unnormalized and node/cell IDs remain tied to
the source. Sorted face incidence derives the actual material boundary and
rejects same-side adjacent tetrahedra and faces with more than two owners.

Thirteen little-endian buffers plus a hash manifest retain geometry, connectivity,
source regional IDs, fibres/sheets, four UVC fields including their -10 sentinel,
source-cell orientation mapping, boundary triangles, owners and components.
Existing changed output is rejected. The manifest publishes last and explicitly
denies native-package, physical-step, calibration, body-registration and blood-mass
claims. Numerical valve and vein closure cells are retained with their source
roles; they are not relabeled as physical myocardium or valves.

## Actual geometry and qualification

Mac mini imported all **300,965 nodes and 1,470,083 tetrahedra**. The final import
took **23.27 seconds**, with **1,076,068,352 bytes peak resident memory** and no
swaps. All source tetrahedra have positive volume; the smallest is
`8.591853181060137e-12 m³`. There are 230,364 exterior faces, 2,824,984 shared
interior faces and six material-boundary components.

| Boundary component | Geometric enclosed volume | Boundary defects | Source closure faces |
| --- | ---: | --- | ---: |
| LV candidate | 92.890323 mL | 4 four-face edges; 4 bad vertex links | 2,583 |
| LA candidate | 43.346388 mL | No edge or vertex-link defects | 1,773 |
| Exterior material surface | 487.198728 mL, outward sign | 25 four-face edges; 40 bad vertex links | 3,755 |
| Mixed LV/RV-labelled ventricular candidate | 119.168310 mL | 2 four-face edges; 3 bad vertex links | 3,743 |
| RA candidate | 52.851852 mL | No edge or vertex-link defects | 2,446 |
| Extra internal component | 0.004009 mL | No edge or vertex-link defects | 0 |

Volumes above describe extracted geometry, including authored numerical closure
surfaces. They are not assigned hydraulic initial conditions. The two ventricular
labels on a boundary can reflect septal ownership; the importer does not select
the right ventricle from majority voting. All 31 defective edges have four
incident boundary faces. The additional small component remains in the asset.
No vertices are shared across the six components. Global tetrahedral embedding
and pressure-boundary admission remain unqualified.

The [exact atrial boundary audit](media/cardiac-wall-anatomy-20260912/atrial-audit/rodero-boundary/REPORT.md)
additionally finds zero forbidden self-intersections in either atrium, zero
intersections between them, and reciprocal exact outside classifications.
Every face retains its source tetrahedron and its opposite vertex is on the
material side. These are embedded, mutually disjoint, source-conforming atrial
references. Exclusion of every distant tetrahedron from their interiors has not
been checked; neither atrium is admitted as a native physical pressure boundary.

All source fibre/sheet components are retained. No zero or parallel frames occur;
the largest ventricular fibre norm error is about `1.07e-5` and the largest
normalized fibre-sheet dot magnitude is about `1.49e-6`. Those are source-field
measurements, not a license to silently normalize or treat fibres as measured
individual microstructure.

Ten focused tests pass locally and on the Mac mini, covering source parsing,
unit conversion, field preservation, reversible orientation, shared-face removal,
degenerate/duplicate/overlapping adjacency rejection, a real hollow test mesh,
retained nonmanifold defects, configuration tampering, and immutable output.
The independent C++ checker reads emitted buffers directly and checks determinant,
owner, connectivity and volume identities. It runs in 1.18 seconds at about
200 MB peak resident memory. Global and maximum regional conditioned volume
errors are `1.21e-16` and `7.88e-15` against `1e-10`. Sixteen C++ control cases
pass, including 15 malformed-buffer rejections. Four further receipt controls
reject changed source semantics, missing buffer records, altered bytes and a
false mechanics promotion. Its result is geometry evidence only.
See [evidence](media/cardiac-wall-anatomy-20260912/qualification.json).

### Coordinate-preserving node separation does not resolve the source defects

The [independent tetrahedral-star audit](media/cardiac-material-field-20260912/source-defects/REPORT.md)
tests the finest coincident-node separation that preserves every existing shared
tetrahedral face. Nineteen added node identities reduce the 31 edge defects and
47 bad vertex links to two and four. The remaining LV and exterior edges have
explicit shared-face paths proving that further node separation must break an
existing tissue-face connection. All source coordinates, cells, labels and
per-label volumes remain unchanged in this scratch experiment.

The RV candidate becomes an abstract closed boundary, but eight exact forbidden
face-contact pairs remain in its geometry. Across the 29 abstractly resolved
edges, 116 such pairs remain. Node separation therefore does not establish an
embedded cavity and also releases the original shared-node mechanical constraints.
No repaired mesh was admitted. The RV tissue-volume discrepancy remains unresolved;
neither adjacent valve-layer labels nor the supplied ventricular coordinate field
identifies the missing 2.415755 mL. Commands, hashes, exact witnesses and the retained
failed script attempt are preserved with the audit.

## Mechanics and calibration ownership

The [source mechanics supplement](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1008851.s004&type=supplementary)
provides an isochoric Guccione ventricular law and distinct nonventricular
neo-Hookean parameters, active-tension settings and circulation loads. These are
recorded as source-model inputs, not an implemented native cardiac material or
subject calibration. Source atrial, RV and vessel thicknesses include prescribed
values; fibre fields are rule-based and valve layers are numerical closures.

The cohort used shared parameters to isolate anatomical variation. Its CSV is
simulation output, not measured patient pressure-volume data. Even exact source
cycle reproduction still lacks the case-specific unloaded reference, density/
inertial convention, numerical spring stiffness and unambiguous vein anchoring.
The source's zero forward valve resistances also need an explicit native
constraint law; replacing them by an arbitrary small resistance is a different
model. Adding CVSim21 would create a separately qualified hybrid.

The [source-output comparison](media/cardiac-wall-anatomy-20260912/source-output-comparison.json)
also preserves a discrepancy: label 2's imported RV tissue volume is
45.024903 mL, while the source CSV reports 47.440658 mL (5.09% difference).
LV tissue volume agrees to about `2.06e-9` relative. The RV difference is
unresolved; no cells were relabeled and no scale factor was fitted to hide it.

At native revision `f73b07133d09ceb9ed7ff94d48d5863157f2527b`, Matter's scalar
material/AD owner and active-stress derivatives exist, but the source API enforces
one material and one reference fibre per continuum object. The
[native audit](media/cardiac-wall-anatomy-20260912/atrial-audit/NATIVE_MATERIAL_CONTRACT.md)
identifies the complete extension: immutable per-element material fields through
passive/active residuals, derivatives, mixed fields, mass assembly, certification,
serialization and replay, while retaining shared FEM nodes and one accepted
transaction. Cloning one object per element would break continuity and ownership.

Next gates are explicit source-topology resolution, physical port/closure work
ownership, per-element native material fields, sourced load/reference completion,
and anatomical calibration. Nonduplicated spatial blood mass/momentum, body
registration and whole-Human standing/walking remain separate open work.
