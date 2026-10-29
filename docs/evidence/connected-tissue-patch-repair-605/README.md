# Connected skin/fat authoring checkpoint

The remesher now compares exact Float32 integer-lattice AABBs in the same
units; world-metre bounds previously omitted retained/accepted patches.
It checks directed edge multiplicity, duplicate faces, connectedness, Euler
characteristic and current geometry before reusing any saved operation.
Each replacement face retains complete parent support.

Corrected batch019 plus two negative-orientation closed-component repairs
produced candidate020, SHA0911a301... . The complete exact self audit021
scanned 258,095 faces and reduced original self pairs 1,619 to 200.
All 200 are unchanged original pairs in eight failed transactions.
There are zero replacement-to-retained, replacement-to-replacement or
closed-component-to-any self pairs. Fine O2958a704... and I5d46ee99...
are unchanged, including the accepted back repair and existing head loops.

This was NOT an anatomical pass. The independent changed-face gate608 found
369 bone interface triangle pairs across 40 replacement transactions in EACH
coordinate domain: actual source-order native Float32 replay and pinned
Float64 operands. These are matching tests, not 738 distinct physical crossings.
Those proposals are unadmitted. Complete unchanged-face, solid-containment
and D-inside-I checks are still required.

The 609 helper adds both bone coordinate versions as exact offline remesh
barriers using the existing predicate owner and a shared 2^149 integer
lattice. Float64 inputs are never rounded to Float32. Bone vertex identities
are disjoint from tissue identities, and negative face IDs prevent query IDs from omitting them. Regression controls detect coincident
distinct structures, preserve disjoint structures, retain Float64 differences,
and reject nonfinite/unrepresentable inputs. The fixture pass alone does not
qualify the Human remeshing or physical scene.

The corrected bone-and-inner-skin search (engine 8392ce5a...) rejects stale
saved operations against both bone coordinate versions and accepted open fine I.
It rebuilt 33 of the 40 bone-unsafe patches and all three previously inner-skin
crossing patches. Seven bone-unsafe patches remain unrepaired and retain their
original faces. The combined candidate025, SHA9790ed1f..., has 258,217 faces
and exactly the same vertex coordinates as baseline007.

Independent gates026/027 tested all 1,997 newly generated faces: zero exact
pairs with all 185 bone banks in either coordinate domain, and zero with the
205,239-face open fine-I surface. These are changed-face checks only. Whole
surface audit028 found 237 self-intersection pairs, so the all-185 full-surface
stage was correctly not run. Exact Float32 triangle-bit attribution029 confirms
all 237 pairs use unchanged original geometry and none touch replacement faces.
This guarded candidate has more retained self
pairs than the bone-unsafe candidate020; the lower 200 count is not an
acceptable alternative because its replacements penetrate bones. Complete
unchanged-face, closed-solid and nesting checks remain pending. No candidate
in this package passes whole-anatomy admission.

The 607 bridge-and-ear fallback repaired six Human annuli under self checks.
Its square/concave-hole fixtures also pass directed boundary, Euler, exact
self, oriented area and negative controls. It follows section 3 of
[Eberly's ear clipping reference](
https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf).
Bone qualification remains separate.

Failed cavity 033 preflight is preserved: enlarging the cavity retained 52
collar-to-patch and 29 collar self-crossings; that zipper route is retired.
Current006 detail localization still exposes excessive physical-tet weights.
The 1.5 weight gate was retained. Regional capacity screens use floating
Manifold CSG as diagnostics; they do not establish exact-predicate admission
or prove that every alternative regional reconstruction is impossible.

Physical volume reconstruction, immutable source330 mass/density mapping,
detailed physical-skin support and accepted native motion remain unfinished.
Runtime creation and accepted steps for these authoring candidates are zero.
No simulation force/state owner or accepted scene is changed.

Large JSON receipts and exact CSV traces are gzip archives of exact original bytes. The manifest
records compressed and uncompressed SHA-256 identities. Frozen authoring
scripts preserve executed bytes and require retained pinned input assets and
owner packages at their recorded paths. These are construction/evidence tools,
not a new general artifact format. The mixed-source adult remains a reference
assembly, not measurements of one individual.
