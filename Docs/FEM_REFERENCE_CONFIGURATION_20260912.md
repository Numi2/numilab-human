# Supplied FEM reference configuration, 12 September 2026

Matter now accepts separate reference and initial coordinates for immutable
FEM objects. A supplied loaded shape can carry elastic prestress while its
reference volume, rest operators and lumped mass remain unchanged. This closes
the native representation gap; it does not recover an unloaded cardiac mesh.

**PASS for bounded native reference-configuration ownership.** All 27 selected
checks pass on the physical Apple M4 Pro, with no failures or skips, in
80.10 seconds. The [qualification receipt](media/fem-reference-20260912/qualification.json)
binds 16 source files, the compiled binaries, production metallib and shared
runtime library. Source hashes are unchanged across build and execution.
The [native change](https://github.com/Numi2/numi-lab/commit/cb34a4a3e0ba961fc68b3e027fca3b03c0f48706)
is published on `coupled`.

## Native ownership

`ObjectSource::femNodes` remains the initial/current geometry.
`femReferenceNodes`, when supplied, contains one reference coordinate per node
with unchanged tetrahedral connectivity. A nonzero 256-bit
`femReferenceSourceIdentity` binds that declaration. Empty reference fields
preserve the previous initial-equals-reference behavior.

The cooker converts reference coordinates to the executable Float32 geometry
before forming inverse rest matrices and reference volumes. Each cell uses its
selected material density and reference volume to contribute one quarter of
its mass to each incident node. Shared-node contributions are accumulated in
Float64 and rounded once to Float32. Initial deformation does not change that
mass. Reference and initial centers of mass use their respective coordinates
with the same canonical nodal masses.

Initial coordinates remain the runtime reset state. Fixed nodes retain their
initial loaded positions rather than jumping to the reference shape.
The production FEM force and tangent kernels consume current edges and the
cooked reference inverse; no separate host mechanical solve is introduced.

The reference may be combined with per-cell material indices and material
frames. Frames must be declared in the reference basis. This interface does
not infer how a measured fibre field should be transported to an unknown
unloaded geometry.

This increment admits fixed-parameter, immutable FEM. Adaptive representation,
topology mutation, mixed FEM, multiphysics and identification remain excluded
from this reference configuration. Invalid identities, mismatched counts,
nonfinite coordinates, collapsed or inverted cells and invalid material-domain
geometry are rejected. The cooked validator independently reconstructs rest
operators, reference volume and mass from serialized geometry.
The determinant interval mode also rejects nonzero subnormal coordinate or
inverse entries and potential intermediate overflow; it does not promise
admission for every representable Float32 configuration.

Reference-only worlds receive the existing immutable topology, mass, rest,
material-parameter and scheduler guards. Restore also binds reference-center
metadata and checks evolving current geometry. Package fingerprints include
both coordinate sets and their source identity.

Compatibility is **Matter ABI 32, package 17, snapshot archive 10 and accepted
proof manifest 10**. Older cooked payloads must be recooked. Historical receipts
retain the versions and source revisions they actually tested.

## Evidence boundary

The focused driver uses two tetrahedra sharing three nodes, in two environments.
It supplies an independent affine preload and synthetic densities of 900 and
1200 kg/m³. The reference-only case and the combined reference/regional/frame
case exercise the same production force and tangent kernels used by Matter.
An independent FP64 constitutive oracle and finite differences check initial
prestress and directional tangents. Translated reference geometry with a full
non-axis transform separately checks general inverse construction and package
roundtrip.

The compiler driver passes 96 controls. The final 27-check selection also
rebuilds and checks regional/frame materials, source cardiac laws, vascular
coupling, Human tendon/FEM transfer, stateful materials, mutation, snapshots,
identification and accepted transaction rollback.

| Production FEM case | Relative force error | Relative tangent error | Finite-difference error |
| --- | ---: | ---: | ---: |
| Reference only | 8.52e-8 | 9.71e-8 | 8.76e-4 |
| Reference with regional materials and frames | 8.58e-8 | 3.06e-7 | 1.34e-3 |

The force/tangent gate is 3e-4 and the finite-difference gate is 3e-3.
Initial maximum nodal prestress forces are 0.6285 and 0.6443 N; the supplied
reference-shape controls produce zero force. These are fixture measurements,
not anatomical tissue calibration.

Integrated tests cover 16 accepted steps per case at 0.1 ms per step, bitwise
replay, isolated rollback, reset to loaded initial state, immutable mass and
snapshot rejection. This is **1.6 ms of synthetic motion per trajectory**.
It does not qualify anatomical equilibrium, a cardiac cycle, physiological
calibration or sustained Human behavior.

Review found that an initial-volume ratio can pass a material boundary while
the executed Float32 deformation determinant falls outside it. Admission now
uses a conservative roundoff interval for current edges multiplied by the
serialized inverse rest matrix. New source and cooked initial states require
that entire interval to lie inside the selected material's declared domain.
Snapshot restore rejects geometry whose interval cannot intersect the domain;
it preserves potentially valid accepted states at an ambiguous rounding
boundary. That overlap check is not a certificate of the precise Metal
determinant for an arbitrary restored boundary state. Production stepping
continues to apply the actual material-domain check.

The retained boundary controls distinguish the original admission defect from
a physical solver failure. Final source hashes and exact regression results
are recorded with the qualification receipt.

## Remaining source gates

Rodero case18 is supplied as loaded end-diastolic geometry. The source does not
include its unloaded reference mesh or a complete loading reconstruction deck.
The separate [material attribution](CARDIAC_MATERIAL_ATTRIBUTION_20260912.md)
resolves supported passive classes and LV geometric mass accounting; it leaves
artificial closure stiffnesses and native inertial densities explicit.

An anatomical execution still needs an admitted wall/lumen mesh, physical port
boundaries, resolved material/density ownership, a source-backed reference or
qualified loading construction, reference-basis fibres, quantitative supports
and complete activation data. Native representation support does not supply
those missing inputs. The anatomical import continues to have zero physical
steps; blood mass partition, standing and walking remain separate open gates.

The remaining work also includes engineering that does not depend entirely on
new measurements: the source valve's zero forward resistance with finite
reverse resistance needs a native constraint law; pressure-port and spatial
support assembly need conservative work/tangent ownership; loading/unloading
and source activation need qualified native workflows. The numerical source
coefficients already known must be retained rather than approximated by an
epsilon resistance or borrowed active-law defaults. These owners can advance
before full subject calibration, but their anatomical acceptance still depends
on the missing geometry and boundary inputs.
