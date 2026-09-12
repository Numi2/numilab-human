# Native cardiac passive laws and material frames, 12 September 2026

Matter now evaluates a separate material basis for each tetrahedron while keeping
the original shared nodes, mass, rest geometry and accepted-step transaction.
The Rodero ventricular Guccione and nonventricular neo-Hookean passive energies
are also available as native material programs. This closes a constitutive
infrastructure gap; the imported anatomical heart is not yet a simulated or
calibrated heart.

Native publication: [`29a4d3e`](https://github.com/Numi2/numi-lab/commit/29a4d3ec81f2e265c0a709400559e16185ef13a1)
on `coupled`. All 16 changed source blobs match the qualified file hashes.

## Native interface and ownership

`ObjectSource::femMaterialFrameRotations` supplies one unit Hamilton quaternion
`(x,y,z,w)` for every authored tetrahedron, in source order. It maps material axes
to reference-world axes. `femMaterialFrameSourceIdentity` binds a nonzero SHA256
identity of the source field and its declared conversion. Empty frames preserve
the existing world-basis behavior. Cooking rejects missing identity, unmatched
counts, nonfinite/nonunit frames and non-FEM owners.

The production FEM kernels evaluate deformation, rate and directional inputs as
`F Q`, `Fdot Q` and `dF Q`; material stress and its tangent return through `Q^T`.
Projected internal state and learned-material derivatives use the same basis.
The existing mixed active-fibre path rotates its reference fibre by `Q` while
keeping pressure, cofactor and volume operators in world coordinates. Contact,
geometry, mass assembly and shared-node continuity remain under Matter ownership.
There is no host physical stepper or new cardiac solver.

Frames currently require an immutable world. Adaptive representations and
topology mutation are rejected until field transfer is defined. Snapshot restore
pins the entire initialized tetrahedron record, its material-frame association,
allocation generation and representation. Altering connectivity or a rest
operator cannot rebind the source field. The CPU retains one complete cooked
element array for these opt-in worlds; legacy worlds allocate no such copy.
The per-element GPU record grows from 80 to 96 bytes. Matter ABI is **30**,
package format **15**, snapshot archive **8**, and accepted-proof manifest **8**.
Older executable packages require recooking and previous receipts retain their
historical identities.

## Source constitutive laws

The implementation transcribes Rodero et al., PLOS Computational Biology
17:e1008851, supporting text S4 equations 3–8 and Tables B/C, retained with
SHA256 `5699feb01b7f91f93fd75e26a9a5043ca6bfb5da1803f010c45c71270992ebc6`.
The ventricular law uses the source isochoric Guccione strain, `a=1.7 kPa`,
`bf=8`, `bt=3`, `bfs=4`, and `kappa=1000 kPa`. The nonventricular law retains
the isochoric neo-Hookean expression and source regional `c` values 7.45,
26.66, 3.7 and 1000 kPa. The last value represents artificial source valve planes.
These are source model parameters, not subject-specific fits.

Both programs use the complete source volumetric energy and require the caller
to select `mixedFEM=false`. The existing mixed pressure constraint/traction pair
does not reproduce either source volumetric potential. A numerical control
retains that non-equivalence; it is not corrected by substituting a bulk modulus.
No source activation, viscosity, prestress or unloading is added.

Density is absent from the source supplement. Both material files therefore
carry a zero unresolved-input sentinel: constitutive compilation works, but
world cooking rejects it. Qualification uses an explicitly synthetic positive
density. This neither supplies anatomical tissue inertia nor partitions blood
mass. The Human importer still preserves raw fibre/sheet fields; a declared,
source-bound orthonormal conversion and regional material assembly remain open.

## Physical Mac mini qualification

The new tests run on the physical Apple M4 Pro through `ssh macmini`. The
[receipt](media/cardiac-material-field-20260912/qualification.json) binds the
native revision, source and binary hashes, commands, logs and scope. Results:

| Check | Result |
| --- | --- |
| Source energy/stress/tangent versus independent FP64 tensor oracle | 95 multiaxial states; 1,710 stress/tangent scalar comparisons pass |
| Maximum normalized source stress / tangent error | `6.94e-15` / `1.65e-15` |
| Compiler, cooked-layout and package controls | 34 assertions pass; missing density and invalid frame inputs rejected |
| Production Metal framed Guccione force / tangent error | `9.03e-7` / `2.41e-7`, against `3e-4` |
| Force finite-difference tangent error | `1.90e-3`, against `3e-3` |
| Identity-frame versus legacy control | Zero measured force and tangent difference |
| Synthetic state/rate and mixed active-fibre controls | Independent force/tangent and finite-difference checks pass |
| Synthetic framed learned-material force / tangent error | `2.51e-7` / `4.22e-8`; equivalent world-fibre and finite-difference controls pass |
| Existing physics and transaction regressions | 19 checks pass after rebuilding against ABI 30 |

The trajectory fixture has five shared nodes and two conforming tetrahedra with
different material frames. It accepts 16 steps in each of two environments over
**1.6 ms**, with **43.96 micrometres** maximum motion, exact nodal mass,
bitwise replay/reset and isolated rollback. Normal-generation snapshots with
changed quaternion, connectivity, rest operator, material/object ID, active flag,
representation or allocation generation are rejected without altering state.
These are bounded synthetic mechanics, not anatomical wall loading or an
active cardiac source reproduction.

The learned-material check uses a small authored invariant network with a
closed-form FP64 oracle. It does not qualify trained weights or calibration.
The initial source-law test exposed a near-zero energy covariance cancellation
of `2.22e-10 Pa` at the stiffest source value. Its comparison now uses the same
explicit `1000 Pa` normalization as stress/tangent checks; the source equation
and normalized gate remain unchanged. Both failed attempts are retained. The
first native build also retained an invalid-target failure before using the
actual `numi_matter_runtime` target. No final qualification check is skipped.

The active runtime uses the staged production FEM force/operator kernels. The
unused `fusedFEMOperatorAtNode` helper has consistent basis transformations but
no call sites, so this receipt makes no executed coverage claim for that helper.

## Remaining anatomical admission

The independent [source-defect audit](media/cardiac-material-field-20260912/source-defects/REPORT.md)
shows that ordinary coordinate-preserving node separation cannot produce a clean
ventricular cavity. It removes 29 of 31 abstract edge defects but leaves two
continuity obstructions. The abstract RV sphere still has eight exact forbidden
contact pairs; node separation also releases original FEM nodal constraints.
The RV tissue-volume difference from the published CSV remains −2.415755 mL.
No such variant was admitted or substituted for the source.

The next anatomical gates are a source-supported embedded wall/port topology,
regional material and density ownership on shared nodes, an explicit conversion
of source fibre/sheet axes, unloaded/loading and boundary data, then native
passive pressure–volume loading with convergence. Source activation and subject
calibration follow those gates. Mechanical blood mass, whole-body integration,
standing and walking remain unqualified by this increment.
