# Native cardiac material field audit

Read-only audit of native commit `f73b07133d09ceb9ed7ff94d48d5863157f2527b` at `ssh macmini:/Users/n/MetalRobo-human-completion-20260907`. The checkout was clean. No native files were edited, no build was run, and no physical trajectory was executed.

## What exists

- `matter/include/numi/matter/matter.hpp:278`: `TetrahedronSource` contains only four node indices. `ObjectSource` owns one `materialIndex` and one shared FEM node arena (`:405`). There is no per-element frame or parameter-field source contract.
- `matter/include/numi/matter/shared.h:911`: `NMTetrahedronGPU.identity.x` already denotes the element's material. Passive FEM evaluation loads `materials[tetrahedron.identity.x]` (`matter/src/metal/fem.metalinc:859,1144`). However, `compiler_impl.cpp:1742` always writes the object's material index, and `validation_impl.cpp:1252` requires equality. This is currently a homogeneous-material invariant, not an exposed heterogeneous-material feature.
- Generic material energy, dissipation, validity, and implicit-state programs already produce stress and consistent tangent programs. The common evaluator reads an environment-wide parameter arena through `material.parameterOffset` (`common.metalinc:547`). It receives deformation, deformation rate, internal state, time step and temperature, but no element field/frame context.
- `MixedMaterialSource::fibreDirection` and `maximumActiveTension` (`matter.hpp:190`) lower into `NMMixedMaterialGPU.fibre` (`shared.h:382`). The mixed FEM stress uses active Cauchy stress `Tmax * activation * m⊗m`, with `m=F*f/|F*f|`; its Piola transformation, deformation derivative, and activation derivative exist (`fem.metalinc:165–267,1216–1248`). The fibre comes from `mixedMaterials[object.materialIndex]`, so it is uniform within an object. Learned material invariants use the same uniform fibre.
- Nodal activation is an accepted/candidate field. The existing mixed owner has scalar diffusion and a sigmoid electrical drive with on/off rates (`mixed_fem.metalinc:80–166`). This is not evidence of a cardiac excitation–contraction model, calcium dynamics, length-dependent active tension, or a sourced activation waveform.

## Smallest complete extension

Add an immutable, source-bound **per-element material field context**, initially supporting a reference fibre (and an explicitly supplied full material frame when required). Keep the shared node arena and one physical continuum owner. The passive expression/AD evaluator must be able to consume that context without cloning a complete material program and environment parameter block for every tetrahedron. The existing generic energy/AD owner can remain the constitutive authority; Guccione does not require a second mechanical solver.

An explicit reference-frame evaluator is another viable implementation: transform `F` and every tangent direction consistently into the authored orthonormal frame and transform stress/tangent back. This needs an explicit material capability declaration, including internal-state frame semantics. It must not quietly rotate arbitrary existing materials or their stored tensors. A field operand supplying the fibre to an invariant energy avoids that ambiguity for a fibre-only transversely isotropic law.

Every relevant path must resolve the same element context: passive stress; algorithmic and rate tangent; local state/validity evaluation; mixed active stress and its deformation/activation derivative; learned invariants if supported; preconditioner; final certification. Existing important call sites are `fem.metalinc:869,1151,1639,1790` plus the mixed/learned branches immediately beside them. Fixing only the first residual would create an inconsistent Newton operator.

Keep global/object-level fibre behavior as an explicit legacy fallback. A material that requires an authored element field must reject absent fields, bad counts, missing source identities, nonfinite or zero vectors, invalid frame handedness, and bad normalization. Preserve original source vectors and record any deterministic normalization separately. If the selected Guccione law is transversely isotropic about the fibre, a separately measured sheet vector is unnecessary; an orthotropic sheet-dependent law requires that additional source and cannot invent it from the fibre alone.

Start with immutable FEM. Reject adaptive transfer and topology mutation for element fields until their frame/state transfer has an explicit owner. Bind the exact field array and source mapping into package serialization, physical fingerprint, immutable-buffer alias protection and reset/restore identity. Bump the relevant ABI/package versions for the actual new layout; do not use unchecked padding.

## Regional materials are a second invariant

Relaxing `tet.identity.x == object.materialIndex` alone is insufficient. Compiler nodal mass assembly currently uses one object material density (`compiler_impl.cpp:1750`), rate selection uses one object material (`:1019–1031`), and mixed field assembly/active coefficients repeatedly use `object.materialIndex`. A regional-material extension must select per-element material for mass, stress, tangent and mixed coefficients, and define consistent assembly at shared nodes. Its CFL/stability estimate must bound every region. Splitting a conforming cardiac mesh into independently owned objects to obtain different fibres or tags would duplicate interface nodes and lose tissue continuity unless an additional constraint law owned that connection.

For an initial passive admission with one explicitly declared tissue law/density, cell region tags can be retained as source identity without being falsely interpreted as distinct constitutive materials. Spatial fibres still require the element field path above.

## Guccione qualification gates

`mixedFirstPiola` removes the analytic stress's hydrostatic Kirchhoff component and inserts the independent mixed pressure. Therefore the selected Guccione energy, isochoric/volumetric split, bulk law and pressure sign must be explicit. It is not sufficient to insert an arbitrary exponential strain energy and call the resulting mixed response source reproduction. No coefficients, stress-free reference state or density are established by this code audit.

Required focused checks are two neighbouring elements with distinct known fibre directions, rotational covariance, fibre-sign invariance where the law has it, sourced-frame preservation through cooking, passive stress/energy and Jv against an independent formula, active deformation and activation tangents, shared-node mass assembly, final residual consistency, and rejection of missing/forged field data. A passive inflation comparison should precede any claim about active cardiac contraction. Geometry admission, material source reproduction, numerical convergence and biological calibration remain separate evidence.

## Source hashes inspected

| Native owner | SHA256 |
|---|---|
| `matter/include/numi/matter/matter.hpp` | `4d18268d9cbff3da993f317975b5584f942d3bfc2e2f8015bdaf0c9e4e880b67` |
| `matter/include/numi/matter/shared.h` | `755c91ee8a94851496d37cd6c113546b8b17147c4fbb739bb9d75e271a94666d` |
| `matter/src/compiler_impl.cpp` | `b7bf038861d8fdf3b510a7e76619990b3410346ad8bb467776f617a0da03ddfd` |
| `matter/src/validation_impl.cpp` | `b04cfaaee731c53be682a614b49bb10cb39036d257559daf54f0679b9d0c4dcd` |
| `matter/src/metal/fem.metalinc` | `72625459d0f26a87d4b20201dd6eb18b427711b127135c58286edf50daff0623` |
| `matter/src/metal/common.metalinc` | `d6000c6319ea991870053b0348029e451cae1dcf7dcb5f55fe5224e298a50847` |
| `matter/src/metal/mixed_fem.metalinc` | `5042632a1d44d12cabbc3d09fa6dc9c9f40ac51b3f81ab733ccedd95b58b4f0a` |
