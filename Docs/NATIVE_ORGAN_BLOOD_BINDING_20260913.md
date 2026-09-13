# Native organ-to-blood binding receipt — 2026-09-13

The Human organ/blood bridge now reaches the native ABI38 vascular compiler through an explicit test target. The check reads the immutable four-chamber bridge receipt, retains the moments receipt identity and each BodyParts3D member SHA in VascularCavitySource.sourceIdentity, and compiles four deforming cavity compartments, four real FEM shell objects, and four single-owner blood regions. It verifies canonical compartment-to-cavity ownership, the anatomical member strings, 32 FEM binding nodes, and cooked initial owner mass.

The native admission is intentionally a compiler/identity gate. The shell geometry and 1060 kg/m^3 density are explicit synthetic fixtures; they do not promote the atlas surface integrals to physical organ volumes or claim subject calibration.

Native evidence:

- Mac mini: Apple M4 Pro, macOS 26.6, Matter ABI38.
- Branch: human-blood-mass-20260913.
- Commit: dfd3696444a8badc65e140b04b95f9fe395054df.
- Source bridge: Docs/media/organ-blood-cavity-bridge-20260913/bridge.json.
- Log: Docs/media/organ-blood-cavity-bridge-20260913/native/source-native-binding.log.
- Log and prior owner recheck hashes: Docs/media/organ-blood-cavity-bridge-20260913/native/SHA256SUMS.
- Receipt: Docs/media/organ-blood-cavity-bridge-20260913/native/source-native-binding-receipt.json.

The isolated native run passed:

    vascular_compiler=pass
    cavity_geometry_oracle=pass
    cavity_coupled_equations=pass
    cavity_production_operators=pass
    cavity_moving_wall=pass
    human_organ_blood_native_binding=pass
    abi=38 chamber_bindings=4 tissue_owners=4 fem_binding_nodes=32

The remaining gates are still explicit: subject-specific material and density calibration, body-frame anatomical registration, pressure-gradient blood momentum, two-way blood/tissue mass transfer, force convergence, the 12.5 microsecond clock at the full Human source, sustained standing/walking, and the 420-trial behavior gate.
