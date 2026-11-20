"""Build a shader-only witness for the largest rejecting equilibrium block."""
import difflib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
base = root.parent / 'resting-clearance-and-atrium-011/failure-owner-shader'
build = root / 'residual-bank-shader'
build.mkdir()
for entry in (base / 'src').rglob('*'):
    target = build / entry.relative_to(base)
    if entry.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    elif entry.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(entry.resolve(), target)

path = build / 'src/metal/mixed_fem.metalinc'
original = path.read_text()
updated = original
definition = """
// Diagnostic-only local witness. The normal certificate remains authoritative.
inline void nmMixedObserveResidualRow(
    thread float& sum, thread float& peak, thread uint& witness,
    float normSquared, uint row) {
    sum += normSquared;
    if (normSquared > peak || (normSquared == peak && row < witness)) {
        peak = normSquared;
        witness = row;
    }
}
"""
anchor = '// Legacy and tiled measurement share one publication/acceptance authority.'
assert updated.count(anchor) == 1
updated = updated.replace(anchor, definition + '\n' + anchor)
anchor = '    const float minimumJ, const float volumeNorm2, const uint activeElements\n) {'
assert updated.count(anchor) == 1
updated = updated.replace(anchor, anchor.replace('activeElements\n','activeElements,\n    const float2 residualWitness\n'))
anchor = '                             volumeResidual, -902.0f));'
assert updated.count(anchor) == 1
updated = updated.replace(anchor, '''                             residualWitness.y == 0.0f ? volumeResidual : residualWitness.x,
                             residualWitness.y == 0.0f ? -902.0f : residualWitness.y));''')

witness = """    float2 residualWitness(0.0f);
    if (lane == 0u && isfinite(relativeResidual) &&
        relativeResidual > solver.residualTolerances.x &&
        atomic_load_explicit(reinterpret_cast<device atomic_uint*>(
            &statuses[environment].code), memory_order_relaxed) == NM_STATUS_SUCCESS &&
        dispatch.femNodeCount < 1048575u &&
        dispatch.rigidGeneralizedCapacity < 1048575u &&
        layout.supportContactCount < 1048575u &&
        dispatch.mpmActiveNodeCapacity == 0u) {
        // Banks 1..5 encode FEM velocity, FEM pressure, transport,
        // rigid generalized momentum, and fixed support constraints.
        float norms[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float peaks[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        uint witnesses[5] = {NM_INVALID_INDEX, NM_INVALID_INDEX, NM_INVALID_INDEX,
                            NM_INVALID_INDEX, NM_INVALID_INDEX};
        for (uint i = 0u; i < dispatch.objectCount; ++i) {
            const NMContinuumObjectGPU localObject = objects[i];
            if (localObject.representation != NM_REPRESENTATION_FEM) continue;
            for (uint j = localObject.stateOffset;
                 j < localObject.stateOffset + localObject.stateCount; ++j) {
                const float4 r = generalizedResidual[nodeBase + j];
                nmMixedObserveResidualRow(norms[0], peaks[0], witnesses[0], dot(r.xyz,r.xyz), j);
                nmMixedObserveResidualRow(norms[1], peaks[1], witnesses[1], r.w*r.w, j);
                if ((localObject.flags & NM_OBJECT_MULTIPHYSICS) != 0u) {
                    const float4 f = physicalFieldResidual[nodeBase + j];
                    nmMixedObserveResidualRow(norms[2], peaks[2], witnesses[2], dot(f,f), j);
                }
            }
        }
        for (uint j = 0u; j < dispatch.rigidGeneralizedCapacity; ++j) {
            const float r = generalizedResidual[fgmresRigidIndex(dispatch, environment, j)].x;
            nmMixedObserveResidualRow(norms[3], peaks[3], witnesses[3], r*r, j);
        }
        for (uint j = 0u; j < layout.supportContactCount; ++j) {
            const float3 r = generalizedResidual[layout.supportBase +
                environment * layout.supportContactCount + j].xyz;
            nmMixedObserveResidualRow(norms[4], peaks[4], witnesses[4], dot(r,r), j);
        }
        uint selected = 0u;
        for (uint j = 1u; j < 5u; ++j) {
            if (norms[j] > norms[selected]) selected = j;
        }
        if (isfinite(norms[selected]) && witnesses[selected] != NM_INVALID_INDEX) {
            const uint tag = ((selected + 1u) << 20u) + witnesses[selected] + 1u;
            residualWitness = float2(sqrt(norms[selected]) / initialNorm, -float(tag));
        }
    }
"""
anchor = '    nmMixedPublishCertificate(dispatch, solver, object, objectIndex,'
assert updated.count(anchor) == 2
updated = updated.replace(anchor, witness + anchor, 1)
anchor = '        minimumJ, volumeNorm2, activeElements);'
assert updated.count(anchor) == 2
updated = updated.replace(anchor,'        minimumJ, volumeNorm2, activeElements, float2(0.0f));')
updated = updated.replace('        minimumJ, volumeNorm2, activeElements, float2(0.0f));',
                          '        minimumJ, volumeNorm2, activeElements, residualWitness);', 1)
path.unlink()
path.write_text(updated)
(root / 'residual-bank.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True), updated.splitlines(keepends=True),
    'a/matter/src/metal/mixed_fem.metalinc', 'b/matter/src/metal/mixed_fem.metalinc')))

commands = json.loads((base / 'build-commands.json').read_text())
commands = [[str(build / Path(argument).relative_to(base))
             if argument.startswith(str(base) + '/') else argument for argument in command]
            for command in commands]
(build / 'build-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
with (root / 'residual-bank-build.log').open('wb') as log:
    for command in commands:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
Path(commands[0][-1]).unlink()
