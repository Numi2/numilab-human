"""Test lifted Human attachment motion in FEM determinant backtracking."""
import difflib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
base = root/'residual-bank-shader'
build = root/'moving-attachment-shader'
build.mkdir()
for entry in (base/'src').rglob('*'):
    target = build/entry.relative_to(base)
    if entry.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    elif entry.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(entry.resolve(), target)

path = build/'src/metal/fem.metalinc'
original = path.read_text()
helper = r'''
// The Human attachment correction is already lifted by the exact point
// Jacobian. The rigid owner integrates it over the frame timestep, whereas
// free FEM nodes advance on the object's own clock. This is the linearized
// body motion for backtracking. Exact body kinematics and fresh FEM
// certification still own the resulting candidate.
inline NMFEMNodeStateGPU femBacktrackingNodeAtAlpha(
    const NMFEMNodeStateGPU acceptedState,
    NMFEMNodeStateGPU candidate,
    const float3 correction,
    const float alpha,
    const float objectDt,
    const float bodyDt
) {
    if (!femHumanAttachmentConstraint(candidate.restAndFixed.w))
        return femTrialNodeAtAlpha(
            acceptedState, candidate, correction, alpha, objectDt, true);

    const float3 deltaVelocity = alpha * correction;
    candidate.velocityAndInverseMass.xyz += deltaVelocity;
    if (candidate.referenceDisplacementAndMode.w != 1.0f) {
        candidate.positionAndMass.xyz += bodyDt * deltaVelocity;
        return candidate;
    }
    for (uint axis = 0u; axis < 3u; ++axis) {
        const auto update = mrCompensatedProduct(bodyDt, deltaVelocity[axis]);
        const auto displacement = mrCompensatedAdd(
            {candidate.referenceDisplacementAndMode[axis],
             candidate.referenceDisplacementLow[axis]}, update);
        candidate.referenceDisplacementAndMode[axis] = displacement.high;
        candidate.referenceDisplacementLow[axis] = displacement.low;
        candidate.positionAndMass[axis] = femReferencePositionProjection(
            candidate.restAndFixed[axis], displacement.high, displacement.low);
    }
    return candidate;
}

'''
anchor = 'inline bool femBacktrackingConstitutiveValid('
assert original.count(anchor) == 1
updated = original.replace(anchor, helper + anchor)
for direction in ('nodeCorrection.xyz', 'correction[index].xyz'):
    before = ('const NMFEMNodeStateGPU trialNode = femTrialNodeAtAlpha(\n'
              f'                    accepted[index], candidate[index], {direction},\n'
              '                    alpha, dt, true);')
    after = ('const NMFEMNodeStateGPU trialNode = femBacktrackingNodeAtAlpha(\n'
             f'                    accepted[index], candidate[index], {direction},\n'
             '                    alpha, dt, dispatch.gravityAndTimestep.w);')
    assert updated.count(before) == 1
    updated = updated.replace(before, after)
path.unlink()
path.write_text(updated)
(root/'moving-attachment-geometry.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True), updated.splitlines(keepends=True),
    'a/matter/src/metal/fem.metalinc', 'b/matter/src/metal/fem.metalinc')))
commands = json.loads((base/'build-commands.json').read_text())
commands = [[str(build/Path(arg).relative_to(base)) if arg.startswith(str(base)+'/') else arg
             for arg in command] for command in commands]
(build/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
with (root/'moving-attachment-build.log').open('wb') as log:
    for command in commands:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
Path(commands[0][-1]).unlink()
