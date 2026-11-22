"""Build an isolated Metal experiment using the same FEM node update for trial and application."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from material_context import sha


def build(base,control,out):
    launch=json.loads((control/'launch.json').read_text())
    metal=Path(launch['environment']['NUMI_MATTER_METALLIB_OVERRIDE'])
    if metal!=base/'NumiMatter.metallib' or sha(metal)!=launch['pins'][str(metal)]:raise ValueError('Different Matter owner')
    orig=base/'src/metal/fem.metalinc'
    cpp=orig.read_text()
    begin=cpp.index('inline NMFEMNodeStateGPU femTrialNodeAtAlpha('); finish=cpp.index('    return next;\n}',begin)
    old=cpp[begin:finish]
    a=old.index('    next.velocityAndInverseMass.xyz +=')
    b=old.index('            next.referenceDisplacementAndMode[axis] = displacement.high;')
    seed=old[a:b]
    new=old.replace(seed,'''    if (next.referenceDisplacementAndMode.w == 1.0f && dt > 0.0f) {
        // Update the authoritative displacement along the physical Newton direction.
        // Recover velocity and the inertial increment from this same displacement.
        for (uint axis = 0u; axis < 3u; ++axis) {
            const MRCompensatedScalar dv = mrCompensatedProduct(alpha, correction[axis]);
            const MRCompensatedScalar du = mrCompensatedMultiply({dt, 0.0f}, dv);
            const MRCompensatedScalar displacement = mrCompensatedAdd(
                {next.referenceDisplacementAndMode[axis], next.referenceDisplacementLow[axis]}, du);
            const MRCompensatedScalar stepDisplacement = mrCompensatedAdd(displacement,
                {-acceptedState.referenceDisplacementAndMode[axis], -acceptedState.referenceDisplacementLow[axis]});
            const MRCompensatedScalar velocity = mrCompensatedDivide(stepDisplacement, {dt, 0.0f});
            const MRCompensatedScalar stepVelocity = mrCompensatedAdd(velocity,
                {-acceptedState.velocityAndInverseMass[axis], 0.0f});
            next.velocityAndInverseMass[axis] = velocity.high + velocity.low;
            next.deltaVelocity[axis] = stepVelocity.high + stepVelocity.low;
''').replace('    } else {\n        next.positionAndMass','    } else {\n        next.velocityAndInverseMass.xyz += alpha * correction;\n        next.deltaVelocity.xyz = next.velocityAndInverseMass.xyz - acceptedState.velocityAndInverseMass.xyz;\n        next.positionAndMass')
    if seed.count('mrCompensatedProduct')!=1 or old.count('    } else {')!=1 or cpp.count(old)!=1:raise ValueError('Unknown FEM update')
    out.mkdir(exist_ok=False);src=out/'src';sub=src/'metal';sub.mkdir(parents=True)
    for p in (base/'src').iterdir():
        if p.name!='metal':(src/p.name).symlink_to(p)
    for p in (base/'src/metal').iterdir():
        if p.name!=orig.name:(sub/p.name).symlink_to(p)
    (sub/orig.name).write_text(cpp[:begin]+new+cpp[finish:])
    diff=subprocess.run(['diff','-u','--label','a/matter/src/metal/fem.metalinc','--label','b/matter/src/metal/fem.metalinc',str(orig),str(sub/orig.name)],capture_output=True,text=True)
    if diff.returncode!=1:raise ValueError('Missing isolated change')
    (out/'reference-update.patch').write_text(diff.stdout)
    commands=json.loads((base.parent/'contact-prefix-certification/build-commands.json').read_text())
    c=commands[0].copy();c[c.index('-c')+1]=str(src/'matter_kernels.metal');c[-1]=str(out/'NumiMatter.air')
    link=commands[1].copy();link[4]=c[-1];link[-1]=str(out/'NumiMatter.metallib')
    for cmd in (c,link):subprocess.run(cmd,check=True)
    rec=dict(source=str(orig),source_sha256=sha(orig),new_sha256=sha(sub/orig.name),control_metallib_sha256=sha(metal),metallib_sha256=sha(Path(link[-1])),commands=[c,link],patch_sha256=sha(out/'reference-update.patch'))
    (out/'identities.json').write_text(json.dumps(rec,indent=2)+'\n')


if __name__=='__main__':build(*(Path(s).resolve() for s in sys.argv[1:]))
