"""Build the routed force kernel inside the current NumiMatter library with the original material evaluator."""
from pathlib import Path
import json,sys,subprocess
from compile_connected_joint import digest
def build(manifest,src,out):
    env=json.loads(manifest.read_text())['environment'];base=Path(env['NUMI_MATTER_METALLIB_OVERRIDE']).parent
    if digest(base/'NumiMatter.metallib')!=json.loads(manifest.read_text())['pins'][str(base/'NumiMatter.metallib')]:raise ValueError('Shader generation mismatch')
    out.mkdir();dst=out/'src';dst.mkdir();(dst/'metal').mkdir()
    for path in (base/'src').iterdir():
        if path.name!='metal':(dst/path.name).symlink_to(path.resolve())
    path=base/'src/metal/numi_human.metalinc';shader=path.read_text()
    for f in path.parent.iterdir():
        if f!=path:(dst/'metal'/f.name).symlink_to(f.resolve())
    text=shader[shader.index('    effectiveStretch = ligament.reference.z * currentLength /'):shader.index('    firstForce = tension * delta / currentLength;')]
    if shader.count(text)!=1:raise ValueError('Unknown native passive fiber law')
    helper='inline bool nhCorticalLigamentLaw(NMNumiHumanPassiveLigamentGPU ligament,float currentLength,thread float& effectiveStretch,thread float& tension){\n'+text+'    return true;\n}\n'
    shader=shader.replace(text,'    if(!nhCorticalLigamentLaw(ligament,currentLength,effectiveStretch,tension))return false;\n')
    block=(src/'cortical_ligament_path.metalinc').read_text();split=block.index('inline bool nhCorticalCone')
    opening='namespace numi_matter_metal {'
    shader=shader.replace(opening,opening+'\n'+block[:split],1)
    at=shader.index('inline bool passiveLigamentState(');shader=shader[:at]+helper+'\n'+block[split:]+'\n'+shader[at:]
    at=shader.rindex('} // namespace numi_matter_metal')
    shader=shader[:at]+(src/'cortical_ligament_force.metalinc').read_text()+'\n'+shader[at:];(dst/'metal'/path.name).write_text(shader)
    commands=[[a.replace(str(base)+'/',str(out)+'/') for a in c] for c in json.loads((base/'receipt.json').read_text())['commands'][:2]]
    # Compile through our staged include tree even if the original translation unit is a symlink.
    unit=dst/'matter_kernels.metal'
    if unit.is_symlink():
        contents=unit.read_bytes();unit.unlink();unit.write_bytes(contents)
    with (out/'build.log').open('wb') as f:
        for cmd in commands:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    (out/'build.json').write_text(json.dumps(dict(source=digest(path),shader=digest(dst/'metal'/path.name),path=digest(src/'cortical_ligament_path.metalinc'),kernel=digest(src/'cortical_ligament_force.metalinc'),commands=commands,metallib=digest(out/'NumiMatter.metallib')),indent=2)+'\n')
if __name__=='__main__':build(*map(lambda v:Path(v).resolve(),sys.argv[1:]))
