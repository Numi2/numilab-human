"""Compile the bounded native material direction against the complete P1 trial segment."""
import json, shutil, subprocess, sys
from pathlib import Path
from bind_gluteal_receivers import digest, require


def sub(source,before,after):
    require(source.count(before)==1,'Unknown source block '+before[:65])
    return source.replace(before,after)


def build(parent, out, unchanged=False):
    config=json.loads((parent/'build.json').read_bytes())
    orig=parent/'metal'
    wrapper=out/'src'
    wrapper.mkdir(parents=True)
    (wrapper/'metal').mkdir()
    for item in (orig/'src').iterdir():
        if item.name!='metal':
            (wrapper/item.name).symlink_to(item.resolve())
    for item in (orig/'src/metal').iterdir():
        if item.name!='fem.metalinc':
            (wrapper/'metal'/item.name).symlink_to(item.resolve())
    # The translation unit must be local so quoted Metal includes resolve to the changed source.
    root=wrapper/'matter_kernels.metal'
    root.unlink()
    shutil.copyfile(orig/'src/matter_kernels.metal',root)
    fem=(orig/'src/metal/fem.metalinc').read_text()
    before,tail=fem.split('inline uint registeredVolumeCellDirection(',1)
    helper=Path(__file__).with_name('swept_material_direction.metalinc').read_text()
    fn,after=tail.split('kernel void nm_fem_refresh_volume_witness(',1)
    fn=sub(fn,'const float current=determinant(registeredTrialF(p,inv,reference));',
           'const float3x3 startF=registeredTrialF(p,inv,reference);\n    const float current=determinant(startF);')
    fn=sub(fn,'uint refinementsAfterBalance=0u;', 'uint refinementsAfterBalance=0u, priorTerm=0u;')
    fn=sub(fn,'float J=determinant(F);const float unbalancedJ=J;', '''auto control=registeredSweptControl(startF,F,lo,hi);
        float J=control.value;const float unbalancedJ=J;''')
    fn=sub(fn,'if(refine==0u){','if(refine==0u || (control.term!=priorTerm && !(J>lo && J<hi))){\n            lastError=INFINITY;')
    fn=sub(fn,'const float actual=determinant(registeredTrialF(p,inv,reference));',
           'const auto balanced=registeredSweptControl(startF,registeredTrialF(p,inv,reference),lo,hi);\n            const float actual=balanced.value;')
    fn=sub(fn,'J=actual;', 'J=actual;\n            control=balanced;')
    fn=sub(fn,'const float3x3 cofactor=cofactorMatrix(F);',
           'priorTerm=control.term;\n        const float3x3 cofactor=registeredControlGradient(startF,F,control.term);')
    refresh,rest=after.split('kernel void nm_fem_registered_volume_active_incident64(',1)
    refresh=sub(refresh,'NMFEMNodeStateGPU trial[4];bool movable=true;', 'NMFEMNodeStateGPU trial[4],base[4];bool movable=true;')
    refresh=sub(refresh,'const auto state=candidate[n];','const auto state=candidate[n];\n            base[k]=state;')
    refresh=sub(refresh,'const float J=determinant(registeredTrialF(trial,inv,reference));','''const float J=registeredSweptControl(registeredTrialF(base,inv,reference),
            registeredTrialF(trial,inv,reference),lo,hi).value;''')
    src=wrapper/'metal/fem.metalinc'
    src.write_text(before+helper+'\ninline uint registeredVolumeCellDirection('+fn+'kernel void nm_fem_refresh_volume_witness('+refresh+'kernel void nm_fem_registered_volume_active_incident64('+rest)
    if unchanged:
        shutil.copyfile(orig/'src/metal/fem.metalinc',src)
    commands=[]
    for command in config['commands']:
        if not (command[0]=='xcrun' and ('metal' in command or 'metallib' in command)):continue
        args=[x.replace(str(orig)+'/',str(out)+'/') for x in command]
        with (out/f'compile-{len(commands)}.log').open('w') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
        commands.append(args)
    require(len(commands)==2,'Missing native Metal commands')
    info={'original_metallib':digest(orig/'NumiMatter.metallib'),'metallib':digest(out/'NumiMatter.metallib'),
          'original_fem':digest(orig/'src/metal/fem.metalinc'),'fem':digest(src),'helper':digest(Path(__file__).with_name('swept_material_direction.metalinc')),
          'commands':commands}
    (out/'build.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent',type=lambda x:Path(x).resolve())
    parser.add_argument('out',type=lambda x:Path(x).resolve())
    parser.add_argument('--unchanged',action='store_true')
    build(**vars(parser.parse_args()))
