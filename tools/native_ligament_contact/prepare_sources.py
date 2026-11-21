"""Prepare a private build from the published, hash-pinned SSH Numi owners."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

def digest(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','equilibrium','capsule','project','evidence','work'):
        parser.add_argument(name,type=Path)
    a=parser.parse_args()
    src=Path(__file__).resolve().parent
    p,e,c,app=a.scene,a.equilibrium,a.capsule,a.project
    ids=json.loads((a.evidence/'build-identities.json').read_text())
    files={'include/numi/matter/numi_human_shared.h':app/'matter/include/numi/matter/numi_human_shared.h',
           'include/numi/matter/numi_human.hpp':app/'matter/include/numi/matter/numi_human.hpp',
           'numi_human.mm':p/'pressure-with-abdominal-cavity-build/numi_human.mm',
           'src/metal/numi_human.metalinc':e/'src/metal/numi_human.metalinc',
           'NumiHumanRegisteredLigaments.hpp':c/'emitted-native/build/NumiHumanRegisteredLigaments.hpp',
           'numilab_human_myosim_visual_probe.mm':c/'emitted-native/build/numilab_human_myosim_visual_probe.mm'}
    out=a.work/'build'
    out.mkdir(parents=True,exist_ok=False)
    for name,input_path in files.items():
        if digest(input_path)!=ids['original'][name]:raise ValueError(f'Native source drift in {name}')
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(input_path,target)
    subprocess.run(['git','apply','--unidiff-zero','--whitespace=error',str(src/'native-owner.patch')],cwd=out,check=True)
    for name,file in files.items():
        if digest(out/name)!=ids['patched'][name]:raise ValueError(f'Patched source mismatch in {name}')
    for name,expected in ids['shaders'].items():
        dep=e/'src/metal'/name
        if digest(dep)!=expected:raise ValueError(f'Unexpected shader include {name}')
        (out/'src/metal'/name).symlink_to(dep)
    big=e/'src/matter_kernels.metal'
    if digest(big)!=ids['mattersource'] or digest(src/'RegisteredMaterialBranches.hpp')!=ids['catalog'] or digest(a.evidence/'route.json')!=ids['route']:
        raise ValueError('Catalog, route or Metal library source changed')
    shutil.copyfile(big,out/'src/matter_kernels.metal')
    shutil.copyfile(src/'RegisteredMaterialBranches.hpp',out/'RegisteredMaterialBranches.hpp')
    shutil.copyfile(a.evidence/'route.json',a.work/'strip.json')
    for name in ('audit_gpu.mm','test_source_rows.py','test_straight_rows.py','build_native.py','run_scene.py','compose_current.py','rebind_current.py','verify_trace.py'):
        shutil.copyfile(src/name,a.work/name)
if __name__=='__main__':
    main()
