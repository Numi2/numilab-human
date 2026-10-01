#!/usr/bin/env python3
"""Build a public comparison solver on Apple silicon, without login.

This is an independent reference dependency, not the Human/Matter runtime.
Requires existing Xcode/clang, CMake and libomp. Never installs system packages.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import urllib.request

SOURCES = {
    "febio2": ("Kevin-Mattheus-Moerman/febio2", "fc13f37e72cd0014535e1c2ea5f3d0396c447165",
               "72eca36868f42df23ce9d7035a11031b12f323770d0d2c57f7ac3a24856bc321"),
    "superlu": ("xiaoyeli/superlu", "a8a0caafa593772450b35339c8fa4e76b7bc934f",
                "82fb2b3d7e50728daa22d1e64f0f66545097072a71dbc7711d060ade369d8b36"),
    "febio3": ("febiosoftware/FEBio", "47746329aea1ec32201ba556afe4da863357efad",
               "9a94c34d206c7657caee3a71bf741e9a319a6e5db0ec25b410dd4794ae38ee89"),
}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--openmp-root",type=Path,default=Path("/opt/homebrew/opt/libomp"))
    p.add_argument("--jobs",type=int,default=4)
    p.add_argument("--version",choices=['2.9.0','3.0.0'],default='3.0.0')
    a=p.parse_args()
    if platform.system()!="Darwin" or platform.machine()!="arm64":
        p.error("this recipe is qualified only on Apple arm64")
    if not 1<=a.jobs<=8 or not (a.openmp_root/'lib/libomp.dylib').is_file():
        p.error("supply an existing libomp installation and 1 to 8 build jobs")
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    recipe=Path(__file__).resolve().parent
    owner='febio3' if a.version=='3.0.0' else 'febio2'
    patch='febio-'+a.version+'-arm64.patch'
    cmake='CMakeLists-febio3.txt' if owner=='febio3' else 'CMakeLists.txt'
    receipt={"status":"building","version":a.version,"original_version":"2.9.1",
             "source_equivalence":False,"sources":{},"commands":[],
             "machine":platform.platform(),"compiler":subprocess.check_output(['clang++','--version'],text=True),
             "linear_backend":"SuperLU 5.2.2 with Apple Accelerate BLAS",
             "openmp_sha256":sha(a.openmp_root/'lib/libomp.dylib'),
             "recipe_sha256":{n:sha(recipe/n) for n in
                 ['build_public_reference.py',cmake,patch,'febio_joint_oracle.cpp']+
                 (['febio_material_oracle.cpp'] if owner=='febio3' else [])},
             "unsupported_optional_backends":['MKL/Pardiso','MKL preconditioners','FEAST','BIPN'],
             "energy_output_limitation":"GSL not enabled; legacy fibre strain-energy output is unavailable. No energy-output equivalence claim."}
    def run(command,cwd=root):
        receipt['commands'].append(command)
        with (root/'build.log').open('a') as log:
            subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        for name in [owner,'superlu']:
            repo,commit,expected=SOURCES[name]
            url=f'https://codeload.github.com/{repo}/tar.gz/{commit}'
            archive=root/(name+'.tar.gz')
            with urllib.request.urlopen(url,timeout=60) as response,archive.open('xb') as out:
                shutil.copyfileobj(response,out)
            if sha(archive)!=expected:raise ValueError(f'{name} archive hash mismatch')
            receipt['sources'][name]={'url':url,'commit':commit,'sha256':expected}
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    path=(root/member.name).resolve()
                    if not path.is_relative_to(root) or not (member.isfile() or member.isdir()):
                        raise ValueError('unsafe source archive member')
                tar.extractall(root,filter='data')
        source=root/(SOURCES[owner][0].split('/')[-1]+'-'+SOURCES[owner][1]);superlu=root/('superlu-'+SOURCES['superlu'][1])
        run(['patch','--batch','-p1','-i',str(recipe/patch)],source)
        shutil.copy2(recipe/cmake,source/'CMakeLists.txt')
        shutil.copy2(recipe/'febio_joint_oracle.cpp',source/'OpenKneeJointOracle.cpp')
        if owner=='febio3':
            shutil.copy2(recipe/'febio_material_oracle.cpp',source/'OpenKneeMaterialOracle.cpp')
        run(['cmake','-S',str(superlu),'-B',str(root/'superlu-build'),'-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
             '-DCMAKE_BUILD_TYPE=Release','-DXSDK_ENABLE_Fortran=OFF','-Denable_tests=OFF',
             '-Denable_single=OFF','-Denable_complex=OFF','-Denable_complex16=OFF',
             '-Denable_internal_blaslib=OFF','-DTPL_BLAS_LIBRARIES=-framework Accelerate'])
        run(['cmake','--build',str(root/'superlu-build'),'-j',str(a.jobs)])
        build=root/'febio-build'
        run(['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
             '-DOPENMP_ROOT='+str(a.openmp_root.resolve())])
        run(['cmake','--build',str(build),'-j',str(a.jobs)])
        info=subprocess.run([str(build/owner),'-info','-norun'],text=True,capture_output=True)
        # FEBio 3.0 returns 1 for -norun; preserve that status and require the
        # actual version text. A successful compiler exit alone is insufficient.
        if info.returncode not in (0,1) or 'FEBio version  = '+a.version not in info.stdout:
            raise ValueError('built binary did not identify its expected version')
        receipt.update(version_output=info.stdout,version_command_returncode=info.returncode)
        names=['febio2','febio-joint-oracle'] if owner=='febio2' else ['febio3','febio-material-oracle']
        receipt['binaries']={name:{'file':str(build/name),'sha256':sha(build/name)} for name in names}
        receipt['status']='built_comparison_dependency_not_original_binary'
    except BaseException as error:
        receipt.update(status='failed_build',error=str(error));raise
    finally:
        (root/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['binaries'],indent=2))
if __name__=='__main__':main()
