#!/usr/bin/env python3
"""Execute independent compiled FEBio 3.0/Matter stress and tangent comparisons.

This does not qualify reported legacy energy, tissue equilibrium, or a knee.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import numpy as np

SOURCE_COMMIT='47746329aea1ec32201ba556afe4da863357efad'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matter',type=Path,required=True)
    p.add_argument('--reference-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build=a.reference_build.resolve()
    source=build/('FEBio-'+SOURCE_COMMIT);oracle=build/'febio-build/febio-material-oracle'
    build_receipt=json.loads((build/'build-receipt.json').read_text())
    if (build_receipt['status']!='built_comparison_dependency_not_original_binary' or
        build_receipt['version']!='3.0.0' or build_receipt['sources']['febio3']['commit']!=SOURCE_COMMIT):
        raise ValueError('requires the pinned public FEBio 3.0 comparison build')
    # A later-added oracle target is allowed, but its probe must match this
    # recipe. Hash actual binaries; never reuse the original solver identity.
    recipe=Path(__file__).resolve().parent
    if sha(source/'OpenKneeMaterialOracle.cpp')!=sha(recipe/'febio_material_oracle.cpp'):
        raise ValueError('compiled oracle source does not match the comparison recipe')
    # Verify the material implementation against the immutable public archive,
    # not merely the patch's stated intent. Include shared frame/point classes.
    archive=build/'febio3.tar.gz'
    if sha(archive)!=build_receipt['sources']['febio3']['sha256']:
        raise ValueError('public source archive identity drift')
    source_hashes={}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts=Path(member.name).parts
            if len(parts)!=3 or not member.isfile() or parts[1] not in ('FEBioMech','FECore'):
                continue
            if parts[2].endswith(('.cpp','.h')):
                relative='/'.join(parts[1:]);original=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                if sha(source/relative)!=original:raise ValueError('source mechanics modified: '+relative)
                source_hashes[relative]=original
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'running','source_equivalence':False,'assembled_equilibrium':False,
             'legacy_energy_output_equivalence':False,'oracle_version':'public FEBio 3.0.0, not archived 2.9.1',
             'scope':'compiled constitutive stress and directional tangent at identical deformations',
             'stress_measure':'first Piola P=J sigma F^-T, Pa',
             'tangent_measure':'dP[dF]=J (c:sym(dF F^-1)+(dF F^-1) sigma) F^-T',
             'normalization':'Frobenius norm / max(1e6 Pa, reference Frobenius norm)',
             'gates':{'stress':1e-10,'directional_tangent':1e-10},
             'unmodified_mechanics_file_count':len(source_hashes),
             'unmodified_mechanics_manifest_sha256':hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest(),
             'identities':{name:{'file':str(path.resolve()),'sha256':sha(path)} for name,path in
                [('matter_binary',a.matter),('febio_oracle',oracle),('oracle_source',recipe/'febio_material_oracle.cpp'),
                 ('comparison_recipe',Path(__file__)),('reference_build_receipt',build/'build-receipt.json')]}}
    try:
        commands=[('matter',[str(a.matter.resolve()),'--reference-export',str(out/'inputs.txt'),str(out/'matter.txt')]),
                  ('febio',[str(oracle),str(out/'inputs.txt'),str(out/'febio.txt')])]
        receipt['commands']=[command for _,command in commands]
        for name,command in commands:
            with (out/(name+'.stdout')).open('x') as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        inputs=np.loadtxt(out/'inputs.txt');native=np.loadtxt(out/'matter.txt');reference=np.loadtxt(out/'febio.txt')
        if inputs.shape!=(486,29) or native.shape!=(486,18) or reference.shape!=native.shape:
            raise ValueError('incomplete material comparison')
        if not all(np.isfinite(x).all() for x in [inputs,native,reference]):raise ValueError('nonfinite comparison')
        for name,sl in [('stress',slice(None,None,2)),('directional_tangent',slice(1,None,2))]:
            error=np.linalg.norm(native[:,sl]-reference[:,sl],axis=1)/np.maximum(1e6,np.linalg.norm(reference[:,sl],axis=1))
            receipt['maximum_'+name+'_error']=float(error.max())
            if error.max()>=receipt['gates'][name]:raise ValueError(name+' source equations disagree')
        receipt.update(status='passed_constitutive_equations_only',rows=486,
                       tissues=['ACL','PCL','MCL','LCL','PTL','QAT','MNS-L','MNS-M','cartilage'])
    except BaseException as error:
        receipt.update(status='failed_comparison',error=str(error));raise
    finally:
        receipt['artifact_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
