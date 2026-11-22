"""Test separation on every nonincident vertex/face and edge/edge feature of the failed face pairs."""
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
from material_context import sha

def split(p):
    hi=np.float32(p)
    lo=np.float32(p-hi.astype(float))
    return ' '.join(format(float(v),'.9g') for v in np.r_[hi,lo])

def test(run,source,probe):
    j=json.loads((run/'launch.json').read_text());report=json.loads(source.read_text())
    start=Path(j['environment']['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'])
    end=run/'physical-accepted.node';lib=Path(j['environment']['NUMI_MATTER_METALLIB_OVERRIDE'])
    ep=json.loads((run/'verified-endpoint.json').read_text())
    if (sha(start)!=report['installed_sha256'] or sha(end)!=report['physical_position_sha256']
            or ep['frame_sha256']!=report['frame_sha256'] or sha(lib)!=j['pins'][str(lib)]):
        raise ValueError('Unknown trajectory or kernel library')
    points=[np.loadtxt(f,skiprows=1,usecols=(1,2,3)) for f in (start,end)]
    result=[];case=[]
    for i,row in enumerate(report['rows']):
        if not row['accepted_piercings'] or row['installed_piercings']:raise ValueError('Missing transverse crossing')
        a,b=row['material_nodes']
        result.append(dict(faces=row['deep_faces'],shared_nodes=row['shared_nodes'],tested_features=0,unproven_separation=[]))
        for left,right in ((a,b),(b,a)):
            for n in left:
                if n not in right:case.append((i,0,[n,*right]))
        for ai,n in enumerate(a):
            for bi,m in enumerate(b):
                e=[n,a[(ai+1)%3],m,b[(bi+1)%3]]
                if len(set(e))==4:case.append((i,1,e))
    for offset in range(0,len(case),16):
        batch=case[offset:offset+16];data=[str(len(batch))]
        for _,kind,ids in batch:
            data.append('0 '+str(kind))
            for pose in points:data.extend(split(pose[n]) for n in ids)
        res=subprocess.run([str(probe),str(lib)],input='\n'.join(data)+'\n',text=True,capture_output=True,check=True)
        lines=res.stdout.splitlines()
        if len(lines)!=len(batch):raise ValueError('Incomplete GPU results')
        for (k,kind,ids),line in zip(batch,lines):
            m=re.fullmatch(r'witness=(\d+) clearance_m=\S+ full_sweep_separated=([01])',line.strip())
            if not m:raise ValueError('Unexpected native probe output '+line)
            result[k]['tested_features']+=1
            if m[2]=='0':result[k]['unproven_separation'].append(dict(kind=kind,nodes=ids))
    return dict(accepted_sha256=sha(end),library_sha256=sha(lib),probe_sha256=sha(probe),rows=result,
        scope='Installed GPU unresolved-sweep separation test on the straight chord from the loaded state to the physical endpoint. Zero clearance tests geometric intersection. An unresolved result is conservative. This does not reconstruct the swept trajectory of individual Newton trials or establish which candidate was emitted at runtime.')

if __name__=='__main__':print(json.dumps(test(*(Path(p).resolve() for p in sys.argv[1:])),indent=2))
