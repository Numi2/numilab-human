"""Intersect the source-registered surfaces before native integration."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_material_interface import read_inputs,solid
from compare_hand_interfaces import query
from count_accepted_cortical_crossings import dyadic,plane,straddles
from qualify_material_motion import frame
from material_context import sha
def measure(v,f,b,g,exe):
    x=np.vstack((v,b)).astype(float);faces=np.vstack((f,g+len(v)));cut=np.arange(len(faces))<len(f)
    pairs=query(exe,x,faces,cut);xyz=dyadic(x);proper=0;boundary=0
    for a,j in pairs:
        if not(a<len(f)<=j):continue
        A=[xyz[k] for k in faces[a]];B=[xyz[k] for k in faces[j]]
        if straddles(A,B[0],plane(*B)) and straddles(B,A[0],plane(*A)):proper+=1
        else:boundary+=1
    a=solid(v,f,'source material')[0];c=solid(b,g,'source cortex')[0]
    return dict(transverse=proper,boundary=boundary,closed_overlap_ml=(a^c).volume()*1.e6)
def run(part,assets,cfg,receipt,cap,base,exe,out):
    source,lineage=read_inputs(cfg,receipt,cap,base)
    d=json.loads((assets/'manifest.json').read_text());banks=d['matched_cortical_banks'];frames={r['body_index']:frame(r) for r in json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']}
    samples=np.fromfile(assets/'cortical-samples.bin','f4').reshape(2,256,5,4);tris=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)
    report={}
    for side,sid in enumerate((29,30)):
        mesh=np.load(part/'conforming-boundary'/f'{sid}.npz');v=mesh['vertices'].astype(float);f=mesh['face'];ports=np.load(part/'ports'/f'{sid}.npz')
        bank={str(ports['target_bank'][i]):tuple(samples[side,j,4,:2].copy().view('u4').astype(int)) for j,i in enumerate(d['sides'][side]['quadrature_indices'])}
        for name,(first,n) in bank.items():
            b,g=source[name];owner=banks[name.split('_')[-1]]['owner']
            local=Rotation.from_quat(frames[owner][3:]).inv().apply(v-frames[owner][:3]);native,idx=np.unique(tris[first:first+n,:3,:3].reshape(-1,3),axis=0,return_inverse=True);native=native.astype(float)
            samples_by_format={'source_world_binary64':(v,f,b,g),'rounded_source_world':(v.astype('f4').astype(float),f,b.astype('f4').astype(float),g),
                               'rounded_cortical_local':(local.astype('f4').astype(float),f,native,idx.reshape(-1,3))}
            for mode,args in samples_by_format.items():report.setdefault(mode,{})[f'{sid}/{owner}']=measure(*args,exe)
    result=dict(source=lineage,registered_assets=sha(assets/'manifest.json'),predicate=sha(exe),regions=report)
    result['totals']={mode:{k:sum(x[k] for x in banks.values()) for k in ['transverse','boundary','closed_overlap_ml']} for mode,banks in report.items()}
    out.write_text(json.dumps(result,indent=2)+'\n');print(result['totals'],flush=True)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
