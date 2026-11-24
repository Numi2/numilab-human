"""Test exact source-driven child re-emission without installing or moving the registered reference."""
from pathlib import Path
import json,sys,struct
import numpy as np
from audit_current_tissue_parents import records
from compile_connected_joint import digest

def run(job,scene,out):
    rd=lambda p:json.loads(p.read_bytes());args=rd(scene/'launch.json')['argv'];path=Path(args[args.index('--soft-tissue-payload')+1]);audit=job/'tissue-pose-diagnosis';report=rd(audit/'result.json')
    if digest(path)!=report['input']:raise ValueError('Incorrect child')
    raw=bytearray(path.read_bytes());_,_,N,B,V,I,_,_=struct.unpack_from('<8s6I32s',raw);rows=np.frombuffer(raw,'<u4',8*N,64).reshape(-1,8);verts=np.ndarray((V,3),'<f4',buffer=raw,offset=64+32*N+36*B,strides=(56,4))
    faces=np.frombuffer(raw,'<u4',I,64+32*N+36*B+56*V)
    ids={int(k) for k,v in report['surfaces'].items() if v['failing']};poses=[r['rest'] for r in rd(scene/'registered-initial-bodies.json')['all_bodies']]
    before=records(path,poses,ids);out.mkdir();changes={};summary={}
    for sid in sorted(ids):
        z=np.load(audit/f'{sid}.npz');which=z['nodes'];first=int(z['first']);delta=np.linalg.solve(z['jacobian'],(z['target']-z['current'])[:,:,None])[:,:,0]
        verts[which+first]=(verts[first+which].astype(float)+delta).astype('f4')
        row,=rows[rows[:,6]==sid].astype(int);_,_,_,count,fi,ic,_,_=row
        tris=(faces[fi:fi+ic].reshape(-1,3)-first).astype(int);touch=np.any(np.isin(tris,which),axis=1);t=tris[touch]
        current,M,_,_=before[sid];proposed=current.copy()
        # Project the actual emitted float coordinates using the current frozen linear map.
        old=np.ndarray((V,3),'<f4',buffer=path.read_bytes(),offset=64+32*N+36*B,strides=(56,4))
        proposed[which]+=np.einsum('nij,nj->ni',M[which],verts[first+which].astype(float)-old[first+which])
        def normals(x):a,b,c=x[t].transpose(1,0,2);return np.cross(b-a,c-a)
        a=normals(current);b=normals(proposed);ma=np.linalg.norm(a,axis=1);mb=np.linalg.norm(b,axis=1);valid=(ma>0)&(mb>0);cos=np.einsum('ij,ij->i',a[valid],b[valid])/(ma[valid]*mb[valid])
        e=np.linalg.norm(proposed[which]-z['target'],axis=1);changes[sid]=[which,proposed[which]]
        summary[sid]=dict(vertices=len(which),incident_faces=len(t),original_degenerate=int((ma==0).sum()),new_degenerate=int(((ma>0)&(mb==0)).sum()),normal_reversals=int((cos<=0).sum()),minimum_normal_cosine=float(cos.min()),minimum_area_ratio=float((mb[valid]/ma[valid]).min()),maximum_area_ratio=float((mb[valid]/ma[valid]).max()),max_rest_error_um=float(e.max()*1e6),maximum_world_move_mm=float(np.linalg.norm(proposed[which]-current[which],axis=1).max()*1000))
    # Store the trial changes as a compact diagnostic. Never overwrite the source.
    np.savez_compressed(out/'positions.npz',**{str(k):np.column_stack(v) for k,v in changes.items()})
    result=dict(source=report['input'],surfaces=summary);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(summary)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
