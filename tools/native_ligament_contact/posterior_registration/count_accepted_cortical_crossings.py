"""Separate exact intersecting triangle pairs with transverse plane straddles from boundary contact."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from compare_hand_interfaces import query
from material_context import sha

def dyadic(vertices):
    value=[[float(x).as_integer_ratio() for x in v] for v in vertices]
    common=max(d for v in value for _,d in v)
    return [[n*(common//d) for n,d in v] for v in value]
def plane(a,b,c):
    e=[j-i for i,j in zip(a,b)];f=[j-i for i,j in zip(a,c)]
    return [e[1]*f[2]-e[2]*f[1],e[2]*f[0]-e[0]*f[2],e[0]*f[1]-e[1]*f[0]]
def straddles(points,origin,n):
    d=[sum((x-y)*c for x,y,c in zip(p,origin,n)) for p in points];return min(d)<0<max(d)
def test(assets,surface,snapshot,predicate,out):
    with snapshot.open() as f:rows=list(csv.DictReader(f))
    def field(key):
        return {int(r['index']):np.array([float(r[a]) for a in 'xyzw']) for r in rows if r['field']==key}
    m=field('material_node');x=np.stack([m[i][:3] for i in range(len(m))]);pos=field('relative_position');rot={b:Rotation.from_quat(q) for b,q in field('orientation').items()}
    d=json.loads((assets/'manifest.json').read_text());bone=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3]
    bank=np.fromfile(assets/'cortical-samples.bin','f4').reshape(2,256,5,4);stats={};start=0
    for side,sid in enumerate((29,30)):
        s=np.load(surface/f'{sid}.npz');V=np.einsum('ni,nij->nj',s['vertex_shape'],x[s['node']+start]);F=s['face'];start+=1190
        entries={tuple(r[4].copy().view('u4')[:2]):int(r[3].copy().view('u4')[2]) for r in bank[side,:d['sides'][side]['rows']]}
        for (lo,n),owner in entries.items():
            vx,rev=np.unique(bone[lo:lo+n].reshape(-1,3),axis=0,return_inverse=True)
            Vb=rot[128].inv().apply(rot[owner].apply(vx)+pos[owner][:3]);v=np.vstack((V,Vb));faces=np.vstack((F,rev.reshape(-1,3)+len(V)))
            changed=np.zeros(len(faces),bool);changed[:len(F)]=True
            pairs=sorted((i,j) for i,j in query(predicate,v,faces,changed) if i<len(F)<=j)
            v=dyadic(v);trans=[];boundary=[]
            for i,j in pairs:
                A=[v[k] for k in faces[i]];B=[v[k] for k in faces[j]]
                exact=straddles(B,A[0],plane(*A)) and straddles(A,B[0],plane(*B))
                (trans if exact else boundary).append([i,j-len(F)])
            stats[f'{sid}/{owner}']=dict(intersecting_pairs=len(pairs),transverse_pairs=len(trans),touch_or_coplanar_pairs=len(boundary),transverse_examples=trans[:5])
    out.write_text(json.dumps(dict(accepted_step=int(rows[0]['accepted_step']),snapshot_sha256=sha(snapshot),predicate_sha256=sha(predicate),regional=stats,total_transverse=sum(r['transverse_pairs'] for r in stats.values())),indent=2)+'\n')
if __name__=='__main__':test(*map(Path,sys.argv[1:]))
