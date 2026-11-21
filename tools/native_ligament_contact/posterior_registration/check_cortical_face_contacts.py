"""Evaluate reverse vertex/face gaps independently on an accepted native material snapshot."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cortical_distance import CorticalSurface
from material_context import sha

def check(assets,snapshot,library,out):
    with snapshot.open() as f:rows=list(csv.DictReader(f))
    def vals(k):
        return {int(r['index']):np.array([float(r[a]) for a in 'xyzw']) for r in rows if r['field']==k}
    d=json.loads((assets/'manifest.json').read_text());base=d['cortical_face_contacts']['first']
    pool=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)[base:]
    faces=np.fromfile(assets/'cortical-face-weights.bin','f4').reshape(-1,3,4).astype(float)
    lineage=np.fromfile(assets/'cortical-face-lineage.bin','u4').reshape(-1,4)
    t=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3]
    m=vals('material_node');x=np.stack([m[i][:3] for i in range(len(m))]);pos=vals('relative_position');rot={b:Rotation.from_quat(q) for b,q in vals('orientation').items()}
    pts=np.einsum('nki,nij->nkj',faces,x[pool[:,5].copy().view('u4')])
    e=pts[:,1]-pts[:,0];g=pts[:,2]-pts[:,0];n=np.cross(e,g);n2=np.einsum('ij,ij->i',n,n);stats={}
    banks=pool[:,4,:2].copy().view('u4')
    for side,owner,start,count in np.unique(np.column_stack((lineage[:,0],lineage[:,2],banks)),axis=0):
        idx=np.flatnonzero((lineage[:,0]==side)&(lineage[:,2]==owner)&(banks[:,0]==start))
        bone=rot[128].inv().apply(rot[owner].apply(pool[idx,2,:3])+pos[owner][:3]);v=bone-pts[idx,0]
        alpha=np.einsum('ni,ni->n',np.cross(v,g[idx]),n[idx])/n2[idx];beta=np.einsum('ni,ni->n',np.cross(e[idx],v),n[idx])/n2[idx]
        unit=n[idx]/np.sqrt(n2[idx,None]);gap=np.einsum('ni,ni->n',v,unit)
        selected=np.flatnonzero((alpha>=0)&(beta>=0)&(alpha+beta<=1)&(gap<0));proj=bone[selected]-gap[selected,None]*unit[selected]
        mesh=t[start:start+count].reshape(-1,3);query=CorticalSurface(library,mesh,np.arange(len(mesh)).reshape(-1,3))
        if len(proj):
            inside=query.query(rot[owner].inv().apply(rot[128].apply(proj)-pos[owner][:3]))[0]<=2.e-8;selected=selected[inside]
        if len(selected):
            j=selected[np.argmin(gap[selected])];witness=dict(gap_m=gap[j],material_facet=int(lineage[idx[j],1]),bone_vertex=int(lineage[idx[j],3]),barycentric=[1-alpha[j]-beta[j],alpha[j],beta[j]])
        else:witness=None
        stats[f'{side}/{owner}']=dict(reachable_pairs=len(idx),penetrations_above_0_5_um=int((gap[selected]<-5.e-7).sum()),worst=witness)
    out.write_text(json.dumps(dict(accepted_step=int(rows[0]['accepted_step']),snapshot_sha256=sha(snapshot),regions=stats),indent=2)+'\n')
if __name__=='__main__':check(*map(Path,sys.argv[1:]))
