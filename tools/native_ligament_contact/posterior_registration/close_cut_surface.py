"""Resolve hanging edges on the cut integration surface with compatible P1 coordinates."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial import cKDTree
from fit_contact_domain import points
from project_material_cells import barycentric
from material_context import sha

def edges(tri):
    e=np.concatenate([tri[:,[k,(k+1)%3]] for k in range(3)]);u,i,c=np.unique(np.sort(e,axis=1),axis=0,return_index=True,return_counts=True)
    return u,c,e,i

def join(part,prev,out):
    out.mkdir(exist_ok=True);stats={}
    for sid in (29,30):
        p=np.load(part/'projected'/f'{sid}.npz');_,_,xyz,owner,bary=points(part,prev,sid)
        packed=xyz.reshape(-1,3);_,first,index=np.unique(np.round(packed,11),axis=0,return_index=True,return_inverse=True)
        v=packed[first];tri=index.reshape(-1,3);table,c,*_=edges(tri);open_e=table[c==1];ends=np.unique(open_e)
        tree=cKDTree(v[ends]);split={}
        for a,b in open_e:
            e=v[b]-v[a];length=np.linalg.norm(e);candidates=ends[tree.query_ball_point((v[a]+v[b])/2,.5*length+1.e-10)]
            q=candidates[(candidates!=a)&(candidates!=b)];delta=v[q]-v[a];alpha=delta@e/(e@e)
            take=(alpha>0)&(alpha<1)&(np.linalg.norm(delta-alpha[:,None]*e,axis=1)<2.e-11)
            ids=q[take][np.argsort(alpha[take])];split[(a,b)]=ids.tolist();split[(b,a)]=ids[::-1].tolist()
        verts=list(v);cells=[];faces=[];source=[]
        for i,row in enumerate(tri):
            ring=[]
            for j in range(3):
                a,b=row[j],row[(j+1)%3];ring.append(a);ring.extend(split.get((a,b),()))
            if len(ring)==3:faces.append(row);cells.append(owner[i]);source.append(i)
            else:
                new=len(verts);verts.append(xyz[i].mean(axis=0))
                for j,a in enumerate(ring):faces.append([new,a,ring[(j+1)%len(ring)]]);cells.append(owner[i]);source.append(i)
        verts=np.array(verts);faces=np.array(faces,'u4');cells=np.array(cells,'u4')
        _,cnt,oriented,_=edges(faces);assert (cnt==2).all(),(sid,np.unique(cnt,return_counts=True))
        assert len(np.unique(oriented,axis=0))==len(oriented)
        t=p['cells'];vref=p['vertices'].astype(float)
        weights=np.array([barycentric(verts[f],vref[t[c]]) for f,c in zip(faces,cells)])
        if weights.min() < -1.e-8:raise ValueError(('P1 surface mapping',weights.min()))
        ancestry={};werror=0.
        for j,f in enumerate(faces):
            for k,i in enumerate(f):
                val={int(n):w for n,w in zip(t[cells[j]],weights[j,k]) if abs(w)>1.e-9}
                if i not in ancestry:ancestry[i]=(cells[j],weights[j,k],val)
                else:
                    old=ancestry[i][2];diff=max((abs(old.get(m,0)-val.get(m,0)) for m in old.keys()|val.keys()),default=0)
                    werror=max(werror,diff)
        if werror>5.e-7:raise ValueError(('Conflicting material interpolation',sid,werror))
        vc=np.array([ancestry[i][0] for i in range(len(verts))],'u4');vw=np.stack([ancestry[i][1] for i in range(len(verts))])
        area=lambda z:.5*np.linalg.norm(np.cross(z[:,1]-z[:,0],z[:,2]-z[:,0]),axis=1)
        old=area(xyz);per=np.bincount(source,weights=area(verts[faces]),minlength=len(xyz))
        if max(abs(old-per))>1.e-14:raise ValueError('Changed registered area')
        shifted=verts[faces]-verts.mean(0);vol=np.einsum('ij,ij->i',shifted[:,0],np.cross(shifted[:,1],shifted[:,2])).sum()/6
        if abs(vol-p['cell_volumes'].sum())>1.e-12:raise ValueError(('Material volume mismatch',vol,p['cell_volumes'].sum()))
        dest=out/f'{sid}.npz'
        np.savez_compressed(dest,vertices=verts,face=faces,owner=cells,source_face=np.array(source,'u4'),shape=weights,node=t[vc],vertex_shape=vw)
        stats[sid]=dict(source_mesh=sha(part/'projected'/f'{sid}.npz'),output=sha(dest),source_faces=len(tri),source_unmatched_edges=len(open_e),conforming_faces=len(faces),conforming_vertices=len(verts),unmatched_edges=0,maximum_weight_mismatch=werror,maximum_area_change_m2=float(max(abs(old-per))),enclosed_volume_ml=vol*1e6)
        print(stats[sid],flush=True)
    (out/'report.json').write_text(json.dumps(stats,indent=2)+'\n')
if __name__=='__main__':join(*map(Path,sys.argv[1:]))
