"""Intersect same-material tetrahedra through radius-binned source trees and exact solid clipping."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial import cKDTree
from build_carpal_cavity import FACES
from fit_carpal_clearance import geom
from partition_carpal_material import solid
from audit_registered_skeleton import digest

def run(mesh,plc,domain,cut,out):
    z=np.load(domain);tc=z['t'];x=z['x'].astype(float);role=z['role'];c=np.load(plc);src=c['source_tetrahedra'];ledger=np.load(cut)
    raw=np.loadtxt(mesh/'mesh.ele',dtype='i4',skiprows=1)[src];xyz=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3))
    ref=x.mean(0);a=(xyz[raw[:,1:5]-1]-ref)*1000.;b=(x[tc]-ref)*1000.;pr=raw[:,5]
    pa=a.min(1);pb=a.max(1);pc=(pa+pb)*.5;rb=np.linalg.norm((pb-pa)*.5,axis=1);B=b.mean(1);rchild=np.linalg.norm(b-B[:,None],axis=2).max(1)
    faces=a[:,FACES];n=np.cross(faces[:,:,1]-faces[:,:,0],faces[:,:,2]-faces[:,:,0])
    trees=[]
    for material in (15,16):
        for s in np.unique(np.frexp(rb)[1]):
            ids=np.flatnonzero((pr==material)&(np.frexp(rb)[1]==s))
            if len(ids):trees.append((material,ids,cKDTree(pc[ids]),rb[ids].max()))
    def key(v):return tuple(sorted(p.tobytes() for p in v))
    bytet={key(p):i for i,p in enumerate(a)}
    exact=[];parent=[];child=[];volume=[];cache={};pv=geom(a)[0];cv=geom(b)[0];excluded=set(ledger['source_tet'][ledger['exterior_volume_mm3']<=0])
    for j,bj in enumerate(b):
        eq=bytet.get(key(bj))
        if eq is not None and pr[eq]==role[j] and src[eq] not in excluded:
            parent.append(eq);child.append(j);volume.append(cv[j]);exact.append(j);continue
        candidates=[]
        for m,ids,tr,r in trees:
            if m==role[j]:candidates.extend(ids[tr.query_ball_point(B[j],r+rchild[j]+1e-10)])
        ii=np.asarray(candidates,dtype=int);ii=ii[((pa[ii]<=bj.max(0))&(pb[ii]>=bj.min(0))).all(1)]
        bf=bj[FACES];bn=np.cross(bf[:,1]-bf[:,0],bf[:,2]-bf[:,0])
        d=np.einsum('pfi,pfci->pfc',n[ii],bj[None,None]-faces[ii,:,:1]);e=np.einsum('fi,pfci->pfc',bn,a[ii,None]-bf[:,0][None,:,None]);ii=ii[(d.min(2)<=1e-15).all(1)&(e.min(2)<=1e-15).all(1)]
        obj=solid(bj,FACES)
        for i in ii:
            if src[i] in excluded:continue
            if i not in cache:cache[i]=solid(a[i],FACES)
            hit=obj^cache[i]
            if hit.status()!=obj.status():raise ValueError('Indeterminate overlap')
            v=hit.volume()
            if v>0:parent.append(i);child.append(j);volume.append(v)
    parent=np.array(parent);child=np.array(child);volume=np.array(volume);s=np.searchsorted(src,ledger['source_tet'])
    if not np.array_equal(src[s],ledger['source_tet']):raise ValueError('Unmatched source debit')
    pv[s]=ledger['exterior_volume_mm3'];used=np.bincount(parent,weights=volume,minlength=len(src));coverage=np.bincount(child,weights=volume,minlength=len(b))
    out.mkdir();file=out/'overlap.npz';np.savez_compressed(file,parent=src[parent],child=child,volume_mm3=volume,parent_ids=src,parent_volume_mm3=pv,child_volume_mm3=cv)
    report=dict(mesh=digest(domain),source=digest(mesh/'mesh.ele'),cavity=digest(plc),partition=digest(cut),mapping=digest(file),edges=len(volume),exact_unchanged=len(exact),uncovered_children=np.flatnonzero(coverage==0).tolist(),uncovered_parents=src[(used==0)&(pv>0)].tolist(),max_child_volume_error_mm3=float(abs(coverage-cv).max()),max_parent_volume_error_mm3=float(abs(used-pv).max()),by_material={})
    for m in (15,16):
        gap=cv[role==m]-coverage[role==m];report['by_material'][m]=dict(child_volume_mm3=float(cv[role==m].sum()),uncovered_mm3=float(np.maximum(gap,0).sum()),multiple_coverage_mm3=float(np.maximum(-gap,0).sum()))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('edges',len(volume),'uncovered children',len(report['uncovered_children']),'coverage',report['by_material'])
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
