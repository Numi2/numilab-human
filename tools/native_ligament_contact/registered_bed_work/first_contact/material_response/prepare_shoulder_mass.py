"""Reindex the verified quadrature flow onto the wrist replacement and construct permitted shoulder routes."""
import json,os,sys
from pathlib import Path
import numpy as np
from audit_registered_skeleton import digest
from fit_carpal_clearance import geom

def run(original,guarded,flow,primary,wrist,prior,ov,reg,out):
    d=json.loads((reg/'result.json').read_bytes());mapfile=ov/'overlap.npz';z=np.load(mapfile);tg=np.load(reg/'domain.npz')
    if d['original_spatial_map']!=digest(mapfile) or d['domain']!=digest(reg/'domain.npz'):raise ValueError('Mismatched material geometry')
    original_mat=np.loadtxt(original/'mesh.ele',dtype='i4',skiprows=1,usecols=[5]);mat=np.loadtxt(guarded/'mesh.ele',dtype='i4',skiprows=1,usecols=[5]);M=np.fromfile(guarded/'cell-mass.f64',dtype='<f8')
    prior_idx=np.load(guarded/'material-source-indices.npz');drop=prior_idx['old_cavity_tets'];base=np.setdiff1d(np.arange(len(original_mat)),drop);gmap=np.full(len(original_mat),-1,'i8');gmap[base]=np.arange(len(base))
    if len(base)!=prior_idx['child_begin'] or not np.array_equal(original_mat[base],mat[:len(base)]):raise ValueError('Wrong wrist cell map')
    deletion=[v['old_child'] for v in d['removed']];nold=len(tg['t'])+len(deletion);kept=np.setdiff1d(np.arange(nold),deletion);tochild=np.full(nold,-1,int);tochild[kept]=np.arange(len(kept))
    parent=z['parent'];child=tochild[z['child']];weight=z['volume_mm3'];mode=np.zeros(len(child),'u1')
    if np.any(child<0):raise ValueError('Removed fat tet had real spatial mass ancestry')
    for v in d['advected']:
        for p,vol in v['parents']:
            parent=np.r_[parent,p];child=np.r_[child,tochild[v['child']]];weight=np.r_[weight,vol];mode=np.r_[mode,np.uint8(1)]
    if np.any(mat[parent]!=tg['role'][child]) or len(np.unique(child))!=len(tg['t']):raise ValueError('Incomplete physical donor ancestry')
    out.mkdir();sub=out/'ancestry';sub.mkdir();new=sub/'overlap.npz'
    cv,_=geom(tg['x'][tg['t']].astype(float));np.savez_compressed(new,parent=parent,child=child,volume_mm3=weight,edge_is_advected=mode,parent_ids=z['parent_ids'],parent_volume_mm3=z['parent_volume_mm3'],child_volume_mm3=cv*1e9)
    (sub/'result.json').write_text(json.dumps(dict(mesh=digest(reg/'domain.npz'),mapping=digest(new),geometric_origin=digest(mapfile),material_origin=digest(reg/'result.json')),indent=2)+'\n')
    F=out/'flows';F.mkdir();A=out/'allocation';A.mkdir();report={}
    for m,tag in [(15,'SKIN'),(16,'FAT')]:
        rawflow=flow/(tag+'-physical-flow.npz');f=np.load(rawflow);q,oi,w=[f[k] for k in ('carrier','role_cell','mass')];bank=np.flatnonzero(original_mat==m);G=gmap[bank[oi]]
        signed=json.loads((flow/'exact-audit.json').read_bytes())[tag];wa=np.load(wrist/(tag+'-mapping.npz'));old_alloc=np.load(prior/(tag+'-delta.npz'))
        if not signed['verified'] or signed['flow_sha256']!=digest(rawflow) or str(old_alloc['base_flow_sha256'])!=digest(rawflow):raise ValueError('Shuffled wrist and shoulder donors')
        # Compose the physical carpal credit with unchanged source edges. A carrier
        # can legitimately supply both a wrist recipient and a shoulder recipient.
        oldcell=bank[oi];replaced=np.zeros(len(q),bool);replaced[old_alloc['original_edges']]=True
        replaced|=np.isin(oldcell,drop)&(w>0)
        existing=np.flatnonzero(~replaced&(G>=0));we=wa['source_edge'];wd=wa['destination'];external=wd<len(original_mat)
        if np.any(~replaced[we]) or not np.array_equal(wd[external],oldcell[we[external]]):raise ValueError('Wrist flow has an unknown original route')
        wdest=np.empty(len(wd),int);wdest[external]=gmap[wd[external]];wdest[~external]=len(base)+wd[~external]-len(original_mat)
        if np.any(wdest<0) or np.any(mat[wdest]!=m):raise ValueError('Wrist physical recipient disappeared')
        origin_edge=np.r_[existing,we];wrist_route=np.r_[np.full(len(existing),-1,int),np.arange(len(we))]
        orig_quota=np.bincount(q,weights=w);G=np.r_[G[existing],wdest]
        w=np.r_[w[existing],wa['mass_kg']];q=q[origin_edge];oi=oi[origin_edge]
        if np.max(abs(np.bincount(q,weights=w,minlength=len(orig_quota))-orig_quota))>1e-12:raise ValueError('Wrist transfer loses source quadrature mass')
        full=np.bincount(G,weights=w,minlength=len(mat));bank_now=np.flatnonzero(mat==m);delta=abs(full[bank_now]-M[bank_now]);mx=float(delta.max())
        if np.any(delta>np.maximum(3e-12*M[bank_now],3e-19)):raise ValueError(f'Wrist-composed source flow disagrees with current cell masses: {mx}')
        inside=np.isin(G,z['parent_ids'])
        if np.any(inside&(G>=len(base))):raise ValueError('Shoulder geometry unexpectedly includes a replaced wrist cell')
        val=np.load(primary/(tag+'-candidate-flow.npz'));nq=int(q.max()+1);bad=~val['eligible'];frozen=val['role_cell'][bad]*nq+val['carrier'][bad]
        eligible=(G<len(base))&~np.isin(oi*nq+q,frozen);source=np.unique(q[inside&eligible]);free=eligible&np.isin(q,source)
        rows=np.flatnonzero(free|(inside&(w>0)));path=F/(tag+'-physical-flow.npz')
        np.savez_compressed(path,carrier=q[rows],role_cell=np.searchsorted(bank_now,G[rows]),mass=w[rows],original_edge=origin_edge[rows],wrist_route=wrist_route[rows])
        good=np.flatnonzero(free[rows]);np.savez_compressed(A/(tag+'-delta.npz'),original_edges=good,base_flow_sha256=digest(path))
        report[tag]=dict(original_flow=digest(rawflow),candidate_prior=digest(primary/(tag+'-candidate-flow.npz')),protected_wrist=digest(wrist/(tag+'-mapping.npz')),source_rows=len(source),edges=len(rows),free_edges=len(good),reindexed=digest(path),original_mass_error_kg=mx,shoulder_parent_mass_kg=float(M[z['parent_ids'][mat[z['parent_ids']]==m]].sum()))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
