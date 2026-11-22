"""Correct registered deep-fat surface folds against paired native material states."""
import ctypes,json,sys
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha
from probe_deep_surface_motion import gather
from repair_accepted_material_cells import metric

def main():
    d,lib,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
    faces,ref,end,pairs,groups=gather(d);faces=np.ascontiguousarray(faces,dtype='u4')
    poses=[ref.copy(),end.copy()];du=end-ref
    with (d/'mesh/mesh.ele').open() as stream:
        n=int(next(stream).split()[0]);el=np.loadtxt(stream,max_rows=n,usecols=(1,2,3,4,5),dtype='i4')
    cells=el[:,:4]-1
    call=ctypes.CDLL(str(lib)).batch;p=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS')
    call.argtypes=[p,p,p,ctypes.c_uint,p];call.restype=None
    def hits(x,ab):
        out=np.zeros(len(ab),dtype='u1');call(x,faces,ab,len(ab),out);return out.astype(bool)
    bounds=[]
    for x in poses:
        xyz=x[faces];bounds.append((xyz.min(1),xyz.max(1)))
    def near(ids):
        moved=np.flatnonzero(np.isin(faces,ids).any(1));allpairs=[]
        for lo,hi in bounds:
            for i in moved:
                ix=np.flatnonzero(np.all(lo<=hi[i]+4e-5,axis=1) & np.all(hi>=lo[i]-4e-5,axis=1))
                ab=np.column_stack((np.full(len(ix),i),ix));allpairs.extend(np.sort(ab[ab[:,0]!=ab[:,1]],axis=1))
        return np.ascontiguousarray(np.unique(allpairs,axis=0),dtype='u4')
    summary=[]
    for group in np.unique(groups):
        witnesses=np.ascontiguousarray(pairs[groups==group]);ids=np.unique(faces[witnesses])
        held=np.flatnonzero(np.isin(cells,ids).any(1));nearby=near(ids)
        # Check the complete local D-face interaction list at both native endpoints.
        assert not hits(poses[0],nearby).any()
        before=hits(poses[1],nearby)
        if before.sum()!=len(witnesses):raise ValueError(('Unexpected neighboring defect',group,before.sum()))
        accepted=[];initial=poses[0][ids].copy()
        for epoch in range(12):
            old=poses[0][ids].copy();existing=hits(poses[1],witnesses)
            if not existing.any():break
            best=None
            def evaluate(proposals):
                nonlocal best
                for y in proposals:
                    y=y.astype('f4').astype('f8')
                    dist=np.linalg.norm(y-initial,axis=1).max()
                    if dist==0 or dist>2e-5:continue
                    poses[0][ids]=y;poses[1][ids]=y+du[ids]
                    flags=hits(poses[1],witnesses);improvement=int(existing.sum()-flags.sum())
                    if improvement<=0 or np.any(flags&~existing):continue
                    rank=(improvement,-dist)
                    if best is not None and rank<=best[0]:continue
                    if hits(poses[0],witnesses).any():continue
                    if any((lambda v,q:np.any(v<=1e-18) or np.any(q<1e-5))(*metric(x[cells[held]])) for x in poses):continue
                    if hits(poses[0],nearby).any():continue
                    h=hits(poses[1],nearby)
                    if np.any(h & ~before) or h.sum()!=flags.sum():continue
                    best=(rank,y.copy(),h)
                poses[0][ids]=old;poses[1][ids]=old+du[ids]
            direction=du[ids]-np.mean(du[ids],axis=0)
            evaluate([old-scale*direction for scale in (.25,.5,.75,1.,1.25,1.5,2.,3.,4.,6.,8.,12.,16.,32.,64.)])
            if best is None or best[0][0]<existing.sum():
                for a,b in witnesses[existing]:
                    for first,other in ((a,b),(b,a)):
                        tri=ref[faces[other]];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
                        for node in np.setdiff1d(faces[first],faces[other]):
                            pos=np.searchsorted(ids,node);s=np.sign((ref[node]-tri[0])@normal)
                            dirs=[normal*s,-normal*s] if s else [normal,-normal]
                            for direction in dirs:
                                trial=np.broadcast_to(old,(15,*old.shape)).copy();trial[:,pos]+=np.geomspace(2e-9,1.2e-5,15)[:,None]*direction
                                evaluate(trial)
            if best is None:break
            rank,chosen,before=best;poses[0][ids]=chosen;poses[1][ids]=chosen+du[ids];accepted.append(rank[0])
        rem=int(hits(poses[1],witnesses).sum());summary.append(dict(group=int(group),prior=len(witnesses),remaining=rem,reductions=accepted,max_move_m=float(np.linalg.norm(poses[0][ids]-initial,axis=1).max())))
        print(group,len(witnesses),rem,accepted,flush=True)
    moved=np.flatnonzero(np.any(poses[0]!=ref,axis=1))
    held=np.flatnonzero(np.isin(cells,moved).any(1))
    np.savez_compressed(out/'proposal.npz',nodes=moved.astype('i4'),x=poses[0][moved].astype('f4'),cell=held.astype('i4'))
    vol,sh=metric(poses[0][cells[held]]);ev,eq=metric(poses[1][cells[held]])
    invalid=held[(vol<=1e-18)|(sh<1e-5)|(ev<=1e-18)|(eq<1e-5)]
    result=dict(physical_source_sha256=sha(d/'mesh/mesh.node'),accepted_sha256=sha(d/'endpoint-accepted.node'),audited_layers_sha256=sha(d/'accepted-layers.json'),groups=summary,proposal_sha256=sha(out/'proposal.npz'),changed_nodes=len(moved),affected_tets=len(held),failed_cells_frozen_endpoint=invalid.tolist(),requires_new_native_replay=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
