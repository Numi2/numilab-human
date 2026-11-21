"""Factor each cut-muscle stiffness and emit responses only to reachable cortical P1 nodal loads."""
import json,shutil,sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from pelvic_elastic_response import stiffness
from pelvic_specimen import properties
from material_context import sha
from qualify_material_motion import frame

def prepare(part,response,assets,receipt,law,out):
    out.mkdir(exist_ok=True);manifest=json.loads((assets/'manifest.json').read_text())
    for f in assets.iterdir():
        if f.is_file() and f.name!='manifest.json' and not (out/f.name).exists():(out/f.name).hardlink_to(f)
    poses=json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses'];ref=frame(next(b for b in poses if b['body_index']==128))
    rotation=Rotation.from_quat(ref[3:]).inv()
    pool=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)
    meta=[];start=0;block=0;lookup=[];stored=out/'cortical-flexibility.bin';props=properties(law)
    with stored.open('wb') as out_f:
        for side,(sid,owner) in enumerate(((29,131),(30,145))):
            p=np.load(part/'projected'/f'{sid}.npz');reg=np.load(response/f'{sid}.npz');rec=pool[pool[:,3,1].copy().view('u4')==owner]
            x=p['vertices'].astype(float);x=rotation.apply(x-x.mean(0));t=p['cells'];used=p['cell_volumes']>0
            A,_=stiffness(x,t[used],p['cell_volumes'][used],props['mu'],props['lam'])
            free=np.setdiff1d(np.unique(t[used]),reg['fixed']);idx=(3*free[:,None]+np.arange(3)).ravel();K=A[idx][:,idx].tocsc()
            units=1/np.sqrt(K.diagonal());D=sparse.diags(units);factor=splu((D@K@D).tocsc())
            src=np.unique(rec[:,5].copy().view('u4').astype(int))-start
            if 'closed_reverse' in manifest:
                meta=manifest['closed_reverse']['sides'][side];closed=np.fromfile(assets/'closed-material-triangles.bin','u4').reshape(-1,3,4)
                domain=closed[meta['first_face']:meta['first_face']+meta['faces'],2].astype(int)-start
                src=np.union1d(src,domain.ravel())
            map_=np.full(len(x),np.iinfo('u4').max,'u4');map_[src]=np.arange(len(src));lookup.append(map_)
            expr=np.zeros((len(x),len(src),3,4),'f4');freeidx=np.full(3*len(x),-1,int);freeidx[idx]=np.arange(len(idx))
            srcd=(3*src[:,None]+np.arange(3)).ravel()
            largest=0.
            for a in range(0,len(srcd),48):
                b=min(a+48,len(srcd));dofs=freeidx[srcd[a:b]];present=dofs>=0
                rhs=np.zeros((len(idx),b-a));rhs[dofs[present],np.flatnonzero(present)]=1
                ans=units[:,None]*factor.solve(units[:,None]*rhs);largest=max(largest,float(np.max(abs(K@ans-rhs))))
                full=np.zeros((3*len(x),b-a));full[idx]=ans
                expr[:,a//3:b//3,:,:3]=full.reshape(len(x),3,(b-a)//3,3).transpose(0,2,3,1)
            if not np.isfinite(expr).all() or largest>1e-6:raise ValueError(('local factor accuracy',sid,largest))
            sub=expr[src,:,:,:3].transpose(0,3,1,2).reshape(len(src)*3,len(src)*3).astype(float)
            symmetry=np.linalg.norm(sub-sub.T,ord=np.inf)/np.linalg.norm(sub,ord=np.inf)
            rng=np.random.default_rng(sid);f=rng.normal(size=len(src)*3);rhs=np.zeros(3*len(x));rhs[srcd]=f
            target=units*factor.solve(units*rhs[idx]);reconstructed=np.einsum('nski,sk->ni',expr[:,:,:,:3].astype(float),f.reshape(-1,3)).ravel()[idx]
            delta=reconstructed-target;relative=float(np.sqrt(max(0.,delta@(K@delta))/(target@(K@target))))
            if symmetry>1e-6 or relative>1e-3:raise ValueError(('emitted flexibility',sid,symmetry,relative))
            expr.tofile(out_f);meta.append(dict(muscle=sid,first_node=start,nodes=len(x),input_nodes=len(src),first_vector=block,free_dofs=len(idx),largest_factor_residual_N=largest,relative_reciprocity_error=symmetry,float32_energy_error=relative));start+=len(x);block+=len(x)*len(src)*3
    link=out/'cortical-free-index.bin';np.concatenate(lookup).tofile(link)
    manifest['files'][stored.name]=sha(stored);manifest['files'][link.name]=sha(link);manifest['contact_flexibility']=meta
    manifest['model']='Quasistatic cut-muscle elasticity condensed exactly onto the selected cortical loads at the reference material tangent. Articulated donors retain inertia.'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(meta,flush=True)
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
