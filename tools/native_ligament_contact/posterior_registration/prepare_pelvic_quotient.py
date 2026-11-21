"""Factor the existing local free stiffness and compile dependency levels for GPU defect correction."""
from pathlib import Path
import json,sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation
from pelvic_elastic_response import stiffness
from qualify_material_motion import frame
from material_context import sha

def prepare(part,response,base,receipt,out):
    out.mkdir(exist_ok=True)
    for p in base.iterdir():
        if p.name!='manifest.json' and not (out/p.name).exists():(out/p.name).hardlink_to(p)
    spec=json.loads((base/'manifest.json').read_text());root=json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']
    R=Rotation.from_quat(frame(next(x for x in root if x['body_index']==128))[3:]).inv()
    rows=[];col=[];coef=[];levels=[];order=[];perm=[];scale=[];regions=[];nodes=[];volumes=[];corners=[];nodeFirst=0;cellFirst=0
    projectorRows=[];projectorCols=[];projectorVals=[]
    def add_tri(tri,nodeOffset,lower):
        tri=tri.tocsr();start=len(rows);N=tri.shape[0];depth=np.zeros(N,dtype=int)
        for i in (range(N) if lower else reversed(range(N))):
            a,b=tri.indptr[i:i+2];indices=tri.indices[a:b];value=tri.data[a:b];mask=(indices!=i)&(value!=0)
            ids=indices[mask];diag=value[indices==i]
            if len(diag)!=1 or diag[0]==0 or np.any(ids>=i if lower else ids<=i):raise ValueError('Invalid triangular factor')
            if len(ids):depth[i]=1+max(depth[ids])
        for i in range(N):
            a,b=tri.indptr[i:i+2];indices=tri.indices[a:b];value=tri.data[a:b];mask=(indices!=i)&(value!=0);ids=indices[mask];diag=value[indices==i][0]
            # Divide by the diagonal after accumulation. Store reciprocal alongside the CSR row.
            rows.append([len(col),len(ids),nodeOffset+i,np.asarray(1/diag,'f4').view('u4').item()])
            col.extend(nodeOffset+ids);coef.extend(value[mask])
        begin=len(levels)
        for k in range(depth.max()+1):
            ix=np.flatnonzero(depth==k);levels.append([len(order),len(ix)]);order.extend(start+ix)
        return begin,len(levels)-begin
    for side,sid in enumerate((29,30)):
        p=np.load(part/'projected'/f'{sid}.npz');red=np.load(response/f'{sid}.npz');x=p['vertices'].astype(float);x=R.apply(x-x.mean(0))
        mask=p['cell_volumes']>0;t=p['cells'][mask];vol=p['cell_volumes'][mask];volumes.extend(vol)
        # Current global material-cell order is part of the constitutive integral.
        assert np.all(vol>0)
        bucket=[[] for _ in x]
        for i,tt in enumerate(t):
            for j,k in enumerate(tt):bucket[k].append(4*(i+cellFirst)+j)
        for ids in bucket:nodes.append([len(corners),len(ids)]);corners.extend(ids)
        cellFirst+=len(t)
        free=np.setdiff1d(np.unique(t),red['fixed']);dofs=(3*free[:,None]+np.arange(3)).ravel()
        A,_=stiffness(x,t,vol,spec['specimen']['mu'],spec['specimen']['lam']);K=A[dofs][:,dofs].tocsc()
        units=1/np.sqrt(K.diagonal());D=sparse.diags(units);Kscaled=(D@K@D).tocsc()
        graph=sparse.coo_matrix((np.ones(t.size*4),(np.repeat(t,4,axis=1).ravel(),np.tile(t,(1,4)).ravel())),shape=(len(x),len(x)))
        _,component=connected_components(graph,directed=False);patch={};gauge=[];addition=sparse.lil_matrix(Kscaled.shape)
        for owner in np.unique(component[t[:,0]]):
            node=np.flatnonzero(component==owner)
            if np.intersect1d(node,red['fixed']).size:continue
            local=(node[:,None]*3+np.arange(3)).ravel();ff=np.searchsorted(dofs,local);assert np.array_equal(dofs[ff],local)
            block=Kscaled[ff][:,ff].toarray()
            # Rigid translations and infinitesimal rotations form the exact nullspace of a free solid.
            rel=x[node]-x[node].mean(0);Z=np.zeros((len(node),3,6));Z[:,:,:3]=np.eye(3)
            for axis in range(3): Z[:,:,3+axis]=np.cross(np.eye(3)[axis],rel)
            Q=np.linalg.qr(Z.reshape(-1,6)/units[ff,None],mode='reduced')[0];nullError=np.linalg.norm(block@Q,ord=np.inf)
            P=Q@Q.T;ev=np.linalg.eigvalsh(block)
            if nullError>1.e-9 or abs(ev[5])>1.e-9 or ev[6]<1.e-8:raise ValueError(('Unexpected free-body modes',sid,ev))
            addition[np.ix_(ff,ff)]=P
            project=(np.eye(len(ff))-P)*units[ff][None,:]/units[ff][:,None]
            for a,i in enumerate(local):patch[int(i)]=(local,project[a])
            gauge.append(dict(nodes=node.tolist(),rigid_modes=6,scaled_rigid_residual=float(nullError),lowest_elastic_eigenvalue=float(ev[6]),volume_m3=float(vol[component[t[:,0]]==owner].sum())))
        for i in range(3*len(x)):
            if i in patch:
                j,v=patch[i];projectorRows.append([len(projectorCols),len(j)]);projectorCols.extend(3*nodeFirst+j);projectorVals.extend(v)
            else:projectorRows.append([0,0])
        Kscaled=Kscaled+addition.tocsc();fac=splu(Kscaled)
        rr=np.argsort(fac.perm_r);cc=np.argsort(fac.perm_c)
        diff=(fac.L@fac.U-Kscaled[rr][:,cc]).tocoo()
        permError=max(abs(diff.data),default=0.)
        if permError>1e-9:raise ValueError(('Permutation mismatch',sid,permError))
        off=len(scale);n=len(dofs)
        for j in range(n):
            perm.append([3*nodeFirst+int(dofs[rr[j]]),3*nodeFirst+int(dofs[cc[j]])])
            scale.append([units[rr[j]],units[cc[j]]])
        ll=add_tri(fac.L,off,True);uu=add_tri(fac.U,off,False)
        regions.append(dict(muscle=sid,coordinate_start=off,coordinates=n,lower=ll,upper=uu,nnz=int(fac.L.nnz+fac.U.nnz),floating_material_components=gauge,permutation_check=float(permError)))
        nodeFirst+=len(x)
    data={'pelvic-projector-rows':(projectorRows,'u4'),'pelvic-projector-cols':(projectorCols,'u4'),'pelvic-projector-vals':(projectorVals,'f4'),'pelvic-cell-volume':(volumes,'f4'),'pelvic-cell-incidence':(nodes,'u4'),'pelvic-cell-corners':(corners,'u4'),'pelvic-factor-rows':(rows,'u4'),'pelvic-factor-columns':(col,'u4'),'pelvic-factor-values':(coef,'f4'),'pelvic-factor-levels':(levels,'u4'),'pelvic-factor-order':(order,'u4'),'pelvic-factor-perm':(perm,'u4'),'pelvic-factor-scale':(scale,'f4')}
    for name,(value,dt) in data.items():
        p=out/(name+'.bin');a=np.array(value,dt)
        if not np.all(np.isfinite(a)):raise ValueError(name)
        a.tofile(p);spec['files'][p.name]=sha(p)
    spec['regional_factor']=regions
    (out/'manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(regions)
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
