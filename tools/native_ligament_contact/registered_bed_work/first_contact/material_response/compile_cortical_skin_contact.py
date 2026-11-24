"""Lump each registered cortical boundary face once using the native material contact law."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from compile_connected_joint import digest

def run(h,job,tables,out,material=None,pose_file=None,cortex=None):
    src=material or h/'closed-native/native';pose_file=pose_file or job/'posed-app/current.node';cortex=cortex or job/'loaded-cortex'
    out.mkdir();inputs=json.loads((tables/'result.json').read_bytes());z=np.load(cortex/'registered-cortical-facets.npz')
    r=np.array(z['cell_side_material_bank_nodes'],dtype='i4');bodies=z['body'];seeds=z['cortex_faces'];x=np.loadtxt(src/'mesh.node',skiprows=1,usecols=(1,2,3));tet=np.loadtxt(src/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    boundary=np.load(job/'registered-boundary/registered-boundary.npz');env=json.loads((job/'posed-app/scene/launch.json').read_bytes())['environment'];pr='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    area=np.linalg.norm(np.cross(x[r[:,5]]-x[r[:,4]],x[r[:,6]]-x[r[:,4]]),axis=1)*.5
    if not np.isfinite(area).all() or np.any(area<=0):raise ValueError('Degenerate cortical face')
    report={};samples=[]
    for mat,roles,col in [(15,'OI',1),(16,'ID',2)]:
        th=np.zeros(len(x));t=tet[tet[:,4]==mat,:4]-1
        for k in roles:
            key='ROLE_'+k+'_CONTACT_THICKNESS';f=inputs['files'].get(key)
            path=Path(f['path'] if f else env[pr+key+'_CSV']);sha=f['sha256'] if f else env[pr+key+'_SHA256']
            if digest(path)!=sha:raise ValueError('Contact thickness hash changed')
            nd=boundary[k+'_node_joint'];val=np.loadtxt(path,delimiter=',',skiprows=1,usecols=(col,))
            if len(val)!=len(nd) or np.any(val<=0) or np.any(th[nd]!=0):raise ValueError('Invalid boundary thickness')
            th[nd]=val
        used=np.unique(t);free=used[th[used]==0];loc=np.full(len(x),-1,'i4');loc[free]=np.arange(len(free))
        # Only free-incident edges contribute to the Dirichlet system. Keep one
        # undirected edge irrespective of how many tetrahedra contain it.
        local=t[np.any(loc[t]>=0,axis=1)]
        edge=np.concatenate([local[:,[a,b]] for a in range(3) for b in range(a+1,4)])
        edge=np.unique(np.sort(edge,axis=1),axis=0);edge=edge[np.any(loc[edge]>=0,axis=1)]
        length=np.linalg.norm(x[edge[:,0]]-x[edge[:,1]],axis=1)
        if np.any(length<=0):raise ValueError('Zero material edge')
        u,v=np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]];good=loc[u]>=0;u,v=u[good],v[good];weights=np.tile(1/length,2)[good]
        idx=loc[u];interior=loc[v]>=0;diag=np.bincount(idx,weights,minlength=len(free))
        M=diags(diag,format='csr')+coo_matrix((-weights[interior],(idx[interior],loc[v[interior]])),shape=(len(free),len(free))).tocsr()
        rhs=np.bincount(idx[~interior],weights[~interior]*th[v[~interior]],minlength=len(free));values=spsolve(M,rhs)
        bounds=[th[used][th[used]>0].min(),th[used].max()];rel=np.max(abs(M@values-rhs)/np.maximum(diag*values,1e-30))
        if not np.isfinite(values).all() or min(values)<bounds[0]*(1-1e-8) or max(values)>bounds[1]*(1+1e-8) or rel>1e-8:raise ValueError('Invalid extension of material thickness')
        th[free]=values;which=np.flatnonzero(r[:,2]==mat)
        contributions={}
        for i in which:
            for n in r[i,4:]:
                contributions.setdefault(int(n),[]).append((float(area[i]/3),int(r[i,3]),int(bodies[i])))
        total=0
        for n,terms in sorted(contributions.items()):
            local_area=sum(v[0] for v in terms);total+=local_area;_,bank,owner=max(terms);samples.append([n,mat,bank,owner,local_area,th[n]])
        if abs(total-area[which].sum())>1e-12*total:raise ValueError('Cortical area was counted twice or lost')
        report[str(mat)]=dict(facets=len(which),samples=len(contributions),area_m2=total,interpolated_interior_nodes=len(free),relative_residual=float(rel),minimum_thickness_m=min(th[list(contributions)]),maximum_thickness_m=max(th[list(contributions)]))
    np.savez_compressed(out/'material-contact.npz',facets=np.column_stack((r,bodies,seeds)),samples=np.array(samples))
    result=dict(layers=report,material_reference=digest(src/'mesh.node'),elements=digest(src/'mesh.ele'),loaded=digest(pose_file),bone=json.loads((cortex/'result.json').read_bytes())['bone'],surface_binding=digest(tables/'result.json'),cortex=digest(cortex/'registered-cortical-facets.npz'),program=digest(out/'material-contact.npz'),rule='Reference tributary area; three vertex quadrature points per face, accumulated per material and node. Positive thickness uses the existing role values, harmonically extended through same-material edges to interior cortical nodes.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(report)
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
