"""Construct a seated initial guess from one-sided bed heights and bounded P1 displacement gradients."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack,vstack,csr_matrix
from screen_registered_dense_posture import load_geometry
from recover_material_thickness import audit
from untangle_registered_donor_field import differentials
from qualify_loaded_pose_registration import anchors

def main(full,donor,prior,out,load=None):
    x,target,owner,t,mat,outer,_,pins=load_geometry(full,donor)
    q0=(x+np.asarray(pins['root_translation_m'])).astype('f4').astype(float)
    delta=-pins['root_translation_m'][2]
    ref=x.astype('f4').astype(float)
    if not np.array_equal(ref,x):raise ValueError('reference precision mismatch')
    if delta>.005 and load:
        from scipy.spatial import cKDTree
        rows=np.asarray(json.loads(load.read_text())['nodes'])
        radial=cKDTree(x[rows,:2]).query(x[:,:2])[0]
        free=((x[:,2]<.07)&(radial<.055)) | (q0[:,2]<2e-5)
        env=json.loads((full/'launch.json').read_text())['environment']
        free[[a[1] for a in anchors(env).values()]]=False
        ids=np.flatnonzero(free)
    else:ids=np.flatnonzero(x[:,2] < .025)
    ft=np.full(len(x),-1,np.int32);ft[ids]=np.arange(len(ids));selected=(ft[t]>=0).any(1)
    tet=t[selected];N=len(tet);nodes=ft[tet];mov=nodes>=0
    dm=(x[tet[:,1:]]-x[tet[:,:1]]).transpose(0,2,1)
    inv=np.linalg.inv(dm).astype('f4').astype(float)
    gradients=np.concatenate((-inv.sum(axis=1)[:,None,:],inv),axis=1)
    dz=q0[:,2]-x[:,2];f0=((dz[tet[:,1:]]-dz[tet[:,:1]])[:,None,:]@inv)[:,0,:]
    vals=(gradients*delta).transpose(0,2,1);shape=vals.shape
    row=np.broadcast_to(np.arange(3*N).reshape(N,3,1),shape)
    co=np.broadcast_to(nodes[:,None,:],shape);use=co>=0
    M=coo_matrix((vals[use],(row[use],co[use])),shape=(3*N,len(ids))).tocsr()
    ones=csr_matrix(-np.ones((3*N,1)))
    cons=vstack([hstack([M,ones]),hstack([-M,ones])]).tocsr()
    bound=np.r_[-f0.ravel(),f0.ravel()]
    scale=1/np.maximum(1,np.asarray(abs(cons).max(axis=1).toarray()).ravel())
    A=cons.multiply(scale[:,None]).tocsr();b=bound*scale
    e=np.asarray(np.linalg.det(dm))/6
    vol=np.zeros(len(x));np.add.at(vol,tet.ravel(),np.repeat(e/4,4))
    obj=np.r_[.02*vol[ids]/vol[ids].sum(),1.]
    anchors_all=anchors(json.loads((full/'launch.json').read_text())['environment'])
    locked=np.array([i[1] for i in anchors_all.values()])
    assert np.all(ft[locked]<0)
    baseline=x+np.load(prior/'displacement.npy')
    preferred=outer&(baseline[:,2]==0.)
    outer_below=ids[outer[ids] & (q0[ids,2]<0)]
    lowest=outer_below[np.argmin(x[outer_below,2])]
    assert q0[lowest,2]<0 and np.allclose(q0[locked],target[locked],rtol=0,atol=2e-7)
    reference=np.concatenate([differentials(x,t[i:i+32768])[0] for i in range(0,len(t),32768)])
    best=None;hist=[]
    band=json.loads(load.read_text()) if load else None
    sets=[('all_sourced',np.flatnonzero(preferred)),('lower_quarter',outer_below[x[outer_below,2]<np.quantile(x[outer_below,2],.25)]),('first_touch',np.array([lowest]))]
    if band:sets=[('barrier_supported',np.asarray(band['nodes'],dtype=int))]
    for name,contact in sets:
        width=band['band_m'] if band else 0.
        lo=np.maximum(0,(width-q0[ids,2])/delta);hi=np.maximum(lo,1.01)
        ci=ft[contact];assert (ci>=0).all()
        gap=np.asarray(band['gaps']) if band else np.zeros(len(contact))
        lo[ci]=(gap-q0[contact,2])/delta
        provisional=bool(band and band.get('select_unilateral'))
        cap=1.
        objective=obj.copy()
        if provisional:
            objective[:len(ids)]=.002*vol[ids]/vol[ids].sum()
            objective[ci]+=np.asarray(band['individual_force_N'])/band['weight_N']
            objective[-1]=.02
            cap=.55
        else:
            hi[ci]=lo[ci]
        from fit_spectral_bed_rows import solve
        opt=solve(objective,M,f0,A,b,[*zip(lo,hi),(0,cap)],delta>.005)
        entry={'support_set':name,'required_skin_points':len(contact),'solver_status':opt.message};hist.append(entry)
        if not opt.success:continue
        q=q0.copy();q[ids,2]=(q0[ids,2]+delta*opt.x[:-1]).astype('f4');pinned=contact[np.isclose(opt.x[ci],lo[ci],rtol=0,atol=1e-9)]
        if provisional:q[pinned,2]=gap[np.isin(contact,pinned)]
        else:q[contact,2]=gap
        if q[:,2].min() < -1e-12:raise ValueError('penetrating fitted output')
        q[q[:,2]<0,2]=0
        grad=((q[tet[:,1:],2]-x[tet[:,1:],2]-(q[tet[:,:1],2]-x[tet[:,:1],2]))[:,None,:]@inv)[:,0,:]
        F=np.broadcast_to(np.eye(3),(len(tet),3,3)).copy();F[:,2,:]+=grad
        sv=np.linalg.svd(F,compute_uv=False)
        J,shape,det,bad=audit(q,t,reference);fail=int(bad.sum())
        active=(outer & (q[:,2]<width)) if band else (outer & (q[:,2]==0))
        if band and not provisional and not np.array_equal(np.flatnonzero(active),contact):raise ValueError('unaccounted active bed points')
        entry.update(geometry_failures=fail,lp_max_gradient=float(opt.x[-1]),max_emitted_gradient=float(abs(grad).max()),
                     max_stretch=float(sv[:,0].max()),min_stretch=float(sv[:,2].min()),touching=int(active.sum()))
        stretch=max(entry['max_stretch'],1/entry['min_stretch'])
        if provisional:entry['active_nodes']=np.flatnonzero(active).tolist()
        score=(fail,stretch>1.5,-entry['touching'] if stretch<=1.5 else 0,stretch)
        if best is None or score<best[0]:best=(score,q.copy(),dict(entry))
    print(json.dumps(hist),flush=True)
    if best is None:raise ValueError('No admissible normal construction')
    out.mkdir(exist_ok=False);q=best[1];np.save(out/'displacement.npy',q-x)
    shutil.copyfile(prior/'construction-boundary.npz',out/'parent-construction-boundary.npz')
    np.savez_compressed(out/'construction-boundary.npz',node=locked,axis=np.ones((len(locked),3),bool),value=q[locked],trial_contact=np.asarray(best[2].get('active_nodes',band['nodes'])) if band else np.flatnonzero(outer&(q[:,2]==0)))
    result=dict(source=pins,failed=best[2]['geometry_failures'],fits=hist,selected=best[2],movable_nodes=len(ids),checked_incident_cells=N,
                scope='Construction only. All physical contact reactions and elastic forces are solved by the existing native owners.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main(*(Path(v).resolve() for v in __import__('sys').argv[1:]))
