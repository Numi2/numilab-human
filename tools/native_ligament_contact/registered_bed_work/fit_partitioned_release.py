"""Relax the connected prestressed tissue before native release with fixed anatomical boundaries."""
import argparse,csv,json,re
from pathlib import Path
import numpy as np
from scipy.linalg import qr
from scipy.sparse import coo_matrix,csr_matrix,bmat,vstack
from scipy.sparse.linalg import splu
from admissible import emit_conservative
from cavity_mass import cavity_delta
from quantize_tissue_moments import emit
from torso_cell_geometry import geom
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from refine_posterior_contact import mechanical
from supported_material_admission import certify,installed_topology,stress_limits


def gradients(e):
    g=np.linalg.inv(e).transpose(0,2,1)
    return np.concatenate([-g.sum(1,keepdims=True),g],axis=1)


def matrix(ref,pos,t,roles,props,free):
    R=ref[t[:,1:]]-ref[t[:,:1]];D=pos[t[:,1:]]-pos[t[:,:1]]
    G,H=gradients(R),gradients(D)
    vol=np.linalg.det(R)/6
    mu,lam=props[roles-15].T
    ii=np.full(len(ref),-1,int);ii[free]=np.arange(len(free))
    # Positive shear and log-volume terms precondition the exact constitutive
    # energy gradient. The discarded Hessian terms are absent only in this
    # preparation metric. Accepted states use the original energy and stress.
    rows=[];cols=[];vals=[]
    for i in range(4):
        for j in range(4):
            a,b=ii[t[:,i]],ii[t[:,j]]
            good=(a>=0)&(b>=0)
            if not good.any():continue
            B=(lam[good]*vol[good])[:,None,None]*H[good,i,:,None]*H[good,j,None,:]
            s=mu[good]*vol[good]*np.sum(G[good,i]*G[good,j],axis=1)
            for k in range(3):B[:,k,k]+=s
            rows.append(np.broadcast_to(3*a[good,None,None]+np.arange(3)[None,:,None],B.shape).ravel())
            cols.append(np.broadcast_to(3*b[good,None,None]+np.arange(3)[None,None,:],B.shape).ravel())
            vals.append(B.ravel())
    return coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(3*len(free),)*2).tocsc()


def fit(run,ledger,rank,out,seed):
    cfg=json.loads((run/'launch.json').read_text());env=cfg['environment'];owner=Path(cfg['argv'][0]).parent
    x,y,base,tags,pins=inputs(run);x=x.astype(float);y=y.astype(float)
    t,roles=installed_topology(base,tags,env)
    diag=json.loads(rank.read_text())
    if diag['source']!=pins:raise ValueError("different measured release")
    flow=ledger/'fine-mass-routed/current-density-closed-flow.npz'
    labels=ledger/'mesh/current-child.labels.csv'
    if sha(flow)!=json.loads((flow.parent/'mass-transfer.json').read_text())['flow_sha256'] or sha(labels)!=json.loads(labels.with_suffix('.json').read_text())['labels_sha256']:raise ValueError("wrong mass ledger")
    donors=np.loadtxt(labels,delimiter=',',skiprows=1,usecols=2,dtype=int)
    with np.load(flow) as f,np.load(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ']) as z:
        old=f['element_mass_kg'];keep=z['retained_input_cell_rows'];repl=z['replacement_tetrahedra'];new=z['replacement_mass_kg']
        mass=np.r_[old[keep],new];off=z['replacement_parent_offsets'];par=z['replacement_parent_cell_rows']
        dst=donors[par[off[:-1]]]
        if not np.array_equal(np.repeat(dst,np.diff(off)),donors[par]):raise ValueError("mixed anatomical donors")
        groups=np.r_[donors[keep],dst]
        dm,remesh,remass=cavity_delta(base,old,keep,repl,new,len(y))
    layer=np.full(len(x),-1,int)
    node_map=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    if sha(node_map)!=env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_SHA256']:raise ValueError("stale node map")
    with node_map.open() as stream:
        for row in csv.DictReader(stream):
            n=int(row['joint_node_id'])-1;k={'':0,'O':1,'I':2,'D':3}[row['registered_surface_layer']]
            if layer[n] not in (-1,k):raise ValueError("conflicting node layers")
            layer[n]=k
    if min(layer)<0:raise ValueError("missing node layer")
    header=(owner/'NumiHumanLoadedSkinPoseData.inc').read_text()
    anchor=re.search(r'constexpr Anchor anchors\[\]=\{(.*?)\};',header,re.S)
    anchors=np.array(re.findall(r'\{\s*\d+u?\s*,\s*(\d+)u?\s*,',anchor[1]),int)
    if len(anchors)!=14:raise ValueError("mismatched attachments")
    # Source points outside the prepared material registration have zero
    # displacement gradient exactly. Include one material ring beyond the
    # entire nonrigid perturbation, preserving external and deep surfaces.
    delta=y-x
    strained=np.flatnonzero(np.any(delta[t[:,1:]]!=delta[t[:,:1]],axis=(1,2)))
    first=np.unique(t[strained]);ring=np.unique(t[np.isin(t,first).any(1)])
    free=np.setdiff1d(ring[np.isin(layer[ring],[0,2])],anchors)
    star=np.flatnonzero(np.isin(t,free).any(1));nodes=np.unique(t[star]);T=np.searchsorted(nodes,t[star]);f=np.searchsorted(nodes,free)
    assert all(r['installed_row'] in star for r in diag['source_witnesses'])
    refs=x[nodes];start=y[nodes];pos=start.copy();gids,g=np.unique(np.c_[roles[star],groups[star]],axis=0,return_inverse=True)
    wt=np.zeros((len(gids),len(nodes)))
    for corners in T.T:np.add.at(wt,(g,corners),mass[star]/4)
    W=wt[:,f];rel=y[remesh]-y[remesh[0]]
    reserve=(1e-10*remass*abs(rel).max()-abs(dm[remesh]@rel).max())/2
    if reserve<=0:raise ValueError("no cavity moment margin")
    W=np.vstack([W,dm[free]*(8e-11/reserve)])
    scale=np.maximum(np.linalg.norm(W,axis=1),1e-90)
    Q,R,piv=qr((W/scale[:,None]).T,mode='economic',pivoting=True)
    Rmom=R.copy()
    C=csr_matrix(np.kron(Q[:,abs(np.diag(R))>abs(np.diag(R)).max()*1e-11].T,np.eye(3)))
    props,identity=materials(owner);caps=stress_limits(identity)
    targets=np.searchsorted(star,[r['installed_row'] for r in diag['source_witnesses']])
    # Include unchanged elements across fixed-node stars for witness forcing
    # by computing an additive reference for their opening nodal forces.
    target_nodes=np.unique(t[[r['installed_row'] for r in diag['source_witnesses']]])
    halo=np.setdiff1d(np.flatnonzero(np.isin(t,target_nodes).any(1)),star)
    fixedF=np.zeros_like(pos)
    if len(halo):
        ext,ti=np.unique(t[halo],return_inverse=True);_,extra=mechanical((x[ext],y[ext]),ti.reshape(-1,4),roles[halo],props)
        _,i,j=np.intersect1d(nodes,ext,return_indices=True);fixedF[i]=extra[j]
    nodal=np.bincount(t.ravel(),np.repeat(mass*.25,4))[nodes]
    def mode(z,F):
        tt=T[targets];G=gradients(z[tt[:,1:]]-z[tt[:,:1]])
        a=(F+fixedF)/nodal[:,None]
        return np.einsum('ijk,ijk->i',G,a[tt])
    def evaluate(z):return mechanical((refs,z),T,roles[star],props)
    en0,F0=evaluate(pos);m0=mode(pos,F0)
    if not np.allclose(m0,[r['material_logJ_acceleration_s2'] for r in diag['source_witnesses']],rtol=1e-7):raise ValueError("incomplete force stars")
    seed_report=json.loads((seed/'fit.json').read_text())
    if seed_report['source']!=pins or not seed_report['final_admission']['passed']:raise ValueError("unmatched region seed")
    with np.load(seed/'preload.npz') as z:
        if not np.isin(z['rows'],free).all():raise ValueError("seed moved a boundary")
        pos[np.searchsorted(nodes,z['rows'])]=z['coordinates']
    # The first failing cell's complete material force star is preserved while
    # minimizing the excess energy in the rest of the connected release.
    affected=np.isin(T,T[targets[0]]).any(axis=1)
    protected=np.isin(f,np.unique(T[affected]))
    hist=[];out.mkdir()
    print('stressed',len(strained),'movable',len(f),'star cells',len(T),'moments',C.shape[0],flush=True)
    for it in range(15):
        en,F=evaluate(pos);K=matrix(refs,pos,T,roles[star],props,f)
        # Preserve sensitive cells during *preparation* when a finite proposed
        # move leaves their measured domain. Re-solve on all remaining modes
        # together; no restriction is installed into physical time integration.
        locked=protected.copy()
        r0=refs[T[:,1:]]-refs[T[:,:1]]
        stiffness=float(np.median(K.diagonal()))
        baseline_s=np.linalg.svd(np.linalg.solve(r0,pos[T[:,1:]]-pos[T[:,:1]]),compute_uv=False)
        for sweep in range(12):
            ok=np.flatnonzero(~locked);dofs=(3*ok[:,None]+np.arange(3)).ravel()
            activeW=W[:,ok];norms=np.maximum(np.linalg.norm(activeW,axis=1),1e-95)
            Qr,Rr,_=qr((activeW/norms[:,None]).T,mode='economic',pivoting=True)
            basis=Qr[:,abs(np.diag(Rr))>abs(Rr).max()*1e-12]
            eq=csr_matrix(np.kron(basis.T,np.eye(3)))*stiffness
            factors=splu(bmat([[K[dofs][:,dofs],eq.T],[eq,None]],format='csc'))
            flat=factors.solve(np.r_[F[f[ok]].ravel(),np.zeros(eq.shape[0])])[:len(dofs)].reshape(-1,3)
            flat-=basis@(basis.T@flat)
            direction=np.zeros_like(F[f]);direction[ok]=flat
            direction*=min(1.,0.00012/max(1.e-30,np.linalg.norm(flat,axis=1).max()))
            test=pos.copy();test[f]+=direction
            pred=np.linalg.svd(np.linalg.solve(r0,test[T[:,1:]]-test[T[:,:1]]),compute_uv=False)
            relJ=(np.linalg.det(test[T[:,1:]]-test[T[:,:1]])/np.linalg.det(r0))
            bad=(pred.min(1)<2/3-1e-8)|(pred.max(1)>1.5+1e-8)|(relJ<.4)|(relJ>2.5)
            vt,qt=geom(test.astype('f4').astype(float),T)
            bad|=(vt<=1e-18)|(qt<1e-5)
            if not bad.any():break
            caught=np.isin(f,np.unique(T[bad])) & (~locked)
            if not caught.any():break
            locked|=caught
        print('protected during preparation',int(locked.sum()),'of',len(f),'remaining offending cells',int(bad.sum()),flush=True)
        if np.linalg.norm(direction)==0:break
        moved=False
        for j in range(19):
            trial=pos.copy();trial[f]+=2.**-j*direction
            ideal=trial
            emitted=ideal.astype("f4").astype(float)
            Vcheck,Qcheck=geom(emitted,T)
            Fcheck=np.linalg.solve(r0,emitted[T[:,1:]]-emitted[T[:,:1]])
            Jcheck=np.linalg.det(Fcheck);Scheck=np.linalg.svd(Fcheck,compute_uv=False)
            fragile=(Qcheck<1.3e-5)|(Vcheck<1.5e-18)|(Jcheck<.44)|(Jcheck>2.45)|(Scheck[:,0]>1.49)|(Scheck[:,-1]<.68)
            pin=np.isin(f,np.unique(T[fragile]))|locked
            trial=emit(ideal,start,f,W,protected=pin)
            if np.max(abs(W@(trial[f]-start[f])))>8e-11:
                trial=emitted
            if np.max(abs(W@(trial[f]-start[f])))>8e-11 or np.array_equal(trial,pos):
                if j in (0,4,10) and it==0: print('moment/unchanged',j,np.max(abs(W@(trial[f]-start[f]))),np.max(abs(W@direction)),flush=True)
                continue
            try: state=certify(refs,trial,T,roles[star],props,caps)
            except ValueError as error:
                if str(error)!='Nonpositive or nonfinite relative volume':raise
                continue
            if not state['passed']:
                if it==0 and j==0:
                    vv,qq=geom(trial,T);jj=np.linalg.det(np.linalg.solve(r0,trial[T[:,1:]]-trial[T[:,:1]]))
                    w=np.flatnonzero((vv<=1e-18)|(qq<1e-5)|(jj<.4)|(jj>2.5))
                    print('failed values',[(int(k),float(vv[k]),float(qq[k]),float(jj[k]),float(Qcheck[k]),float(Jcheck[k])) for k in w[:12]],flush=True)
                if j in (0,4,10,15):print('declined',j,state,flush=True)
                continue
            en1,F1=evaluate(trial);m1=mode(trial,F1)
            if en1>=en or np.linalg.norm(F1[f])>=np.linalg.norm(F[f]) or np.any(abs(m1)>=abs(m0)):continue
            pos=trial;moved=True
            hist.append(dict(iteration=it,alpha=2.**-j,energy_J=en1,force_l2_N=float(np.linalg.norm(F1[f])),modes_s2=m1.tolist()))
            print(hist[-1],flush=True);break
        if not moved:break
    final,F=evaluate(pos);m=mode(pos,F);moved=np.any(pos[f]!=start[f],axis=1)
    if moved.any():np.savez_compressed(out/'preload.npz',rows=free[moved],coordinates=pos[f[moved]].astype('f4'))
    record=dict(source=pins,mass_ledger=sha(flow),rank=sha(rank),materials=identity,opening_strained_cells=len(strained),
       updated_star_cells=len(T),eligible_nodes=len(f),moved=int(moved.sum()),final_admission=certify(refs,pos,T,roles[star],props,caps),
       initial_modes_s2=m0.tolist(),final_modes_s2=m.tolist(),opening_energy_J=en0,final_energy_J=final,
       opening_force_l2_N=float(np.linalg.norm(F0[f])),final_force_l2_N=float(np.linalg.norm(F[f])),
       first_moment_drift_kg_m=float(abs(wt[:,f]@(pos[f]-start[f])).max()),iterations=hist)
    (out/'fit.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','ledger','rank','out','seed'):p.add_argument(k,type=Path)
    fit(**vars(p.parse_args()))
