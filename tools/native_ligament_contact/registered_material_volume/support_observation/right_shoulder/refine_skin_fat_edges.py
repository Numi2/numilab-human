"""Use conforming edge stars to refine skinny tissue cells, with source-exact material ancestry."""
from itertools import combinations
from pathlib import Path
import json,sys
import numpy as np
from prepare_shoulder import read
from compile_connected_joint import digest
from repair_skin_fat_quality import incident
from supported_material_admission import geom
from certify_interior_motion import response

EDGES=np.array(list(combinations(range(4),2)))
def metrics(p):
    a=p[:,:,1:]-p[:,:,:1]
    D=np.linalg.det(a)
    L=np.linalg.norm(p[:,:,EDGES[:,1]]-p[:,:,EDGES[:,0]],axis=3)
    q=D/L.max(axis=2)**3
    return q,D,L


def propose(target,coords,t,C,mass,role,constants,limit=45):
    n=coords.shape[1];parents={};newpts={};constraints={};splits=[];paths={};originals=set()
    def node(i):return coords[:,i] if i<n else newpts[i]
    def add_cell(k):
        if k not in originals:
            originals.add(k);key=(k,'');parents[key]=np.array(t[k],dtype='i8')
    add_cell(target)
    for it in range(limit+1):
        kk=list(parents);tt=np.asarray(list(parents.values()));used,inv=np.unique(tt,return_inverse=True)
        xyz=np.stack([node(i) for i in used],axis=1)
        local=inv.reshape(-1,4);q,D,L=metrics(xyz[:,local])
        if D.min()<=6e-18 or np.min(D[1:]/D[0])<.35 or np.max(D[1:]/D[0])>2.5:return None,'cell volume or deformation bound'
        bad=q.min(0)<1e-5
        if not np.any(bad):break
        if it==limit:return None,'refinement budget'
        # Only repair the original witness and defects created inside its connected edge stars.
        idx=np.argmin(q.min(0));pose=np.argmin(q[:,idx]);edge=EDGES[L[pose,idx].argmax()]
        a,b=sorted(tt[idx,edge]);m=n+len(newpts)
        if a<n and b<n:
            for k in np.intersect1d(C.getrow(a).indices,C.getrow(b).indices,assume_unique=True):add_cell(int(k))
        # Split every currently incident leaf, including the other material across a bonded interface.
        hits=[k for k,row in parents.items() if a in row and b in row]
        roots=np.unique([k[0] for k in hits])
        ref=coords[0,t[roots]]
        ex=ref[:,1:]-ref[:,:1]
        g=np.stack([np.linalg.solve(ex,(coords[pose,t[roots,1:]]-coords[pose,t[roots,:1]]))for pose in (1,2)])
        candidates=[]
        for fraction in (.5,.4,.6,1/3,2/3,.45,.55,.375,.625):
            nominal=(1-fraction)*node(a)+fraction*node(b)
            point=nominal.copy();point[0]=nominal[0].astype('f4')
            # Evaluate the actual rounded material site in the neighboring affine
            # deformation fields. Shared exact affine deformation is reproduced.
            dz=point[0]-nominal[0]
            corr=dz@g
            point[1:]+=corr.mean(axis=1)
            disp=point[1:]-point[0];high=disp.astype('f4').astype('f8')
            point[1:]=point[0]+(high+(disp-high).astype('f4').astype('f8'))
            direction=node(b)[0]-node(a)[0]
            off=dz-direction*(dz@direction)/(direction@direction)
            noise=np.linalg.norm(off)+np.max(np.linalg.norm(corr-corr.mean(axis=1,keepdims=True),axis=-1))
            candidates.append((noise*(1+8*abs(fraction-.5)),fraction,point))
        noise,fraction,newpts[m]=min(candidates,key=lambda z:z[0])
        constraints[m]=(a,b,fraction)
        for root,path in hits:
            row=parents.pop((root,path));c=row.copy();d=row.copy()
            c[c==b]=m;d[d==a]=m
            parents[root,path+'0']=c;parents[root,path+'1']=d
        splits.append([int(a),int(b),int(m),fraction])
    # Nested child volumes provide the conservative map. Check it independently after Float32 emission.
    # Pull back child forces through the subdivision operator when comparing discretizations.
    allnodes,tl=np.unique(np.array(list(parents.values())),return_inverse=True);tl=tl.reshape(-1,4)
    oldnodes=np.unique(t[sorted(originals)]);slot={int(i):k for k,i in enumerate(oldnodes)}
    maprows={i:np.eye(len(oldnodes))[o]for i,o in slot.items()}
    for m,(a,b,fraction)in constraints.items():maprows[m]=(1-fraction)*maprows[a]+fraction*maprows[b]
    B=np.vstack([maprows[int(i)] for i in allnodes])
    xnew=np.stack([node(i)for i in allnodes],axis=1);oldshape={};checks=[];minimum=float(q.min())
    src=np.array([k for k,s in parents]);childweights=np.array([np.linalg.det(B[leaf][:,np.searchsorted(oldnodes,t[r])]) for leaf,r in zip(tl,src)])
    worst_force=worst_energy=max_fraction_error=0.
    for k in originals:
        mask=src==k;sel=tl[mask];orig=t[k];weights=childweights[mask]
        if abs(weights.sum()-1)>1e-12:raise ValueError('Broken parent partition')
        old=np.searchsorted(oldnodes,orig)[None,:];mat=constants[str(role[k])]['registered'];mu=mat['source_mu_pa'];bulk=mat['source_logJ_coefficient_pa']
        for frame in range(3):
            vs=geom(xnew[frame],sel)[0];v0=geom(coords[frame,oldnodes],old)[0][0]
            max_fraction_error=max(max_fraction_error,float(np.max(abs(vs/v0/weights-1))))
        for frame in (1,2):
            e,f,_=response(coords[0,oldnodes],coords[frame,oldnodes],old,mu,bulk)
            ee,ff,_=response(xnew[0],xnew[frame],sel,mu,bulk)
            ferr=np.linalg.norm(B.T@ff-f);eerr=abs(ee-e)
            worst_force=max(worst_force,float(ferr/max(np.linalg.norm(f),1e-8)))
            worst_energy=max(worst_energy,float(eerr/max(abs(e),2e-13)))
    if worst_force>.01 or worst_energy>.05:return None,f'force/energy transfer {worst_force:.5g}/{worst_energy:.5g}'
    if max_fraction_error>.01:return None,f'source volume split {max_fraction_error:.5g}'
    svals0=np.linalg.svd(coords[:,t[sorted(originals)]]-coords[:,t[sorted(originals)]].mean(axis=2,keepdims=True),compute_uv=False)
    svals1=np.linalg.svd(xnew[:,tl]-xnew[:,tl].mean(axis=2,keepdims=True),compute_uv=False)
    c0=(svals0[:,:,-1]/svals0[:,:,0]).min();c1=(svals1[:,:,-1]/svals1[:,:,0]).min()
    if c1<=c0:return None,'worst local aspect did not improve'
    childmass=mass[src]*childweights
    delta=np.linalg.norm(childmass@(xnew[0,tl].mean(1)-xnew[0,tl[0,0]])-mass[sorted(originals)]@(coords[0,t[sorted(originals)]].mean(1)-xnew[0,tl[0,0]]))
    if delta>1e-11:return None,'mass first moment'
    return dict(target=target,source_cells=sorted(originals),source_per_child=src.tolist(),children=np.array(list(parents.values())).tolist(),mass_fractions=childweights.tolist(),
                material=role[src].tolist(),mass_kg=childmass.tolist(),splits=splits,coordinates={k:v.tolist()for k,v in newpts.items()},
                old_worst_aspect=float(c0),new_worst_aspect=float(c1),minimum_shape=minimum,worst_projected_force_error=worst_force,
                worst_strain_energy_error=worst_energy,maximum_fractional_volume_error=max_fraction_error,first_moment_error_kg_m=float(delta)),None


def run(j,out):
    out.mkdir(exist_ok=True);case=j/'posterior-interface-082/native-pair';vol=j/'conditioned/witness-material/volume';wit=read(case/'material-check.json')
    cfg=read(case/'launch.json');files=[vol/'mesh.node',Path(cfg['environment']['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']),case/'physical-accepted.node']
    if [digest(f)for f in files]!=[wit[k] for k in ('reference','loaded','physical_frame')]:raise ValueError('Wrong capture')
    x=np.stack([np.loadtxt(f,skiprows=1,usecols=(1,2,3))for f in files])
    d=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=d[:,:4]-1;roles=d[:,-1];C=incident(t,x.shape[1]);m=np.fromfile(vol/'cell-mass.f64','<f8')
    params=read(j.parent/'registration-strain.json')['materials']
    rows=sorted(i for s in ('skin','fat') for i in wit['roles'][s]['failing_cells'])
    edits=[];fail={};seen=set()
    for q in rows:
        if q in seen:continue
        z,reason=propose(q,x,t,C,m,roles,params)
        if z and seen.intersection(z['source_cells']):z=None;reason='overlaps retained cavity'
        if z:edits.append(z);seen.update(z['source_cells'])
        else:fail[q]=reason
        print(q,('fixed '+str(len(z['splits']))) if z else reason,flush=True)
    (out/'result.json').write_text(json.dumps(dict(input=digest(case/'material-check.json'),coordinate_base=x.shape[1],proposals=edits,unresolved=fail),indent=2,allow_nan=False)+'\n')
    print('verified candidate repairs',len(set(rows)&seen),'of',len(rows),flush=True)
if __name__=='__main__':run(*(Path(p).resolve()for p in sys.argv[1:]))
