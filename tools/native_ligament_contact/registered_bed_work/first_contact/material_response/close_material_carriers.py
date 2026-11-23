"""Close remaining source and recipient debits over actual current anatomical neighborhoods."""
import json
import subprocess
import sys
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
from register_material_surfaces import sha,tetgen_positions,PREFIX,pinned


def close_role(tag,mat,e,mesh,carriers,support,solver,out,centers,target_mass,labels,source_xyz,source_tet,parent_mass,radius):
    f=np.load(carriers/(tag+'-candidate-flow.npz'))
    packet=np.load(carriers/(tag.lower()+'-carriers.npz'))
    source_ids=f['carrier'];receiver=f['role_cell'];w=f['mass'];distance=f['distance_m'];eligible=f['eligible'];global_id=f['global_cell']
    parent=np.repeat(packet['global_source330_tet_rows'],np.diff(packet['parent_offsets']))
    quota=packet['carrier_weights']*parent_mass[parent]
    location=np.einsum('ij,ijk->ik',packet['carrier_barycentric'],source_xyz[source_tet[parent]])
    exceptional=~eligible
    if np.any(exceptional):
        overrides=packet['regional_compensation_limits']
        if not (np.isin(parent[source_ids[exceptional]],packet['regional_compensation_parent_rows']).all()
                and np.isin(global_id[exceptional],packet['regional_compensation_child_rows']).all()
                and distance[exceptional].max()<=overrides[1] and w[exceptional].sum()<=overrides[0]):
            raise ValueError('Changed regional compensation authorization')
    src_sum=np.bincount(source_ids,weights=w,minlength=len(quota));rcv_sum=np.bincount(receiver,weights=w,minlength=len(target_mass))
    source_error=quota-src_sum;target_error=target_mass-rcv_sum
    src_bad=np.abs(source_error)>64*np.finfo(float).eps*quota
    change=abs(target_error)>64*np.finfo(float).eps*target_mass
    change[packet['carrier_target_child_rows'][src_bad]]=True
    change[receiver[src_bad[source_ids]]]=True
    # Keep exact authorized compensation fixed. Only ordinary positive edges
    # incident to the affected recipients participate in the correction.
    active=change[receiver]&~exceptional
    cid=np.flatnonzero(change)
    lookup=np.full(len(target_mass),-1,dtype='i4');lookup[cid]=np.arange(len(cid))
    frozen_q=np.bincount(source_ids[~active],weights=w[~active],minlength=len(quota))
    frozen_g=np.bincount(receiver[~active],weights=w[~active],minlength=len(target_mass))
    supply=quota-frozen_q;demand=target_mass[cid]-frozen_g[cid]
    active_q=np.unique(np.r_[source_ids[active],np.flatnonzero(src_bad)]);pos=supply[active_q]>0
    active_q=active_q[pos];supply=supply[active_q]
    if (demand < -64*np.finfo(float).eps*target_mass[cid]).any():raise ValueError('Frozen compensation has excess mass')
    demand=np.maximum(demand,0);ids=np.full(len(quota),-1,dtype='i4');ids[active_q]=np.arange(len(active_q))
    p=np.load(Path(json.loads((support/(tag.lower()+'-p1-receipt.json')).read_bytes())['p1_one_hop_path']))
    offsets=p['cell_admissible_region_offsets'];regions=p['cell_admissible_region_ids']
    tree=cKDTree(location[active_q])
    edges=[];init=[];stalled=[]
    existing={}
    for q,g,v in zip(ids[source_ids[active]],lookup[receiver[active]],w[active]):
        if q>=0:existing[(int(q),int(g))]=existing.get((int(q),int(g)),0.)+float(v)
    for j,c in enumerate(cid):
        qs=np.asarray(tree.query_ball_point(centers[c],radius),dtype='i4')
        allowed=regions[offsets[c]:offsets[c+1]]
        qs=qs[np.isin(labels[parent[active_q[qs]],2],allowed)]
        if not len(qs) and demand[j]>0:stalled.append(int(c))
        for q in qs:
            edges.append((q,j));init.append(existing.get((q,j),0.))
    if stalled:raise ValueError(f'{tag} has recipients without admissible sources: {stalled[:20]}')
    E=np.array(edges,np.uint32);s,g=E.T;init=np.asarray(init)
    if len(existing)>np.count_nonzero(init):raise ValueError('Previous eligible local flow lost an edge')
    assigned=np.bincount(s,weights=init,minlength=len(supply))
    for q in np.flatnonzero(assigned==0):
        js=np.flatnonzero(s==q);positive=js[demand[g[js]]>0]
        if not len(positive):raise ValueError(f'Carrier {active_q[q]} cannot transfer to an active receiver')
        i=positive[np.argmin(np.linalg.norm(centers[cid[g[positive]]]-location[active_q[q]],axis=1))]
        init[i]=supply[q];assigned[q]=supply[q]
    init*=supply[s]/assigned[s]
    prefix=out/tag
    with prefix.with_suffix('.bin').open('wb')as stream:
        for a in (np.array([len(supply),len(demand),len(s)],'u4'),s,g,init,supply,demand,demand,np.zeros(len(s),'u1')):a.tofile(stream)
    r=subprocess.run([str(solver),str(prefix.with_suffix('.bin')),str(prefix.with_suffix('.flow')),str(prefix.with_suffix('.cut'))],capture_output=True,text=True,check=True)
    print(tag,len(demand),'recipients',len(supply),'active sources',len(s),'eligible edges',r.stdout.strip(),flush=True)
    y=np.fromfile(prefix.with_suffix('.flow'),dtype='f8')
    if not np.isfinite(y).all() or np.min(y)<0:raise ValueError('Negative or nonfinite mass')
    positive=y>0;Q=np.r_[source_ids[~active],active_q[s[positive]]];G=np.r_[receiver[~active],cid[g[positive]]];M=np.r_[w[~active],y[positive]]
    sq=np.bincount(Q,weights=M,minlength=len(quota));sg=np.bincount(G,weights=M,minlength=len(target_mass))
    report=dict(local_sources=len(supply),local_targets=len(demand),local_edges=len(s),positive_edges=len(M),
        maximum_source_error_kg=float(np.max(abs(sq-quota))),maximum_relative_source_error=float(np.max(abs(sq-quota)/quota)),
        maximum_target_error_kg=float(np.max(abs(sg-target_mass))),maximum_relative_target_error=float(np.max(abs(sg-target_mass)/target_mass)),
        source_zero=int(np.sum(sq<=0)),target_zero=int(np.sum(sg<=0)),fixed_original_exception_source_edges=int(exceptional.sum()),
        fixed_original_exception_kg=float(w[exceptional].sum()),solver=r.stdout.strip().splitlines())
    report['closed']=bool(report['maximum_relative_source_error']<1e-9 and report['maximum_relative_target_error']<1e-9
        and not report['source_zero'] and not report['target_zero'])
    if report['closed']:
        # Canonicalize by recipient then carrier so native streaming verification
        # can reproduce each recipient index and coalesced carrier debit.
        sequence=np.lexsort((Q,G));np.savez_compressed(out/(tag+'-physical-flow.npz'),carrier=Q[sequence],role_cell=G[sequence],mass=M[sequence])
    return report


def run(launch,mesh,carriers,support,solver,out):
    e=json.loads(launch.read_bytes())['environment'];rec=json.loads((mesh/'receipt.json').read_bytes());out.mkdir()
    x=tetgen_positions(mesh/'mesh.node')
    t=np.loadtxt(mesh/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    m=np.fromfile(mesh/'cell-mass.f64',dtype='f8')
    if sha(mesh/'cell-mass.f64')!=rec['cell_mass_sha256']:raise ValueError('Target masses changed')
    lab=np.loadtxt(pinned(e,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
    pp=Path(e['NUMI_H706_FRACTIONAL_SOURCE330_TETS']);pn=Path(e['NUMI_H706_FRACTIONAL_SOURCE330_NODES'])
    pt=np.loadtxt(pp,delimiter=',',skiprows=1,usecols=(3,4,5,6),dtype='i4');pm=np.loadtxt(pp,delimiter=',',skiprows=1,usecols=(8,));px=np.loadtxt(pn,delimiter=',',skiprows=1,usecols=(5,6,7))
    results={'solver_sha256':sha(solver),'materials':{}}
    for tag,mat in [('SKIN',15),('FAT',16)]:
        sel=t[:,4]==mat
        results['materials'][tag]=close_role(tag,mat,e,mesh,carriers,support,solver,out,x[t[sel,:4]-1].mean(1),m[sel],lab,px,pt,pm,rec['radius_m'])
        (out/'closure.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
