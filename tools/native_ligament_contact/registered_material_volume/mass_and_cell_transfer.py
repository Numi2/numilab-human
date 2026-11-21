"""Rebalance the verified fine mesh from the actual native parent and certify all route distances."""
from pathlib import Path
import sys,json,subprocess
import numpy as np
from emit_skin_source_registration import sha
baseline,capture,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True);fine=baseline/'refit-motion-qualification'
flows=fine/'refit-density-closed-flow.npz';position=baseline/'constitutive-witness-repaired-tetgen/mesh.node'
cells_path=position.with_name('mesh.ele');volume=fine/'constitutive-witness-repaired.volumes'
cert=json.loads((fine/'refit-density-closure.json').read_text());wr=json.loads((fine/'witnesses/receipt.json').read_text());assert sha(flows)==cert['candidate_sha256']==wr['verified_flow_sha256'] and sha(volume)==cert['native_volume_sha256']
mesh=json.loads(position.with_name('receipt.json').read_text());assert sha(position)==mesh['mesh_node_sha256'] and sha(cells_path)==mesh['mesh_ele_sha256']
parent=capture/'compiled-parent-cell-mass.csv';old=baseline/'floor-live-parent-capture-002'/parent.name
new_parent,old_parent=[np.loadtxt(f,delimiter=',',skiprows=1) for f in (parent,old)]
assert np.array_equal(new_parent[:,:2],old_parent[:,:2]);assert np.all(new_parent[:,2]>0)
pins=json.loads((capture/'receipt.json').read_text())['outputs']
for n,h in pins.items():assert sha(capture/n)==h
nodes=np.loadtxt(str(parent)+'.nodes.csv',delimiter=',',skiprows=1);tet=np.loadtxt(str(parent)+'.tets.csv',delimiter=',',skiprows=1)
labels,old_labels=[np.loadtxt(str(f)+'.labels.csv',delimiter=',',skiprows=1,dtype=np.int64)for f in (parent,old)]
assert np.array_equal(labels[:,:4],old_labels[:,:4]) and np.array_equal(tet[:,:7],np.loadtxt(str(old)+'.tets.csv',delimiter=',',skiprows=1,usecols=range(7)))
with np.load(flows)as data:flow=dict(data)
raw_nodes=np.loadtxt(position,skiprows=1,usecols=(1,2,3));cells=np.loadtxt(cells_path,skiprows=1,usecols=(1,2,3,4,5),dtype=np.int32);fcorner=cells[:,:4]-1
center=sum(raw_nodes[fcorner[:,i]]*.25 for i in range(4))
rows=[];source=[];weights=[]
for witness in wr['roles']:
    path=Path(witness['carrier_path']);assert sha(path)==witness['carrier_sha256']
    with np.load(path) as carriers:
        pp=np.repeat(carriers['global_source330_tet_rows'],np.diff(carriers['parent_offsets']));rows.append(pp)
        source.append(np.einsum('ni,nij->nj',carriers['carrier_barycentric'],nodes[tet[pp,3:7].astype('i4'),5:8]))
        weights.append(carriers['carrier_weights'])
par=np.concatenate(rows);src=np.concatenate(source);wt=np.concatenate(weights)
ids=flow['carrier_row'];group=flow['group_for_cell'];targets=flow['group'];prev=flow['mass_flow_kg'];arch=flow['archived_compensation'];ng=int(group.max())+1
assert np.array_equal(par,flow['source_parent']);q=wt*new_parent[par,2];old_q=np.bincount(ids,weights=prev,minlength=len(q))
assert np.max(np.abs(wt*old_parent[par,2]-old_q))<2e-12 and np.all(q>0)
bounds=np.bincount(group,minlength=ng);order=np.argsort(group,kind='stable');off=np.concatenate(([0],bounds.cumsum()))
limit=wr['transport_distance_limit_m'];allowed=np.empty(len(ids),dtype=bool);distances=np.empty(len(ids))
for i in range(0,len(ids),30000):
    j=min(i+30000,len(ids));k=targets[i:j];num=bounds[k];edge=np.repeat(np.arange(j-i),num)
    child=order[off[k[edge]]+np.arange(len(edge))-np.repeat(np.r_[0,num.cumsum()[:-1]],num)]
    distances[i:j]=np.maximum.reduceat(np.linalg.norm(center[child]-src[ids[i:j][edge]],axis=1),np.r_[0,num.cumsum()[:-1]])
allowed=distances<=limit
# A historical compensation route cannot acquire mass beyond its recorded allocation.
base=prev*(q/old_q)[ids];base[~allowed]=0;base[arch]=np.minimum(base[arch],prev[arch])
need=q-np.bincount(ids,weights=base,minlength=len(q));ordinary=allowed&~arch;eligible=np.bincount(ids,weights=ordinary*prev,minlength=len(q))
if np.any((need>1e-13)&(eligible==0)):raise ValueError('A changed carrier requires a new verified route')
base[ordinary]+=prev[ordinary]*np.divide(need,eligible,out=np.zeros_like(need),where=eligible>0)[ids[ordinary]]
if np.min(base)<-1e-15:raise ValueError('Negative quota initialization')
base=np.maximum(base,0.);eligible_edges=np.flatnonzero(allowed)
vol=np.fromfile(volume,dtype='f8');skin=cells[:,4]==15;reserve=cert['reserve_kg_m3'];lo=(np.where(skin,900.,850.)+reserve)*vol;hi=(np.where(skin,1100.,950.)-reserve)*vol
lower=np.bincount(group,weights=lo);upper=np.bincount(group,weights=hi);inp=out/'density-input.bin';sol=out/'density-solution.bin';cut=out/'density-cut.bin'
with inp.open('wb') as f:
    for a,dtype in [([len(q),ng,len(eligible_edges)],'u4'),(ids[eligible_edges],'u4'),(targets[eligible_edges],'u4'),(base[eligible_edges],'f8'),(q,'f8'),(lower,'f8'),(upper,'f8'),(arch[eligible_edges],'u1')]:np.asarray(a,dtype=dtype).tofile(f)
solver=baseline/'close_live_parent_density_compiled';status=subprocess.run([solver,inp,sol,cut],capture_output=True,text=True)
r=dict(cooked_parent_rows=len(new_parent),current_total_kg=float(new_parent[:,2].sum()),recorded_total_kg=float(old_parent[:,2].sum()),maximum_parent_cell_mass_change_kg=float(abs(new_parent[:,2]-old_parent[:,2]).max()),changed_face_labels=int(np.sum(labels[:,4]!=old_labels[:,4])),carrier_count=len(q),edges=len(ids),distance_bound_m=limit,maximum_original_route_distance_m=float(distances.max()),blocked= int(np.count_nonzero(~allowed)),blocked_positive=int(np.count_nonzero((prev>0)&~allowed)),solver_exit=status.returncode,solver_output=status.stdout+status.stderr,files={str(f):sha(f) for f in [parent,position,cells_path,volume,flows,solver,inp]},role_import_verified=False)
def write(): (out/'mass-transfer.json').write_text(json.dumps(r,indent=2)+'\n')
write()
if status.returncode:sys.exit(status.returncode)
mass_flow=np.zeros_like(prev);mass_flow[eligible_edges]=np.fromfile(sol,'f8')
assert np.all(np.isfinite(mass_flow)&(mass_flow>=0)) and np.all(mass_flow[arch]<=prev[arch]+1e-17)
carrier_error=np.max(np.abs(np.bincount(ids,weights=mass_flow,minlength=len(q))-q));assigned=np.bincount(targets,weights=mass_flow,minlength=ng);overflow=np.maximum(lower-assigned,assigned-upper)
# Finish the residual minimum cut using a destination that already admits
# the same material, donor and anatomical region. Each new carrier's distance
# is checked against every fine child represented by that destination.
added=[];index={(int(c),int(g)):i for i,(c,g) in enumerate(zip(ids,targets))}
_,kinds=np.unique(labels[:,1:4],axis=0,return_inverse=True);kind=kinds.ravel()[par]
for g in np.flatnonzero(assigned < lower-1e-15):
    dest_rows=np.flatnonzero((targets==g)&(prev>0)&~arch)
    valid_kind=np.unique(kind[ids[dest_rows]])
    fine_rows=order[off[g]:off[g+1]]
    middle=center[fine_rows].mean(axis=0)
    possible=np.flatnonzero(np.isin(kind,valid_kind))
    for c in possible[np.argsort(np.linalg.norm(src[possible]-middle,axis=1),kind='stable')]:
        if assigned[g]>=lower[g]-1e-17:break
        distance=float(np.linalg.norm(center[fine_rows]-src[c],axis=1).max())
        if distance>limit:continue
        dst=index.get((int(c),int(g)))
        if dst is not None and arch[dst]:continue
        from_ids=np.flatnonzero((ids==c)&(targets!=g)&(mass_flow>0))
        for e in from_ids:
            other=targets[e]
            transfer=min(lower[g]-assigned[g],mass_flow[e],assigned[other]-lower[other])
            if transfer<=1e-17:continue
            if dst is None:
                dst=len(ids);index[(int(c),int(g))]=dst
                ids=np.append(ids,np.asarray(c,dtype=ids.dtype));targets=np.append(targets,np.asarray(g,dtype=targets.dtype))
                mass_flow=np.append(mass_flow,0.);distances=np.append(distances,distance);arch=np.append(arch,False)
                flow['prior_edge_row']=np.append(flow['prior_edge_row'],np.asarray(-1,dtype=flow['prior_edge_row'].dtype))
                added.append(dict(carrier=int(c),receiver=int(g),maximum_distance_m=distance,source_tuple=labels[par[c],1:4].tolist(),verified_peer=int(dest_rows[0])))
            mass_flow[e]-=transfer;mass_flow[dst]+=transfer;assigned[other]-=transfer;assigned[g]+=transfer
            if assigned[g]>=lower[g]-1e-17:break
flow.update(carrier_row=ids,group=targets,archived_compensation=arch)
r['qualified_additional_routes']=added
assigned=np.bincount(targets,weights=mass_flow,minlength=ng);carrier_error=float(abs(np.bincount(ids,weights=mass_flow,minlength=len(q))-q).max())
overflow=np.maximum(lower-assigned,assigned-upper)
r.update(maximum_carrier_error_kg=float(carrier_error),unmet_groups=int((overflow>1e-15).sum()),unmet_group_mass_kg=float(np.maximum(overflow,0).sum()));write()
assert carrier_error<1e-14 and r['unmet_groups']==0
mass=np.clip(flow['element_mass_kg'].copy(),lo,hi);delta=assigned-np.bincount(group,weights=mass);room=np.where(delta[group]>=0,hi,lo)-mass;total=np.bincount(group,weights=room);mass+=room*np.clip(np.divide(delta,total,out=np.zeros_like(delta),where=total!=0),0,1)[group];actual=np.bincount(group,weights=mass)
assert np.all(abs(actual-assigned)<=np.maximum(1e-20,assigned*2e-12))
density=mass*(assigned/actual)[group]/vol
assert np.all(density>=np.where(skin,900.,850.)) and np.all(density<=np.where(skin,1100.,950.))
flow.update(mass_flow_kg=mass_flow,element_mass_kg=mass,nodal_mass_kg=sum(np.bincount(fcorner[:,k],weights=mass*.25,minlength=len(raw_nodes))for k in range(4)))
packet=out/'current-density-closed-flow.npz';np.savez_compressed(packet,**flow);r.update(cells=len(vol),density_outliers=0,allocated_mass_kg=float(mass.sum()),new_flow_sha256=sha(packet),used_route_distance_max_m=float(distances[mass_flow>0].max()))
write()
