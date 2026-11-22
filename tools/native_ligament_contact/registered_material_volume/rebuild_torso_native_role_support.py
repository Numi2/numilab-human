"""Recompute actual P1 ancestry and one-hop role support for a changed native volume."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha

def main():
    ap=argparse.ArgumentParser()
    for k in ('base','reg','proposal','out'):ap.add_argument(k,type=Path)
    ap.add_argument("--prior-mesh",type=Path)
    args=ap.parse_args();base,reg,out=args.base,args.reg,args.out;out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(base.parent/'resting-clearance-and-atrium-016'))
    from screen_paired_mass_routes import COMPILER
    from export_repaired_tetgen import array_hash
    entry=Path(json.load((COMPILER/'p1-Skin.command.json').open())['command'][1])
    assert sha(entry)==json.load((base.parent/'resting-clearance-and-atrium-013/fine-local-current-p1.json').open())['owner_sha256']
    spec=importlib.util.spec_from_file_location('physical_role_owner',entry);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);owner=api._load_230(api.OWNER230)
    prior=reg/'material/p1';ps=prior/'current-paired-volume-state.json';st=json.load(ps.open());s=np.load(args.proposal)
    original=args.prior_mesh or base/'constitutive-witness-repaired-tetgen'
    if args.prior_mesh:
        rec=json.loads((original/'receipt.json').read_text())
        if sha(original/'mesh.node')!=rec['mesh_node_sha256'] or sha(original/'mesh.ele')!=rec['mesh_ele_sha256']:
            raise ValueError('Cannot reuse prior mesh with changed identities')
    mesh=out/'mesh';mesh.mkdir(exist_ok=True)
    old=np.loadtxt(original/'mesh.node',skiprows=1,usecols=(1,2,3),dtype='f4');assert array_hash(old,'<f4')==st['geometry_sha256']
    points=old.copy();points[s['nodes']]=s['x'];moved=np.flatnonzero(np.any(points!=old,1))
    dst=mesh/'mesh.node'
    with (original/'mesh.node').open() as source,dst.open('w') as sink:
        sink.write(next(source));changed=set(moved.tolist())
        for i,line in enumerate(source):
            if i not in changed:sink.write(line);continue
            q=points[i];sink.write(f'{i+1} {float(q[0]):.17g} {float(q[1]):.17g} {float(q[2]):.17g}\n')
    ele=mesh/'mesh.ele'
    if not ele.exists():ele.hardlink_to(original/'mesh.ele')
    receipt=dict(baseline_mesh_sha256=sha(original/'mesh.node'),mesh_node_sha256=sha(dst),mesh_ele_sha256=sha(ele),changed_nodes=len(moved),proposal_sha256=sha(args.proposal))
    (mesh/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    p1=out/'material/p1';p1.mkdir(parents=True,exist_ok=True)
    pins=st['source330_identity'];pn=Path(pins['source330_nodes']['path']);pm=Path(pins['source330_mass']['path'])
    assert sha(pn)==pins['source330_nodes']['sha256'] and sha(pm)==pins['source330_mass']['sha256']
    par=owner.native.read_parent(pm);par['points']=np.loadtxt(pn,delimiter=',',skiprows=1,usecols=(2,3,4));index=owner.native.build_index(par);mi=owner.build_material_indices(par)
    st['prior_geometry_state_sha256']=sha(ps);st['geometry_sha256']=array_hash(points,'<f4');st['geometry_receipt']=dict(path=str(mesh/'receipt.json'),sha256=sha(mesh/'receipt.json'))
    st['native_admitted']=False;st['p1_preference_policy']='Exact current same-material projection for moved points. Unchanged points retain the installed source maps and distance certificate.'
    patches={};stats={}
    for role in ('skin','fat'):
        rec=json.load((prior/(role+'-p1-receipt.json')).open());gp=Path(rec['role_geometry_path']);assert sha(gp)==rec['role_geometry_sha256']
        data=dict(np.load(gp));rows=data['combined_node_rows'];local=np.flatnonzero(np.isin(rows,moved));assert np.array_equal(data['points_world_m_f32'],old[rows])
        groups,lookup=api._source_groups(par,role.title());print(role,'project',len(local),flush=True)
        projected=api.build_node_support(points[rows[local]],role.title(),par,index,mi,groups,lookup,owner,max_distance=st['p1_current_distance_limit_m'])
        data['points_world_m_f32']=points[rows];dest=p1/gp.name;np.savez_compressed(dest,**data)
        st['p1_'+role+'_current_distance_upper_bound_m']=max(st['p1_'+role+'_current_distance_upper_bound_m'],float(max(projected['distance'],default=0)))
        patches[role]=(rec,rows,local,projected,groups,dest)
    state=p1/ps.name;state.write_text(json.dumps(st,indent=2)+'\n');sh=sha(state)
    for role,(rec,rows,local,projected,groups,geo) in patches.items():
        geom=np.load(geo);t=geom['tetrahedra'];oldbase=Path(rec['p1_path']);assert sha(oldbase)==rec['p1_sha256'];olddata=dict(np.load(oldbase))
        assert np.array_equal(olddata['group_tuples'],groups)
        off=olddata['node_group_offsets'];gid=olddata['node_group_ids'];po=olddata['node_group_parent_tet_offsets'];pr=olddata['node_group_parent_tet_rows']
        maps=[{int(gid[k]):set(pr[po[k]:po[k+1]].tolist()) for k in range(off[i],off[i+1])} for i in range(len(rows))]
        changed=sum(maps[int(i)]!=g for i,g in zip(local,projected['maps']))
        for i,m in zip(local,projected['maps']):maps[i]=m
        origin=olddata['node_origin_kind'].copy();origin[local]=projected['origin']
        dist=olddata['node_parent_registration_distance_m'].copy();dist[local]=projected['distance']
        print(role,'integrating',len(t),'cells; changed node maps',changed,flush=True)
        arr=api._serialize_node_maps(points[rows],dict(maps=maps,origin=origin,distance=dist),groups,par)
        cell=api.integrate_p1_cells(t,maps,groups,par,role.title());arr.update(cell)
        arr.update(current_state_sha256=np.asarray(sh),current_state_path=np.asarray(str(state)),geometry_sha256=np.asarray(sha(geo)),volume_sha256=np.asarray(sha(geo)),source330_mass_sha256=np.asarray(sha(pm)))
        if set(arr)!=set(olddata):raise ValueError(('Incomplete native P1 keys',set(olddata)^set(arr)))
        dept=p1/role.title();dept.mkdir(exist_ok=True);dst=dept/oldbase.name;np.savez_compressed(dst,**arr);digest=sha(dst)
        adj,_=api.build_region_adjacency(par,role.title());arr.update(api.expand_onehop(cell,groups,adj,role.title(),len(t)));arr['admissibility_source_support_sha256']=np.asarray(digest)
        hp=dept/Path(rec['p1_one_hop_path']).name;np.savez_compressed(hp,**arr)
        rec.update(state_path=str(state),state_sha256=sh,p1_path=str(dst),p1_sha256=digest,p1_one_hop_path=str(hp),p1_one_hop_sha256=sha(hp),role_geometry_path=str(geo),role_geometry_sha256=sha(geo),changed_nodes_exactly_reprojected=rows[local].tolist(),parent_maps_identical=not bool(changed),changed_node_parent_maps=changed)
        (p1/(role+'-p1-receipt.json')).write_text(json.dumps(rec,indent=2)+'\n')
        stats[role]=dict(nodes=len(rows),cells=len(t),moved_nodes=len(local),changed_exact_parent_maps=changed,max_reprojected_distance_m=float(max(projected['distance'],default=0)),integrated_p1_winners=int(sum(cell['cell_semantic_status']==1)),retained_ambiguous=int(len(cell['unresolved_child_tet_rows'])))
        print(role,'complete',stats[role],flush=True)
        del arr,cell,maps,olddata
    (p1/'rebuild.json').write_text(json.dumps(dict(proposal_sha256=sha(args.proposal),owner_sha256=sha(entry),current_state_sha256=sh,roles=stats),indent=2)+'\n')
if __name__=='__main__':main()
