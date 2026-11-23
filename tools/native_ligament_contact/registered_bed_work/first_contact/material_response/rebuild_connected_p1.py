"""Rebuild both material P1 supports, reusing node classifications only on identical surviving source nodes."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from compile_current_material_support import read,sha,array_hash,verify_affected

def loadmod(path):
    s=importlib.util.spec_from_file_location('connected_native_p1_owner',path);a=importlib.util.module_from_spec(s);s.loader.exec_module(a);return a

def run(h,job,control,ownerpath,out):
    out.mkdir();cfg=read(control)['environment'];pref='NUMI_H706_FRACTIONAL_'
    prior=Path(cfg[pref+'P1_SKIN_NPZ']).parents[1];statepath=prior/'current-paired-volume-state.json'
    if sha(statepath)!=cfg[pref+'STATE_SHA256']:raise ValueError('Wrong historical P1 state')
    state=read(statepath);api=loadmod(ownerpath);owner=api._load_230(api.OWNER230);ids=state['source330_identity']
    for k in ['source330_mass','source330_nodes']:
        if sha(Path(ids[k]['path']))!=ids[k]['sha256']:raise ValueError('Mutable material parent')
    parent=owner.native.read_parent(Path(ids['source330_mass']['path']));parent['points']=np.loadtxt(ids['source330_nodes']['path'],delimiter=',',skiprows=1,usecols=(2,3,4))
    idx=owner.native.build_index(parent);mi=owner.build_material_indices(parent);loc=h/'closed-native/native'
    x=np.loadtxt(loc/'mesh.node',skiprows=1,usecols=(1,2,3),dtype='<f4');a=np.loadtxt(loc/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    t=a[:,:4]-1;mat=a[:,4];orig=np.load(h/'scene-registration/node-map.npz')['source_to_emitted'];back=np.full(len(x),-1,'i4');keep=orig>=0;back[orig[keep]]=np.flatnonzero(keep)
    # Existing cells retained after the two cavities have an exact historical cell ID.
    oldcell=np.arange(sum(read(h/'closed-native/source/receipt.json')[k] for k in ['removed_cells'])+int(np.load(h/'closed-native/source/material-source-indices.npz')['child_begin']),dtype='i4')
    for p in [h/'closed-native/source',loc]:
        cut=np.load(p/'material-source-indices.npz');retain=np.ones(len(oldcell),bool);retain[cut['old_cavity_tets']]=False
        oldcell=np.r_[oldcell[retain],np.full(len(cut['child_geometric_parent']),-1,'i4')]
    if len(oldcell)!=len(t):raise ValueError('Wrong two-stage cell order')
    newstate=dict(state);newstate.update(geometry_receipt=dict(path=str(loc/'result.json'),sha256=sha(loc/'result.json')),geometry_sha256=array_hash(x),tetrahedra_sha256=array_hash(t),mass_sha256=sha(loc/'cell-mass.f64'),native_admitted=False,prior_geometry_state_sha256=sha(statepath))
    st=out/'current-paired-volume-state.json';st.write_text(json.dumps(newstate,indent=2)+'\n');results={};pins={pref+'STATE_SHA256':sha(st)}
    for role,m in [('Skin',15),('Fat',16)]:
        k=role.lower();tag=role.upper();oldfile=Path(cfg[pref+'P1_'+tag+'_NPZ']);oldrec=read(prior/(k+'-p1-receipt.json'));g0=Path(oldrec['role_geometry_path'])
        if sha(oldfile)!=cfg[pref+'P1_'+tag+'_SHA256'] or sha(g0)!=oldrec['role_geometry_sha256']:raise ValueError('Role cache identity')
        old=dict(np.load(oldfile));g=np.load(g0);orows=g['combined_node_rows'];op=old['points_world_m_f32'];parent_rows=np.full(len(orig),-1,'i4');parent_rows[orows]=np.arange(len(orows))
        cells=np.flatnonzero(mat==m);rows,L=np.unique(t[cells],return_inverse=True);L=L.reshape(-1,4);points=x[rows]
        old_rows=np.full(len(rows),-1,'i4');ok=back[rows]>=0;old_rows[ok]=parent_rows[back[rows[ok]]];ok=old_rows>=0
        if not np.array_equal(points[ok],op[old_rows[ok]]):raise ValueError('P1 retained coordinates moved')
        maps=[None]*len(rows);distance=np.zeros(len(rows));offset=old['node_group_offsets'];group=old['node_group_ids'];po=old['node_group_parent_tet_offsets'];pp=old['node_group_parent_tet_rows']
        for j in np.flatnonzero(ok):
            i=old_rows[j];maps[j]={int(group[v]):set(pp[po[v]:po[v+1]].tolist()) for v in range(offset[i],offset[i+1])};distance[j]=old['node_parent_registration_distance_m'][i]
        groups,lookup=api._source_groups(parent,role)
        if not np.array_equal(groups,old['group_tuples']):raise ValueError('Source material/region registry moved')
        need=np.flatnonzero(~ok);limit=state['p1_current_distance_limit_m'];print(role,'new node classifications',len(need),flush=True)
        change=api.build_node_support(points[need],role,parent,idx,mi,groups,lookup,owner,max_distance=limit)
        for j,d,v in zip(need,change['distance'],change['maps'],strict=True):distance[j]=d;maps[j]=v
        if not np.isfinite(distance).all() or max(distance)>limit+1e-12:raise ValueError('Out-of-bounds P1 mapping')
        na=api._serialize_node_maps(points,dict(maps=maps,distance=distance,origin=np.where(ok,2,3).astype('u1')),groups,parent)
        print(role,'forming coefficients',flush=True);ca=api.integrate_p1_cells(L,maps,groups,parent,role)
        adj,_=api.build_region_adjacency(parent,role);neighbors=api.expand_onehop(ca,groups,adj,role,len(cells))
        # Exact rational P1 coefficients and region neighborhoods must be identical on every surviving cell.
        compare=np.flatnonzero(oldcell[cells]>=0);old_global=g['combined_child_rows'];prior_rows=np.searchsorted(old_global,oldcell[cells[compare]])
        if not np.array_equal(old_global[prior_rows],oldcell[cells[compare]]):raise ValueError('Material identity differs on a retained cell')
        for item,cols,offset_name in [(ca,['cell_group_ids','cell_group_p1_score_numerators','cell_group_p1_score_denominators'],'cell_group_offsets'),(neighbors,['cell_admissible_region_ids'],'cell_admissible_region_offsets')]:
            for sub in np.array_split(np.arange(len(compare)),12):
                dest=compare[sub];source=prior_rows[sub];o=item[offset_name];p=old[offset_name];cnt=o[dest+1]-o[dest];cnt0=p[source+1]-p[source]
                if not np.array_equal(cnt,cnt0):raise ValueError('P1 support length differs')
                base=np.r_[0,np.cumsum(cnt)];cursor=np.arange(base[-1])-np.repeat(base[:-1],cnt)
                u=cursor+np.repeat(o[dest],cnt);v=cursor+np.repeat(p[source],cnt)
                if any(not np.array_equal(item[c][u],old[c][v]) for c in cols):raise ValueError('P1 coefficient changed on an unchanged cell')
        geom=out/(k+'-role-geometry.npz');np.savez_compressed(geom,points_world_m_f32=points,tetrahedra=L.astype('<u4'),combined_node_rows=rows,combined_child_rows=cells)
        gh=sha(geom);base=dict(**na,**ca,current_state_sha256=np.array(sha(st)),current_state_path=np.array(str(st)),geometry_sha256=np.array(gh),volume_sha256=np.array(gh),source330_mass_sha256=np.array(ids['source330_mass']['sha256']))
        primary,complete,ps=api._write_support_pair(base,neighbors,out/role,role)
        rec=dict(role=role,points=len(rows),cells=len(cells),reused_nodes=int(ok.sum()),reprojected=len(need),identical_retained_cells=len(compare),maximum_registration_distance_m=float(distance.max()),role_geometry_path=str(geom),role_geometry_sha256=gh,p1_path=str(primary),p1_sha256=ps,p1_one_hop_path=str(complete),p1_one_hop_sha256=sha(complete),state_path=str(st),state_sha256=sha(st))
        for key,value in [('P1_'+tag+'_NPZ',str(complete)),('P1_'+tag+'_SHA256',sha(complete)),('GEOMETRY_'+tag+'_SHA256',gh),('VOLUME_'+tag+'_SHA256',gh)]:pins[pref+key]=value
        (out/(k+'-p1-receipt.json')).write_text(json.dumps(rec,indent=2)+'\n');results[k]=rec;print(role,'qualified',len(compare),flush=True)
    (out/'result.json').write_text(json.dumps(dict(roles=results,environment=pins,owner=sha(ownerpath)),indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
