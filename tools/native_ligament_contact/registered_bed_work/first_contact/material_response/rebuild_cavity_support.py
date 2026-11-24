"""Carry unchanged P1 rows exactly and re-integrate only the new finite elements."""
import json
import sys
from pathlib import Path
import numpy as np
from compile_current_material_support import read, sha, array_hash
from rebuild_connected_p1 import loadmod


def splice(old, added, selected, name, values, dest):
    offsets=old[name]
    lengths=offsets[selected+1]-offsets[selected]
    starts=np.r_[0,np.cumsum(lengths)]
    rows=np.repeat(offsets[selected],lengths)+np.arange(starts[-1])-np.repeat(starts[:-1],lengths)
    dest[name]=np.r_[starts,starts[-1]+added[name][1:]].astype(offsets.dtype)
    for field in values:
        dest[field]=np.r_[old[field][rows],added[field]]
    return rows


def run(job,volume,ownerpath,out):
    out.mkdir()
    prior=job/'deep-fat-p1'; registry=read(prior/'result.json')
    if sha(ownerpath)!=registry['owner']:
        raise ValueError('Unexpected P1 source')
    api=loadmod(ownerpath);owner=api._load_230(api.OWNER230)
    statefile=prior/'current-paired-volume-state.json'
    state=read(statefile);ids=state['source330_identity'];mass=Path(ids['source330_mass']['path'])
    if sha(mass)!=ids['source330_mass']['sha256']:
        raise ValueError('Invalid parent source')
    parent=owner.native.read_parent(mass)
    x=np.loadtxt(volume/'mesh.node',skiprows=1,usecols=(1,2,3),dtype='f4')
    e=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    t=e[:,:4]-1
    rec=read(volume/'transfer.json')
    if sha(volume/'mesh.ele')!=rec['elements'] or sha(volume/'cell-mass.f64')!=rec['mass']:
        raise ValueError('Invalid transfer')
    state.update(geometry_receipt=dict(path=str(volume/'transfer.json'),sha256=sha(volume/'transfer.json')),geometry_sha256=array_hash(x),tetrahedra_sha256=array_hash(t),mass_sha256=rec['mass'],native_admitted=False,prior_geometry_state_sha256=sha(statefile))
    now=out/statefile.name;now.write_text(json.dumps(state,indent=2)+'\n')
    mapping=np.load(volume/'overlay.npz')['old_to_new']
    prefix='NUMI_H706_FRACTIONAL_';pins={prefix+'STATE_SHA256':sha(now)}
    oldenv=registry['environment'];roles={}
    for role,m in [('Skin',15),('Fat',16)]:
        tag=role.upper()
        support=Path(oldenv[prefix+'P1_'+tag+'_NPZ']);gpath=prior/(role.lower()+'-role-geometry.npz')
        if sha(support)!=oldenv[prefix+'P1_'+tag+'_SHA256'] or sha(gpath)!=oldenv[prefix+'GEOMETRY_'+tag+'_SHA256']:
            raise ValueError('Basis or geometry changed')
        g=np.load(gpath);old=dict(np.load(support));oldrows=g['combined_child_rows'];kept=np.flatnonzero(mapping[oldrows]>=0)
        globalrows=np.flatnonzero(e[:,4]==m)
        if not np.array_equal(globalrows[:len(kept)],mapping[oldrows[kept]]):
            raise ValueError('Unchanged cells reordered')
        nodes=g['combined_node_rows'];lt=np.searchsorted(nodes,t[globalrows]).astype('i4')
        if not np.array_equal(nodes[lt],t[globalrows]) or not np.array_equal(x[nodes],g['points_world_m_f32']) or not np.array_equal(lt[:len(kept)],g['tetrahedra'][kept]):
            raise ValueError('Node or surviving cell identity changed')
        local=lt[len(kept):];needed=np.unique(local)
        offsets=old['node_group_offsets'];gid=old['node_group_ids']
        po=old['node_group_parent_tet_offsets'];pr=old['node_group_parent_tet_rows']
        maps=[{int(gid[i]):set(pr[po[i]:po[i+1]]) for i in range(offsets[n],offsets[n+1])} for n in needed]
        groups=old['group_tuples']
        added=api.integrate_p1_cells(np.searchsorted(needed,local),maps,groups,parent,role)
        adj,_=api.build_region_adjacency(parent,role)
        added.update(api.expand_onehop(added,groups,adj,role,len(local)))
        new={k:v for k,v in old.items() if k.startswith('node_') or k in ['group_tuples','points_world_m_f32','source330_mass_sha256']}
        for field in ('cell_selected_group_ids','cell_semantic_status','cell_p1_score_denominator','cell_p1_winner_score_numerator'):
            new[field]=np.r_[old[field][kept],added[field]]
        rows=splice(old,added,kept,'cell_group_offsets',['cell_group_ids','cell_group_p1_score_numerators','cell_group_p1_score_denominators'],new)
        for name in ('parent_tet','stable_face'):
            splice(old,added,rows,'cell_group_'+name+'_offsets',['cell_group_'+name+'_rows'],new)
        for mode in ('resolved','unresolved'):
            k=mode+'_child_tet_rows';priorrows=old[k]
            present=np.isin(priorrows,kept);sel=np.flatnonzero(present)
            new[k]=np.r_[np.searchsorted(kept,priorrows[present]),len(kept)+added[k]].astype('i4')
            for field in (['resolved_group_ids','stable_source_face_id'] if mode=='resolved' else ['unresolved_reason_codes']):
                new[field]=np.r_[old[field][sel],added[field]]
            if mode=='resolved':
                for name in ('parent_source_tet','stable_source_face'):
                    splice(old,added,sel,name+'_offsets',[name+'_rows'],new)
        near={};splice(old,added,kept,'cell_admissible_region_offsets',['cell_admissible_region_ids'],near)
        geom=out/gpath.name
        np.savez_compressed(geom,points_world_m_f32=x[nodes],tetrahedra=lt.astype('u4'),combined_node_rows=nodes,combined_child_rows=globalrows)
        gh=sha(geom)
        new.update(current_state_sha256=np.array(sha(now)),current_state_path=np.array(str(now)),geometry_sha256=np.array(gh),volume_sha256=np.array(gh))
        _,complete,_=api._write_support_pair(new,near,out/role,role)
        for k,v in [('P1_'+tag+'_NPZ',str(complete)),('P1_'+tag+'_SHA256',sha(complete)),('GEOMETRY_'+tag+'_SHA256',gh),('VOLUME_'+tag+'_SHA256',gh)]:
            pins[prefix+k]=v
        roles[role]=dict(unchanged_cells=len(kept),new_cells=len(local),unchanged_nodes=len(nodes));print(role,roles[role],flush=True)
    (out/'result.json').write_text(json.dumps(dict(environment=pins,roles=roles,owner=sha(ownerpath)),indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
