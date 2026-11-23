"""Rebuild both whole-body material P1 support tables from the physically registered mesh."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from register_material_surfaces import sha,tetgen_positions


def read(path):
    return json.loads(path.read_bytes())


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).data).hexdigest()


def cache_nodes(old, new_points, rows, old_rows, points, api, owner, parent, index, material_index, groups, lookup, role, bound):
    source=np.full(max(int(rows.max()),int(old_rows.max()))+1,-1,dtype='i4')
    source[old_rows]=np.arange(len(old_rows))
    at=source[rows]
    reuse=at>=0
    reuse[reuse] &= np.all(points[at[reuse]] == new_points[reuse],axis=1)
    offsets=old['node_group_offsets'];ids=old['node_group_ids']
    starts=old['node_group_parent_tet_offsets'];parents=old['node_group_parent_tet_rows']
    distance=np.full(len(rows),np.nan)
    maps=[None]*len(rows)
    prev=old['node_parent_registration_distance_m']
    reuse[reuse] &= np.isfinite(prev[at[reuse]]) & (prev[at[reuse]]<=bound+1e-12)
    for i in np.flatnonzero(reuse):
        j=at[i]
        maps[i]={int(ids[s]):set(parents[starts[s]:starts[s+1]].tolist()) for s in range(offsets[j],offsets[j+1])}
        if not maps[i]:raise ValueError('Empty old coordinate support')
        distance[i]=prev[j]
    new=np.flatnonzero(~reuse)
    print(role,'retained point supports',int(reuse.sum()),'new projections',len(new),flush=True)
    if len(new):
        changed=api.build_node_support(new_points[new],role,parent,index,material_index,groups,lookup,owner,max_distance=bound)
        for i,m in zip(new,changed['maps'],strict=True):maps[i]=m
        distance[new]=changed['distance']
    if not np.isfinite(distance).all() or distance.max()>bound+1e-12:raise ValueError('Unresolved physical parent projection')
    return maps,dict(maps=maps,distance=distance,origin=np.where(reuse,2,3).astype('u1')),reuse


def verify_affected(curr, regional, ix, keys):
    for off, cols in keys:
        target=curr[off]
        counts=target[ix+1]-target[ix]
        if not np.array_equal(np.diff(regional[off]),counts):raise ValueError('Regional supports differ: '+off)
        starts=np.r_[0,np.cumsum(counts)]
        positions=np.arange(starts[-1])+np.repeat(target[ix]-starts[:-1],counts)
        for name in cols:
            if not np.array_equal(curr[name][positions], regional[name]):raise ValueError('Regional coefficients differ: '+name)


def compile_all(control,mesh,regional,owner_path,output):
    e=read(control)['environment'];receipt=read(mesh/'receipt.json')
    original=Path(e['NUMI_H706_FRACTIONAL_P1_SKIN_NPZ']).parents[1]
    src_state=original/'current-paired-volume-state.json';old_state=read(src_state)
    if sha(src_state)!=e['NUMI_H706_FRACTIONAL_STATE_SHA256']:raise ValueError('Unmatched prior state')
    region_report=read(regional/'results.json')
    if sha(owner_path)!=region_report['owner']:raise ValueError('Existing projection compiler changed')
    spec=importlib.util.spec_from_file_location('physical_p1_owner',owner_path)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    owner=api._load_230(api.OWNER230)
    source=old_state['source330_identity']
    for k in ('source330_mass','source330_nodes'):
        if sha(Path(source[k]['path']))!=source[k]['sha256']:raise ValueError('Source parent identity changed')
    parent=owner.native.read_parent(Path(source['source330_mass']['path']))
    parent['points']=np.loadtxt(source['source330_nodes']['path'],delimiter=',',skiprows=1,usecols=(2,3,4))
    index=owner.native.build_index(parent);material_index=owner.build_material_indices(parent)
    x=tetgen_positions(mesh/'mesh.node').astype('<f4')
    data=np.loadtxt(mesh/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    cells=data[:,:4]-1;materials=data[:,4]
    for item in receipt['entries']:
        if sha(mesh/item['file'])!=item['sha256']:raise ValueError('Changed repaired mesh')
    output.mkdir(exist_ok=True)
    state=dict(current_reference=old_state['current_reference'],
        geometry_receipt=dict(path=str(mesh/'receipt.json'),sha256=sha(mesh/'receipt.json')),
        geometry_sha256=array_hash(x),tetrahedra_sha256=array_hash(cells),mass_sha256=sha(mesh/'cell-mass.f64'),
        source330_identity=source,native_admitted=False,prior_geometry_state_sha256=sha(src_state),
        historical_preference_source330=old_state['historical_preference_source330'],
        p1_preference_policy=old_state['p1_preference_policy'],p1_current_distance_limit_m=old_state['p1_current_distance_limit_m'])
    state_path=output/'current-paired-volume-state.json'
    text=json.dumps(state,indent=2)+'\n'
    if state_path.exists() and state_path.read_text()!=text:raise ValueError('Prior state changed')
    state_path.write_text(text)
    result={}
    for role,mat in (('Skin',15),('Fat',16)):
        k,tag=role.lower(),role.upper()
        old_file=Path(e['NUMI_H706_FRACTIONAL_P1_'+tag+'_NPZ'])
        if sha(old_file)!=e['NUMI_H706_FRACTIONAL_P1_'+tag+'_SHA256']:raise ValueError('Cache changed')
        old=np.load(old_file);rec=read(original/(k+'-p1-receipt.json'));gf=Path(rec['role_geometry_path'])
        if sha(gf)!=rec['role_geometry_sha256']:raise ValueError('Original role geometry changed')
        prior=np.load(gf);old_xyz=prior['points_world_m_f32']
        if not np.array_equal(old_xyz,old['points_world_m_f32']):raise ValueError('Cached node domain changed')
        child=np.flatnonzero(materials==mat)
        rows,loc=np.unique(cells[child],return_inverse=True);loc=loc.reshape(-1,4);points=x[rows]
        groups,lookup=api._source_groups(parent,role)
        if not np.array_equal(groups,old['group_tuples']):raise ValueError('Typed parent groups changed')
        maps,info,reused=cache_nodes(old,points,rows,prior['combined_node_rows'],old_xyz,api,owner,parent,index,material_index,groups,lookup,role,state['p1_current_distance_limit_m'])
        na=api._serialize_node_maps(points,info,groups,parent)
        print(role,'integrating',len(child),'material cells',flush=True)
        ca=api.integrate_p1_cells(loc,maps,groups,parent,role)
        adjacency,_=api.build_region_adjacency(parent,role)
        neighbors=api.expand_onehop(ca,groups,adjacency,role,len(child))
        section=next(s for s in receipt['regions'] if s['material']==mat)
        ids=np.arange(section['first_cell'],section['first_cell']+section['cells'])
        ix=np.searchsorted(child,ids)
        if not np.array_equal(child[ix],ids):raise ValueError('Regional material rows changed')
        rf=regional/(k+'-recomputed-support.npz')
        if sha(rf)!=next(r['support']for r in region_report['records']if r['material']==mat):raise ValueError('Regional comparison changed')
        with np.load(rf) as r:
            verify_affected(ca,r,ix,[('cell_group_offsets',['cell_group_ids','cell_group_p1_score_numerators','cell_group_p1_score_denominators'])])
            verify_affected(neighbors,r,ix,[('cell_admissible_region_offsets',['cell_admissible_region_ids'])])
        geom=output/(k+'-role-geometry.npz')
        np.savez_compressed(geom,points_world_m_f32=points,tetrahedra=loc.astype('<u4'),combined_node_rows=rows,combined_child_rows=child)
        h=sha(geom);base=dict(**na,**ca,current_state_sha256=np.asarray(sha(state_path)),current_state_path=np.asarray(str(state_path)),
            geometry_sha256=np.asarray(h),volume_sha256=np.asarray(h),source330_mass_sha256=np.asarray(source['source330_mass']['sha256']))
        primary,complete,ph=api._write_support_pair(base,neighbors,output/role,role)
        entry=dict(role=role,points=len(points),cells=len(child),reused=int(reused.sum()),reprojected=int((~reused).sum()),
            maximum_registration_distance_m=float(info['distance'].max()),regional_coefficients_verified=len(ix),
            role_geometry_path=str(geom),role_geometry_sha256=h,p1_path=str(primary),p1_sha256=ph,
            p1_one_hop_path=str(complete),p1_one_hop_sha256=sha(complete),state_path=str(state_path),state_sha256=sha(state_path))
        (output/(k+'-p1-receipt.json')).write_text(json.dumps(entry,indent=2)+'\n');result[k]=entry
        print(role,'certified',len(child),flush=True)
        del maps,info,base,na,ca,neighbors
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':compile_all(*(Path(v).resolve() for v in sys.argv[1:]))
