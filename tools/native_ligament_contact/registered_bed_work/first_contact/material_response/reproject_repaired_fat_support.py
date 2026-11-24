"""Prove the current P1 material support of moved points against the actual coarse source."""
from pathlib import Path
import json,sys,os
import numpy as np
from rebuild_connected_p1 import loadmod
from compile_current_material_support import read,sha,array_hash
def run(job,geom,field,ownerfile,out):
    previous=job/'material-p1';registry=read(previous/'result.json')
    if sha(ownerfile)!=registry['owner']:raise ValueError('Incorrect material projection owner')
    api=loadmod(ownerfile);owner=api._load_230(api.OWNER230)
    oldstate=previous/'current-paired-volume-state.json';state=read(oldstate);ids=state['source330_identity']
    for key in ['mass','nodes']:
        row=ids['source330_'+key]
        if sha(Path(row['path']))!=row['sha256']:raise ValueError('Source volume changed')
    parent=owner.native.read_parent(Path(ids['source330_mass']['path']));parent['points']=np.loadtxt(ids['source330_nodes']['path'],delimiter=',',skiprows=1,usecols=(2,3,4));idx=owner.native.build_index(parent);mi=owner.build_material_indices(parent)
    out.mkdir();x=np.loadtxt(geom/'mesh.node',skiprows=1,usecols=(1,2,3),dtype='<f4')
    state.update(geometry_receipt=dict(path=str(geom/'result.json'),sha256=sha(geom/'result.json')),geometry_sha256=array_hash(x),mass_sha256=sha(geom/'cell-mass.f64'),prior_geometry_state_sha256=sha(oldstate),native_admitted=False)
    sf=out/'current-paired-volume-state.json';sf.write_text(json.dumps(state,indent=2)+'\n');pr='NUMI_H706_FRACTIONAL_';env={pr+'STATE_SHA256':sha(sf)};proof={}
    for role in ['Skin','Fat']:
        tag=role.upper();k=role.lower();entries=read(previous/(k+'-p1-receipt.json'));geofile=Path(entries['role_geometry_path']);pfile=Path(registry['environment'][pr+'P1_'+tag+'_NPZ'])
        if sha(geofile)!=entries['role_geometry_sha256'] or sha(pfile)!=registry['environment'][pr+'P1_'+tag+'_SHA256']:raise ValueError('Previous support changed')
        geo=dict(np.load(geofile));old=dict(np.load(pfile));rows=geo['combined_node_rows'];new=x[rows];changed=np.flatnonzero(np.any(new!=old['points_world_m_f32'],axis=1))
        f=out/(k+'-role-geometry.npz')
        if len(changed):
            geo['points_world_m_f32']=new;np.savez_compressed(f,**geo);groups,lookup=api._source_groups(parent,role)
            if not np.array_equal(groups,old['group_tuples']):raise ValueError('Source ownership changed')
            solved=api.build_node_support(new[changed],role,parent,idx,mi,groups,lookup,owner,max_distance=state['p1_current_distance_limit_m'])
            off,g,po,pp=(old[s] for s in ['node_group_offsets','node_group_ids','node_group_parent_tet_offsets','node_group_parent_tet_rows'])
            for j,item,dist in zip(changed,solved['maps'],solved['distance'],strict=True):
                before={int(g[i]):set(pp[po[i]:po[i+1]].tolist()) for i in range(off[j],off[j+1])}
                if before!=item:raise ValueError(f'Material groups changed at node {rows[j]}; require reintegration')
                old['node_parent_registration_distance_m'][j]=dist;old['node_origin_kind'][j]=3
            old['points_world_m_f32']=new
        else:os.link(geofile,f)
        path=out/role/pfile.name;path.parent.mkdir();gs=sha(f)
        old.update(current_state_sha256=np.array(sha(sf)),current_state_path=np.array(str(sf)),geometry_sha256=np.array(gs),volume_sha256=np.array(gs))
        np.savez_compressed(path,**old)
        proof[tag]=dict(nodes=len(rows),changed=rows[changed].tolist(),identical_group_and_parent_support=True)
        for key,val in [('P1_'+tag+'_NPZ',str(path)),('P1_'+tag+'_SHA256',sha(path)),('GEOMETRY_'+tag+'_SHA256',gs),('VOLUME_'+tag+'_SHA256',gs)]:env[pr+key]=val
    (out/'result.json').write_text(json.dumps(dict(environment=env,receipts=proof,owner=sha(ownerfile)),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
