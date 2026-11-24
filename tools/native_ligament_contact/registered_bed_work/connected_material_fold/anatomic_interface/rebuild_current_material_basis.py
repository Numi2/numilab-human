"""Rebuild the source-backed P1 basis of the exact imported mesh, including edited vertex positions."""
from pathlib import Path
import json,gc,sys
import numpy as np
from compile_current_material_support import array_hash
from compile_connected_joint import digest
from rebuild_connected_p1 import loadmod

def gather(a,b,rows):
    out=np.empty((len(rows),*a.shape[1:]),a.dtype);k=rows>=0;out[k]=a[rows[k]];out[~k]=b[-1-rows[~k]];return out

def children(a,b,rows):
    # Negative row IDs name entries in the newly evaluated part of the table.
    counts=gather(np.diff(a),np.diff(b),rows);offs=np.r_[0,np.cumsum(counts)].astype(a.dtype)
    begin=gather(a[:-1],b[:-1],rows)
    vals=np.repeat(begin,counts)+np.arange(offs[-1])-np.repeat(offs[:-1],counts)
    sel=np.repeat(rows<0,counts);vals[sel]=-1-vals[sel];return offs,vals

def copy_groups(out,old,fresh,select,root):
    off,source=children(old[root+'_group_offsets'],fresh[root+'_group_offsets'],select);out[root+'_group_offsets']=off
    keys=['ids'] if root=='node'else ['ids','p1_score_numerators','p1_score_denominators']
    for key in keys:
        k=root+'_group_'+key;out[k]=gather(old[k],fresh[k],source)
    for name in ('parent_tet','stable_face'):
        key=root+'_group_'+name;offs,idx=children(old[key+'_offsets'],fresh[key+'_offsets'],source);out[key+'_offsets']=offs;out[key+'_rows']=gather(old[key+'_rows'],fresh[key+'_rows'],idx)
def lineage(q):
    rows=np.load(q/'seam-volume/local-ancestry.npz')['surviving_source_cells'].copy()
    for dst in (q/'fold-volume',q/'remaining-fold/collar-2/fold-volume'):
        a=np.load(dst/'ancestry.npz');n=json.loads((dst/'result.json').read_bytes())['new_elements']
        rows=np.r_[rows[a['unchanged_prior_cells']],np.full(n,-1,'i4')]
    return rows

def run(work,refined=False):
    q=work.parent
    v=work/'witness-material/volume' if refined else q/'qualified-material'
    prior=work/'current-p1' if refined else q.parent/'relative/material-p1'
    reg=json.loads((prior/'result.json').read_bytes());src=Path((work/'owner.locator').read_text().strip())
    if digest(src)!=reg['owner']:raise ValueError('Wrong P1 owner')
    api=loadmod(src);owner=api._load_230(api.OWNER230);statepath=prior/'current-paired-volume-state.json';state=json.loads(statepath.read_bytes());ids=state['source330_identity']
    for k in ('source330_mass','source330_nodes'):
        if digest(Path(ids[k]['path']))!=ids[k]['sha256']:raise ValueError('Source330 asset changed')
    parent=owner.native.read_parent(Path(ids['source330_mass']['path']));parent['points']=np.loadtxt(ids['source330_nodes']['path'],delimiter=',',skiprows=1,usecols=(2,3,4))
    idx=owner.native.build_index(parent);mi=owner.build_material_indices(parent);out=(work/'witness-material' if refined else work)/'current-p1';out.mkdir(exist_ok=True)
    x=np.loadtxt(v/'mesh.node',skiprows=1,usecols=(1,2,3),dtype='f4');cells=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=cells[:,:4]-1
    labels=np.loadtxt(v/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
    if refined:
        overlay=np.load(v/'source-overlay.npz')
        priorCells=np.arange(len(t),dtype='i4')
        priorCells[overlay['new_cells']]=-1
        nd=np.arange(len(x),dtype='i4')
        nd[-1]=-1
    else:
        priorCells=lineage(q)
        nd=np.load(q/'seam-volume/local-ancestry.npz')['surviving_source_nodes']
    cert=json.loads((v/'result.json').read_bytes())
    if len(priorCells)!=len(t) or len(nd)!=len(x) or cert['exit']:raise ValueError('Unverified mesh ancestry')
    for k in ('mesh.node','mesh.ele','cell-mass.f64','current-child.labels.csv'):
        if digest(v/k)!=cert['inputs'][k]:raise ValueError('Unverified material')
    now=out/statepath.name;state.update(geometry_receipt={'path':str(v/'result.json'),'sha256':digest(v/'result.json')},geometry_sha256=array_hash(x),tetrahedra_sha256=array_hash(t),mass_sha256=cert['inputs']['cell-mass.f64'],prior_geometry_state_sha256=digest(statepath),native_admitted=False);now.write_text(json.dumps(state,indent=2)+'\n')
    prefix='NUMI_H706_FRACTIONAL_';pins={prefix+'STATE_SHA256':digest(now)};roles={}
    for role,material in [('Skin',15),('Fat',16)]:
        tag=role.upper();key=prefix+'P1_'+tag;op=Path(reg['environment'][key+'_NPZ']);gf=prior/(role.lower()+'-role-geometry.npz')
        if digest(op)!=reg['environment'][key+'_SHA256'] or digest(gf)!=reg['environment'][prefix+'GEOMETRY_'+tag+'_SHA256']:raise ValueError('Unverified source basis')
        old=dict(np.load(op));g=np.load(gf);rows=np.flatnonzero(cells[:,4]==material);nodes,inv=np.unique(t[rows],return_inverse=True);local=inv.reshape(-1,4).astype('i4');pt=x[nodes]
        og=g['combined_node_rows'];ix=np.searchsorted(og,nd[nodes]);oldrows=g['combined_child_rows']
        existing=nd[nodes]>=0
        if not np.array_equal(og[ix[existing]],nd[nodes][existing]):raise ValueError('Missing role node')
        changed=np.flatnonzero(~existing | np.any(pt!=g['points_world_m_f32'][ix],axis=1));newnode=ix.copy();newnode[changed]=-1-np.arange(len(changed))
        groups,lookup=api._source_groups(parent,role)
        if not np.array_equal(groups,old['group_tuples']):raise ValueError('Source group identity changed')
        newmap=api.build_node_support(pt[changed],role,parent,idx,mi,groups,lookup,owner,max_distance=state['p1_current_distance_limit_m']) if len(changed) else dict(maps=[],origin=np.empty(0,'u1'),distance=np.empty(0))
        new=api._serialize_node_maps(pt[changed],newmap,groups,parent)
        composed=dict(points_world_m_f32=pt,group_tuples=groups,source330_mass_sha256=old['source330_mass_sha256'])
        for k in ['node_origin_kind','node_parent_registration_distance_m']:composed[k]=gather(old[k],new[k],newnode)
        copy_groups(composed,old,new,newnode,'node')
        srcrows=priorCells[rows];ret=np.flatnonzero(srcrows>=0);cr=np.full(len(rows),-1,'i4');oi=np.searchsorted(oldrows,srcrows[ret])
        if not np.array_equal(oldrows[oi],srcrows[ret]):raise ValueError('Missing role element')
        original_nodes=g['combined_node_rows'][g['tetrahedra'][oi]];same=(original_nodes==nd[t[rows[ret]]]).all(1)&(~np.isin(local[ret],changed).any(1))
        cr[ret[same]]=oi[same];add=np.flatnonzero(cr<0);cr[add]=-1-np.arange(len(add))
        needed=np.unique(local[add]);go=composed['node_group_offsets'];pgo=composed['node_group_parent_tet_offsets'];parentrows=composed['node_group_parent_tet_rows'];gids=composed['node_group_ids']
        maps=[{int(gids[u]):set(parentrows[pgo[u]:pgo[u+1]])for u in range(go[n],go[n+1])}for n in needed]
        fresh=api.integrate_p1_cells(np.searchsorted(needed,local[add]),maps,groups,parent,role);adj,_=api.build_region_adjacency(parent,role);fresh.update(api.expand_onehop(fresh,groups,adj,role,len(add)))
        for k in ['cell_selected_group_ids','cell_semantic_status','cell_p1_score_denominator','cell_p1_winner_score_numerator']:composed[k]=gather(old[k],fresh[k],cr)
        copy_groups(composed,old,fresh,cr,'cell');near={}
        name='cell_admissible_region';offs,idx2=children(old[name+'_offsets'],fresh[name+'_offsets'],cr);near[name+'_offsets']=offs;near[name+'_ids']=gather(old[name+'_ids'],fresh[name+'_ids'],idx2)
        near['admissibility_max_source_region_hops']=np.asarray(1,'i4')
        for mode in ['resolved','unresolved']:
            k=mode+'_child_tet_rows';target=np.flatnonzero((composed['cell_selected_group_ids']>=0)==(mode=='resolved'));pr=cr[target];ok=pr>=0
            subset=np.empty(len(target),'i8');subset[ok]=np.searchsorted(old[k],pr[ok]);subset[~ok]=-1-np.searchsorted(fresh[k],-1-pr[~ok])
            if not np.array_equal(old[k][subset[ok]],pr[ok]) or not np.array_equal(fresh[k][-1-subset[~ok]],-1-pr[~ok]):raise ValueError('Unmatched resolved subset')
            composed[k]=target.astype('i4')
            for name in (['stable_source_face_id','resolved_group_ids'] if mode=='resolved'else ['unresolved_reason_codes']):composed[name]=gather(old[name],fresh[name],subset)
            if mode=='resolved':
                for name in ('parent_source_tet','stable_source_face'):
                    z,zr=children(old[name+'_offsets'],fresh[name+'_offsets'],subset);composed[name+'_offsets']=z;composed[name+'_rows']=gather(old[name+'_rows'],fresh[name+'_rows'],zr)
        bad=[]
        for ci in add:
            off=composed['cell_group_offsets'];sl=np.arange(off[ci],off[ci+1]);num=composed['cell_group_p1_score_numerators'][sl];slot=sl[num==num.max()];tagrow=labels[rows[ci],1:]
            valid=False
            for i in slot:
                gid=composed['cell_group_ids'][i];fo=composed['cell_group_stable_face_offsets'];fs=composed['cell_group_stable_face_rows'][fo[i]:fo[i+1]]
                if np.array_equal(groups[gid],tagrow[:3]) and tagrow[3] in fs:valid=True
            if not valid:bad.append(int(rows[ci]))
        rolefile=out/gf.name;np.savez_compressed(rolefile,points_world_m_f32=pt,tetrahedra=local.astype('u4'),combined_node_rows=nodes,combined_child_rows=rows);gh=digest(rolefile)
        composed.update(current_state_sha256=np.array(digest(now)),current_state_path=np.array(str(now)),geometry_sha256=np.array(gh),volume_sha256=np.array(gh))
        _,complete,_=api._write_support_pair(composed,near,out/role,role)
        for k,val in [('P1_'+tag+'_NPZ',str(complete)),('P1_'+tag+'_SHA256',digest(complete)),('GEOMETRY_'+tag+'_SHA256',gh),('VOLUME_'+tag+'_SHA256',gh)]:pins[prefix+k]=val
        roles[role]=dict(nodes=len(nodes),changed_node_classifications=len(changed),elements=len(rows),reintegrated_elements=len(add),unchanged_proven_elements=int((cr>=0).sum()),inadmissible_label_rows=bad)
        print(role,roles[role],flush=True);del composed,old;gc.collect()
    (out/'result.json').write_text(json.dumps(dict(owner=digest(src),roles=roles,environment=pins),indent=2)+'\n')
if __name__=='__main__':run(Path(__file__).resolve().parent, '--motion-witness' in sys.argv[1:])
