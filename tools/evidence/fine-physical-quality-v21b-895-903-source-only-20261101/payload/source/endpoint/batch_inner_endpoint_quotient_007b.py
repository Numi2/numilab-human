#!/usr/bin/env python3
"""Cached local-star endpoint quotient batch with one final global replay.

Candidate edges come from the pinned active-row census. Baseline source and
volume topology/quality are cached once; each trial certifies only its complete
disjoint endpoint stars and updates the global quality ledger by local deltas.
Final arrays are flattened to the exact input state, rebound, and independently
read back once. External 775/805/all185 audits and mass mapping remain pending.
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
CORE=HERE/'inner_endpoint_quotient_002.py'
CORE_SHA='e6ce2c436ce0a4ca700e2a4b1be524403576a066c42efe6c945c6c5cb87cea36'
PROTECTION=H/'quality-partial-cavity-684-root'/'paired_connected_boundary_patch_855_active_rows_lineage.py'
PROTECTION_SHA='5645dbfda4312b6df81b31bca03ba75fdf16af4f4956540c3b92f66b67eb101d'
PROTECTION_OWNER_SHA='68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a'
REBINDER=H/'desired-boundary-source-rebind-001-nativepreflight'/'desired_boundary_source_rebind_877.py'
REBINDER_SHA='78f60687bda5f416604608ef877c7a8c1251bb74e3c53b6929385ce586000a34'
MAX_EDGE_M=50e-6

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def imp(name,path,expected=None):
    path=Path(path).resolve()
    if expected and sha(path)!=expected:raise ValueError(name+' SHA mismatch')
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader:raise RuntimeError('cannot import '+str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def pin(ref,label):
    if not isinstance(ref,dict) or not ref.get('path') or not ref.get('sha256'):raise ValueError(label+' has no path/SHA')
    p=Path(ref['path']).resolve()
    if not p.is_file() or sha(p)!=str(ref['sha256']):raise ValueError(label+' SHA mismatch')
    return p
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError('not JSON serializable: '+type(value).__name__)
def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,default=json_default)+'\n')
def compact_metrics(doc,omit=()):
    skip=set(omit);out={}
    for k,v in doc.items():
        if k in skip or isinstance(v,np.ndarray):continue
        out[k]=v
    return out
def set_text_ref(arrays,path_key,sha_key,parent_ref):
    for key,value in ((path_key,parent_ref['path']),(sha_key,parent_ref['sha256'])):
        if key not in arrays:continue
        old=np.asarray(arrays[key]);dtype=f'<U{max(1,len(value))}'
        arrays[key]=np.asarray(value,dtype=dtype) if old.ndim==0 else np.asarray([value],dtype=dtype)
def exact_chain(faces):return core.face_chain(np.asarray(faces,np.int64))
def typed_union(surface,rows):
    off=np.asarray(surface['quality_star_patch_union_offsets'],np.int64)
    cols=[np.asarray(surface['quality_star_patch_union_face_source_'+k],np.int64) for k in core.LANES]
    tuples=set()
    for r in rows:
        a,b=int(off[int(r)]),int(off[int(r)+1])
        tuples.update(zip(*(c[a:b].tolist() for c in cols)))
    return sorted(tuples)
def ancestry_union(surface,rows):
    off=np.asarray(surface['face_ancestry_offsets'],np.int64)
    cols=[np.asarray(surface['face_ancestry_'+k],np.int64) for k in core.ANCESTRY_LANES]
    tuples=set()
    for r in rows:
        a,b=int(off[int(r)]),int(off[int(r)+1])
        tuples.update(zip(*(c[a:b].tolist() for c in cols)))
    return sorted(tuples)
def scalar_text(a,label):
    x=np.asarray(a)
    if x.dtype.kind not in 'US' or x.shape not in ((),(1,)):
        raise ValueError(label+' must be a single 0-D or 1-D string')
    value=str(x.reshape(-1)[0])
    if not value:raise ValueError(label+' is empty')
    return value
def verify_parent_pin(arrays,path_key,sha_key,parent_ref,label):
    if path_key in arrays and scalar_text(arrays[path_key],label+' parent path')!=parent_ref['path']:
        raise ValueError(label+' immediate parent path mismatch')
    if sha_key in arrays and scalar_text(arrays[sha_key],label+' parent SHA')!=parent_ref['sha256']:
        raise ValueError(label+' immediate parent SHA mismatch')
def verify_surface_step(parent,child,sres,parent_ref,operation_id):
    p=np.asarray(parent['points_world_m_f32'],np.float32);f=np.asarray(parent['faces'],np.int64)
    q=np.asarray(sres['endpoint_quotient_output_to_current_rows'],np.int64)
    expected_q=np.arange(len(p),dtype=np.int64);expected_q[int(sres['drop_row'])]=int(sres['keep_row'])
    if not np.array_equal(q,expected_q):raise ValueError('source quotient map is not exactly one endpoint redirect')
    expected=q[f];expected=expected[~((expected[:,0]==expected[:,1])|(expected[:,1]==expected[:,2])|(expected[:,2]==expected[:,0]))]
    if parent['points_world_m'].tobytes()!=child['points_world_m'].tobytes() or p.tobytes()!=np.asarray(child['points_world_m_f32'],np.float32).tobytes():
        raise ValueError('independent source point-byte replay failed')
    if not np.array_equal(expected,np.asarray(child['faces'],np.int64)):raise ValueError('independent source quotient face replay failed')
    parent_rows=np.asarray(sres['face_output_to_parent_rows'],np.int64)
    for k in core.LANES:
        if len(child['face_source_'+k])!=len(child['faces']):raise ValueError('typed source primary lane length mismatch '+k)
        if not np.array_equal(np.asarray(child['face_source_'+k]),np.asarray(parent['face_source_'+k])[parent_rows]):
            raise ValueError('typed source primary lane replay mismatch '+k)
        if len(child['quality_star_patch_union_face_source_'+k])!=int(child['quality_star_patch_union_offsets'][-1]):raise ValueError('typed union lane mismatch '+k)
    if len(child['quality_star_patch_union_offsets'])!=len(child['faces'])+1 or len(child['face_ancestry_offsets'])!=len(child['faces'])+1:
        raise ValueError('typed CSR output face domain mismatch')
    star=np.asarray(sres['complete_operation_parent_face_rows'],np.int64)
    for k in ('quality_star_face_output_to_input_row','source_input_face_rows'):
        if k in parent and not np.array_equal(np.asarray(child[k]),np.asarray(parent[k])[parent_rows]):
            raise ValueError('immediate face parent row map mismatch '+k)
    if 'source_collapse_operation_rows' in parent:
        expected_ops=np.asarray(parent['source_collapse_operation_rows'])[parent_rows].copy()
        expected_ops[np.asarray(sres['changed_output_rows'],np.int64)]=int(operation_id)
        if not np.array_equal(np.asarray(child['source_collapse_operation_rows']),expected_ops):
            raise ValueError('source collapse operation-row replay mismatch')
    verify_parent_pin(child,'typed_parent_candidate_path','typed_parent_candidate_sha256',parent_ref,'typed source')
    verify_parent_pin(child,'endpoint_quotient_parent_path','endpoint_quotient_parent_sha256',parent_ref,'source quotient')
    u=typed_union(parent,star);a=ancestry_union(parent,star)
    changed=set(map(int,sres['changed_output_rows']))
    uoff=np.asarray(child['quality_star_patch_union_offsets'],np.int64)
    au=np.asarray(child['face_ancestry_offsets'],np.int64)
    for outrow,prow in enumerate(parent_rows):
        got=sorted(zip(*(np.asarray(child['quality_star_patch_union_face_source_'+k],np.int64)[uoff[outrow]:uoff[outrow+1]].tolist() for k in core.LANES)))
        expected_u=u if outrow in changed else typed_union(parent,[int(prow)])
        if got!=expected_u:raise ValueError('typed face parent union mismatch at output row '+str(outrow))
        gota=sorted(zip(*(np.asarray(child['face_ancestry_'+k],np.int64)[au[outrow]:au[outrow+1]].tolist() for k in core.ANCESTRY_LANES)))
        expected_a=a if outrow in changed else ancestry_union(parent,[int(prow)])
        if gota!=expected_a:raise ValueError('typed face ancestry mismatch at output row '+str(outrow))
    return {'point_arrays_byte_identical':True,'source_faces_recomputed_from_parent_quotient':True,
            'typed_primary_lanes_replayed':True,'complete_face_unions_replayed':True,'complete_face_ancestry_replayed':True}
def verify_volume_step(parent,child,vres,rebinder,role,typed_surfaces,parent_ref):
    if parent['points_world_m'].tobytes()!=child['points_world_m'].tobytes():raise ValueError(role+' point array bytes changed')
    if 'points_world_m_f32' in parent and parent['points_world_m_f32'].tobytes()!=child['points_world_m_f32'].tobytes():raise ValueError(role+' F32 point array bytes changed')
    t=np.asarray(parent['tetrahedra'],np.int64);q=np.asarray(vres['input_node_to_output_node_rows'],np.int64)
    expected_q=np.arange(len(parent['points_world_m']),dtype=np.int64)
    expected_q[int(vres['drop_volume_row'])]=int(vres['keep_volume_row'])
    if not np.array_equal(q,expected_q):raise ValueError(role+' node quotient is not exactly one endpoint redirect')
    mapped=q[t];s=np.sort(mapped,axis=1);valid=~np.any(s[:,1:]==s[:,:-1],axis=1)
    if not np.array_equal(mapped[valid],np.asarray(child['tetrahedra'],np.int64)):raise ValueError(role+' tetrahedron quotient replay mismatch')
    if not np.array_equal(np.asarray(parent['material_indices'])[valid],np.asarray(child['material_indices'])):
        raise ValueError(role+' material row replay mismatch')
    if not np.array_equal(np.asarray(child['retained_input_cell_rows'],np.int64),np.flatnonzero(valid)):
        raise ValueError(role+' retained-cell row map mismatch')
    target0=np.asarray(parent['desired_boundary_faces'],np.int64)
    mapped_target=q[target0];ts=np.sort(mapped_target,axis=1)
    target_valid=~np.any(ts[:,1:]==ts[:,:-1],axis=1)
    if not np.array_equal(mapped_target[target_valid],np.asarray(child['desired_boundary_faces'],np.int64)):
        raise ValueError(role+' desired source boundary quotient row replay mismatch')
    actual=core.boundary_faces(np.asarray(child['tetrahedra'],np.int64))
    if exact_chain(actual)!=exact_chain(child['desired_boundary_faces']):raise ValueError(role+' exact output boundary chain mismatch')
    if exact_chain(actual)!=exact_chain(child['boundary_faces']):raise ValueError(role+' stored actual boundary differs from readback')
    target=np.asarray(child['desired_boundary_faces'],np.int64)
    aliases=('desired_boundary_source_group','desired_boundary_source_rows','desired_boundary_source_face_rows',
             'proposed_current_desired_boundary_source_group','proposed_current_desired_boundary_source_rows',
             'proposed_current_desired_boundary_source_face_rows')
    for k in aliases:
        if k not in child or np.asarray(child[k]).shape!=(len(target),):raise ValueError(role+' current target alias row domain mismatch '+k)
    prefix='skin' if role=='Skin' else 'fat'
    if not np.array_equal(child['mapped_current_boundary_faces'],child['boundary_faces']):raise ValueError(role+' mapped actual-boundary alias stale')
    points=np.asarray(child['points_world_m_f32'] if 'points_world_m_f32' in child else child['points_world_m'],np.float32)
    if not np.array_equal(points[np.asarray(child[f'{prefix}_current_target_rebind_wall_faces_volume_rows'],np.int64)],
                          np.asarray(child[f'{prefix}_current_target_rebind_wall_face_points_world_m_f32'],np.float32)):
        raise ValueError(role+' authenticated wall face coordinate bindings changed')
    oldpoints=np.asarray(parent['points_world_m_f32'] if 'points_world_m_f32' in parent else parent['points_world_m'],np.float32)
    olddet,oldq=core.M.measurements(oldpoints,t)
    oldgood=np.asarray(core.M.passes(olddet,oldq),bool)
    det,qnative=core.M.measurements(points,np.asarray(child['tetrahedra'],np.int64))
    if np.any(det<=0) or np.any(det/6<=core.VMIN):raise ValueError(role+' final volume has nonpositive/subminimum tetrahedra')
    newgood=np.asarray(core.M.passes(det,qnative),bool)
    retained=np.flatnonzero(valid)
    if np.any(oldgood[retained]&~newgood):raise ValueError(role+' independent replay found regression of an old-good cell')
    oldmat=np.asarray(parent['material_indices']);newmat=np.asarray(child['material_indices'])
    for m in np.unique(oldmat):
        if np.count_nonzero((~newgood)&(newmat==m))>np.count_nonzero((~oldgood)&(oldmat==m)):
            raise ValueError(role+' independent replay found material native-bad-count regression')
    changed=np.asarray(vres['changed_output_cell_rows'],np.int64)
    affected=np.asarray(vres['affected_input_cell_rows'],np.int64)
    if not np.array_equal(np.asarray(child['replacement_tetrahedra'],np.int64),np.asarray(child['tetrahedra'],np.int64)[changed]):
        raise ValueError(role+' replacement tetrahedra do not match changed rows')
    roff=np.asarray(child['replacement_parent_offsets'],np.int64)
    expected_roff=np.arange(len(changed)+1,dtype=np.int64)*len(affected)
    if not np.array_equal(roff,expected_roff) or not np.array_equal(np.asarray(child['replacement_parent_cell_rows'],np.int64),np.tile(affected,len(changed))):
        raise ValueError(role+' full affected-star parent union replay mismatch')
    for k in ('current_geometry_parent_path','endpoint_quotient_parent_path'):
        verify_parent_pin(child,k,k.replace('_path','_sha256'),parent_ref,role+' volume')
    exact=core._exact_positive(core.M,points,np.asarray(child['tetrahedra'],np.int64)[changed])
    if any(x<=0 for x in exact):raise ValueError(role+' final exact F32 positivity replay failed')
    old_exact=core._exact_positive(core.M,oldpoints,t[affected])
    oldlocal=t[affected];newlocal=mapped[affected];nl=np.sort(newlocal,axis=1)
    newlocal=newlocal[~np.any(nl[:,1:]==nl[:,:-1],axis=1)]
    new_exact=core._exact_positive(core.M,points,newlocal)
    exact_delta=sum(new_exact)-sum(old_exact)
    if str(exact_delta)!=str(vres['exact_signed_volume_delta_det6_integer']):
        raise ValueError(role+' exact affected-star signed-volume delta replay mismatch')
    # Recompute current typed alias arrays independently from the serialized
    # source surfaces and the authenticated wall rows/coordinates.
    wallrows=np.asarray(child[f'{prefix}_current_target_rebind_wall_faces_volume_rows'],np.int64)
    wallxyz=np.asarray(child[f'{prefix}_current_target_rebind_wall_face_points_world_m_f32'],np.float32)
    wallids=np.arange(90,dtype=np.int64)+(0 if role=='Skin' else 90)
    rebound,receipt=rebinder.rebind_source_aliases(role,child,typed_surfaces,wallxyz,wallrows,wallids)
    for k in aliases:
        if not np.array_equal(np.asarray(rebound[k]),np.asarray(child[k])):raise ValueError(role+' independent target source alias reconstruction mismatch '+k)
    return {'point_arrays_byte_identical':True,'tetrahedra_replayed':True,'exact_positive_F32':True,
            'oriented_target_boundary_exact':True,'current_source_aliases_reconstructed':True,
            'wall_rows_exact':True,'full_affected_star_parent_union_replayed':True,
            'bad_cells':int(np.count_nonzero(~newgood)),
            'rebinder_receipt':receipt}
def load_state_inputs(state_path):
    state=json.loads(Path(state_path).read_text())
    if state.get('physical_admission') is True or int(state.get('native_steps',0))!=0:raise ValueError('input state must be construction-only')
    surfaces={};srefs={}
    for r in ('O','I','D'):
        p=pin(state['source_surfaces'][r],r+' source');srefs[r]=ref(p);surfaces[r]=load_npz(p)
    volumes={};vrefs={}
    for r in ('Skin','Fat'):
        p=pin(state[r+'_current'],r+' volume');vrefs[r]=ref(p);volumes[r]=load_npz(p)
    return state,surfaces,srefs,volumes,vrefs

def protected_rows(state_path,state,role,surfaces,volumes,out):
    P=imp('endpoint_quotient_protection_005',PROTECTION,PROTECTION_SHA)
    if sha(P.P.__file__)!=PROTECTION_OWNER_SHA:raise ValueError('795 owner pin mismatch')
    pc={}
    for vr,v in volumes.items():
        pts=np.asarray(v['points_world_m_f32'] if 'points_world_m_f32' in v else v['points_world_m'],np.float32)
        pc[vr]={'points':pts,'coordinate_index':P.point_key_index(pts)}
    prot=P.protected_scope(state_path,state,role,pc,out/'protection')
    src=prot['Fat']['report']['sources'][role]
    fixed=np.unique(np.r_[np.asarray(src['rim_rows'],np.int64),np.asarray(src['route_rows'],np.int64)]).astype(np.int64)
    back=np.asarray(src['back_rows'],np.int64)
    return fixed,back,prot,P

def source_star_vertices(surface,edge):
    f=np.asarray(surface['faces'],np.int64);e=np.asarray(edge,np.int64)
    rows=np.flatnonzero(np.any(np.isin(f,e),axis=1))
    return rows,np.unique(f[rows]).astype(np.int64)
def volume_star_vertices(v,source,edge):
    pts=np.asarray(v['points_world_m_f32'] if 'points_world_m_f32' in v else v['points_world_m'],np.float32)
    sp=np.asarray(source['points_world_m_f32'],np.float32);lut={np.ascontiguousarray(x).view(np.uint32).tobytes():i for i,x in enumerate(pts)}
    node=[]
    for r in edge:
        key=np.ascontiguousarray(sp[int(r)]).view(np.uint32).tobytes()
        if key not in lut:raise ValueError('source endpoint missing from volume')
        node.append(int(lut[key]))
    t=np.asarray(v['tetrahedra'],np.int64);rows=np.flatnonzero(np.any(np.isin(t,node),axis=1))
    return rows,np.unique(t[rows]).astype(np.int64),node

def _simplicial_link(tets,required):
    req=set(map(int,required));out={()}
    for tet in np.asarray(tets,np.int64):
        cells=set(map(int,tet))
        if req.issubset(cells):
            rest=sorted(cells-req)
            for n in range(1,len(rest)+1):
                out.update(tuple(x) for x in combinations(rest,n))
    return out

def volume_edge_link_ok(volume,source,drop,keep):
    vp=np.asarray(volume['points_world_m_f32'] if 'points_world_m_f32' in volume else volume['points_world_m'],np.float32)
    sp=np.asarray(source['points_world_m_f32'],np.float32)
    lookup={np.ascontiguousarray(x).view(np.uint32).tobytes():i for i,x in enumerate(vp)}
    kd=lambda x:np.ascontiguousarray(np.asarray(x,np.float32)).view(np.uint32).tobytes()
    if kd(sp[int(drop)]) not in lookup or kd(sp[int(keep)]) not in lookup: return False
    a,b=lookup[kd(sp[int(drop)])],lookup[kd(sp[int(keep)])]
    if a==b:return False
    t=np.asarray(volume['tetrahedra'],np.int64)
    ma=np.any(t==a,axis=1);mb=np.any(t==b,axis=1)
    la=_simplicial_link(t[ma],(a,));lb=_simplicial_link(t[mb],(b,));lab=_simplicial_link(t[ma&mb],(a,b))
    return la.intersection(lb)==lab

def finalize_surface(current,base,source_node_map,face_map,op_records,base_ref):
    out={k:np.array(v,copy=True) for k,v in current.items()}
    oldfaces=np.asarray(base['faces'],np.int64);q=np.asarray(source_node_map,np.int64)
    mapped=q[oldfaces];valid=~((mapped[:,0]==mapped[:,1])|(mapped[:,1]==mapped[:,2])|(mapped[:,2]==mapped[:,0]))
    if not np.array_equal(mapped[valid],np.asarray(out['faces'],np.int64)):
        raise ValueError('batch source quotient does not reconstruct from base faces')
    fmap=np.asarray(face_map,np.int64)
    if not np.array_equal(fmap,np.flatnonzero(valid)):raise ValueError('batch source output-face map differs from direct base quotient')
    out['endpoint_quotient_output_to_current_parent_rows']=q.copy()
    if 'quality_star_face_output_to_input_row' in out:out['quality_star_face_output_to_input_row']=fmap.copy()
    set_text_ref(out,'typed_parent_candidate_path','typed_parent_candidate_sha256',base_ref)
    set_text_ref(out,'endpoint_quotient_parent_path','endpoint_quotient_parent_sha256',base_ref)
    out['endpoint_quotient_batch_operation_ids']=np.asarray([int(x['operation_id']) for x in op_records],np.int64)
    out['endpoint_quotient_batch_drop_source_rows']=np.asarray([int(x['drop_source_row']) for x in op_records],np.int64)
    out['endpoint_quotient_batch_keep_source_rows']=np.asarray([int(x['keep_source_row']) for x in op_records],np.int64)
    out['endpoint_quotient_batch_output_face_to_input_rows']=fmap.copy()
    out['endpoint_quotient_batch_operation_log_json']=np.asarray(json.dumps(op_records,sort_keys=True,separators=(',',':')),dtype='U')
    return out

def finalize_volume(current,base,node_map,cell_map,cell_unions,op_records,base_ref,role):
    out={k:np.array(v,copy=True) for k,v in current.items()}
    q=np.asarray(node_map,np.int64);cm=np.asarray(cell_map,np.int64)
    if len(cm)!=len(out['tetrahedra']):
        raise ValueError('batch volume output-cell map length mismatch')
    nbase=len(np.asarray(base['tetrahedra']))
    inv=np.full(nbase,-1,np.int64);inv[cm]=np.arange(len(cm),dtype=np.int64)
    for k in ('retained_input_cell_rows','output_cell_to_input_cell','output_cell_to_input_cell_rows'):
        out[k]=cm.copy()
    for k in ('input_cell_to_output_cell','input_cell_to_output_cell_rows'):
        out[k]=inv.copy()
    for k in ('input_node_to_output_node','input_node_to_output_node_rows','input_point_to_output_point_row','input_point_to_output_point_rows'):
        out[k]=q.copy()
    ident=np.arange(len(q),dtype=np.int64)
    for k in ('output_node_to_input_node','output_node_to_input_node_rows','output_point_to_input_point_row','output_point_to_input_point_rows'):
        out[k]=ident.copy()
    out['endpoint_quotient_input_node_map']=q.copy()
    changed=np.asarray(sorted(map(int,cell_unions)),np.int64)
    unions=[sorted(map(int,cell_unions[int(r)])) for r in changed]
    offsets=np.zeros(len(changed)+1,np.int64)
    if len(changed):offsets[1:]=np.cumsum([len(x) for x in unions],dtype=np.int64)
    out['endpoint_quotient_changed_output_cell_rows']=changed.copy()
    out['endpoint_quotient_changed_input_cell_rows']=cm[changed].copy()
    affected=np.asarray(sorted({x for rows in unions for x in rows}),np.int64)
    out['endpoint_quotient_affected_input_cell_rows']=affected
    out['replacement_tetrahedra']=np.asarray(out['tetrahedra'],np.int64)[changed].copy()
    out['replacement_parent_offsets']=offsets
    out['replacement_parent_cell_rows']=np.asarray([x for rows in unions for x in rows],np.int64)
    set_text_ref(out,'current_geometry_parent_path','current_geometry_parent_sha256',base_ref)
    set_text_ref(out,'endpoint_quotient_parent_path','endpoint_quotient_parent_sha256',base_ref)
    out['endpoint_quotient_batch_operation_ids']=np.asarray([int(x['operation_id']) for x in op_records],np.int64)
    out['endpoint_quotient_batch_drop_volume_rows']=np.asarray([int(x['drop_volume_rows'][role]) for x in op_records],np.int64)
    out['endpoint_quotient_batch_keep_volume_rows']=np.asarray([int(x['keep_volume_rows'][role]) for x in op_records],np.int64)
    out['endpoint_quotient_batch_output_cell_to_input_rows']=cm.copy()
    out['endpoint_quotient_batch_input_node_to_output_rows']=q.copy()
    out['endpoint_quotient_batch_operation_log_json']=np.asarray(json.dumps(op_records,sort_keys=True,separators=(',',':')),dtype='U')
    return out

def verify_final_surface(base,child,node_map,face_map,op_records,base_ref):
    q=np.asarray(node_map,np.int64);f=np.asarray(base['faces'],np.int64)
    mapped=q[f];valid=~((mapped[:,0]==mapped[:,1])|(mapped[:,1]==mapped[:,2])|(mapped[:,2]==mapped[:,0]))
    if not np.array_equal(mapped[valid],np.asarray(child['faces'],np.int64)):raise ValueError('final source faces do not replay from batch quotient')
    fm=np.asarray(face_map,np.int64)
    if not np.array_equal(fm,np.flatnonzero(valid)):raise ValueError('final source immediate face map mismatch')
    if base['points_world_m'].tobytes()!=child['points_world_m'].tobytes() or base['points_world_m_f32'].tobytes()!=child['points_world_m_f32'].tobytes():
        raise ValueError('final source point bytes changed')
    if not np.array_equal(np.asarray(child['endpoint_quotient_output_to_current_parent_rows'],np.int64),q):
        raise ValueError('final source quotient map mismatch')
    if 'quality_star_face_output_to_input_row' in child and not np.array_equal(np.asarray(child['quality_star_face_output_to_input_row'],np.int64),fm):
        raise ValueError('final source face-to-input map mismatch')
    for lane in core.LANES:
        if not np.array_equal(np.asarray(child['face_source_'+lane]),np.asarray(base['face_source_'+lane])[fm]):
            raise ValueError('final typed source primary lane mismatch '+lane)
    changed_unions={};changed_ancestry={};changed_ops={}
    for rec in op_records:
        star=np.asarray(rec['star_face_base_rows'],np.int64)
        u=set(map(tuple,typed_union(base,star)));a=set(map(tuple,ancestry_union(base,star)))
        for row in rec['changed_face_base_rows']:
            changed_unions.setdefault(int(row),set()).update(u)
            changed_ancestry.setdefault(int(row),set()).update(a)
            changed_ops[int(row)]=int(rec['operation_id'])
    uoff=np.asarray(child['quality_star_patch_union_offsets'],np.int64);aoff=np.asarray(child['face_ancestry_offsets'],np.int64)
    if len(uoff)!=len(fm)+1 or len(aoff)!=len(fm)+1:raise ValueError('final source typed CSR row domain mismatch')
    ucols={k:np.asarray(child['quality_star_patch_union_face_source_'+k],np.int64) for k in core.LANES}
    acols={k:np.asarray(child['face_ancestry_'+k],np.int64) for k in core.ANCESTRY_LANES}
    base_ops=np.asarray(base.get('source_collapse_operation_rows',np.full(len(f),-1)),np.int64)
    expected_ops=base_ops[fm].copy()
    for outrow,base_row in enumerate(fm):
        if int(base_row) in changed_unions:
            eu=sorted(changed_unions[int(base_row)]);ea=sorted(changed_ancestry[int(base_row)])
            expected_ops[outrow]=changed_ops[int(base_row)]
        else:
            eu=typed_union(base,[int(base_row)]);ea=ancestry_union(base,[int(base_row)])
        gu=sorted(zip(*(ucols[k][uoff[outrow]:uoff[outrow+1]].tolist() for k in core.LANES)))
        ga=sorted(zip(*(acols[k][aoff[outrow]:aoff[outrow+1]].tolist() for k in core.ANCESTRY_LANES)))
        if gu!=eu or ga!=ea:raise ValueError('final typed source face union/ancestry mismatch at row '+str(outrow))
    if 'source_collapse_operation_rows' in child and not np.array_equal(np.asarray(child['source_collapse_operation_rows'],np.int64),expected_ops):
        raise ValueError('final source operation-row lineage mismatch')
    verify_parent_pin(child,'typed_parent_candidate_path','typed_parent_candidate_sha256',base_ref,'typed source')
    verify_parent_pin(child,'endpoint_quotient_parent_path','endpoint_quotient_parent_sha256',base_ref,'source quotient')
    q0=core._face_qbound(np.asarray(base['points_world_m_f32'],np.float32),f)
    q1=core._face_qbound(np.asarray(child['points_world_m_f32'],np.float32),np.asarray(child['faces'],np.int64))
    if np.count_nonzero(q1<core.QMIN)>np.count_nonzero(q0<core.QMIN):raise ValueError('final source facet quality-count regression')
    return {'source_face_quotient_replayed':True,'source_point_bytes_identical':True,'typed_primary_lanes_replayed':True,
            'full_typed_union_and_ancestry_replayed':True,'source_operation_rows_replayed':True,
            'facet_bad_before':int(np.count_nonzero(q0<core.QMIN)),'facet_bad_after':int(np.count_nonzero(q1<core.QMIN))}

def verify_final_volume(base,child,node_map,cell_map,cell_unions,op_records,rebinder,role,typed_surfaces,base_ref):
    q=np.asarray(node_map,np.int64);old_t=np.asarray(base['tetrahedra'],np.int64);old_m=np.asarray(base['material_indices'])
    mapped=q[old_t];s=np.sort(mapped,axis=1);valid=~np.any(s[:,1:]==s[:,:-1],axis=1)
    if not np.array_equal(mapped[valid],np.asarray(child['tetrahedra'],np.int64)):raise ValueError(role+' final tets do not replay from batch quotient')
    cm=np.asarray(cell_map,np.int64)
    if not np.array_equal(cm,np.flatnonzero(valid)) or not np.array_equal(np.asarray(child['retained_input_cell_rows'],np.int64),cm):
        raise ValueError(role+' final output-to-input cell map mismatch')
    if not np.array_equal(old_m[valid],np.asarray(child['material_indices'])):raise ValueError(role+' final material rows mismatch')
    target=q[np.asarray(base['desired_boundary_faces'],np.int64)];ts=np.sort(target,axis=1)
    tv=~np.any(ts[:,1:]==ts[:,:-1],axis=1)
    if not np.array_equal(target[tv],np.asarray(child['desired_boundary_faces'],np.int64)):
        raise ValueError(role+' final desired boundary does not replay from batch quotient')
    actual=core.boundary_faces(np.asarray(child['tetrahedra'],np.int64))
    if exact_chain(actual)!=exact_chain(child['desired_boundary_faces']) or exact_chain(actual)!=exact_chain(child['boundary_faces']):
        raise ValueError(role+' final exact oriented target boundary mismatch')
    inv=np.full(len(old_t),-1,np.int64);inv[cm]=np.arange(len(cm),dtype=np.int64)
    if not np.array_equal(np.asarray(child['input_cell_to_output_cell_rows'],np.int64),inv):raise ValueError(role+' final inverse cell map mismatch')
    changed=np.asarray(sorted(map(int,cell_unions)),np.int64)
    unions=[sorted(map(int,cell_unions[int(r)])) for r in changed]
    off=np.zeros(len(changed)+1,np.int64)
    if len(changed):off[1:]=np.cumsum([len(x) for x in unions],dtype=np.int64)
    flat=np.asarray([x for a in unions for x in a],np.int64)
    if not np.array_equal(np.asarray(child['endpoint_quotient_changed_output_cell_rows'],np.int64),changed):raise ValueError(role+' final changed-cell rows mismatch')
    if not np.array_equal(np.asarray(child['endpoint_quotient_changed_input_cell_rows'],np.int64),cm[changed]):raise ValueError(role+' final changed input-cell rows mismatch')
    if not np.array_equal(np.asarray(child['replacement_tetrahedra'],np.int64),np.asarray(child['tetrahedra'],np.int64)[changed]):raise ValueError(role+' final replacement tet rows mismatch')
    if not np.array_equal(np.asarray(child['replacement_parent_offsets'],np.int64),off) or not np.array_equal(np.asarray(child['replacement_parent_cell_rows'],np.int64),flat):
        raise ValueError(role+' final replacement parent union mismatch')
    affected=np.asarray(sorted({x for a in unions for x in a}),np.int64)
    if not np.array_equal(np.asarray(child['endpoint_quotient_affected_input_cell_rows'],np.int64),affected):raise ValueError(role+' final affected-parent rows mismatch')
    verify_parent_pin(child,'current_geometry_parent_path','current_geometry_parent_sha256',base_ref,role+' volume')
    verify_parent_pin(child,'endpoint_quotient_parent_path','endpoint_quotient_parent_sha256',base_ref,role+' volume quotient')
    p=np.asarray(child['points_world_m_f32'] if 'points_world_m_f32' in child else child['points_world_m'],np.float32)
    oldp=np.asarray(base['points_world_m_f32'] if 'points_world_m_f32' in base else base['points_world_m'],np.float32)
    if base['points_world_m'].tobytes()!=child['points_world_m'].tobytes():raise ValueError(role+' point arrays changed')
    det0,q0=core.M.measurements(oldp,old_t);det1,q1=core.M.measurements(p,np.asarray(child['tetrahedra'],np.int64))
    if np.any(det1<=0) or np.any(det1/6<=core.VMIN):raise ValueError(role+' final volume has nonpositive/subminimum cells')
    good0=np.asarray(core.M.passes(det0,q0),bool);good1=np.asarray(core.M.passes(det1,q1),bool)
    if np.any(good0[cm]&~good1):raise ValueError(role+' final volume regressed a previously good cell')
    for m in np.unique(old_m):
        if np.count_nonzero((~good1)&(np.asarray(child['material_indices'])==m))>np.count_nonzero((~good0)&(old_m==m)):
            raise ValueError(role+' final native bad count regressed for a material')
    exact=core._exact_positive(core.M,p,np.asarray(child['tetrahedra'],np.int64)[changed])
    if any(x<=0 for x in exact):raise ValueError(role+' final exact F32 positivity failed')
    local_deltas=[]
    for rec in op_records:
        rows=np.asarray(rec['affected_base_cells'][role],np.int64)
        oldlocal=old_t[rows];newlocal=q[oldlocal];nl=np.sort(newlocal,axis=1)
        nv=~np.any(nl[:,1:]==nl[:,:-1],axis=1);newlocal=newlocal[nv]
        d0=core._exact_positive(core.M,oldp,oldlocal);d1=core._exact_positive(core.M,p,newlocal)
        delta=sum(d1)-sum(d0)
        expected=str(rec['volume_delta_det6_integer'][role])
        if str(delta)!=expected:raise ValueError(role+' final exact local volume delta replay mismatch')
        local_deltas.append(str(delta))
    prefix='skin' if role=='Skin' else 'fat'
    wallrows=np.asarray(child[f'{prefix}_current_target_rebind_wall_faces_volume_rows'],np.int64)
    wallxyz=np.asarray(child[f'{prefix}_current_target_rebind_wall_face_points_world_m_f32'],np.float32)
    if not np.array_equal(p[wallrows],wallxyz):raise ValueError(role+' wall point bindings changed')
    wallids=np.arange(90,dtype=np.int64)+(0 if role=='Skin' else 90)
    rebound,receipt=rebinder.rebind_source_aliases(role,child,typed_surfaces,wallxyz,wallrows,wallids)
    alias_keys=('desired_boundary_source_group','desired_boundary_source_rows','desired_boundary_source_face_rows',
        'proposed_current_desired_boundary_source_group','proposed_current_desired_boundary_source_rows','proposed_current_desired_boundary_source_face_rows')
    if any(not np.array_equal(rebound[k],child[k]) for k in alias_keys):raise ValueError(role+' final source aliases do not independently reconstruct')
    return {'final_target_chain_exact':True,'final_tets_replayed_from_parent':True,'full_native_gate_replayed':True,
        'all_exact_positive_changed_cells':True,'full_replacement_parent_union_replayed':True,'source_aliases_reconstructed':True,
        'native_bad_before':int(np.count_nonzero(~good0)),'native_bad_after':int(np.count_nonzero(~good1)),
        'log_deficit_before':core._log_deficit(q0),'log_deficit_after':core._log_deficit(q1),
        'exact_local_volume_deltas_det6':local_deltas,'rebinder_receipt':receipt}

def _csr_incidence(rows,vertices,nvertices):
    rows=np.asarray(rows,np.int64).reshape(-1);vertices=np.asarray(vertices,np.int64).reshape(-1)
    order=np.lexsort((rows,vertices));v=vertices[order];r=rows[order]
    counts=np.bincount(v,minlength=int(nvertices));off=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    return off,r

def _csr_rows(offsets,rows,vertex):
    i=int(vertex);return np.asarray(rows[int(offsets[i]):int(offsets[i+1])],np.int64)

def _simplex_link(tets,required):
    req=set(map(int,required));out={()}
    for tet in np.asarray(tets,np.int64):
        cells=set(map(int,tet))
        if req.issubset(cells):
            rest=sorted(cells-req)
            for n in range(1,len(rest)+1):out.update(tuple(x) for x in combinations(rest,n))
    return out

class LocalQualityCache:
    """Immutable input-mesh incidence/quality cache for disjoint edge stars."""
    def __init__(self,owner,role,surface,volumes,targets):
        self.owner=owner;self.role=role;self.surface=surface;self.targets=tuple(targets)
        self.sp=np.asarray(surface['points_world_m_f32'],np.float32);self.sf=np.asarray(surface['faces'],np.int64)
        if self.sf.ndim!=2 or self.sf.shape[1]!=3:raise ValueError('malformed source faces')
        self.edge_faces=owner._edge_faces(self.sf);self.neighbors=owner._vertex_neighbors(self.sf)
        sr=np.repeat(np.arange(len(self.sf),dtype=np.int64),3)
        sv=self.sf.reshape(-1);self.soff,self.srows=_csr_incidence(sr,sv,len(self.sp))
        self.sboundary=owner._surface_boundary_chain(self.sf)
        self.sface_q=owner._face_qbound(self.sp,self.sf)
        self.source_maps={};self.volumes={}
        for vr in self.targets:
            v=volumes[vr];p=np.asarray(v['points_world_m_f32'] if 'points_world_m_f32' in v else v['points_world_m'],np.float32)
            t=np.asarray(v['tetrahedra'],np.int64);mat=np.asarray(v['material_indices'])
            if len(np.unique(p.view(np.dtype((np.void,p.dtype.itemsize*p.shape[1]))).ravel()))!=len(p):
                raise ValueError(vr+' input F32 points are ambiguous')
            st=np.sort(t,axis=1)
            if len(np.unique(st,axis=0))!=len(t):raise ValueError(vr+' input contains duplicate tetrahedra')
            actual=owner.boundary_faces(t)
            target=np.asarray(v.get('desired_boundary_faces',v.get('boundary_faces')),np.int64)
            if owner.face_chain(actual)!=owner.face_chain(target):raise ValueError(vr+' input exact target boundary mismatch')
            det,q=owner.M.measurements(p,t);good=np.asarray(owner.M.passes(det,q),bool)
            if np.any(det/6<=owner.VMIN):raise ValueError(vr+' input has nonpositive/subminimum cells')
            cr=np.repeat(np.arange(len(t),dtype=np.int64),4);cv=t.reshape(-1)
            coff,crows=_csr_incidence(cr,cv,len(p))
            fr=np.repeat(np.arange(len(target),dtype=np.int64),3);fv=target.reshape(-1)
            toff,trows=_csr_incidence(fr,fv,len(p))
            # Coordinate-based mapping is exact F32 and fails closed on ambiguous rows.
            lookup={};ambiguous=set()
            for i,x in enumerate(p):
                k=core.f32key(x)
                if k in lookup:ambiguous.add(k)
                else:lookup[k]=int(i)
            smap=np.full(len(self.sp),-1,np.int64)
            for i in np.unique(self.sf.reshape(-1)):
                k=core.f32key(self.sp[int(i)])
                if k in lookup and k not in ambiguous:smap[int(i)]=lookup[k]
            self.source_maps[vr]=smap
            permat={int(m):int(np.count_nonzero((~good)&(mat==m))) for m in np.unique(mat)}
            self.volumes[vr]={'points':p,'tets':t,'materials':mat,'target':target,'det':det,'q':q,'good':good,
                'cell_offsets':coff,'cell_rows':crows,'target_offsets':toff,'target_rows':trows,
                'bad':int(np.count_nonzero(~good)),'per_material_bad':permat,'score':owner._log_deficit(q)}
        self.current={vr:{'bad':d['bad'],'per_material_bad':dict(d['per_material_bad']),'score':d['score']} for vr,d in self.volumes.items()}
        self.initial={vr:{'bad':int(d['bad']),'per_material_bad':dict(d['per_material_bad']),'score':float(d['score'])} for vr,d in self.current.items()}

    def _surface_star(self,drop,keep):
        return np.unique(np.r_[_csr_rows(self.soff,self.srows,drop),_csr_rows(self.soff,self.srows,keep)]).astype(np.int64)

    def _surface_trial(self,edge,drop,keep):
        p=self.sp;f=self.sf;u,v=map(int,edge);d,k=int(drop),int(keep)
        inc=self.edge_faces.get(tuple(sorted((u,v))),[])
        if len(inc)!=2:raise ValueError('source edge is not incident to exactly two faces')
        common=(self.neighbors.get(d,set())&self.neighbors.get(k,set()))-{d,k}
        opposite={int(x) for r in inc for x in f[int(r)] if int(x) not in (d,k)}
        if common!=opposite:raise ValueError('source surface simplicial link condition fails')
        star=self._surface_star(d,k);old=f[star]
        mapped=old.copy();mapped[mapped==d]=k
        deg=(mapped[:,0]==mapped[:,1])|(mapped[:,1]==mapped[:,2])|(mapped[:,2]==mapped[:,0])
        new=mapped[~deg]
        if len(np.unique(np.sort(new,axis=1),axis=0))!=len(new):raise ValueError('local source quotient creates duplicate faces')
        changed=(old[~deg]==d).any(axis=1)
        if np.any(changed):
            a=old[~deg][changed];b=new[changed]
            xyz=p.astype(np.float64)
            n0=np.cross(xyz[a[:,1]]-xyz[a[:,0]],xyz[a[:,2]]-xyz[a[:,0]])
            n1=np.cross(xyz[b[:,1]]-xyz[b[:,0]],xyz[b[:,2]]-xyz[b[:,0]])
            if np.any(np.einsum('ij,ij->i',n0,n1)<=0):raise ValueError('local source face winding/area changed')
        q0=self.sface_q[star];q1=self.owner._face_qbound(p,new)
        if np.count_nonzero(q1<self.owner.QMIN)>np.count_nonzero(q0<self.owner.QMIN):
            raise ValueError('local source native facet failure count regressed')
        # Exact 1D boundary quotient on the complete endpoint star.
        expected=Counter()
        for (a,b),sgn in self.sboundary.items():
            if d not in (a,b) and k not in (a,b):continue
            x=k if a==d else a;y=k if b==d else b
            if x!=y:
                e=tuple(sorted((x,y)));expected[e]+=sgn*(1 if x<y else -1)
        expected={e:n for e,n in expected.items() if n}
        local=self.owner._surface_boundary_chain(new)
        local={e:n for e,n in local.items() if k in e}
        expected={e:n for e,n in expected.items() if k in e}
        if local!=expected:raise ValueError('local source boundary chain quotient mismatch')
        changed_base=star[np.flatnonzero(~deg)[np.flatnonzero(changed)]] if False else star[np.flatnonzero((old==d).any(axis=1)&~deg)]
        return {'star_rows':star,'changed_parent_rows':changed_base,'retired_parent_rows':star[np.flatnonzero(deg)],
            'facet_bad_before':int(np.count_nonzero(q0<self.owner.QMIN)),
            'facet_bad_after':int(np.count_nonzero(q1<self.owner.QMIN)),
            'minimum_facet_q_before':float(q0.min()),'minimum_facet_q_after':float(q1.min())}

    def _volume_trial(self,vr,drop_source,keep_source):
        drow=int(drop_source);krow=int(keep_source);smap=self.source_maps[vr]
        dv=int(smap[drow]);kv=int(smap[krow])
        if dv<0 or kv<0:raise ValueError(vr+' source endpoint has no unique exact volume row')
        d=self.volumes[vr];t=d['tets'];p=d['points'];mat=d['materials']
        drop_cells=_csr_rows(d['cell_offsets'],d['cell_rows'],dv)
        keep_cells=_csr_rows(d['cell_offsets'],d['cell_rows'],kv)
        if len(drop_cells)==0 or len(keep_cells)==0:
            raise ValueError(vr+' endpoint quotient has an orphan volume endpoint')
        star=np.union1d(drop_cells,keep_cells).astype(np.int64)
        old=t[star];mapped=old.copy();mapped[mapped==dv]=kv
        ss=np.sort(mapped,axis=1);valid=~np.any(ss[:,1:]==ss[:,:-1],axis=1);new=mapped[valid]
        sorted_new=np.sort(new,axis=1)
        if len(np.unique(sorted_new,axis=0))!=len(new):raise ValueError(vr+' local volume quotient creates duplicate tetrahedra')
        if not len(new):raise ValueError(vr+' local volume quotient removes the complete endpoint star')
        # Three-dimensional simplicial link condition (empty simplex included).
        la=_simplex_link(old,(dv,));lb=_simplex_link(old,(kv,));lab=_simplex_link(old,(dv,kv))
        if la.intersection(lb)!=lab:raise ValueError(vr+' tetrahedral simplicial link condition fails')
        detnew,qnew=self.owner.M.measurements(p,new)
        if np.any(~np.isfinite(detnew)) or np.any(detnew/6<=self.owner.VMIN):
            raise ValueError(vr+' local quotient creates nonpositive/subminimum tetrahedra')
        # The only changed survivors are dropped-node incident cells. Keep-only
        # cells are included in boundary/duplicate checks but cancel in metrics.
        changed_mask=np.any(old==dv,axis=1)&valid
        changed_in_new=changed_mask[valid]
        new_changed=mapped[changed_mask];detc,qc=detnew[changed_in_new],qnew[changed_in_new]
        oldgood=d['good'][drop_cells];oldmat=mat[drop_cells]
        oldq=d['q'][drop_cells]
        # Match changed survivors in parent row order for old-good preservation.
        drop_positions=np.searchsorted(star,drop_cells)
        valid_drop=valid[drop_positions]
        oldgood_survive=oldgood[valid_drop]
        newgood=np.asarray(self.owner.M.passes(detc,qc),bool)
        if np.any(oldgood_survive&~newgood):raise ValueError(vr+' local quotient regresses a previously good tetrahedron')
        oldbad_by={int(m):int(np.count_nonzero((~oldgood)&(oldmat==m))) for m in np.unique(mat)}
        newmat=mat[drop_cells[valid_drop]]
        newbad_by={int(m):int(np.count_nonzero((~newgood)&(newmat==m))) for m in np.unique(mat)}
        if any(newbad_by.get(m,0)>oldbad_by.get(m,0) for m in oldbad_by):
            raise ValueError(vr+' local quotient increases native bad cells for a material')
        before_bad=int(np.count_nonzero(~oldgood));after_bad=int(np.count_nonzero(~newgood))
        before_score=self.owner._log_deficit(oldq);after_score=self.owner._log_deficit(qc)
        # Local exact oriented boundary quotient: faces containing keep are
        # wholly covered by this vertex-star union; outside faces are invariant.
        local_boundary=self.owner.boundary_faces(new)
        actual_keep=local_boundary[np.any(local_boundary==kv,axis=1)]
        tr=np.union1d(_csr_rows(d['target_offsets'],d['target_rows'],dv),_csr_rows(d['target_offsets'],d['target_rows'],kv)).astype(np.int64)
        target=d['target'][tr];target=target.copy();target[target==dv]=kv
        ts=np.sort(target,axis=1);target=target[~np.any(ts[:,1:]==ts[:,:-1],axis=1)]
        target_keep=target[np.any(target==kv,axis=1)]
        if self.owner.face_chain(actual_keep)!=self.owner.face_chain(target_keep):
            raise ValueError(vr+' local exact target boundary chain quotient mismatch')
        # Exact F32 positivity only needs recomputation on changed surviving tets;
        # all other cells were positive in the pinned input or previous disjoint star.
        exact=self.owner._exact_positive(self.owner.M,p,new_changed)
        if any(x<=0 for x in exact):raise ValueError(vr+' local exact F32 positivity failed')
        oldexact=self.owner._exact_positive(self.owner.M,p,t[drop_cells])
        oldlocal=t[drop_cells].copy();oldlocal[oldlocal==dv]=kv
        os=np.sort(oldlocal,axis=1);oldlocal=oldlocal[~np.any(os[:,1:]==os[:,:-1],axis=1)]
        newexact=self.owner._exact_positive(self.owner.M,p,oldlocal)
        delta=sum(newexact)-sum(oldexact)
        return {'drop_volume_row':dv,'keep_volume_row':kv,'affected_input_cell_rows':drop_cells,
            'star_cell_rows':star,'native_bad_before':before_bad,'native_bad_after':after_bad,
            'native_deficit_before':before_score,'native_deficit_after':after_score,
            'native_bad_by_material_before':oldbad_by,'native_bad_by_material_after':newbad_by,
            'minimum_normalized_det_after':float(qnew.min()),'minimum_volume_m3_after':float(detnew.min()/6),
            'changed_cell_count':int(np.count_nonzero(changed_mask)),
            'changed_input_cell_rows':drop_cells[valid_drop].copy(),
            'exact_signed_volume_delta_det6_integer':str(delta)}

    def evaluate(self,edge,drop,keep,immutable_rows,max_edge_m,*,allow_local_tie=False):
        d,k=int(drop),int(keep);u,v=map(int,edge)
        if d in set(map(int,immutable_rows)):raise ValueError('retired endpoint is an immutable rim/route')
        length=float(np.linalg.norm(self.sp[d].astype(np.float64)-self.sp[k].astype(np.float64)))
        if length<=0 or length>max_edge_m:raise ValueError('source endpoint displacement exceeds cap')
        sres=self._surface_trial(edge,d,k);vres={vr:self._volume_trial(vr,d,k) for vr in self.targets}
        for vr,r in vres.items():
            local_before=int(r['native_bad_before']);local_after=int(r['native_bad_after'])
            local_score_before=float(r['native_deficit_before']);local_score_after=float(r['native_deficit_after'])
            if local_after>local_before:raise ValueError(vr+' local native bad count regressed')
            if local_after==local_before:
                if allow_local_tie:
                    if local_score_after>local_score_before+1e-8:
                        raise ValueError(vr+' paired local tie worsens log deficit')
                elif not local_score_after<local_score_before-1e-8:
                    raise ValueError(vr+' unpaired local quality has no strict progress')
        before=sum(int(self.current[vr]['bad']) for vr in self.targets)
        after=before+sum(int(x['native_bad_after'])-int(x['native_bad_before']) for x in vres.values())
        bscore=sum(float(self.current[vr]['score']) for vr in self.targets)
        ascore=bscore+sum(float(x['native_deficit_after'])-float(x['native_deficit_before']) for x in vres.values())
        if not (after<before or (after==before and ascore<bscore-1e-8)):
            raise ValueError('paired local quotient has no strict combined native progress')
        return {'role':self.role,'edge_rows':(u,v),'drop_source_row':d,'keep_source_row':k,'edge_length_m':length,
            'surface':sres,'volumes':vres,'combined_native_bad_before':before,'combined_native_bad_after':after,
            'combined_log_deficit_before':bscore,'combined_log_deficit_after':ascore,'paired':self.role=='I'}

    def commit(self,result):
        for vr,r in result['volumes'].items():
            state=self.current[vr]
            state['bad']+=int(r['native_bad_after'])-int(r['native_bad_before'])
            state['score']+=float(r['native_deficit_after'])-float(r['native_deficit_before'])
            for m in state['per_material_bad']:
                state['per_material_bad'][m]+=int(r['native_bad_by_material_after'].get(m,0))-int(r['native_bad_by_material_before'].get(m,0))

def build_source_output(base,node_map,op_records,base_ref):
    out={k:np.array(v,copy=True) for k,v in base.items()};f=np.asarray(base['faces'],np.int64);q=np.asarray(node_map,np.int64)
    mapped=q[f];valid=~((mapped[:,0]==mapped[:,1])|(mapped[:,1]==mapped[:,2])|(mapped[:,2]==mapped[:,0]))
    fmap=np.flatnonzero(valid).astype(np.int64);out['faces']=mapped[valid].copy()
    for key,value in base.items():
        a=np.asarray(value)
        if key not in ('faces','quality_star_patch_union_offsets','face_ancestry_offsets') and a.ndim>=1 and a.shape[0]==len(f):out[key]=a[fmap].copy()
    out['endpoint_quotient_output_to_current_parent_rows']=q.copy()
    if 'quality_star_face_output_to_input_row' in out:out['quality_star_face_output_to_input_row']=fmap.copy()
    if 'source_input_vertex_quotient' in base:
        vq=np.asarray(base['source_input_vertex_quotient'],np.int64).copy()
        for op in op_records:vq[int(op['drop_source_row'])]=vq[int(op['keep_source_row'])]
        out['source_input_vertex_quotient']=vq
    changed_union={};changed_anc={};changed_op={}
    for op in op_records:
        tu=set(map(tuple,b6.typed_union(base,np.asarray(op['star_face_base_rows'],np.int64))))
        an=set(map(tuple,b6.ancestry_union(base,np.asarray(op['star_face_base_rows'],np.int64))))
        for r in op['changed_face_base_rows']:
            changed_union.setdefault(int(r),set()).update(tu);changed_anc.setdefault(int(r),set()).update(an)
            changed_op[int(r)]=int(op['operation_id'])
    uoff=[0];ucols={k:[] for k in core.LANES};aoff=[0];acols={k:[] for k in core.ANCESTRY_LANES}
    for outrow,base_row in enumerate(fmap):
        if int(base_row) in changed_union:
            u=sorted(changed_union[int(base_row)]);a=sorted(changed_anc[int(base_row)])
        else:
            u=b6.typed_union(base,[int(base_row)]);a=b6.ancestry_union(base,[int(base_row)])
        for j,k in enumerate(core.LANES):ucols[k].extend(int(x[j]) for x in u)
        for j,k in enumerate(core.ANCESTRY_LANES):acols[k].extend(int(x[j]) for x in a)
        uoff.append(len(ucols[core.LANES[0]]));aoff.append(len(acols[core.ANCESTRY_LANES[0]]))
    out['quality_star_patch_union_offsets']=np.asarray(uoff,np.int64)
    for k in core.LANES:out['quality_star_patch_union_face_source_'+k]=np.asarray(ucols[k],np.int64)
    out['face_ancestry_offsets']=np.asarray(aoff,np.int64)
    for k in core.ANCESTRY_LANES:out['face_ancestry_'+k]=np.asarray(acols[k],np.int64)
    if 'source_collapse_operation_rows' in out:
        ops=np.asarray(out['source_collapse_operation_rows'],np.int64).copy()
        for row,opid in changed_op.items():ops[np.flatnonzero(fmap==row)]=opid
        out['source_collapse_operation_rows']=ops
    out['endpoint_quotient_output_to_current_parent_rows']=q.copy()
    set_text_ref(out,'typed_parent_candidate_path','typed_parent_candidate_sha256',base_ref)
    set_text_ref(out,'endpoint_quotient_parent_path','endpoint_quotient_parent_sha256',base_ref)
    out['endpoint_quotient_batch_operation_ids']=np.asarray([int(x['operation_id']) for x in op_records],np.int64)
    out['endpoint_quotient_batch_drop_source_rows']=np.asarray([int(x['drop_source_row']) for x in op_records],np.int64)
    out['endpoint_quotient_batch_keep_source_rows']=np.asarray([int(x['keep_source_row']) for x in op_records],np.int64)
    out['endpoint_quotient_batch_output_face_to_input_rows']=fmap.copy()
    out['endpoint_quotient_batch_operation_log_json']=np.asarray(json.dumps(op_records,sort_keys=True,separators=(',',':')),dtype='U')
    out['endpoint_quotient_retired_parent_face_rows']=np.flatnonzero(~valid).astype(np.int64)
    out['endpoint_quotient_changed_output_face_rows']=np.asarray([i for i,r in enumerate(fmap) if int(r) in changed_op],np.int64)
    if op_records:
        last=op_records[-1]
        out['endpoint_quotient_operation_id']=np.asarray([int(last['operation_id'])],np.int64)
        out['endpoint_quotient_drop_source_row']=np.asarray([int(last['drop_source_row'])],np.int64)
        out['endpoint_quotient_keep_source_row']=np.asarray([int(last['keep_source_row'])],np.int64)
    return out,fmap

def build_volume_output(base,node_map,cell_map,cell_unions,op_records,base_ref,role):
    t=np.asarray(base['tetrahedra'],np.int64);ncell=len(t);q=np.asarray(node_map,np.int64)
    mapped=q[t];ss=np.sort(mapped,axis=1);valid=~np.any(ss[:,1:]==ss[:,:-1],axis=1)
    cm=np.flatnonzero(valid).astype(np.int64)
    if not np.array_equal(cm,np.asarray(cell_map,np.int64)):raise ValueError(role+' final cell map disagrees with composed node quotient')
    out={k:np.array(v,copy=True) for k,v in base.items()}
    special={'tetrahedra','boundary_faces','desired_boundary_faces','material_indices','retained_input_cell_rows',
        'output_cell_to_input_cell','output_cell_to_input_cell_rows','input_cell_to_output_cell','input_cell_to_output_cell_rows',
        'replacement_tetrahedra','replacement_parent_offsets','replacement_parent_cell_rows'}
    for key,value in base.items():
        a=np.asarray(value)
        if key not in special and a.ndim>=1 and a.shape[0]==ncell:out[key]=a[cm].copy()
    out['tetrahedra']=mapped[valid].copy();out['material_indices']=np.asarray(base['material_indices'])[cm].copy()
    target0=np.asarray(base['desired_boundary_faces'],np.int64);mt=q[target0];ts=np.sort(mt,axis=1)
    tv=~np.any(ts[:,1:]==ts[:,:-1],axis=1);out['desired_boundary_faces']=mt[tv].copy()
    for key,value in base.items():
        a=np.asarray(value)
        if key not in ('desired_boundary_faces','boundary_faces','desired_boundary_degenerate_source_rows') and a.ndim>=1 and a.shape[0]==len(target0):
            out[key]=a[tv].copy()
    out['boundary_faces']=core.boundary_faces(out['tetrahedra'])
    out['mapped_current_boundary_faces']=out['boundary_faces'].copy()
    out['proposed_current_desired_boundary_faces']=out['desired_boundary_faces'].copy()
    if 'source_interior_vertex_quotient' in base and np.asarray(base['source_interior_vertex_quotient']).shape==(len(q),):
        sq=np.asarray(base['source_interior_vertex_quotient'],np.int64).copy()
        for op in op_records:sq[int(op['drop_volume_rows'][role])]=sq[int(op['keep_volume_rows'][role])]
        out['source_interior_vertex_quotient']=sq
    out=b6.finalize_volume(out,base,q,cm,cell_unions,op_records,base_ref,role)
    if op_records:
        last=op_records[-1]
        out['endpoint_quotient_operation_id']=np.asarray([int(last['operation_id'])],np.int64)
        out['endpoint_quotient_drop_volume_row']=np.asarray([int(last['drop_volume_rows'][role])],np.int64)
        out['endpoint_quotient_keep_volume_row']=np.asarray([int(last['keep_volume_rows'][role])],np.int64)
    return out

def main():
    ap=argparse.ArgumentParser(description='Cached local-star endpoint quotient batch with one final global replay.')
    ap.add_argument('--state',type=Path,required=True);ap.add_argument('--roster',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--role',choices=('I','D'),required=True);ap.add_argument('--max-edge-m',type=float,default=MAX_EDGE_M)
    ap.add_argument('--max-ops',type=int,default=64);ap.add_argument('--max-trials',type=int,default=300);ap.add_argument('--seconds',type=float,default=90.0)
    args=ap.parse_args();tic=time.monotonic();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    global core,b6
    core=imp('endpoint_quotient_core_007',CORE,CORE_SHA)
    b6=imp('endpoint_batch_flattened_006',HERE/'batch_inner_endpoint_quotient_006.py','595ee0252bf80451f9024ea7ab54e758e14182c02a67beb6626400b60f183c2d')
    b6.core=core
    rebinder=imp('endpoint_source_alias_rebinder_007',REBINDER,REBINDER_SHA)
    state_path=args.state.resolve();roster_path=args.roster.resolve();state,surfaces,srefs,volumes,vrefs=load_state_inputs(state_path)
    roster=json.loads(roster_path.read_text())
    if roster.get('status')!='PASS_READ_ONLY_ACTIVE_ROW_BAD_CELL_CENSUS' or roster.get('role')!=args.role:raise ValueError('roster role/status mismatch')
    if roster.get('input_state',{}).get('sha256')!=sha(state_path):raise ValueError('roster does not bind exact input state')
    role=args.role;targets=('Skin','Fat') if role=='I' else ('Fat',)
    fixed,back,protection,P=protected_rows(state_path,state,role,surfaces,volumes,out)
    cache_t0=time.monotonic();cache=LocalQualityCache(core,role,surfaces[role],volumes,targets);cache_setup_s=time.monotonic()-cache_t0
    source_node_map=np.arange(len(surfaces[role]['points_world_m_f32']),dtype=np.int64)
    node_maps={vr:np.arange(len(volumes[vr]['points_world_m']),dtype=np.int64) for vr in targets}
    used_source_star=set();used_volume_star={vr:set() for vr in targets}
    actions=[];op_records=[];rejected=[];rejections=Counter();trial_count=0;local_eval_total_s=0.0
    for cand in roster.get('eligible_bad_cell_hit_edges',[]):
        if time.monotonic()-tic>=args.seconds or trial_count>=args.max_trials or len(actions)>=args.max_ops:break
        edge=tuple(map(int,cand['edge_rows']));length=float(cand['length_m'])
        if length>args.max_edge_m:rejections['over_edge_cap']+=1;continue
        try:
            frows=cache._surface_star(edge[0],edge[1]);sverts=np.unique(cache.sf[frows]).astype(np.int64)
        except Exception as exc:
            rejections['source_edge_not_current']+=1
            if len(rejected)<512:rejected.append({'edge_rows':list(edge),'reason':str(exc)[:200]})
            continue
        if set(map(int,sverts))&used_source_star:rejections['overlapping_source_star']+=1;continue
        vstars={};overlap=False
        try:
            for vr in targets:
                smap=cache.source_maps[vr];dv=int(smap[edge[0]]);kv=int(smap[edge[1]])
                if dv<0 or kv<0:raise ValueError(vr+' endpoint lacks exact volume mapping')
                vc=cache.volumes[vr];rows=np.union1d(_csr_rows(vc['cell_offsets'],vc['cell_rows'],dv),
                    _csr_rows(vc['cell_offsets'],vc['cell_rows'],kv)).astype(np.int64)
                verts=np.unique(vc['tets'][rows]).astype(np.int64);vstars[vr]=set(map(int,verts))
                if vstars[vr]&used_volume_star[vr]:overlap=True
        except Exception as exc:
            rejections['volume_edge_not_current']+=1
            if len(rejected)<512:rejected.append({'edge_rows':list(edge),'reason':str(exc)[:200]})
            continue
        if overlap:rejections['overlapping_volume_star']+=1;continue
        passes=[];failure_reasons=[];direction_eval_s=0.0
        dirs=[(int(d['drop']),int(d['keep'])) for d in cand.get('eligible_directions',[]) if d.get('eligible_endpoint')]
        dirs=sorted(set(dirs+[(edge[0],edge[1]),(edge[1],edge[0])]))
        for drop,keep in dirs:
            if drop in set(map(int,fixed)):
                failure_reasons.append({'drop':drop,'reason':'immutable_rim_or_route'});continue
            trial_count+=1
            try:
                eval_t0=time.monotonic()
                got=cache.evaluate(edge,drop,keep,fixed,args.max_edge_m,allow_local_tie=(role=='I'))
                for vr in targets:
                    prefix='skin' if vr=='Skin' else 'fat'
                    wall=set(map(int,np.asarray(volumes[vr][prefix+'_current_target_rebind_wall_faces_volume_rows'],np.int64).reshape(-1)))
                    if int(got['volumes'][vr]['drop_volume_row']) in wall:raise ValueError(vr+' retired endpoint belongs to authenticated wall row')
                passes.append((got,drop,keep))
            except Exception as exc:
                failure_reasons.append({'drop':drop,'reason':str(exc)[:240]})
            finally:
                dt=time.monotonic()-eval_t0;direction_eval_s+=dt;local_eval_total_s+=dt
            if trial_count>=args.max_trials:break
        if not passes:
            rejections['no_orientation_passed']+=1
            if len(rejected)<512:rejected.append({'edge_rows':list(edge),'length_m':length,'directions':failure_reasons})
            continue
        passes.sort(key=lambda x:(x[0]['combined_native_bad_after'],x[0]['combined_log_deficit_after'],x[1],x[2]))
        got,drop,keep=passes[0];sid=int(np.max(np.asarray(surfaces[role].get('source_collapse_operation_rows',[-1]),np.int64)))+len(actions)+1
        sres=got['surface'];affected={};changed={};deltas={};drop_nodes={};keep_nodes={}
        for vr in targets:
            vrres=got['volumes'][vr];dv=int(vrres['drop_volume_row']);kv=int(vrres['keep_volume_row'])
            affected[vr]=np.asarray(vrres['affected_input_cell_rows'],np.int64).tolist()
            changed[vr]=np.asarray(vrres['changed_input_cell_rows'],np.int64).tolist()
            deltas[vr]=str(vrres['exact_signed_volume_delta_det6_integer'])
            node_maps[vr][dv]=node_maps[vr][kv];drop_nodes[vr]=dv;keep_nodes[vr]=kv
        source_node_map[int(drop)]=source_node_map[int(keep)]
        op={'operation_id':sid,'role':role,'edge_rows':list(edge),'edge_length_m':float(got['edge_length_m']),
            'drop_source_row':int(drop),'keep_source_row':int(keep),'changed_face_base_rows':np.asarray(sres['changed_parent_rows'],np.int64).tolist(),
            'star_face_base_rows':np.asarray(sres['star_rows'],np.int64).tolist(),'affected_base_cells':affected,
            'changed_base_cells':changed,'volume_delta_det6_integer':deltas,'drop_volume_rows':drop_nodes,'keep_volume_rows':keep_nodes}
        cache.commit(got);op_records.append(op)
        actions.append({'operation_id':sid,'edge_rows':list(edge),'edge_length_m':float(got['edge_length_m']),
            'drop_source_row':int(drop),'keep_source_row':int(keep),'protected_back_endpoint_dropped':bool(int(drop) in set(map(int,back))),
            'source_star_face_count':int(len(frows)),'source_star_vertex_count':int(len(sverts)),
            'volume_star_vertex_counts':{vr:int(len(vstars[vr])) for vr in targets},
            'combined_quality':{k:got[k] for k in ('combined_native_bad_before','combined_native_bad_after','combined_log_deficit_before','combined_log_deficit_after')},
            'surface_metrics':compact_metrics(sres),'volume_metrics':{vr:compact_metrics(got['volumes'][vr]) for vr in targets},
            'local_direction_evaluation_s':direction_eval_s,'rejected_direction_reasons':failure_reasons,
            'per_operation_external_audit':'PENDING_FINAL_775_805_ALL185'})
        used_source_star.update(map(int,sverts))
        for vr in targets:used_volume_star[vr].update(vstars[vr])
    if not actions:
        report={'schema':'h706.cached-local-endpoint-quotient-batch.v1','status':'NO_QUALIFIED_ENDPOINT_QUOTIENT_PROGRESS',
            'physical_admission':False,'native_steps':0,'mass_mapping':'PENDING','input_state':ref(state_path),'roster':ref(roster_path),
            'role':role,'trial_count':trial_count,'accepted_count':0,'rejection_counts':dict(rejections),'rejected_directions':rejected,
            'external_775_805_full_source_audit':'PENDING','elapsed_s':time.monotonic()-tic}
        write_json(out/'report.json',report);print(json.dumps({'status':report['status'],'trials':trial_count,'rejections':dict(rejections)}));return
    # Compose all point quotients, then build and certify one flattened output.
    final_t0=time.monotonic();serialization_s=0.0;rebinder_s=0.0
    build_t0=time.monotonic();final_surface,face_map=build_source_output(surfaces[role],source_node_map,op_records,srefs[role]);output_build_s=time.monotonic()-build_t0
    ser_t0=time.monotonic();spath=out/f'candidate-{role}-typed-current.npz';np.savez_compressed(spath,**final_surface);surface_ref=ref(spath);serialization_s+=time.monotonic()-ser_t0
    final_surfaces=dict(surfaces);final_surfaces[role]=load_npz(spath)
    cell_maps={};cell_unions={};final_volumes={};volume_refs={};bind_reports={}
    for vr in targets:
        t=np.asarray(volumes[vr]['tetrahedra'],np.int64);mt=node_maps[vr][t];st=np.sort(mt,axis=1);valid=~np.any(st[:,1:]==st[:,:-1],axis=1)
        cm=np.flatnonzero(valid).astype(np.int64);cell_maps[vr]=cm;inv=np.full(len(t),-1,np.int64);inv[cm]=np.arange(len(cm),dtype=np.int64)
        unions={}
        for op in op_records:
            parent_union=set(map(int,op['affected_base_cells'][vr]))
            for basecell in op['changed_base_cells'][vr]:
                outrow=int(inv[int(basecell)])
                if outrow<0:raise ValueError(vr+' changed survivor unexpectedly removed by a disjoint operation')
                unions.setdefault(outrow,set()).update(parent_union)
        cell_unions[vr]=unions
        output_build_s_t0=time.monotonic();vdict=build_volume_output(volumes[vr],node_maps[vr],cm,unions,op_records,vrefs[vr],vr);output_build_s+=time.monotonic()-output_build_s_t0
        prefix='skin' if vr=='Skin' else 'fat'
        wallpoints=np.asarray(vdict[prefix+'_current_target_rebind_wall_face_points_world_m_f32'],np.float32)
        wallrows=np.asarray(vdict[prefix+'_current_target_rebind_wall_faces_volume_rows'],np.int64)
        wallids=np.arange(90,dtype=np.int64)+(0 if vr=='Skin' else 90)
        rebind_t0=time.monotonic();vdict,bind_reports[vr]=rebinder.rebind_source_aliases(vr,vdict,final_surfaces,wallpoints,wallrows,wallids,
            input_pins={'parent_state_path':str(state_path),'parent_state_sha256':sha(state_path),'source_'+role+'_path':surface_ref['path'],'source_'+role+'_sha256':surface_ref['sha256']})
        rebinder_s+=time.monotonic()-rebind_t0
        ser_t0=time.monotonic();vpath=out/f'candidate-{vr.lower()}-current.npz';np.savez_compressed(vpath,**vdict);serialization_s+=time.monotonic()-ser_t0
        final_volumes[vr]=load_npz(vpath);volume_refs[vr]=ref(vpath)
    readback_t0=time.monotonic()
    source_check=verify_final_surface(surfaces[role],final_surfaces[role],source_node_map,face_map,op_records,srefs[role])
    volume_checks={vr:verify_final_volume(volumes[vr],final_volumes[vr],node_maps[vr],cell_maps[vr],cell_unions[vr],op_records,
        rebinder,vr,final_surfaces,vrefs[vr]) for vr in targets}
    independent_readback_s=time.monotonic()-readback_t0
    before_bad=sum(cache.initial[vr]['bad'] for vr in targets);after_bad=sum(volume_checks[vr]['native_bad_after'] for vr in targets)
    before_score=sum(cache.initial[vr]['score'] for vr in targets);after_score=sum(volume_checks[vr]['log_deficit_after'] for vr in targets)
    if not (after_bad<before_bad or (after_bad==before_bad and after_score<before_score-1e-8)):
        raise ValueError('independent final replay found no strict combined quality progress')
    for vr in targets:
        if cache.current[vr]['bad']!=volume_checks[vr]['native_bad_after'] or abs(cache.current[vr]['score']-volume_checks[vr]['log_deficit_after'])>1e-6:
            raise ValueError(vr+' local quality ledger differs from final global readback')
    output_state=dict(state);output_state['source_state']=ref(state_path)
    output_state['source_surfaces']=dict(state['source_surfaces']);output_state['source_surfaces'][role]=surface_ref
    for vr in targets:
        rec=dict(state[vr+'_current']);rec.update(volume_refs[vr]);rec['nodes']=len(final_volumes[vr]['points_world_m']);rec['cells']=len(final_volumes[vr]['tetrahedra'])
        rec['native_quality_failures']=volume_checks[vr]['native_bad_after'];rec['nonpositive']=0;rec['oriented_volume_target_boundary_equal']=True
        output_state[vr+'_current']=rec
    output_state.update({'physical_admission':False,'native_steps':0,'source_audit_status':'PENDING_EXTERNAL_775_805_FULL_CONTEXT',
        'boundary_source_binding_status':'PASS_EXACT_CURRENT_TYPED_ALIAS_REBIND','surface_crossings':'PENDING_EXTERNAL_775_805',
        'mass_mapping':'PENDING','actual_697_217':'PENDING','construction_status':'ENDPOINT_QUOTIENT_BATCH_PROPOSAL_UNQUALIFIED'})
    final_state=out/'candidate-state-proposal.json';write_json(final_state,output_state)
    journal={'schema':'h706.cached-local-endpoint-quotient-journal.v1','status':'PASS_FINAL_READBACK_PENDING_EXTERNAL_QUALIFICATION',
        'input_state':ref(state_path),'roster':ref(roster_path),'role':role,'operations':actions,'operation_parent_lineage':op_records,
        'precomputed_full_mesh_quality_and_incidence_once':True,'per_trial_global_measurement_passes':0,
        'stage_seconds':{'cache_setup':cache_setup_s,'local_direction_evaluations':local_eval_total_s,
            'output_build':output_build_s,'final_alias_rebinding':rebinder_s,'final_serialization':serialization_s,
            'final_independent_readback':independent_readback_s,'post_loop_finalization_total':time.monotonic()-final_t0},
        'final_source_alias_rebinds':len(targets),'final_independent_readbacks':1+len(targets),'independent_surface_replay':source_check,
        'independent_volume_replay':volume_checks,'rejected_directions':rejected,'mass_mapping':'PENDING','physical_admission':False,'native_steps':0}
    write_json(out/'operation-journal.json',journal)
    report={'schema':'h706.cached-local-endpoint-quotient-batch.v1','status':'PASS_BOUNDED_ENDPOINT_QUOTIENT_PROPOSAL_PENDING_EXTERNAL_QUALIFICATION',
        'physical_admission':False,'native_steps':0,'mass_mapping':'PENDING','native_quality_thresholds_unchanged':True,
        'input_state':ref(state_path),'roster':ref(roster_path),'role':role,'max_edge_m':args.max_edge_m,'seconds_budget':args.seconds,
        'trial_count':trial_count,'accepted_count':len(actions),'rejection_counts':dict(rejections),'rejected_directions':rejected,
        'candidate_state_proposal':ref(final_state),'operation_journal':ref(out/'operation-journal.json'),
        'external_775_805_full_source_audit':'PENDING','all185_intersection_audit':'PENDING','mass_conservation':'PENDING',
        'stage_seconds':{'cache_setup':cache_setup_s,'local_direction_evaluations':local_eval_total_s,
            'output_build':output_build_s,'final_alias_rebinding':rebinder_s,'final_serialization':serialization_s,
            'final_independent_readback':independent_readback_s},'elapsed_s':time.monotonic()-tic}
    write_json(out/'report.json',report)
    print(json.dumps({'status':report['status'],'accepted':len(actions),'trials':trial_count,'rejections':dict(rejections),
        'candidate_state':str(final_state),'elapsed_s':report['elapsed_s']},indent=2))

if __name__=='__main__':main()
