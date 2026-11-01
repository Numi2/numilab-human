#!/usr/bin/env python3
"""O-only endpoint quotient batch using Skin-only 855 lineage-aware protection; no scene/runtime admission.

This is a versioned role extension. It imports frozen batch005 readback helpers,
but only edits O->Skin, caps the ordinary source-surface Hausdorff bound at
0.5 mm, and rejects any source/volume endpoint star touching authenticated
O back, rim, or route rows.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
CORE=HERE.parent/'quality-o-endpoint-quotient-007-nativepreflight'/'outer_endpoint_quotient_007.py'
CORE_SHA='6d7cec119eb6e2e752f59c015ebce53a12575c878d5b20f511bc628ff351d27f'
PROTECT_ADAPTER=HERE/'o_skin_only_protection_010.py'
PROTECT_ADAPTER_SHA='5422ec971a193f317290f03114f72a11ee822fa8dbebceb872a1b6a3ef5676f3'
LEGACY_BATCH=H/'quality-endpoint-collapse-001'/'batch_inner_endpoint_quotient_005.py'
LEGACY_BATCH_SHA='eb57b4ad8a91feec2ed84ca7cf483093ec2b49f367bb1ff55977f927a4997619'
MAX_HAUSDORFF_M=500e-6

def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError('cannot serialize '+type(value).__name__)
def json_dumps(value,**kwargs):
    return json.dumps(value,default=json_default,**kwargs)
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def imp(name,path,expected=None):
    path=Path(path).resolve()
    if expected and sha(path)!=expected:raise ValueError(name+' SHA mismatch')
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader:raise RuntimeError('cannot load '+str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def role_o_protection(surface,skin,protection):
    """Role-local O protection; ignore colliding I IDs in the shared Skin domain."""
    rec=protection.get('report',{}).get('sources',{}).get('O')
    if not isinstance(rec,dict):raise ValueError('Skin protected-scope report lacks O source rows')
    src=np.asarray(surface['points_world_m_f32'],np.float32)
    allrows=np.unique(np.r_[np.asarray(rec.get('back_rows',[]),np.int64),
        np.asarray(rec.get('rim_rows',[]),np.int64),np.asarray(rec.get('route_rows',[]),np.int64)]).astype(np.int64)
    if allrows.size and (allrows.min()<0 or allrows.max()>=len(src)):raise ValueError('O protection row outside source domain')
    mapped=COREMOD._exact_rows_to_volume(src,np.asarray(skin.get('points_world_m_f32',skin['points_world_m']),np.float32),allrows)
    return {'source_rows':set(map(int,allrows)),'volume_rows':set(map(int,mapped.values())),
            'back_source_rows':set(map(int,np.asarray(rec.get('back_rows',[]),np.int64))),
            'rim_source_rows':set(map(int,np.asarray(rec.get('rim_rows',[]),np.int64))),
            'route_source_rows':set(map(int,np.asarray(rec.get('route_rows',[]),np.int64)))}

def must_be_o_to_skin(role,targets):
    if role!='O' or tuple(targets)!=('Skin',):raise ValueError('O quotient must target Skin only')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--roster',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-ops',type=int,default=64)
    ap.add_argument('--max-trials',type=int,default=300)
    ap.add_argument('--seconds',type=float,default=90.0)
    args=ap.parse_args();tic=time.monotonic()
    if args.output.exists():raise FileExistsError(args.output)
    out=args.output.resolve();out.mkdir(parents=True)
    global COREMOD,B,PROTECTMOD
    COREMOD=imp('outer_endpoint_core007',CORE,CORE_SHA)
    B=imp('frozen_endpoint_batch005',LEGACY_BATCH,LEGACY_BATCH_SHA)
    PROTECTMOD=imp('o_skin_only_protection_adapter010',PROTECT_ADAPTER,PROTECT_ADAPTER_SHA)
    B.core=COREMOD.CORE
    state_path=args.state.resolve();roster_path=args.roster.resolve()
    state,surfaces,srefs,volumes,vrefs=B.load_state_inputs(state_path)
    roster=json.loads(roster_path.read_text())
    if roster.get('status')!='PASS_READ_ONLY_ACTIVE_ROW_BAD_CELL_CENSUS' or roster.get('role')!='O':
        raise ValueError('O census roster role/status mismatch')
    if roster.get('input_state',{}).get('sha256')!=sha(state_path):
        raise ValueError('O roster does not bind exact input state')
    role='O';targets=('Skin',);must_be_o_to_skin(role,targets)
    fixed,back,prot,_=PROTECTMOD.protected_rows(state_path,state,role,surfaces,volumes,out,COREMOD)
    skin_prot=role_o_protection(surfaces['O'],volumes['Skin'],prot['Skin'])
    protected_source=skin_prot['source_rows'];protected_volume=skin_prot['volume_rows']
    if not back.size or not set(map(int,back)).issubset(protected_source):
        raise ValueError('authenticated O back rows are absent from explicit protection set')
    current_state=state;current_state_path=state_path
    current_surfaces=dict(surfaces);current_volumes=dict(volumes)
    current_srefs=dict(srefs);current_vrefs=dict(vrefs)
    used_source_star=set();used_volume_star={'Skin':set()}
    actions=[];rejections=Counter();trials=0
    candidates=roster.get('eligible_bad_cell_hit_edges',[])
    rebinder=imp('endpoint_source_alias_rebinder_877_for_O006',B.REBINDER,B.REBINDER_SHA)
    for cand in candidates:
        if time.monotonic()-tic>=args.seconds or trials>=args.max_trials or len(actions)>=args.max_ops:break
        edge=tuple(map(int,cand['edge_rows']))
        length=float(cand['length_m'])
        if not np.isfinite(length) or length<=0 or length>MAX_HAUSDORFF_M:
            rejections['over_ordinary_hausdorff_bound']+=1;continue
        frows,sverts=B.source_star_vertices(current_surfaces['O'],edge)
        if set(map(int,sverts))&used_source_star:rejections['overlapping_source_star']+=1;continue
        if set(map(int,sverts))&protected_source:
            rejections['protected_O_back_rim_route_source_star']+=1;continue
        vrows,vverts,nodes=B.volume_star_vertices(current_volumes['Skin'],current_surfaces['O'],edge)
        if set(map(int,vverts))&used_volume_star['Skin']:rejections['overlapping_skin_volume_star']+=1;continue
        if set(map(int,vverts))&protected_volume:
            rejections['protected_O_back_rim_route_Skin_star']+=1;continue
        dirs=[(int(d['drop']),int(d['keep'])) for d in cand.get('eligible_directions',[]) if d.get('eligible_endpoint')]
        dirs=sorted(set(dirs+[(edge[0],edge[1]),(edge[1],edge[0])]))
        passing=[];failures=[]
        for drop,keep in dirs:
            trials+=1
            try:
                got=COREMOD.propose_outer_endpoint_quotient(current_surfaces['O'],current_volumes['Skin'],edge,
                    drop=drop,keep=keep,protected_source_rows=protected_source,
                    protected_volume_rows=protected_volume,max_hausdorff_m=MAX_HAUSDORFF_M)
                passing.append((got,drop,keep))
            except Exception as exc:
                failures.append({'drop':drop,'reason':str(exc)[:220]})
        if not passing:
            rejections['no_direction_passed']+=1;continue
        passing.sort(key=lambda x:(x[0]['volumes']['Skin']['native_bad_after'],
            x[0]['volumes']['Skin']['native_deficit_after'],x[1],x[2]))
        got,drop,keep=passing[0];stage=out/f'stage-{len(actions)+1:03d}';stage.mkdir()
        sid=int(np.max(np.asarray(current_surfaces['O'].get('source_collapse_operation_rows',[-1]),np.int64)))+1
        sdict=COREMOD.CORE.endpoint_surface_arrays(current_surfaces['O'],got['surface'],
            parent_path=current_srefs['O']['path'],parent_sha=current_srefs['O']['sha256'],operation_id=sid)
        spath=stage/'candidate-O-typed-current.npz';np.savez_compressed(spath,**sdict)
        next_surfaces=dict(current_surfaces);next_surfaces['O']=B.load_npz(spath)
        next_srefs=dict(current_srefs);next_srefs['O']=B.ref(spath)
        vres=got['volumes']['Skin']
        vdict=COREMOD.CORE.endpoint_volume_arrays(current_volumes['Skin'],vres,
            parent_path=current_vrefs['Skin']['path'],parent_sha=current_vrefs['Skin']['sha256'])
        vdict['mapped_current_boundary_faces']=np.asarray(vdict['boundary_faces'],np.int64).copy()
        vdict['proposed_current_desired_boundary_faces']=np.asarray(vdict['desired_boundary_faces'],np.int64).copy()
        prefix='skin'
        wallxyz=np.asarray(vdict[prefix+'_current_target_rebind_wall_face_points_world_m_f32'],np.float32)
        wallrows=np.asarray(vdict[prefix+'_current_target_rebind_wall_faces_volume_rows'],np.int64)
        wallids=np.arange(90,dtype=np.int64)
        vdict,binding=rebinder.rebind_source_aliases('Skin',vdict,next_surfaces,wallxyz,wallrows,wallids,
            input_pins={'parent_state_sha256':B.sha(current_state_path),'source_O_sha256':next_srefs['O']['sha256']})
        vpath=stage/'candidate-skin-current.npz';np.savez_compressed(vpath,**vdict)
        vread=B.load_npz(vpath)
        vcheck=B.verify_volume_step(current_volumes['Skin'],vread,vres,rebinder,'Skin',next_surfaces,current_vrefs['Skin'])
        sread=B.load_npz(spath)
        scheck=B.verify_surface_step(current_surfaces['O'],sread,got['surface'],current_srefs['O'],sid)
        next_volumes=dict(current_volumes);next_volumes['Skin']=vread
        next_vrefs=dict(current_vrefs);next_vrefs['Skin']=B.ref(vpath)
        next_state=dict(current_state);next_state['source_state']={'path':str(current_state_path),'sha256':B.sha(current_state_path)}
        next_state['source_surfaces']=dict(current_state['source_surfaces']);next_state['source_surfaces']['O']=next_srefs['O']
        rec=dict(current_state['Skin_current']);rec.update(next_vrefs['Skin'])
        rec.update(nodes=len(vread['points_world_m']),cells=len(vread['tetrahedra']),
                   native_quality_failures=vcheck['bad_cells'],nonpositive=0,oriented_volume_target_boundary_equal=True)
        next_state['Skin_current']=rec
        next_state.update({'physical_admission':False,'native_steps':0,'source_audit_status':'PENDING_EXTERNAL_775_805_FULL_CONTEXT',
            'boundary_source_binding_status':'PASS_EXACT_CURRENT_TYPED_ALIAS_REBIND','surface_crossings':'PENDING_EXTERNAL_775_805',
            'mass_mapping':'PENDING','actual_697_217':'PENDING','construction_status':'O_ENDPOINT_QUOTIENT_PROPOSAL_UNQUALIFIED'})
        state_stage=stage/'state.json';state_stage.write_text(json_dumps(next_state,indent=2,sort_keys=True)+'\n')
        op={'operation':len(actions),'role':'O','target_volume':'Skin','edge_rows':list(edge),'edge_length_m':length,
            'drop_source_row':drop,'keep_source_row':keep,'ordinary_surface_hausdorff_upper_bound_m':got['ordinary_surface_hausdorff_upper_bound_m'],
            'ordinary_surface_hausdorff_limit_m':MAX_HAUSDORFF_M,'hausdorff_bound_method':got['ordinary_surface_hausdorff_bound_method'],
            'protected_O_back_rim_route_source_rows':len(protected_source),'protected_O_back_rim_route_Skin_rows':len(protected_volume),
            'protected_source_star_faces':got['protected_source_star_faces'],'protected_volume_star_cells':got['protected_volume_star_cells'],
            'source_star_rows':frows.tolist(),'source_star_vertices':sverts.tolist(),'Skin_volume_star_vertices':vverts.tolist(),
            'quality':{'native_bad_before':vres['native_bad_before'],'native_bad_after':vres['native_bad_after'],
                       'native_deficit_before':vres['native_deficit_before'],'native_deficit_after':vres['native_deficit_after']},
            'source_readback':scheck,'Skin_readback':vcheck,'alias_rebind':binding,
            'serialized_surface':B.ref(spath),'serialized_volume':B.ref(vpath),'serialized_state':B.ref(state_stage),
            'physical_admission':False,'native_steps':0}
        (stage/'operation.json').write_text(json_dumps(op,indent=2,sort_keys=True)+'\n')
        actions.append(op);used_source_star.update(map(int,sverts));used_volume_star['Skin'].update(map(int,vverts))
        current_state=next_state;current_state_path=state_stage;current_surfaces=next_surfaces;current_volumes=next_volumes
        current_srefs=next_srefs;current_vrefs=next_vrefs
        if trials>=args.max_trials:break
    final_state=out/'candidate-state-proposal.json'
    final_state.write_text(json_dumps(current_state,indent=2,sort_keys=True)+'\n')
    report={'schema':'h706.O-endpoint-quotient-batch.v1',
        'status':'PASS_O_ENDPOINT_QUOTIENT_PROPOSAL_PENDING_EXTERNAL_QUALIFICATION' if actions else 'NO_QUALIFIED_O_ENDPOINT_QUOTIENT_PROGRESS',
        'role':'O','target_volumes':['Skin'],'input_state':B.ref(state_path),'roster':B.ref(roster_path),
        'accepted_operations':actions,'accepted_count':len(actions),'trial_count':trials,'rejection_counts':dict(rejections),
        'max_ordinary_surface_hausdorff_m':MAX_HAUSDORFF_M,'back_rim_route_endpoint_and_star_protection':True,
        'points_unchanged':True,'physical_admission':False,'native_steps':0,'mass_mapping':'PENDING',
        'external_775_805_full_source_audit':'PENDING','candidate_state_proposal':B.ref(final_state),
        'elapsed_s':time.monotonic()-tic,'batch_source_sha256':sha(Path(__file__)),'core_source_sha256':sha(CORE),
        'frozen_batch005_sha256':sha(LEGACY_BATCH),
        'protection_scope_adapter_sha256':sha(PROTECT_ADAPTER),
        'protection_scope_mode':'O_ONLY_SKIN_VOLUME'}
    (out/'report.json').write_text(json_dumps(report,indent=2,sort_keys=True)+'\n')
    print(json_dumps({'status':report['status'],'accepted':len(actions),'trials':trials,
        'rejections':dict(rejections),'candidate_state':str(final_state),'elapsed_s':report['elapsed_s']},indent=2))

if __name__=='__main__':main()
