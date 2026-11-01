#!/usr/bin/env python3
"""State-bound, single-operation endpoint quotient proposal wrapper.

This emits candidate source/volume arrays only. It never emits a qualified or
physical state; the full 775/805 source audit and mass mapping are external.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
CORE=HERE/'inner_endpoint_quotient_001.py'
CORE_SHA='87f7e980ce80a84a5380cec5c840e924489fa221ab0f435bcd91490c394054a5'
PROTECTION=H/'quality-partial-cavity-684-root'/'paired_connected_boundary_patch_853_active_rows_root.py'
PROTECTION_SHA='9104480449f67718f5e26d8248d48c32abf9145ab20015b9043122d9d94b5fce'
PROTECTION_OWNER_SHA='68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a'

def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
def imp(name,path,expected=None):
    path=Path(path).resolve()
    if expected and sha(path)!=expected:raise ValueError(f'{name} source SHA mismatch')
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader:raise RuntimeError(f'cannot import {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def check_ref(ref,label):
    if not isinstance(ref,dict) or not ref.get('path') or not ref.get('sha256'):raise ValueError(label+' has no path/SHA')
    p=Path(ref['path']).resolve()
    if not p.is_file() or sha(p)!=str(ref['sha256']):raise ValueError(label+' SHA mismatch')
    return p

def point_cache(Pmod, volume):
    p=np.asarray(volume['points_world_m_f32'] if 'points_world_m_f32' in volume else volume['points_world_m'],np.float32)
    return {'points':p,'coordinate_index':Pmod.point_key_index(p)}

def derive_anchor_rows(Pmod,state_path,state,role,surface,volumes,output_dir):
    """Use the pinned 795 lineage owner; freeze only rim/route endpoints.

    Actual inner-back endpoints are intentionally not frozen as a whole star.
    Their displacement is bounded by the operation edge length (<=50 um).
    """
    protection=Pmod.protected_scope(state_path,state,role,volumes,output_dir,fixture=False)
    pick='Fat' if role in ('D','I') and 'Fat' in protection else next(iter(protection))
    src=protection[pick]['report']['sources'][role]
    rim=np.asarray(src['rim_rows'],np.int64);route=np.asarray(src['route_rows'],np.int64)
    back=np.asarray(src['back_rows'],np.int64)
    if len(np.intersect1d(rim,route))!=len(np.union1d(rim,route)):
        # Overlap is fine; this branch exists only to force explicit set derivation.
        pass
    immutable=np.unique(np.r_[rim,route]).astype(np.int64)
    return immutable,back,protection

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--role',choices=('I','D'),required=True)
    ap.add_argument('--edge',nargs=2,type=int,required=True)
    ap.add_argument('--drop',type=int,required=True)
    ap.add_argument('--keep',type=int,required=True)
    ap.add_argument('--max-edge-m',type=float,default=50e-6)
    a=ap.parse_args();tic=time.monotonic()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    core=imp('endpoint_quotient_core_001',CORE,CORE_SHA)
    Pmod=imp('endpoint_quotient_protection_853',PROTECTION,PROTECTION_SHA)
    if sha(Pmod.P.__file__)!=PROTECTION_OWNER_SHA:raise ValueError('795 protected-row owner hash mismatch')
    state_path=a.state.resolve();state=json.loads(state_path.read_text())
    if state.get('physical_admission') is True or int(state.get('native_steps',0))!=0:
        raise ValueError('input must be a construction-only state with no native admission')
    role=a.role
    if role=='O':raise ValueError('outer O endpoint changes are outside this proposal')
    source_refs=state.get('source_surfaces',{})
    surface_paths={r:check_ref(source_refs[r],r+' current typed source') for r in ('O','I','D')}
    surface_arrays={r:read_npz(p) for r,p in surface_paths.items()}
    volume_refs={r:state[r+'_current'] for r in ('Skin','Fat')}
    volume_paths={r:check_ref(volume_refs[r],r+' current volume') for r in ('Skin','Fat')}
    volume_arrays={r:read_npz(p) for r,p in volume_paths.items()}
    protect_volumes={r:point_cache(Pmod,volume_arrays[r]) for r in ('Skin','Fat')}
    immutable,back,protection=derive_anchor_rows(Pmod,state_path,state,role,surface_arrays[role],protect_volumes,out/'protection')
    targets={'I':('Skin','Fat'),'D':('Fat',)}[role]
    result=core.propose_endpoint_quotient(role,surface_arrays[role],
        {vr:volume_arrays[vr] for vr in targets},tuple(a.edge),drop=a.drop,keep=a.keep,
        immutable_source_rows=immutable,max_edge_m=a.max_edge_m)
    if result['surface']['edge_length_m']>a.max_edge_m:raise ValueError('source edge exceeded declared distance cap')
    # A collapsed endpoint is the only source-space displacement. The quotient
    # moves any point in the affected patch by at most this edge length.
    opid=int(np.max(np.asarray(surface_arrays[role].get('source_collapse_operation_rows',[-1]),np.int64)))+1
    sarrays=core.endpoint_surface_arrays(surface_arrays[role],result['surface'],
        parent_path=str(surface_paths[role]),parent_sha=source_refs[role]['sha256'],operation_id=opid)
    spath=out/f'candidate-{role}-endpoint-quotient.npz';np.savez_compressed(spath,**sarrays)
    volume_refs_out={}
    volume_reports=[]
    for vr in targets:
        vresult=result['volumes'][vr]
        varrays=core.endpoint_volume_arrays(volume_arrays[vr],vresult,
            parent_path=str(volume_paths[vr]),parent_sha=volume_refs[vr]['sha256'])
        vpath=out/f'candidate-{vr}-endpoint-quotient.npz';np.savez_compressed(vpath,**varrays)
        volume_refs_out[vr]={'path':str(vpath),'sha256':sha(vpath)}
        alias_mismatches={}
        nboundary=len(volume_arrays[vr]['desired_boundary_faces'])
        for k in ('desired_boundary_source_group','desired_boundary_source_rows','desired_boundary_source_face_rows',
                  'proposed_current_desired_boundary_source_group','proposed_current_desired_boundary_source_rows',
                  'proposed_current_desired_boundary_source_face_rows'):
            if k in volume_arrays[vr] and len(np.asarray(volume_arrays[vr][k]))!=nboundary:
                alias_mismatches[k]={'length':int(len(np.asarray(volume_arrays[vr][k]))),'target_faces':int(nboundary)}
        volume_reports.append({'role':vr,'input':{'path':str(volume_paths[vr]),'sha256':volume_refs[vr]['sha256']},
            'output':volume_refs_out[vr],'metrics':vresult,'boundary_source_metadata_alias_mismatches':alias_mismatches,
            'source_metadata_rebind_required':bool(alias_mismatches)})
    # Independent output readback: compare exact files against the candidate
    # rows before any external audit or candidate-state promotion.
    saved_s=read_npz(spath)
    if saved_s['points_world_m'].tobytes()!=surface_arrays[role]['points_world_m'].tobytes() or saved_s['points_world_m_f32'].tobytes()!=surface_arrays[role]['points_world_m_f32'].tobytes():
        raise ValueError('source point arrays changed in serialized proposal')
    if not np.array_equal(saved_s['faces'],result['surface']['faces']):raise ValueError('source faces differ on readback')
    if len(saved_s['quality_star_patch_union_offsets'])!=len(saved_s['faces'])+1 or len(saved_s['face_ancestry_offsets'])!=len(saved_s['faces'])+1:
        raise ValueError('source typed ancestry CSR readback mismatch')
    for vr in targets:
        saved_v=read_npz(Path(volume_refs_out[vr]['path']))
        if saved_v['points_world_m'].tobytes()!=volume_arrays[vr]['points_world_m'].tobytes():raise ValueError(vr+' point array changed in serialized proposal')
        if not np.array_equal(saved_v['tetrahedra'],result['volumes'][vr]['tetrahedra']):raise ValueError(vr+' tets differ on readback')
        if core.face_chain(core.boundary_faces(saved_v['tetrahedra']))!=core.face_chain(saved_v['desired_boundary_faces']):raise ValueError(vr+' exact output boundary readback mismatch')
    candidate_state=dict(state)
    candidate_state['source_surfaces']=dict(source_refs)
    candidate_state['source_surfaces'][role]={'path':str(spath),'sha256':sha(spath)}
    for vr in targets:
        rec=dict(state[vr+'_current']);rec.update(volume_refs_out[vr])
        rec['nodes']=len(volume_arrays[vr]['points_world_m']);rec['cells']=len(result['volumes'][vr]['tetrahedra'])
        rec['native_quality_failures']=int(result['volumes'][vr]['native_bad_after'])
        rec['nonpositive']=0;rec['oriented_volume_target_boundary_equal']=True
        candidate_state[vr+'_current']=rec
    candidate_state.update({'physical_admission':False,'native_steps':0,
        'construction_status':'ENDPOINT_QUOTIENT_PROPOSAL_PENDING_EXTERNAL_QUALIFICATION',
        'source_audit_status':'PENDING_EXTERNAL_775_805_FULL_CONTEXT',
        'mass_mapping':'PENDING','actual_697_217':'PENDING'})
    state_out=out/'candidate-state-proposal.json';state_out.write_text(json.dumps(candidate_state,indent=2,sort_keys=True)+'\n')
    report={'schema':'h706.inner-endpoint-quotient-state-proposal.v1','status':'PASS_BOUNDED_TOPOLOGY_ONLY_PROPOSAL_PENDING_EXTERNAL_AUDITS',
        'physical_admission':False,'native_steps':0,'mass_mapping':'PENDING','native_quality_thresholds_unchanged':True,
        'input_state':{'path':str(state_path),'sha256':sha(state_path)},
        'core':{'path':str(CORE),'sha256':CORE_SHA},'795_anchor_owner':{'path':str(Pmod.P.__file__),'sha256':sha(Pmod.P.__file__)},
        'protection_helper':{'path':str(PROTECTION),'sha256':sha(PROTECTION)},
        'role':role,'edge_rows':list(map(int,a.edge)),'drop_source_row':int(a.drop),'keep_source_row':int(a.keep),
        'surface_displacement_upper_bound_m':float(result['surface']['edge_length_m']),
        'inner_back_displacement_cap_m':float(a.max_edge_m),'endpoint_only_back_policy':True,
        'rim_route_endpoint_rows_fixed':True,'protected_back_endpoint_dropped':bool(int(a.drop) in set(map(int,back))),
        'protected_back_rows_count':int(len(back)),'immutable_rim_route_rows_count':int(len(immutable)),
        'surface_result':{k:v for k,v in result['surface'].items() if k not in ('points_world_m_f32','faces','face_output_to_parent_rows','changed_output_rows')},
        'combined_quality':{k:result[k] for k in ('paired','combined_native_bad_before','combined_native_bad_after','combined_log_deficit_before','combined_log_deficit_after')},
        'volume_outputs':volume_reports,'surface_output':{'path':str(spath),'sha256':sha(spath)},
        'candidate_state_proposal':{'path':str(state_out),'sha256':sha(state_out)},
        'independent_serialized_readback':True,'external_775_805_full_source_audit':'PENDING',
        'all185_intersection_audit':'PENDING','mass_conservation_and_mapping':'PENDING',
        'elapsed_s':time.monotonic()-tic}
    rp=out/'report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'output':str(out),'role':role,'edge':list(a.edge),
                      'quality':report['combined_quality'],'volume_roles':targets,
                      'external_audit':'PENDING','elapsed_s':report['elapsed_s']},indent=2))

if __name__=='__main__':main()
