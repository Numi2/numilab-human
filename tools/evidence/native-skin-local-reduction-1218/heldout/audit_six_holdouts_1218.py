#!/usr/bin/env python3
"""Read-only six-pose held-out exact skin audit for 1218 fit-attempt-001.

No candidate is fitted here and no native state is advanced. The candidate
source positions are forward-mapped into the six retained accepted 1191 states,
then checked against every 1172 target and the complete skin self surface.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, importlib.util, json, mmap, os, struct, sys, time
from pathlib import Path
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005'); R=Path('/Users/n/numi-human-retained-delivery-20261009')
BASE=R/'skin-resting-multipose-clearance-1218'; FIT=BASE/'fit-attempt-001'
FIT_STEPS=(0,9983,20000)
BASE_PREF=R/'skin-resting-multipose-clearance-1206/zero_delta_preflight.json'; BASE_PREF_SHA='cd0b2809cf4cf3bf427ad090f2fb463465cbf195dccf0ee1a66c3df052e0a809'
SKIN=E/'native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin'
NHA=E/'native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy'
FORWARD=E/'native-common-skin-multipose-forward-model-915.py'
FORWARD_SHA='81a5532f634278a45aa9aaf1457d0fdb3be71906256f3490941a4885b74d4cdc'
ROOT=Path('/Users/n/numi-human-conforming-composition-source-1216')
OWNER_REVISION=ROOT/'source-revision.json'
OWNER_REVISION_SHA='51e3a732e18fbdb6922871b435d98f42d01c00511d2d08a0fc4cac1009faba6f'
CLEARANCE=ROOT/'src/numilab_human/common_atlas_skin_clearance.py'
CLEARANCE_SHA='ec198693fce562db27c4b761e3dbbc4b1bff4f0b9458ae67ab7ba596a97a000f'
CARDIAC=ROOT/'src/numilab_human/cardiac_cavity_intersections.py'
CARDIAC_SHA='934c64fa6a64a06044aee6db44cdcd71dc497fcec97a20a80029575faaaff4b4'
WRAPPER=R/'native-hip-reference-nominal-1193-attempt002/audit_skin_1193_attempt002.py'
WRAPPER_SHA='4c9e121530900e2c5fce74d41dbcaf891a139c50f1c187cbd6a8e03d89270782'
RUNNER=E/'native-lung-late-skin-audit-runner-1172/revision-003/audit_full_skin_cycle_1172.py'
RUNNER_SHA='b1b9be5959610325067f703c32bd55e2f568c11124ec232f0d066ed75a6fdcbb'
RUN=R/'native-lung1178-thumb1187-smoke-1191/native-run'
AUDIT=E/'native-skin-epl143-clearance-1187'
HELDOUT=(4991,5375,5759,6111,6495,7743)
OUT_DEFAULT=BASE/'heldout-001/scan-output'
sys.path.insert(0,str(ROOT/'src'))
from numilab_human import common_atlas_skin_clearance as clearance_current
from numilab_human import cardiac_cavity_intersections as prepared_ci
INDEX_PROOF=R/'skin-resting-multipose-clearance-1206/prepared-index-predicate-comparison.json'
INDEX_PROOF_SHA='44245299282ce64cd6f118d35b75f4bf5cb250f20853363b08b2392ec4694c98'
INDEX_EQ=R/'skin-prepared-first-equivalence-1210-attempt003/equivalence-report.json'
INDEX_EQ_SHA='0714743b609f2f540998afefb658a91cd0ccc1e0798111eb5ac545344e0abaa3' 

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
def pin(path):
    p=Path(path).resolve()
    if not p.is_file() or p.is_symlink(): raise RuntimeError('required input missing or symlinked: '+str(p))
    return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def readj(path): return json.loads(Path(path).read_text())
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def load_skin(path):
    raw=Path(path).read_bytes(); magic,abi,nb,nv,ni,fp,archive=struct.unpack_from('<8s5I32s',raw); vo=60+36*nb; io=vo+56*nv; wo=io+4*ni
    require(magic==b'NHSKIN1\0' and abi==5 and len(raw)==wo+4*nv*nb,'bad NHSKIN ABI/length')
    vr=np.frombuffer(raw,'<f4',14*nv,vo).reshape(nv,14)
    faces=np.frombuffer(raw,'<u4',ni,io).reshape(-1,3).astype(np.int64)
    bind=np.frombuffer(raw,'<f4',9*nb,60).reshape(nb,9).copy()
    bind_u=np.frombuffer(raw,'<u4',9*nb,60).reshape(nb,9).copy()
    weights=np.frombuffer(raw,'<f4',nv*nb,wo).reshape(nv,nb).astype(float)
    return {'raw':raw,'nb':nb,'nv':nv,'ni':ni,'fp':fp,'bind':bind,'bind_u':bind_u,'weights':weights,'pos':vr[:,:3].astype(float),'faces':faces}
def load_wrapper():
    spec=importlib.util.spec_from_file_location('_holdout1193_wrapper',WRAPPER)
    require(spec is not None and spec.loader is not None,'cannot load pinned 1193 static-source wrapper')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module
def validate_final_accepted_trial(candidate, referenced_ids, report, runner, keys):
    solve_path=FIT/'solve-result.json'; attempts_path=FIT/'attempts.jsonl'
    solve=readj(solve_path)
    require(solve.get('status')==report.get('status')=='inferred_engineering_clearance_candidate_pending_native_replay','solve result and candidate report do not agree on successful source-only status')
    require(solve.get('candidate_source_positions_npy')==str(FIT/'candidate-source-positions-f32.npy')
            and solve.get('candidate_source_positions_f32_sha256')==report.get('candidate_source_positions_f32_sha256')
            and solve.get('candidate_source_positions_npy_sha256')==report.get('candidate_source_positions_npy_sha256'),
            'solve result does not bind the candidate report and final NPY')
    require(solve.get('final_nonocular_pair_count_by_pose')==[0,0,0]
            and report.get('final_nonocular_pair_count_by_pose')==[0,0,0],
            'final solver result/report do not both record zero nonocular pairs')
    require(solve_path.is_file() and attempts_path.is_file(),'fit completion receipt or trial log is missing')
    lines=attempts_path.read_bytes().splitlines(keepends=True)
    events=[json.loads(line) for line in lines if line.strip()]
    accepted=[(i,e) for i,e in enumerate(events) if e.get('status')=='accepted']
    require(accepted,'fit trial log has no accepted final event')
    index,event=accepted[-1]
    require(event.get('nonocular_pair_counts_candidate_by_pose')==[0,0,0],
            'last accepted solver event does not record zero nonocular pairs')
    attempt_npy=Path(event.get('candidate_npy_path','')).resolve()
    attempts_root=(FIT/'attempts').resolve()
    require(attempt_npy.is_file() and attempt_npy.parent==attempts_root,'last accepted event candidate NPY is missing or outside the fit attempts directory')
    require(sha(attempt_npy)==event.get('candidate_npy_sha256'),'last accepted event NPY hash mismatch')
    event_candidate=np.load(attempt_npy,allow_pickle=False)
    referenced_candidate=np.asarray(candidate[referenced_ids],dtype='<f4')
    require(event_candidate.dtype==np.dtype('<f4') and event_candidate.shape==referenced_candidate.shape
            and np.array_equal(event_candidate,referenced_candidate),'final candidate referenced positions differ from the last accepted compact solver event')
    candidate_bytes=referenced_candidate.tobytes(order='C')
    require(hashlib.sha256(candidate_bytes).hexdigest()==event.get('source_positions_f32_sha256'),
            'last accepted event Float32 source digest differs from final candidate')
    audit_path=Path(event.get('target_audits_path','')).resolve()
    require(audit_path.is_file() and sha(audit_path)==event.get('target_audits_sha256'),
            'last accepted event exact target-audit receipt is missing or changed')
    pose_audits=json.loads(audit_path.read_text())
    require(len(pose_audits)==3,'last accepted event does not bind three pose audits')
    require(event.get('closed_target_interior_counts_candidate_by_pose')==[0,0,0],
            'last accepted event does not show zero closed-target interior vertices at all fit poses')
    key_strings={f'{int(a)}:{int(b)}' for a,b in keys}
    key_tuples={tuple(map(int,k)) for k in keys}
    ocular={tuple(map(int,k)) for k in getattr(runner,'OCULAR',set())}
    require(ocular.issubset(key_tuples),'1172 ocular set includes a surface outside the target inventory')
    nonocular_tuples=key_tuples-ocular
    require(nonocular_tuples|ocular==key_tuples and not (nonocular_tuples&ocular),'1172 ocular/nonocular partition is inconsistent')
    nonocular_strings={f'{a}:{b}' for a,b in nonocular_tuples}
    ocular_strings={f'{a}:{b}' for a,b in ocular}
    event_nonocular=[]; event_ocular=[]
    for pose_audit in pose_audits:
        require(set(pose_audit)==key_strings,'last accepted event omits or adds a target surface')
        require(all(int(pose_audit[k].get('count',-1))==0 and not pose_audit[k].get('degenerate_face_rows')
                    for k in key_strings), 'last accepted event target audit contains a pair or degenerate target')
        event_nonocular.append(sum(int(pose_audit[k].get('count',-1)) for k in nonocular_strings))
        event_ocular.append(sum(int(pose_audit[k].get('count',-1)) for k in ocular_strings))
    require(event_nonocular==[0,0,0] and event_ocular==[0,0,0],
            'last accepted event exact target audits do not show zero nonocular and ocular pairs')
    return {
        'solve_result':pin(solve_path),
        'attempt_log':pin(attempts_path),
        'accepted_event_index':index,
        'accepted_event_attempt':event.get('attempt'),
        'accepted_event_iteration':event.get('iteration'),
        'accepted_event_status':event.get('status'),
        'accepted_event_source_npy':pin(attempt_npy),
        'accepted_event_target_audits':pin(audit_path),
        'accepted_event_final_nonocular_pairs_by_pose':event_nonocular,
        'accepted_event_final_ocular_pairs_by_pose':event_ocular,
        'shared_owner_self_gate':'The pinned shared solver rejects any trial with a nonzero exact nonadjacent skin self-intersection count before emitting an accepted event; no geometry_quality_by_margin field is expected in this shared-solver report.',
    }

def validate_1218_final_fit_proofs(report, preflight, fit_event):
    post_path=FIT/'input-postcheck.json'
    require(post_path.is_file(),'fit input-postcheck is missing')
    post=readj(post_path)
    require(post.get('status')=='unchanged' and post.get('mismatches')==[]
            and int(post.get('input_count',-1))==len(preflight.get('input_pins',[])),
            'fit input-postcheck does not prove all preflight inputs unchanged')
    proof_path=Path(report['differential_audit_proof_path']).resolve()
    require(proof_path==(FIT/'differential-scan-proof.jsonl').resolve()
            and proof_path.is_file() and sha(proof_path)==report.get('differential_audit_proof_sha256'),
            'fit report does not bind the differential scan proof')
    proof=[json.loads(line) for line in proof_path.read_text().splitlines() if line.strip()]
    require(len(proof)>=3,'differential proof has fewer than three fit-pose records')
    final_proof=proof[-3:]
    require([int(x.get('accepted_step',-1)) for x in final_proof]==list(FIT_STEPS)
            and [int(x.get('pose_index',-1)) for x in final_proof]==[0,1,2],
            'final differential proof rows do not correspond to all three declared fit poses')
    verified_by_pose={}
    for row in proof:
        if row.get('full_scan_verified') is True:
            pose=int(row.get('pose_index',-1))
            verified_by_pose.setdefault(pose,[]).append(row)
    for pose,step in enumerate(FIT_STEPS):
        matches=[x for x in verified_by_pose.get(pose,[]) if int(x.get('accepted_step',-1))==step]
        require(matches and all(x.get('complete_target_count')==859
                and int(x.get('targets_narrowphase_scanned',-1))+int(x.get('targets_with_disjoint_F32_bounds_or_no_changed_faces',-1))==859
                and not x.get('full_scan_exact_pair_mismatches') for x in matches),
                'differential proof lacks an independent exact full-859 comparison for fit pose '+str(pose))
    for row in final_proof:
        require(row.get('complete_target_count')==859 and row.get('fresh_pairs')==0
                and row.get('inherited_pairs')==0 and row.get('unchanged_faces_byte_exact') is True
                and int(row.get('targets_narrowphase_scanned',-1))+int(row.get('targets_with_disjoint_F32_bounds_or_no_changed_faces',-1))==859,
                'final differential row lacks complete 859-target coverage at pose '+str(row.get('pose_index')))
    return {
        'input_postcheck':pin(post_path),
        'differential_proof':pin(proof_path),
        'differential_full_scan_verified_pose_indices':[0,1,2],
        'final_differential_rows':[{'pose_index':int(x['pose_index']),'accepted_step':int(x['accepted_step']),
            'complete_target_count':int(x['complete_target_count']),'fresh_pairs':int(x['fresh_pairs']),
            'inherited_pairs':int(x['inherited_pairs']),'full_scan_parity_proven_earlier_for_pose':any(
                r.get('pose_index')==x.get('pose_index') and r.get('accepted_step')==x.get('accepted_step')
                and r.get('full_scan_verified') is True for r in proof)} for x in final_proof],
        'final_event_target_audits':fit_event['accepted_event_target_audits'],
        'final_event_nonocular_pairs_by_pose':fit_event['accepted_event_final_nonocular_pairs_by_pose'],
        'final_event_ocular_pairs_by_pose':fit_event['accepted_event_final_ocular_pairs_by_pose']
    }

def baseline_audit_rows(preflight):
    rows={int(x['step']):x for x in preflight['audits']}
    require(set(rows)=={0,4991,5375,5759,6111,6495,7743,10000,20000},'baseline preflight does not enumerate all nine accepted states')
    checks=[]; pins=[]
    for step in HELDOUT:
        row=rows[step]
        require(int(row['count'])==0,'retained baseline target crossings are not zero at '+str(step))
        result=readj(row['result']['path']); targets=readj(row['target_results']['path'])
        require(result.get('full_skin_self_audit_performed') is True and int(result.get('skin_self_unallowed_intersection_count',-1))==0,'retained baseline self audit failed at '+str(step))
        require(len(targets)==859 and all(int(x.get('unallowed_intersections',-1))==0 for x in targets),'retained baseline 859-target audit failed at '+str(step))
        for field in ('result','target_results'):
            q=row[field]; require(sha(q['path'])==q['sha256'],'baseline heldout audit hash drift '+str(q['path'])); pins.append(pin(q['path']))
        checks.append({'step':step,'baseline_target_unallowed_pairs':0,'baseline_skin_self_unallowed_pairs':0,'target_count':len(targets)})
    return checks,pins

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit-report',type=Path,default=FIT/'candidate-report.json')
    ap.add_argument('--candidate-npy',type=Path,default=FIT/'candidate-source-positions-f32.npy')
    ap.add_argument('--out',type=Path,default=OUT_DEFAULT)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-check',action='store_true',help='validate static pins only; no candidate scans')
    mode.add_argument('--candidate-check',action='store_true',help='validate completed fit, final differential proof, and inputs; no pose forwarding or intersection scans')
    mode.add_argument('--scan',action='store_true',help='run the six retained heldout-pose exact audits')
    args=ap.parse_args(); out=args.out.expanduser().resolve()
    require(out==OUT_DEFAULT.resolve(),'output must use the designated fresh sibling path')
    require(not out.exists(),'refuse existing heldout output '+str(out))
    report_path=args.fit_report.resolve(); candidate_path=args.candidate_npy.resolve()
    require(report_path== (FIT/'candidate-report.json').resolve() and candidate_path==(FIT/'candidate-source-positions-f32.npy').resolve(),'candidate paths must be the designated fit-attempt-001 outputs')
    require(OWNER_REVISION.is_file() and sha(OWNER_REVISION)==OWNER_REVISION_SHA,'frozen1216 source revision record missing or hash drift')
    for path, expected_sha, label in ((FORWARD,FORWARD_SHA,'915 forward model'),(WRAPPER,WRAPPER_SHA,'1193 wrapper'),(CLEARANCE,CLEARANCE_SHA,'frozen1216 clearance owner'),(CARDIAC,CARDIAC_SHA,'frozen1216 intersection owner'),(RUNNER,RUNNER_SHA,'1172 predicate runner')):
        require(path.is_file() and not path.is_symlink() and sha(path)==expected_sha,label+' missing or hash drift')
    report=readj(report_path); pre_path=Path(report['preflight_path']).resolve(); pre=readj(pre_path)
    require(pre_path==(FIT/'preflight.json').resolve(),'fit preflight path differs from attempt001')
    require(report.get('accepted_pose_steps')==list(FIT_STEPS) and report.get('fit_pose_steps')==list(FIT_STEPS)
            and report.get('held_out_pose_steps')==list(HELDOUT),'fit report does not declare exact 0/9983/20000 fit and six-heldout split')
    require(pre.get('fit_steps')==list(FIT_STEPS) and pre.get('held_out_steps')==list(HELDOUT),
            'fit preflight does not bind the exact fit/held-out pose split')
    require(pre.get('explicit_outer_envelope_keys')==['51005:63','51005:64']
            and pre.get('selected_closed_target_face_counts')=={'51005:63':6688,'51005:64':6690},
            'fit preflight does not bind both complete explicit outer-envelope targets')
    require(pre.get('solver_parameters')=={'selected_margin_mm':0.25,'support_radius_edge_multiple':4.0,'max_iterations':12,'backtrack_count':7},
            'fit preflight solver parameters differ from the declared 0.25 mm/4x bounded fit')
    selected=report.get('selected_closed_target_interior',{})
    require(selected.get('target_keys')==['51005:63','51005:64']
            and selected.get('explicit_outer_envelope_keys')==['51005:63','51005:64']
            and selected.get('final_inside_vertex_counts_by_pose')==[0,0,0],
            'fit report does not prove zero final inside vertices for both selected outer envelopes')
    require(report.get('differential_full_scan_verified_pose_indices')==[0,1,2],
            'fit report does not bind full differential verification for all three fit poses')
    require(float(report.get('selected_margin_mm',-1))==0.25
            and float(report.get('support_radius_edge_multiple',-1))==4.0
            and int(report.get('fixed_support_vertex_count',-1))==32
            and int(report.get('preserved_source_anchor_vertex_count',-1))==15,
            'fit report does not preserve the declared margin, support radius, 32 support vertices, and 15 anchors')
    require(report.get('status')=='inferred_engineering_clearance_candidate_pending_native_replay','fit candidate did not complete with the expected source-only status')
    require(report.get('final_nonocular_pair_count_by_pose')==[0,0,0],'fit candidate does not clear all three fit-pose nonocular scans')
    require(report.get('initial_ocular_pair_count_by_pose')==[0,0,0] and report.get('final_ocular_pair_count_by_pose')==[0,0,0],'fit candidate does not preserve the zero ocular pair sets at all three fit poses')
    require(report.get('runtime_asset_status')=='not composed or admitted; holdout six-pose audit and native replay pending','fit report status does not establish an unadmitted source-only candidate')
    require(report.get('preflight_path')==str(pre_path),'candidate report does not bind its preflight')
    require(pre_path==(FIT/'preflight.json').resolve(),'fit preflight path changed')
    candidate= np.load(candidate_path,allow_pickle=False)
    require(candidate.dtype==np.dtype('<f4') and candidate.shape==(54949,3) and np.isfinite(candidate).all(),'candidate NPY type/shape/finiteness mismatch')
    raw_candidate=np.asarray(candidate,dtype='<f4').tobytes(order='C')
    require(hashlib.sha256(raw_candidate).hexdigest()==report.get('candidate_source_positions_f32_sha256'),'candidate Float32 content hash differs from report')
    require(sha(candidate_path)==report.get('candidate_source_positions_npy_sha256'),'candidate NPY file hash differs from report')
    require(report.get('candidate_source_positions_npy')==str(candidate_path),'candidate NPY path differs from report')
    base_pref=readj(BASE_PREF); require(sha(BASE_PREF)==BASE_PREF_SHA and base_pref.get('all_nine_zero_delta_positions_bit_exact') is True,'zero-delta baseline preflight missing/stale')
    baseline_checks,baseline_audit_pins=baseline_audit_rows(base_pref)
    skin=load_skin(SKIN); ref=np.unique(skin['faces'])
    require((skin['nv'],skin['nb'],len(skin['faces']))==(54949,86,109211),'base skin inventory changed')
    start_path=FIT/'starting-source-positions-f32.npy'; starting=np.load(start_path,allow_pickle=False)
    require(starting.dtype==np.dtype('<f4') and starting.shape==candidate.shape,'fit start source array mismatch')
    require(np.array_equal(starting,np.asarray(skin['pos'],dtype='<f4')),'fit starting coordinates are not the exact base NHSKIN positions')
    unreferenced=np.ones(len(starting),dtype=bool); unreferenced[ref]=False
    require(np.array_equal(candidate[unreferenced],starting[unreferenced]),'candidate changed unreferenced source vertices')
    fixed=np.asarray(pre['fixed_support_source_vertex_ids'],dtype=np.int64)
    anchors=np.asarray(pre['preserved_source_anchor_vertex_ids'],dtype=np.int64)
    require(len(fixed)==32 and len(set(map(int,fixed)))==32 and np.all(np.isin(fixed,ref)),'fit fixed support witness set invalid')
    require(np.all(np.isin(anchors,ref)),'fit preserved anchors invalid')
    require(np.array_equal(candidate[fixed],starting[fixed]),'candidate changed fixed support witness source coordinates')
    require(np.array_equal(candidate[anchors],starting[anchors]),'candidate changed preserved thorax anchor coordinates')
    # Recheck fit inputs, candidate files, static package, heldout packs/receipts, and exact baseline audit evidence.
    require(Path(clearance_current.__file__).resolve()==CLEARANCE.resolve()
            and sha(clearance_current.__file__)==CLEARANCE_SHA,
            'loaded clearance module does not come from the frozen1216 source snapshot')
    wrapper=load_wrapper(); static=wrapper.validate_static_sources(); runner=wrapper.load_runner()
    require(Path(wrapper.RUNNER).resolve()==RUNNER.resolve() and wrapper.RUNNER_SHA==RUNNER_SHA,'1193 wrapper does not bind the declared 1172 runner')
    require(Path(runner.__file__).resolve()==RUNNER.resolve() and sha(runner.__file__)==wrapper.RUNNER_SHA==RUNNER_SHA,'loaded 1172 runner module identity is absent or invalid')
    ci,clearance_pred,validator,core=runner.load_predicates(); keys,expected=runner.inventory()
    require(sha(INDEX_PROOF)==INDEX_PROOF_SHA and sha(INDEX_EQ)==INDEX_EQ_SHA,'prepared-index comparison changed')
    index_proof=readj(INDEX_PROOF)
    require(index_proof['legacy_owner']['sha256']==sha(Path(ci.__file__)) and index_proof['prepared_owner']['sha256']==sha(Path(prepared_ci.__file__)),'index/narrow owner mismatch')
    require(Path(prepared_ci.__file__).resolve()==CARDIAC.resolve(),'prepared owner loaded from wrong source')

    require(len(keys)==859,'1172 target inventory not exactly 859')
    require(sha(FORWARD)==FORWARD_SHA,'915 forward source hash drift')
    fit_pin_map={str(Path(x['path']).resolve()):x['sha256'] for x in pre.get('input_pins',[]) if isinstance(x,dict) and 'path' in x}
    require(fit_pin_map,'fit preflight has no source input pins')
    for p,h in fit_pin_map.items(): require(sha(p)==h,'fit preflight input drift '+p)
    initial_pack=RUN/'accepted-geometry/step-0.mrvpack'; initial_receipt=initial_pack.with_suffix('.receipt.json'); initial=readj(initial_receipt)
    fit_event=validate_final_accepted_trial(candidate,ref,report,runner,keys)
    fit_proof=validate_1218_final_fit_proofs(report,pre,fit_event)
    if args.prepare_check:
        holdout_pins=[]
        for step in HELDOUT:
            pack=RUN/'accepted-geometry'/f'step-{step}.mrvpack'; rec=pack.with_suffix('.receipt.json')
            accepted=runner.verify_pack(step,pack,rec,validator,wrapper.NHA_SHA)
            holdout_pins.extend([pin(pack),pin(rec)])
        print(json.dumps({'status':'prepared_only_no_forward_or_scan','fit_candidate_report':pin(report_path),'candidate_npy':pin(candidate_path),'fit_preflight':pin(pre_path),'starting_source_npy':pin(start_path),'baseline_preflight':pin(BASE_PREF),'heldout_steps':list(HELDOUT),'baseline_heldout_checks':baseline_checks,'final_accepted_solver_event':fit_event,'target_count':len(keys),'1172_predicate_runner':pin(RUNNER),'915_forward_model':pin(FORWARD),'1193_wrapper':pin(WRAPPER),'fit_clearance_owner':pin(CLEARANCE),'fit_intersection_owner':pin(CARDIAC),'heldout_audit_script':pin(Path(__file__)),'heldout_pack_receipt_pins':holdout_pins,'output_path':str(out)},sort_keys=True),flush=True)
        return 0
    if args.candidate_check:
        check={'status':'candidate_validation_pass_no_heldout_scan','fit_report':pin(report_path),
               'candidate_npy':pin(candidate_path),'fit_preflight':pin(pre_path),
               'frozen_owner_source_revision':pin(OWNER_REVISION),
               'frozen_clearance_owner':pin(CLEARANCE),'frozen_intersection_owner':pin(CARDIAC),
               'fit_proof_validation':fit_proof,'fit_steps':list(FIT_STEPS),
               'heldout_steps':list(HELDOUT),'closed_target_keys':selected['target_keys'],
               'explicit_outer_envelope_keys':selected['explicit_outer_envelope_keys'],
               'closed_target_final_inside_counts_by_pose':selected['final_inside_vertex_counts_by_pose'],
               'final_nonocular_pairs_by_pose':report['final_nonocular_pair_count_by_pose'],
               'final_ocular_pairs_by_pose':report['final_ocular_pair_count_by_pose'],
               'final_accepted_solver_event':fit_event,'target_count':len(keys),
               'interpretation':'Full candidate/report/NPY/input/differential validation only; no forward reconstruction or held-out intersection scan was run.'}
        print(json.dumps(check,sort_keys=True),flush=True)
        return 0
    # Scan-time tracked inputs are collected before any output writes.
    tracked={}
    paths=[INDEX_PROOF,INDEX_EQ,Path(__file__),report_path,candidate_path,pre_path,BASE_PREF,start_path,SKIN,NHA,FORWARD,WRAPPER,OWNER_REVISION,FIT/'solve-result.json',FIT/'attempts.jsonl',Path(fit_event['accepted_event_source_npy']['path']),Path(fit_event['accepted_event_target_audits']['path'])]
    paths.extend(Path(x) for x in static)
    paths.extend(Path(p) for p in fit_pin_map)
    paths.extend(Path(x['path']) for x in baseline_audit_pins)
    for step in HELDOUT:
        paths.extend([RUN/'accepted-geometry'/f'step-{step}.mrvpack',RUN/'accepted-geometry'/f'step-{step}.receipt.json'])
    for p in paths: tracked[str(Path(p).resolve())]=sha(p)
    out.mkdir()
    decl={'schema':'numi.human.skin-resting-multipose-heldout-six-pose-audit.declaration.v1','status':'running_exact_six_pose_holdout_audit','candidate_fit_report':pin(report_path),'candidate_source_positions_npy':pin(candidate_path),'candidate_source_positions_f32_sha256':report['candidate_source_positions_f32_sha256'],'candidate_source_positions_qualification':report['qualification'],'fit_preflight':pin(pre_path),'baseline_zero_delta_preflight':pin(BASE_PREF),'base_skin':pin(SKIN),'NHA':pin(NHA),'forward_model_915':pin(FORWARD),'prepared_index_owner':pin(CARDIAC),'prepared_index_narrow_predicate_proof':pin(INDEX_PROOF),'prepared_index_pair_equivalence':pin(INDEX_EQ),'registered_1172_predicate_runner':str(RUNNER),'registered_1172_predicate_runner_sha256':RUNNER_SHA,'steps':list(HELDOUT),'target_count':len(keys),'target_inventory':pin(runner.INV),'baseline_heldout_checks':baseline_checks,'final_accepted_solver_event':fit_event,'inputs_before_scan':tracked,'scope':'Source-position candidate forward-mapped into six held-out accepted 1191 states only. Exact Float32-lattice triangle intersections against all 859 surfaces, full skin self intersections, and degenerate triangles using unchanged exact 1172 narrow predicates and the existing current-owner prepared index. No fitting, native execution, contact exemptions, or between-frame claim.'}
    (out/'declaration.json').write_text(json.dumps(decl,indent=2,sort_keys=True)+'\n')
    fwd_spec=importlib.util.spec_from_file_location('_forward_model_915_for_six_holdouts',FORWARD); fm=importlib.util.module_from_spec(fwd_spec); sys.modules[fwd_spec.name]=fm; fwd_spec.loader.exec_module(fm)
    captures=[]; state_receipts=[]; pack_data=[]; initial=readj(initial_receipt)
    for step in HELDOUT:
        pack=RUN/'accepted-geometry'/f'step-{step}.mrvpack'; rec=pack.with_suffix('.receipt.json'); rd=readj(rec)
        accepted=runner.verify_pack(step,pack,rec,validator,wrapper.NHA_SHA)
        if (rd['skin_source_mapping']['vertex_map']['sha256'],rd['skin_source_mapping']['anatomy_parameters']['sha256'])!=(initial['skin_source_mapping']['vertex_map']['sha256'],initial['skin_source_mapping']['anatomy_parameters']['sha256']):
            raise RuntimeError('heldout runtime maps differ at '+str(step))
        positions,surfaces,pack_counts=clearance_pred._pack_surfaces(pack,set(keys))
        runner.validate_pack_surface_keys(surfaces,keys)
        skin_faces=surfaces[runner.EXPECTED_SKIN_KEY]['faces']; skin_base=int(skin_faces.min())
        if len(skin_faces)!=len(skin['faces']) or not np.array_equal(skin_faces-skin_base,skin['faces']): raise RuntimeError('heldout skin topology mismatch at '+str(step))
        captures.append(positions[skin_base:skin_base+skin['nv']].astype('<f4').astype(float)[ref].copy())
        state_receipts.append({'step':step,'body_poses':rd['accepted_registered_body_poses'],'respiratory_motion':rd['accepted_respiratory_motion']})
        pack_data.append({'step':step,'pack':pack,'receipt':rec})
        del positions,surfaces
        print(json.dumps({'phase':'heldout_capture_loaded','step':step,'pack_sha256':tracked[str(pack.resolve())]}),flush=True)
    captures=np.asarray(captures,dtype='<f4').astype(float)
    forward,map_report,selectors=fm.build_forward(skin=skin,source_positions=skin['pos'],referenced_ids=ref,captured_by_pose=captures,state_receipts=state_receipts,initial_body_poses=initial['initial_anatomical_registration']['body_poses'],map_receipt=initial,anatomy_parameters_path=initial['skin_source_mapping']['anatomy_parameters']['path'],respiration_source=Path('/Users/n/numi-human-resting-resp-source-20261006'),clearance_module=clearance_current)
    fwd=forward(np.asarray(candidate,dtype='<f4'))
    world=np.asarray(fwd.get('world_positions_by_pose'),dtype='<f4')
    if fwd.get('diagnostics',{}).get('admissible') is not True or world.shape!=(len(HELDOUT),len(ref),3) or not np.isfinite(world).all(): raise RuntimeError('candidate 915 heldout forward was not admissible/finite')
    result_rows=[]; ocular=set(runner.OCULAR)
    for pose_index,item in enumerate(pack_data):
        step=item['step']; positions,surfaces,pack_counts=clearance_pred._pack_surfaces(item['pack'],set(keys)); runner.validate_pack_surface_keys(surfaces,keys)
        skin_faces=surfaces[runner.EXPECTED_SKIN_KEY]['faces']; skin_base=int(skin_faces.min())
        if len(skin_faces)!=len(skin['faces']) or not np.array_equal(skin_faces-skin_base,skin['faces']): raise RuntimeError('heldout skin topology mismatch during scan at '+str(step))
        positions=positions.copy(); positions[skin_base+ref]=world[pose_index]
        skin_ids=np.unique(skin_faces); skin_local=np.searchsorted(skin_ids,skin_faces)
        skin_records,skin_record_rows,skin_degenerate=core.exact_records(positions[skin_ids],skin_local,ci)
        skin_index=prepared_ci._prepare_surface_aabb(skin_records)
        target_path=out/f'step-{step}.targets.jsonl'; cross_path=out/f'step-{step}.crossing-witnesses.jsonl'; self_path=out/f'step-{step}.self-witnesses.jsonl'; invalid_path=out/f'step-{step}.invalid-triangles.jsonl'
        totals={'target_count':0,'all_pairs':0,'ocular_pairs':0,'nonocular_pairs':0,'invalid_target_surfaces':0,'invalid_target_triangles':0}
        hit_rows=[]
        with target_path.open('x') as ts, cross_path.open('x') as xs, self_path.open('x') as ss, invalid_path.open('x') as ins:
            for face_row in skin_degenerate:
                ids=[int(v) for v in skin_faces[face_row]]
                ins.write(json.dumps({'role':'skin_degenerate_face','skin_source_face_row':int(face_row),'pack_vertex_ids':ids,'triangle_xyz_f32_m':positions[ids].astype(float).tolist()},separators=(',',':'),allow_nan=False)+'\n')
            for ordinal,key in enumerate(keys,1):
                target_faces=surfaces[key]['faces']; target_ids=np.unique(target_faces); target_local=np.searchsorted(target_ids,target_faces)
                target_records,target_record_rows,target_degenerate=core.exact_records(positions[target_ids],target_local,ci)
                audit=prepared_ci._audit_pair_prepared_first(skin_index,target_records,same_surface=False); count=int(audit['count'])
                totals['target_count']+=1; totals['all_pairs']+=count
                if key in ocular: totals['ocular_pairs']+=count
                else: totals['nonocular_pairs']+=count
                if target_degenerate:
                    totals['invalid_target_surfaces']+=1; totals['invalid_target_triangles']+=len(target_degenerate)
                    for face_row in target_degenerate:
                        ids=[int(v) for v in target_faces[face_row]]
                        ins.write(json.dumps({'role':'target_degenerate_face','target_surface':list(key),'target_surface_face_row':int(face_row),'pack_vertex_ids':ids,'triangle_xyz_f32_m':positions[ids].astype(float).tolist()},separators=(',',':'),allow_nan=False)+'\n')
                ts.write(json.dumps({'surface':[key[0],key[1]],'source_owner_or_label':expected[key]['source_owner_or_label'],'face_count':int(len(target_faces)),'pinned_890_face_count':int(expected[key]['face_count_other']),'face_count_delta_from_890':int(len(target_faces))-int(expected[key]['face_count_other']),'aabb_candidate_pairs':int(audit['aabb_candidate_pairs']),'intersecting_triangle_pairs':count,'degenerate_face_rows':[int(x) for x in target_degenerate],'pair_coverage_complete':not bool(target_degenerate or skin_degenerate),'ocular_monitor':key in ocular},separators=(',',':'),allow_nan=False)+'\n')
                if count:
                    hit_rows.append({'surface':list(key),'pairs':count})
                    for skin_i,target_i in audit['triangle_pairs']:
                        core.crossing_witness(xs,semantic=key[0],stable_id=key[1],skin_pair_index=skin_i,target_pair_index=target_i,skin_record=skin_records[skin_i],target_record=target_records[target_i],skin_row=skin_record_rows[skin_i],target_row=target_record_rows[target_i],skin_faces=skin_faces,target_faces=target_faces,positions=positions,ci=ci)
                if ordinal%100==0 or ordinal==len(keys): print(json.dumps({'phase':'heldout_target_scan','step':step,'targets_done':ordinal,'targets_total':len(keys),'pairs':totals['all_pairs']}),flush=True)
                del target_records,target_record_rows,target_degenerate,audit,target_ids,target_local
            self_audit=ci._audit_pair(skin_records,skin_records,same_surface=True)
            for skin_i,other_i in self_audit['triangle_pairs']:
                core.crossing_witness(ss,semantic=runner.EXPECTED_SKIN_KEY[0],stable_id=runner.EXPECTED_SKIN_KEY[1],skin_pair_index=skin_i,target_pair_index=other_i,skin_record=skin_records[skin_i],target_record=skin_records[other_i],skin_row=skin_record_rows[skin_i],target_row=skin_record_rows[other_i],skin_faces=skin_faces,target_faces=skin_faces,positions=positions,ci=ci,role='skin_self_intersection')
        complete=(totals['target_count']==859 and not skin_degenerate and totals['invalid_target_triangles']==0)
        row={'step':step,'status':'complete_pair_coverage' if complete else 'incomplete_degenerate_input_fail_closed','all_skin_crossing_pair_count':totals['all_pairs'],'nonocular_crossing_pair_count':totals['nonocular_pairs'],'ocular_crossing_pair_count':totals['ocular_pairs'],'surface_target_count':totals['target_count'],'skin_self_crossing_pair_count':int(self_audit['count']),'skin_degenerate_face_rows':[int(x) for x in skin_degenerate],'invalid_target_surface_count':totals['invalid_target_surfaces'],'invalid_target_triangle_count':totals['invalid_target_triangles'],'pair_coverage_complete':complete,'hit_surfaces':hit_rows,'targets_sha256':sha(target_path),'crossing_witnesses_sha256':sha(cross_path),'self_witnesses_sha256':sha(self_path),'invalid_triangles_sha256':sha(invalid_path),'predicate':'exact_float32_lattice_triangle_intersection','contact_exemptions':[]}
        result_rows.append(row); (out/f'step-{step}.result.json').write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'phase':'heldout_pose_complete',**{k:row[k] for k in ('step','status','all_skin_crossing_pair_count','nonocular_crossing_pair_count','ocular_crossing_pair_count','skin_self_crossing_pair_count','pair_coverage_complete')}},sort_keys=True),flush=True)
        del positions,surfaces,skin_index,skin_records,skin_record_rows,self_audit; gc.collect()
    after={p:sha(p) for p in tracked}
    unchanged=tracked==after
    complete=(len(result_rows)==len(HELDOUT) and all(r['pair_coverage_complete'] for r in result_rows) and unchanged)
    summary={'schema':'numi.human.skin-resting-multipose-heldout-six-pose-audit.summary.v1','status':'complete_exact_intersection_free' if complete and all(r['all_skin_crossing_pair_count']==0 and r['skin_self_crossing_pair_count']==0 for r in result_rows) else ('complete_with_intersections_or_invalid_geometry' if len(result_rows)==len(HELDOUT) and unchanged else 'incomplete_or_input_changed'),'candidate_fit_report':str(report_path),'candidate_fit_report_sha256':sha(report_path),'candidate_source_positions_npy':str(candidate_path),'candidate_source_positions_npy_sha256':sha(candidate_path),'candidate_source_positions_f32_sha256':report['candidate_source_positions_f32_sha256'],'steps':list(HELDOUT),'target_surface_count':859,'pair_coverage_complete':len(result_rows)==len(HELDOUT) and all(r['pair_coverage_complete'] for r in result_rows),'all_steps_intersection_free':all(r['all_skin_crossing_pair_count']==0 and r['skin_self_crossing_pair_count']==0 for r in result_rows),'inputs_unchanged':unchanged,'inputs_before':tracked,'inputs_after':after,'prepared_index_owner':pin(CARDIAC),'prepared_index_narrow_predicate_proof':pin(INDEX_PROOF),'prepared_index_pair_equivalence':pin(INDEX_EQ),'forward_model_report':map_report,'baseline_heldout_checks':baseline_checks,'pose_results':result_rows,'qualification':'Six held-out accepted 1191 poses only; source-position forward and exact geometry predicates. Not native asset admission, physics qualification, continuous-time assurance, or physiological acceptance.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':summary['status'],'steps':list(HELDOUT),'pair_coverage_complete':summary['pair_coverage_complete'],'all_steps_intersection_free':summary['all_steps_intersection_free'],'inputs_unchanged':unchanged,'summary':str(out/'summary.json'),'sha256':sha(out/'summary.json')},sort_keys=True),flush=True)
    return 0 if summary['pair_coverage_complete'] and unchanged else 2
if __name__=='__main__': raise SystemExit(main())
