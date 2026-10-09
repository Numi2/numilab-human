#!/usr/bin/env python3
"""Exact native changed-star skin clearance audit; untouched skin-target pairs transfer only by exact geometry identity."""
import argparse, csv, gzip, hashlib, importlib.util, json, mmap, os, resource, struct, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

E = Path('/Users/n/numi-human-resting-evidence-20261005')
CAND_RUN = Path('/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/native-run')
BASE_RUN = Path('/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/skin-927-lung-1178-viewer-018-v015-attempt3/native-run')
CAND_SKIN = E/'native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin'
BASE_SKIN = E/'native-common-skin-multipose-clearance-candidate-927/asset-candidate-001/bodyparts3d-myosim-skinned-shell.nhskin'
CAND_REPORT = E/'native-skin-epl143-clearance-1187/attempt-006/report.json'
IDENTITY = Path('/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/geometry-identity-v2/non_skin_identity-v2.json')
BASE_SUMMARY = E/'final-native-scene-preflight-936/skin-927-lung-1178-viewer-018-v015-attempt3/lung-geometry-937-full/summary.json'
INV = E/'native-complete-skin-containment-audit-890/pair-summary-v3.csv'
RECEIPT_HELPER = E/'cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py'
AUDIT_MODULE = E/'native-common-skin-combined-cycle-audit-908/audit_cycle.py'
ROOT = Path('/Users/n/numi-human-resting-final-integration-001')
PRED = ROOT/'src/numilab_human/cardiac_cavity_intersections.py'
CLEARANCE = ROOT/'src/numilab_human/common_atlas_skin_clearance.py'
STEPS = [0,4991,5375,5759,6111,6495,7743,10000]
EXPECTED_CAND_SKIN_SHA='b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b'
EXPECTED_BASE_SKIN_SHA='bd4bfbbf0e071e24a1bb9eea7b9cd9f34ef862ef20417b5891cd24cec1b009d1'
EXPECTED_IDENTITY_SHA='6478a22fe17f7ecb67e5e3042dcf0d46a6403ada7d18f236f7ff9c58a22ce4e9'
EXPECTED_BASE_SUMMARY_SHA='c5776d8d0d144c024a670ea65885aa4191e123a26d2920fa76ec89950afae61f'
EXPECTED_REPORT_SHA='9df248cf4cdcad92dc28a7ad5b0336c365e93ecf712e5811b8d392a11b708e69'
EXPECTED_CLEARANCE_SHA='1321c31c22e1b1dbf0062947c6767c2cfcae9aae35d29e9ea775862feed5ddd1'
EXPECTED_PRED_SHA='11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def writej(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def load_mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def skin_layout(path):
    raw=Path(path).read_bytes()
    magic,abi,bindings,nv,ni,reg_fp,archive=struct.unpack_from('<8s5I32s',raw)
    require(magic==b'NHSKIN1\0' and abi==5,'unexpected NHSKIN magic/ABI')
    vo=60+36*bindings
    io=vo+56*nv
    wo=io+4*ni
    require(wo<=len(raw),'NHSKIN sections exceed payload')
    indices=np.frombuffer(raw,dtype='<u4',count=ni,offset=io).reshape(-1,3).copy()
    records=np.frombuffer(raw,dtype='u1',count=56*nv,offset=vo).reshape(nv,56)
    return {'raw':raw,'magic':magic,'abi':abi,'bindings':bindings,'nv':nv,'ni':ni,'reg_fp':reg_fp,
            'archive':archive,'vo':vo,'io':io,'wo':wo,'indices':indices,'records':records}

def verify_skin_sources():
    cr=json.loads(CAND_REPORT.read_text())
    changed_pos=set(map(int,cr['source_frame_and_field']['changed_source_vertex_ids']))
    changed_faces=list(map(int,cr['source_frame_and_field']['changed_incident_face_rows']))
    changed_norm=set(map(int,cr['candidate_assets']['changed_normal_vertex_ids']))
    b=skin_layout(BASE_SKIN); c=skin_layout(CAND_SKIN)
    require(sha(BASE_SKIN)==EXPECTED_BASE_SKIN_SHA and sha(CAND_SKIN)==EXPECTED_CAND_SKIN_SHA,'skin payload hash changed')
    require((b['magic'],b['abi'],b['bindings'],b['nv'],b['ni'],b['reg_fp'],b['archive']) ==
            (c['magic'],c['abi'],c['bindings'],c['nv'],c['ni'],c['reg_fp'],c['archive']),'skin header/source identity changed')
    require(b['raw'][60:b['vo']]==c['raw'][60:c['vo']],'skin binding records changed')
    require(b['indices'].shape==(109211,3) and np.array_equal(b['indices'],c['indices']),'skin face indices/order changed')
    require(b['raw'][b['io']:b['wo']]==c['raw'][c['io']:c['wo']],'skin face-index bytes changed')
    require(b['raw'][b['wo']:]==c['raw'][c['wo']:] and len(b['raw'][b['wo']:])==86*b['nv']*4,'full 86-weight payload changed')
    posdiff=set(np.flatnonzero(np.any(b['records'][:,:12]!=c['records'][:,:12],axis=1)).tolist())
    normdiff=set(np.flatnonzero(np.any(b['records'][:,12:24]!=c['records'][:,12:24],axis=1)).tolist())
    require(posdiff==changed_pos,'changed source position support differs from recipe')
    require(normdiff==changed_norm,'changed normal support differs from owner report')
    require(np.array_equal(b['records'][:,24:],c['records'][:,24:]),'per-vertex weight records changed')
    expected_faces=np.flatnonzero(np.any(np.isin(b['indices'],list(changed_pos)),axis=1)).astype(int).tolist()
    require(expected_faces==changed_faces and len(changed_faces)==90,'changed-face star differs from source index support')
    return {'base_sha256':sha(BASE_SKIN),'candidate_sha256':sha(CAND_SKIN),'source_position_vertex_count':len(posdiff),
            'changed_position_vertex_ids':sorted(posdiff),'changed_normal_vertex_count':len(normdiff),
            'changed_normal_vertex_ids':sorted(normdiff),'source_face_count':len(b['indices']),
            'changed_face_count':len(changed_faces),'changed_face_rows':changed_faces,
            'bindings_bytes_equal':True,'weights_bytes_equal':True,'indices_bytes_equal':True,
            'archive_identity_equal':True}

def target_inventory():
    rows=list(csv.DictReader(INV.open()))
    keys=[(int(x['second_semantic']),int(x['second_stable_id'])) for x in rows]
    require(len(keys)==859 and len(set(keys))==859,'pinned 859-target inventory invalid')
    return rows,keys

def verify_run(run, skin_path, expected_skin_sha, identity_steps, base=False):
    meta=json.loads((run/'run-metadata.json').read_text()); inv=json.loads((run/'invocation.json').read_text())
    require(meta.get('exit_code')==0 and meta.get('source_files_changed_during_run')==[],'native run did not pass unchanged-source terminal')
    require(meta.get('argv')==inv.get('argv') and meta.get('asset_sha256')==inv.get('asset_sha256'),'invocation and owner run metadata mismatch')
    require(inv.get('asset_sha256',{}).get(str(skin_path))==expected_skin_sha,'native invocation does not bind expected skin payload')
    nhas=[(p,h) for p,h in inv.get('asset_sha256',{}).items() if p.endswith('.nhanatomy')]
    require(len(nhas)==1,'expected exactly one NHA payload in invocation')
    require(sha(nhas[0][0])==nhas[0][1],'NHA asset changed after native run')
    for step in STEPS:
        pack=run/'accepted-geometry'/f'step-{step}.mrvpack'; receipt=run/'accepted-geometry'/f'step-{step}.receipt.json'
        require(pack.is_file() and receipt.is_file(),f'missing capture files at step {step}')
        identity=identity_steps[str(step)]
        expected_pack=identity['base_pack_sha256'] if base else identity['candidate_pack_sha256']
        require(sha(pack)==expected_pack,f'identity report pack hash mismatch at step {step}')
    return {'metadata_sha256':sha(run/'run-metadata.json'),'invocation_sha256':sha(run/'invocation.json'),
            'asset_nha_path':nhas[0][0],'asset_nha_sha256':nhas[0][1],'skin_path':str(skin_path),'skin_sha256':expected_skin_sha,
            'run_path':str(run),'exit_code':meta['exit_code'],'source_files_changed_during_run':meta['source_files_changed_during_run']}

def validate_receipt(pack,receipt_path,step,rv):
    doc=json.loads(receipt_path.read_text())
    require(doc.get('accepted_pack_path')==str(pack) and doc.get('physical_endpoint')=='accepted' and doc.get('surface_audit_endpoint')=='passed',f'not an accepted endpoint at {step}')
    with pack.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        sections=rv._pack_sections(mm)
        vo=next(s[2] for s in sections if s[0]==2)
        accepted=rv.validate_accepted_receipt(pack,receipt_path,step,mm,vo,{})
    require(accepted.get('accepted_step')==step,f'receipt step mismatch at {step}')
    return accepted,doc

def witness(stream, role, step, skin_row, other_row, skin_faces, other_faces, pos, ci, key=None, rec_a=None, rec_b=None):
    skin_vids=[int(v) for v in skin_faces[skin_row]]
    other_vids=[int(v) for v in other_faces[other_row]]
    points=sorted(set(ci.triangle_intersection_points(rec_a[0],rec_b[0])))
    denom=1<<149
    stream.write(json.dumps({'role':role,'step':step,'target_surface':list(key) if key else None,
        'skin_face_row':int(skin_row),'other_face_row':int(other_row),'skin_pack_vertex_ids':skin_vids,
        'other_pack_vertex_ids':other_vids,'skin_triangle_xyz_f32_m':pos[skin_vids].astype(float).tolist(),
        'other_triangle_xyz_f32_m':pos[other_vids].astype(float).tolist(),
        'intersection_points_m':[[float(x/denom) for x in p] for p in points],
        'predicate':'existing exact Float32-lattice triangle intersection predicate'},separators=(',',':'),allow_nan=False)+'\n')

def step_scan(step,args,ctx):
    t0=time.monotonic(); c=ctx['clearance']; ci=ctx['predicate']; rv=ctx['receipt']
    cand=args.candidate_run/'accepted-geometry'/f'step-{step}.mrvpack'; cand_r=cand.with_suffix('.receipt.json')
    base=args.baseline_run/'accepted-geometry'/f'step-{step}.mrvpack'; base_r=base.with_suffix('.receipt.json')
    acc,rdoc=validate_receipt(cand,cand_r,step,rv); bacc,brdoc=validate_receipt(base,base_r,step,rv)
    require(sha(cand)==ctx['capture_hashes'][str(args.candidate_run)][str(cand)] and sha(cand_r)==ctx['capture_hashes'][str(args.candidate_run)][str(cand_r)],f'candidate capture changed after declaration at {step}')
    require(sha(base)==ctx['capture_hashes'][str(args.baseline_run)][str(base)] and sha(base_r)==ctx['capture_hashes'][str(args.baseline_run)][str(base_r)],f'baseline capture changed after declaration at {step}')
    cp,csurf,_=c._pack_surfaces(cand,set(ctx['keys']))
    bp,bsurf,_=c._pack_surfaces(base,set(ctx['keys']))
    cskin=csurf[(51007,1)]['faces']; bskin=bsurf[(51007,1)]['faces']
    require(np.array_equal(cskin,bskin),'skin face rows or pack vertex IDs changed between baseline/candidate pack')
    base_ids=ctx['skin_source']['changed_position_vertex_ids']; changed=np.asarray(ctx['skin_source']['changed_face_rows'],dtype=np.int64)
    skin_base=int(cskin.min())
    require(np.array_equal(cskin-skin_base,ctx['source_faces']),'captured skin indices differ from bound NHSKIN payload')
    require(skin_base==int(bskin.min()),'skin pack vertex base changed')
    nverts=ctx['skin_layout']['nv']
    # Candidate and baseline skin positions may differ only at the declared source position support.
    ids=np.arange(skin_base,skin_base+nverts,dtype=np.int64)
    require(ids.max()<len(cp) and ids.max()<len(bp),'skin pack source vertex range out of bounds')
    posdiff=np.flatnonzero(np.any(cp[ids]!=bp[ids],axis=1))
    require(set(map(int,posdiff)).issubset(set(base_ids)),f'candidate native skin moved outside declared 33-vertex support at step {step}')
    # All target geometry and indices must be byte-equivalent in actual accepted packs.
    target_used=set()
    for key in ctx['keys']:
        cf=csurf[key]['faces']; bf=bsurf[key]['faces']
        require(np.array_equal(cf,bf),f'target {key} face indices differ at {step}')
        if cf.size: target_used.update(map(int,np.unique(cf)))
    tu=np.asarray(sorted(target_used),dtype=np.int64)
    require(np.array_equal(cp[tu],bp[tu]),f'target surface Float32 XYZ differs at {step}')

    skin_pack_ids=np.unique(cskin)
    skin_local=np.searchsorted(skin_pack_ids,cskin)
    skin_records=ci._records([ci.float32_point_lattice_key(p) for p in cp[skin_pack_ids]],skin_local)
    # Record indices from _records preserve original face rows because all skin faces are present and nondegenerate.
    require(len(skin_records)==len(cskin),'skin has degenerate faces; fail closed')
    skin_bad=[]
    for row,idsrow in enumerate(skin_local):
        tri=tuple(ci.float32_point_lattice_key(cp[skin_pack_ids[k]]) for k in idsrow)
        if not any(ci._cross(ci._sub(tri[1],tri[0]),ci._sub(tri[2],tri[0]))): skin_bad.append(row)
    require(not skin_bad,f'candidate skin has degenerate triangles at step {step}: {skin_bad[:8]}')
    # Full skin self-intersection audit on this candidate capture.
    self_audit=ci._audit_pair(skin_records,skin_records,same_surface=True)
    changed_set=set(map(int,changed))
    step_dir=args.out/f'step-{step}'; step_dir.mkdir(exist_ok=False)
    self_path=step_dir/'skin-self-unallowed.jsonl.gz'
    with gzip.open(self_path,'wt',encoding='utf-8',compresslevel=1) as f:
        for ia,ib in self_audit['triangle_pairs']:
            ra=skin_records[ia]; rb=skin_records[ib]
            witness(f,'skin_self_unallowed_intersection',step,ra[3],rb[3],cskin,cskin,cp,ci,rec_a=ra,rec_b=rb)

    changed_records=[r for r in skin_records if int(r[3]) in changed_set]
    require(len(changed_records)==90,f'changed source face count differs at {step}')
    target_results=[]; cross_count=0; changed_target_aabb=0; invalid_targets=[]
    cross_path=step_dir/'changed-skin-target-intersections.jsonl.gz'
    with gzip.open(cross_path,'wt',encoding='utf-8',compresslevel=1) as f:
        for key in ctx['keys']:
            faces=csurf[key]['faces']; local=np.searchsorted(np.unique(faces),faces) if len(faces) else faces.copy()
            ids2=np.unique(faces)
            target_records=ci._records([ci.float32_point_lattice_key(p) for p in cp[ids2]],local) if len(faces) else []
            require(len(target_records)==len(faces),f'target {key} has degenerate triangle at step {step}')
            audit=ci._audit_pair(changed_records,target_records,same_surface=False)
            n=int(audit['count']); cross_count+=n; changed_target_aabb+=int(audit['aabb_candidate_pairs'])
            target_results.append({'surface':[int(x) for x in key],'face_count':len(faces),'changed_face_aabb_candidates':int(audit['aabb_candidate_pairs']),'unallowed_intersections':n})
            for ia,ib in audit['triangle_pairs']:
                ra=changed_records[ia]; rb=target_records[ib]
                witness(f,'changed_skin_target_intersection',step,ra[3],rb[3],cskin,faces,cp,ci,key,ra,rb)

    reqstep=acc.get('accepted_step'); reqtimes=(acc.get('accepted_time_s'),bacc.get('accepted_time_s'))
    require(reqstep==step and bacc.get('accepted_step')==step,f'accepted receipts mismatch capture step {step}')
    require(acc.get('accepted_body_state_sha256')==bacc.get('accepted_body_state_sha256') and acc.get('accepted_respiration_state_sha256')==bacc.get('accepted_respiration_state_sha256') and reqtimes[0]==reqtimes[1],f'accepted body/respiration/time state differs at {step}')
    result={'step':step,'candidate_pack_sha256':sha(cand),'candidate_receipt_sha256':sha(cand_r),
        'baseline_pack_sha256':sha(base),'baseline_receipt_sha256':sha(base_r),
        'accepted_step':reqstep,'candidate_accepted_time_s':reqtimes[0],'baseline_accepted_time_s':reqtimes[1],
        'candidate_accepted_body_state_sha256':acc.get('accepted_body_state_sha256'),
        'baseline_accepted_body_state_sha256':bacc.get('accepted_body_state_sha256'),
        'candidate_accepted_respiration_state_sha256':acc.get('accepted_respiration_state_sha256'),
        'baseline_accepted_respiration_state_sha256':bacc.get('accepted_respiration_state_sha256'),
        'candidate_skin_positions_differing_from_baseline':len(posdiff),
        'candidate_changed_face_count':len(changed_records),'full_candidate_skin_face_count':len(skin_records),
        'skin_self_unallowed_intersection_count':int(self_audit['count']),
        'skin_self_allowed_shared_vertex_or_edge_pair_count':int(self_audit['allowed_shared_vertex_or_edge_pairs']),
        'skin_self_aabb_candidates':int(self_audit['aabb_candidate_pairs']),
        'changed_star_target_surfaces_scanned':len(target_results),'target_inventory_count':len(ctx['keys']),
        'changed_star_target_aabb_candidates':changed_target_aabb,
        'changed_star_target_unallowed_intersection_count':cross_count,
        'changed_star_intersections_by_target':[x for x in target_results if x['unallowed_intersections']],
        'target_results_sha256':None,'target_results':target_results,
        'self_witnesses_sha256':sha(self_path),'cross_witnesses_sha256':sha(cross_path),
        'target_geometry_identical_baseline':True,'full_skin_self_audit_performed':True,
        'pair_completeness':'fresh full skin self audit plus fresh all-859 target scans against every changed source-face row; untouched skin-target pairs transferred only after exact accepted-state, target geometry, unchanged skin vertex, and topology identity proof',
        'elapsed_wall_seconds':time.monotonic()-t0,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    target_path=step_dir/'target-results.json'
    writej(target_path,target_results); result['target_results_sha256']=sha(target_path)
    writej(step_dir/'result.json',result)
    print(json.dumps({'step':step,'self_unallowed':result['skin_self_unallowed_intersection_count'],'self_allowed_adjacent':result['skin_self_allowed_shared_vertex_or_edge_pair_count'],'target_unallowed':cross_count,'target_surfaces':len(target_results),'elapsed_wall_seconds':result['elapsed_wall_seconds'],'rss_peak_bytes':result['rss_peak_bytes']}),flush=True)
    return result

def preflight(args):
    for p,h in [(IDENTITY,EXPECTED_IDENTITY_SHA),(BASE_SUMMARY,EXPECTED_BASE_SUMMARY_SHA),(CAND_REPORT,EXPECTED_REPORT_SHA),(CLEARANCE,EXPECTED_CLEARANCE_SHA),(PRED,EXPECTED_PRED_SHA)]:
        require(sha(p)==h,f'input hash mismatch: {p}')
    identity=json.loads(IDENTITY.read_text()); require(identity.get('status')=='pass_exact_state_and_nonskin_geometry_identity','invalid exact identity report')
    baseline=json.loads(BASE_SUMMARY.read_text()); require(baseline.get('all_requested_steps_complete') and baseline.get('intersection_free') and len(baseline.get('steps',[]))==8,'baseline full-skin audit incomplete or not clear')
    require([int(x['step']) for x in baseline['steps']]==STEPS,'baseline capture schedule mismatch')
    require(all(x['all_skin_crossing_pair_count']==0 and x['skin_self_crossing_pair_count']==0 and x['pair_coverage_complete'] for x in baseline['steps']),'baseline source audit did not prove clean untouched pairs')
    idsteps={str(x['step']):x for x in identity['steps']}; require(set(idsteps)==set(map(str,STEPS)),'identity report steps mismatch')
    require(all(x['non_skin_primitives_bitwise_xyz_equal']==860 and x['candidate_skin_source_face_rows_identical'] and x['index_section_sha256_equal'] for x in idsteps.values()),'identity report does not cover unchanged geometry/indices')
    require(all(v.get('bytes_equal') is True and v.get('all_rows_equal') is True for v in identity.get('csv_comparison',{}).values()) and len(identity.get('csv_comparison',{}))==4,'identity report does not bind all four unchanged native CSV streams')
    skin_source=verify_skin_sources(); rows,keys=target_inventory()
    require(args.out.exists() is False,'refusing to overwrite output')
    args.out.mkdir(parents=True)
    require(sha(Path(__file__).resolve())==EXPECTED_SCRIPT_SHA if 'EXPECTED_SCRIPT_SHA' in globals() else True,'script hash mismatch')
    base_bind=verify_run(args.baseline_run,BASE_SKIN,EXPECTED_BASE_SKIN_SHA,idsteps,True)
    cand_bind=verify_run(args.candidate_run,CAND_SKIN,EXPECTED_CAND_SKIN_SHA,idsteps,False)
    require(base_bind['asset_nha_sha256']==cand_bind['asset_nha_sha256'],'baseline/candidate NHA asset hash differs')
    capture_hashes={}
    for run in (args.baseline_run,args.candidate_run):
        capture_hashes[str(run)]={}
        for step in STEPS:
            for suffix in ('mrvpack','receipt.json'):
                p=run/'accepted-geometry'/f'step-{step}.{suffix}' if suffix=='mrvpack' else run/'accepted-geometry'/f'step-{step}.receipt.json'
                capture_hashes[str(run)][str(p)]=sha(p)
    rv=load_mod('receipt1172',RECEIPT_HELPER)
    sys.path.insert(0,str(ROOT/'src'))
    from numilab_human import common_atlas_skin_clearance as clearance
    from numilab_human import cardiac_cavity_intersections as predicate
    skin=skin_layout(CAND_SKIN)
    source_faces=skin['indices']
    report={'schema':'numi.human.native-skin-epl143-changed-star-differential.v1','status':'preflighted','qualification':'native captured geometry only; unchanged skin-target pairs transferred only after exact dependencies; candidate changed face rows and complete candidate skin self surface freshly audited at all eight accepted poses; no continuous-time, treatment, contact-selector, or full-body qualification',
        'candidate_run':cand_bind,'baseline_run':base_bind,'candidate_skin_sha256':sha(CAND_SKIN),'baseline_skin_sha256':sha(BASE_SKIN),
        'candidate_skin_source_verification':skin_source,'candidate_composition_report_path':str(CAND_REPORT),'candidate_composition_report_sha256':sha(CAND_REPORT),
        'candidate_skin_changed_face_rows':skin_source['changed_face_rows'],'target_inventory_path':str(INV),'target_inventory_sha256':sha(INV),'target_count':len(keys),
        'baseline_full_skin_audit_path':str(BASE_SUMMARY),'baseline_full_skin_audit_sha256':sha(BASE_SUMMARY),
        'exact_identity_report_path':str(IDENTITY),'exact_identity_report_sha256':sha(IDENTITY),
        'predicate_source_sha256':sha(PRED),'clearance_source_sha256':sha(CLEARANCE),'audit_runner_path':str(Path(__file__).resolve()),'audit_runner_sha256':sha(Path(__file__).resolve()),
        'capture_steps':STEPS,'worker_cap':2,'transfer_rule':'For unchanged candidate skin faces against unchanged target surfaces, transfer only the baseline zero witness result after exact all-vertex coordinates, topology, accepted body/resp state, and NHA identity checks. Freshly scan all 90 candidate-modified source faces against every one of 859 targets and perform full candidate skin self audit.',
        'source_hashes':{str(p):sha(p) for p in [IDENTITY,BASE_SUMMARY,CAND_REPORT,INV,RECEIPT_HELPER,AUDIT_MODULE,CLEARANCE,PRED,BASE_SKIN,CAND_SKIN]},
        'candidate_skin_invariants':skin_source,'capture_file_sha256':capture_hashes,'completed_steps':[]}
    writej(args.out/'declaration.json',report)
    ctx={'identity_steps':idsteps,'skin_source':skin_source,'keys':keys,'source_faces':source_faces,'skin_layout':skin,'clearance':clearance,'predicate':predicate,'receipt':rv,'capture_hashes':capture_hashes}
    return report,ctx

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate-run',type=Path,default=CAND_RUN); ap.add_argument('--baseline-run',type=Path,default=BASE_RUN); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--step',type=int)
    args=ap.parse_args(); require(1<=args.workers<=2,'worker count must be 1..2')
    report,ctx=preflight(args)
    if args.step is not None: results=[step_scan(args.step,args,ctx)]
    else:
        failures=[]; results=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs={pool.submit(run_one,args,ctx,s):s for s in STEPS}
            for fut in as_completed(futs):
                step=futs[fut]
                try:
                    r=fut.result(); results.append(r)
                    report['completed_steps'].append(step); writej(args.out/'declaration.json',report)
                    print(json.dumps({'completed_step':step,'completed_count':len(results),'expected_count':8}),flush=True)
                except BaseException as e:
                    failures.append({'step':step,'error':repr(e)})
                    print(json.dumps({'failed_step':step,'error':repr(e)}),flush=True)
        results.sort(key=lambda x:x['step'])
        report['steps']=results; report['status']='complete_differential_coverage' if not failures and len(results)==8 else 'failed_or_incomplete'; report['failures']=failures
        report['all_eight_steps_complete']=not failures and len(results)==8
        report['all_targets_scanned_for_changed_faces']=all(x['changed_star_target_surfaces_scanned']==859 for x in results)
        report['full_skin_self_scanned_all_steps']=all(x['full_skin_self_audit_performed'] for x in results)
        report['intersection_free']=report['all_eight_steps_complete'] and all(x['skin_self_unallowed_intersection_count']==0 and x['changed_star_target_unallowed_intersection_count']==0 for x in results)
        report['source_hashes_unchanged_after_scan']=report['source_hashes']=={p:sha(Path(p)) for p in report['source_hashes']}
        report['capture_hashes_unchanged_after_scan']=all(sha(Path(p))==h for runmap in report['capture_file_sha256'].values() for p,h in runmap.items())
        require(report['source_hashes_unchanged_after_scan'] and report['capture_hashes_unchanged_after_scan'],'one or more input/source/capture files changed during audit')
        writej(args.out/'summary.json',report)
        print(json.dumps({'status':report['status'],'all_eight_steps_complete':report['all_eight_steps_complete'],'intersection_free':report['intersection_free'],'output':str(args.out/'summary.json')}),flush=True)
        return 0 if report['all_eight_steps_complete'] else 2
    return 0

def run_one(args,ctx,step):
    return step_scan(step,args,ctx)

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print(f'exact native skin differential failed closed: {e}',file=sys.stderr); raise
