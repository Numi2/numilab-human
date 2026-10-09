#!/usr/bin/env python3
"""Diagnostic exact-predicate saved-pose screen for NHSKIN weight candidates.

Reconstructs candidate skin points as (captured baseline Float32 + the exact CPU
LBS weight-delta prediction), then runs the unchanged exact Float32-lattice
intersection predicates against all 859 registered targets and within skin.
This is not a native-candidate replay or admission.
"""
from __future__ import annotations
import csv, hashlib, importlib, importlib.util, json, mmap, os, resource, struct, sys, time, types
from pathlib import Path
import numpy as np

E = Path('/Users/n/numi-human-resting-evidence-20261005')
TRIAL = Path('/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/trials/resting-baseline')
SCENE = TRIAL / 'output/scene'
NHA = Path('/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-thorax.nhanatomy')
# The current 1170 scene is pinned to the 1159 NHA, not the P18 planning output.
NHA_SHA = 'c10dce4609be99fdc569801c2705e23c46120b610c8aa86c631165d38baf4713'
SOURCE_SKIN = E/'native-common-skin-multipose-clearance-candidate-927/asset-candidate-001/bodyparts3d-myosim-skinned-shell.nhskin'
SOURCE_SHA = 'bd4bfbbf0e071e24a1bb9eea7b9cd9f34ef862ef20417b5891cd24cec1b009d1'
BUILD = E/'native-common-skin-bone-proximity-weight-candidate-1185/candidate-build-008'
CANDIDATES = {
  'compact-2.5x-half': (BUILD/'compact-2.5x-half.nhskin','e0e9c0dbf405ec76590d1e4960014f6ec80d8ca9439d264a23a5950d0a8b26c4'),
  'compact-2.5x-full': (BUILD/'compact-2.5x-full.nhskin','7a6d47f92ac4ee153e5444c14ac9b4bfba03b278e1cb8aa7623d046a6c985117'),
  'compact-3x-full': (BUILD/'compact-3x-full.nhskin',None),
}
CANDIDATE_REPORT=BUILD/'report.json'
CANDIDATE_REPORT_SHA=''
INV=E/'native-complete-skin-containment-audit-890/pair-summary-v3.csv'
BASELINE_AUDIT=E/'native-lung-late-skin-audit-runner-1172/closed-baseline-1170/full-attempt-002'
ROOT=Path('/Users/n/numi-human-resting-final-integration-001')
PRED=Path('/Users/n/numi-human-free-apex-publication-1159/src/numilab_human/cardiac_cavity_intersections.py')
CLEARANCE=ROOT/'src/numilab_human/common_atlas_skin_clearance.py'
AUDIT_CORE=E/'native-common-skin-combined-cycle-audit-908/audit_cycle.py'
POSES=(47519,155000)
OCULAR={(51010,i) for i in range(381,398)}
OUT=E/'native-common-skin-bone-proximity-saved-pose-screen-1185/attempt-001'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_mod(path,name):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def qrotate(q,v):
    q=np.asarray(q,dtype=np.float64); v=np.asarray(v,dtype=np.float64)
    xyz=q[:3]; t=2.0*np.cross(np.broadcast_to(xyz,v.shape),v)
    return v+q[3]*t+np.cross(np.broadcast_to(xyz,v.shape),t)

def parse_skin(path):
    raw=Path(path).read_bytes()
    magic,abi,nb,nv,ni,reg,srcsha=struct.unpack_from('<8s5I32s',raw,0)
    if magic!=b'NHSKIN1\0' or abi!=5: raise ValueError(f'unsupported skin {path}: {magic!r} ABI {abi}')
    bindoff=60; voff=bindoff+36*nb; ioff=voff+56*nv; woff=ioff+4*ni
    if len(raw)-woff!=nv*nb*4: raise ValueError('full weight matrix length mismatch')
    xyz=np.ndarray((nv,3),dtype='<f4',buffer=raw,offset=voff,strides=(56,4)).copy()
    face=np.frombuffer(raw,dtype='<u4',count=ni,offset=ioff).reshape(-1,3).copy()
    weights=np.frombuffer(raw,dtype='<f4',count=nv*nb,offset=woff).reshape(nv,nb).copy()
    binds=[struct.unpack_from('<I3f4ff',raw,bindoff+36*i) for i in range(nb)]
    return {'raw':raw,'sha256':sha(path),'nb':nb,'nv':nv,'ni':ni,'xyz':xyz,'faces':face,'weights':weights,'bindings':binds}

def pose_dict(receipt):
    d=json.loads(Path(receipt).read_text())
    if d.get('physical_endpoint')!='accepted' or d.get('surface_audit_endpoint')!='passed': raise ValueError('not accepted surface-audited pose')
    return {int(x['body_index']):(np.array(x['position_m'],dtype=np.float64),np.array(x['quaternion_xyzw'],dtype=np.float64)) for x in d['accepted_registered_body_poses']},d

def moved_positions(source,candidate,changed_ids,bodyposes,pack_xyz):
    sw=source['weights'][changed_ids].astype(np.float64)
    cw=candidate['weights'][changed_ids].astype(np.float64)
    d=cw-sw
    cols=np.flatnonzero(np.any(d!=0,axis=0))
    delta=np.zeros((len(changed_ids),3),dtype=np.float64)
    body_changes={}
    for col in cols:
        b,tx,ty,tz,qx,qy,qz,qw,scale=source['bindings'][int(col)]
        if int(b) not in bodyposes: raise ValueError(f'body pose missing for binding {b}')
        bp,bq=bodyposes[int(b)]
        local=scale*qrotate((qx,qy,qz,qw),source['xyz'][changed_ids].astype(np.float64))+np.array((tx,ty,tz))
        world=qrotate(bq,local)+bp
        delta+=d[:,int(col),None]*world
        body_changes[str(int(b))]=body_changes.get(str(int(b)),0)+int(np.count_nonzero(d[:,int(col)]))
    moved=(pack_xyz[changed_ids].astype(np.float64)+delta).astype('<f4')
    baseline_pred=np.zeros((len(changed_ids),3),dtype=np.float64)
    for col,binding in enumerate(source['bindings']):
        b,tx,ty,tz,qx,qy,qz,qw,scale=binding
        w=sw[:,col]
        if not np.any(w): continue
        bp,bq=bodyposes[int(b)]
        local=scale*qrotate((qx,qy,qz,qw),source['xyz'][changed_ids].astype(np.float64))+np.array((tx,ty,tz))
        baseline_pred+=w[:,None]*(qrotate(bq,local)+bp)
    baseline_error=np.linalg.norm(pack_xyz[changed_ids].astype(np.float64)-baseline_pred,axis=1)
    return moved,delta,baseline_error,body_changes,cols.tolist()

def load_targets():
    with INV.open(newline='') as f: rows=list(csv.DictReader(f))
    keys=[(int(x['second_semantic']),int(x['second_stable_id'])) for x in rows]
    if len(keys)!=859 or len(set(keys))!=859: raise ValueError('target inventory not 859 unique surfaces')
    return keys

def baseline_pairs(step):
    path=BASELINE_AUDIT/f'step-{step}.crossing-witnesses.jsonl'
    pairs={}
    with path.open() as f:
      for line in f:
        x=json.loads(line); k=tuple(x['target_surface'])
        pairs.setdefault(k,set()).add((int(x['skin_source_face_row']),int(x['target_surface_face_row'])))
    return pairs

def receipt_for(step):
    pack=SCENE/'accepted-geometry'/f'step-{step}.mrvpack'; rec=pack.with_suffix('.receipt.json')
    d=json.loads(rec.read_text())
    if d.get('accepted_step')!=step or d.get('physical_endpoint')!='accepted' or d.get('surface_audit_endpoint')!='passed': raise ValueError(f'{step}: unaccepted pack')
    if d.get('common_field_source_anatomy_payload_sha256')!=NHA_SHA: raise ValueError(f'{step}: NHA mismatch')
    if d.get('accepted_pack_path')!=str(pack): raise ValueError(f'{step}: receipt pack path mismatch')
    return pack,rec,d

def run_one(variant, candidate_path, step, np, ci, core, clearance, source, keys, candidate_hash):
    t0=time.monotonic(); pack,receipt,receipt_doc=receipt_for(step)
    positions,surfaces,counts=clearance._pack_surfaces(pack,set(keys))
    if set(surfaces)!=set(keys)|{(51007,1)}: raise ValueError('pack target/skin set mismatch')
    skin_faces=surfaces[(51007,1)]['faces']
    base=int(skin_faces.min()); nv=source['nv']
    if int(skin_faces.max())>=base+nv or not np.array_equal(skin_faces-base,source['faces']): raise ValueError('captured skin topology/index mapping differs from exact source')
    source_pos=positions[base:base+nv].copy()
    changed=np.flatnonzero(np.any(source['weights']!=candidate['weights'],axis=1))
    bodyposes,rdoc=pose_dict(receipt)
    new_xyz,delta,base_error,body_changes,changed_cols=moved_positions(source,candidate,changed,bodyposes,source_pos)
    candidate_positions=positions.copy()
    candidate_positions[base+changed]=new_xyz
    skin_ids=np.unique(skin_faces)
    skin_local=np.searchsorted(skin_ids,skin_faces)
    skin_records,skin_record_rows,skin_degenerate=core.exact_records(candidate_positions[skin_ids],skin_local,ci)
    skin_index=ci._prepare_surface_aabb(skin_records)
    target_rows=[]; added=[]; removed=[]; total=ocular=nonocular=0; new_pair_rows=[]
    bp=baseline_pairs(step)
    for ordinal,key in enumerate(keys,1):
        faces=surfaces[key]['faces']; ids=np.unique(faces); local=np.searchsorted(ids,faces)
        recs,rec_rows,deg=core.exact_records(positions[ids],local,ci)
        if deg: raise ValueError(f'target {key} has exact-degenerate faces {len(deg)}')
        audit=ci._audit_pair_prepared_first(skin_index,recs,same_surface=False)
        pairs={(int(skin_record_rows[int(si)]),int(rec_rows[int(ti)])) for si,ti in audit['triangle_pairs']}
        old=bp.get(key,set()); add=pairs-old; rem=old-pairs
        total+=len(pairs); added+=len(add); removed+=len(rem)
        if key in OCULAR: ocular+=len(pairs)
        else: nonocular+=len(pairs)
        target_rows.append({'surface':[key[0],key[1]],'count':len(pairs),'baseline_count':len(old),'added_pair_count':len(add),'removed_pair_count':len(rem),'aabb_candidate_pairs':int(audit['aabb_candidate_pairs'])})
        if add:
            skin_record_by_row={int(row):rec for rec,row in zip(skin_records,skin_record_rows)}
            target_record_by_row={int(row):rec for rec,row in zip(recs,rec_rows)}
            for sf,tf in sorted(add):
                # Report exact candidate intersections and source/target index rows for audit.
                srec=skin_record_by_row.get(sf)
                trec=target_record_by_row.get(tf)
                if srec is None or trec is None: raise RuntimeError('face-record lineage missing')
                pts=ci.triangle_intersection_points(srec[0],trec[0])
                new_pair_rows.append({'surface':[key[0],key[1]],'skin_face_row':sf,'target_face_row':tf,'intersection_point_count':len(pts),'points_lattice':[list(map(int,p)) for p in pts]})
        if ordinal%100==0: print(json.dumps({'variant':variant,'step':step,'targets':ordinal,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
    self_audit=ci._audit_pair_prepared_first(skin_index,skin_records,same_surface=True)
    if skin_degenerate: raise ValueError(f'candidate skin has exact zero-area faces: {len(skin_degenerate)}')
    # Include exact candidate/source movement and changed fullweight columns for replay.
    out=OUT/variant; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'step-{step}-changed-xyz.npz',source_vertex_ids=changed.astype('<u4'),candidate_xyz_f32=new_xyz.astype('<f4'),delta_m=delta.astype('<f8'))
    with (out/f'step-{step}-target-summary.jsonl').open('w') as f:
        for row in target_rows: f.write(json.dumps(row,separators=(',',':'))+'\n')
    with (out/f'step-{step}-new-pairs.jsonl').open('w') as f:
        for row in new_pair_rows: f.write(json.dumps(row,separators=(',',':'))+'\n')
    result={
      'schema':'numi.human.skin-weight-candidate-saved-pose-screen.v1','status':'complete_exact_predicate_diagnostic_not_native_candidate_replay',
      'variant':variant,'candidate_skin':{'path':str(candidate_path),'sha256':candidate_hash},
      'source_skin':{'path':str(SOURCE_SKIN),'sha256':SOURCE_SKIN_SHA},'step':step,'time_s':float(rdoc.get('accepted_time_s',-1)),
      'pack':{'path':str(pack),'sha256':sha(pack),'receipt_path':str(receipt),'receipt_sha256':sha(receipt),'accepted_root_fingerprint':rdoc.get('accepted_root_fingerprint')},
      'nha':{'path':str(NHA),'sha256':NHA_SHA},'target_inventory':{'path':str(INV),'sha256':sha(INV),'count':len(keys)},
      'pose_model':'candidate positions = actual captured source-skin Float32 positions + double-precision CPU LBS displacement delta from exact candidate-minus-source full Float32 weights, rounded to Float32; this is a saved-pose prediction, not native Metal execution',
      'weight_change':{'changed_float64_source_rows':int(len(changed)),'changed_fullweight_columns':changed_cols,'changed_binding_body_ids':body_changes,
        'delta_point_count':int(len(changed)),'delta_norm_quantiles_m':np.quantile(np.linalg.norm(delta,axis=1),[0,.5,.9,.99,1]).tolist(),
        'max_source_lbs_prediction_residual_to_captured_m':float(base_error.max()),'source_lbs_residual_quantiles_m':np.quantile(base_error,[.5,.9,.99,1]).tolist()},
      'exact_predicate':'unchanged Human Float32-lattice triangle intersection predicate, strict shared-index rules; no tolerance/exemption',
      'coverage':{'target_surface_count':len(target_rows),'ocular_surface_count':len(OCULAR),'nonocular_surface_count':len(keys)-len(OCULAR),'target_degenerate_face_count':0,'skin_degenerate_face_count':0},
      'results':{'crossing_pair_count':int(total),'ocular_pair_count':int(ocular),'nonocular_pair_count':int(nonocular),
         'baseline_crossing_pair_count':int(sum(len(x) for x in bp.values())),'new_vs_baseline_pair_count':int(added),'removed_vs_baseline_pair_count':int(removed),
         'skin_self_crossing_pair_count':int(self_audit['count']),'new_pair_witnesses_path':str(out/f'step-{step}-new-pairs.jsonl'),'new_pair_witnesses_sha256':sha(out/f'step-{step}-new-pairs.jsonl'),
         'target_summary_path':str(out/f'step-{step}-target-summary.jsonl'),'target_summary_sha256':sha(out/f'step-{step}-target-summary.jsonl')},
      'candidate_changed_xyz_archive':{'path':str(out/f'step-{step}-changed-xyz.npz'),'sha256':sha(out/f'step-{step}-changed-xyz.npz')},
      'elapsed_seconds':time.monotonic()-t0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/f'step-{step}.result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result['results'][k] for k in ('crossing_pair_count','baseline_crossing_pair_count','new_vs_baseline_pair_count','removed_vs_baseline_pair_count','skin_self_crossing_pair_count')} | {'variant':variant,'step':step,'elapsed_seconds':result['elapsed_seconds'],'peak_rss_bytes':result['peak_rss_bytes']}),flush=True)
    return result

def main():
    global OUT
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT); args=ap.parse_args(); OUT=args.out
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'src'))
    clearance=importlib.import_module('numilab_human.common_atlas_skin_clearance')
    pubpkg='skin_screen_predicate_pkg'
    pkg=types.ModuleType(pubpkg); pkg.__path__=[str(PRED.parent)]; sys.modules[pubpkg]=pkg
    ci=importlib.import_module(pubpkg+'.cardiac_cavity_intersections')
    core=load_mod(AUDIT_CORE,'screen_core_908')
    source=parse_skin(SOURCE_SKIN); keys=load_targets()
    invdoc=json.loads((SCENE/'invocation.json').read_text()); argv=invdoc.get('argv',[]); asset_hashes=invdoc.get('asset_sha256',{})
    if argv[argv.index('--skin-payload')+1]!=str(SOURCE_SKIN) or asset_hashes.get(str(SOURCE_SKIN))!=SOURCE_SHA: raise ValueError('1170 invocation does not bind exact source skin')
    if argv[argv.index('--torso-anatomy-payload')+1]!=str(NHA) or asset_hashes.get(str(NHA))!=NHA_SHA: raise ValueError('1170 invocation does not bind expected NHA')
    report=json.loads(CANDIDATE_REPORT.read_text())
    if report.get('input_sha256_before_after')!=SOURCE_SHA: raise ValueError('candidate builder source identity mismatch')
    if sha(NHA)!=NHA_SHA: raise ValueError('baseline NHA pin mismatch')
    decl={'schema':'numi.human.skin-weight-candidate-saved-pose-screen.declaration.v1','trial':str(TRIAL),'scene':str(SCENE),
      'nha':{'path':str(NHA),'sha256':NHA_SHA},'source_skin':{'path':str(SOURCE_SKIN),'sha256':sha(SOURCE_SKIN)},
      'candidate_report':{'path':str(CANDIDATE_REPORT),'sha256':sha(CANDIDATE_REPORT)},'candidate_builder':report.get('code'),
      'predicate_source':{'path':str(PRED),'sha256':sha(PRED)},'audit_core':{'path':str(AUDIT_CORE),'sha256':sha(AUDIT_CORE)},
      'pack_reader':{'path':str(CLEARANCE),'sha256':sha(CLEARANCE)},'native_invocation':{'path':str(SCENE/'invocation.json'),'sha256':sha(SCENE/'invocation.json')},'target_inventory':{'path':str(INV),'sha256':sha(INV),'count':len(keys)},
      'poses':list(POSES),'scope':'CPU predicted candidate-minus-source skin LBS displacement added to exact accepted source Float32 pack; all 859 targets and skin self intersections; exact predicate and no exemptions; no native candidate run.'}
    (OUT/'declaration.json').write_text(json.dumps(decl,indent=2,sort_keys=True)+'\n')
    results=[]
    for name,(path,expected) in CANDIDATES.items():
      if not path.is_file(): raise FileNotFoundError(path)
      h=sha(path)
      if expected and h!=expected: raise ValueError(f'candidate hash mismatch {name} {h}')
      if name=='compact-3x-full':
        # Pin dynamically to candidate report row for this variant; compare to source report entry.
        entry=next(x for x in report['variants'] if x['variant']['name']==name)
        expected=entry['candidate_payload']['sha256']
        if expected!=h: raise ValueError('third candidate differs from its builder report')
      candidate=parse_skin(path)
      if candidate['faces'].tobytes()!=source['faces'].tobytes() or candidate['xyz'].tobytes()!=source['xyz'].tobytes(): raise ValueError('candidate changed source positions/topology')
      if candidate['weights'].shape!=source['weights'].shape: raise ValueError('candidate weight matrix shape changed')
      results.extend(run_one(name,path,step,np,ci,core,clearance,source,keys,h) for step in POSES)
    summary={'schema':'numi.human.skin-weight-candidate-saved-pose-screen.summary.v1','status':'complete_exact_predicate_diagnostic_not_native_candidate_replay','variants':list(CANDIDATES),'steps':list(POSES),'results':[{'variant':r['variant'],'step':r['step'],'crossing_pair_count':r['results']['crossing_pair_count'],'new_vs_baseline_pair_count':r['results']['new_vs_baseline_pair_count'],'removed_vs_baseline_pair_count':r['results']['removed_vs_baseline_pair_count'],'skin_self_crossing_pair_count':r['results']['skin_self_crossing_pair_count'],'elapsed_seconds':r['elapsed_seconds']} for r in results], 'limits':['CPU LBS displacement prediction is not a native Metal candidate capture.','Only steps 47519 and 155000 were screened.','A zero or lower crossing count would not establish full-cycle, support-Jacobian, or physics qualification.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,sort_keys=True),flush=True)

if __name__=='__main__': main()
