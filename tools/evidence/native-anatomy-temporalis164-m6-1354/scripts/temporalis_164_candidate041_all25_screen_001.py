from __future__ import annotations
import gc, hashlib, importlib.util, json, os, struct, sys, time, mmap
from pathlib import Path
import numpy as np
W=Path('/Users/n/numi-human-exact-normalization-m6-1348')
R=Path('/Users/n/numi-human-retained-delivery-20261009')
A=R/'anatomy-completion-1276'
BASE=W/'reports/temporalis-clearance-002'
OUT=BASE/'all25-candidate-screen-164-043/run-002/result-001'
C=BASE/'native-witness-repair-164-042/run-001/result-001/temporalis-164-left-skin-clearance-candidate-041.npz'
REPAIR=BASE/'native-witness-repair-164-042/run-001/result-001/report.json'
C040=BASE/'skin-clip-clearance-164-039/run-002/result-001/temporalis-164-left-skin-clearance-candidate-040.npz'
C040_REPORT=BASE/'skin-clip-clearance-164-039/run-002/result-001/report.json'
NATIVE164=A/'temporalis164-native-m6-001/native-verification-001/report.json'
TISS=A/'temporalis163-native-m6-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
FORWARD=A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'
CURRENT8=A/'temporalis163-native-m6-001/baseline/native-run/accepted-geometry'
K1=A/'mtp-passive-native-sensitivity-one-001/baseline/native-run/accepted-geometry'
P011=R/'skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011'
STEPS_CURRENT8=[0,4767,5023,5599,6207,6815,7423,8000]
STEPS_P011=[0,47519,151999,152607,153215,153823,154431,155000]
MAX_RSS_BYTES=2_500_000_000
REQUIRED_HEAD_CORES=(20,23,29,79)
sys.path.insert(0,str(W/'src'))
from numilab_human import common_atlas_skin_clearance as ca
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human import cardiac_cavity_partition as cp

def req(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rss_bytes():
    raw=__import__('subprocess').check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip()
    return int(raw)*1024
def atomic_json(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def emit(path,obj):
    with Path(path).open('ab') as f:f.write((json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
def volume(v,f):
    tri=np.asarray(v,dtype=np.float64)[np.asarray(f,dtype=np.int64)]
    return float(__import__('math').fsum(float(x) for x in np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]),optimize=False)/6.0))
def mesh_records(world,faces):
    world=np.ascontiguousarray(world,dtype='<f4');faces=np.asarray(faces,dtype=np.int64)
    unique,inverse=np.unique(world,axis=0,return_inverse=True);qf=np.ascontiguousarray(inverse[faces],dtype=np.int64)
    req(not np.any((qf[:,0]==qf[:,1])|(qf[:,1]==qf[:,2])|(qf[:,2]==qf[:,0])),'Float32 candidate face collapsed')
    rec=ci._records([ci.float32_point_lattice_key(tuple(float(x) for x in p)) for p in unique],qf.tolist())
    return rec,unique,qf
def parse_pack_keys(path):
    with Path(path).open('rb') as f:mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
    h=ca._PACK_HEADER.unpack_from(mm,0);keys=set()
    for i in range(h[2]):
        d=ca._PACK_DIRECTORY.unpack_from(mm,ca._PACK_HEADER.size+i*ca._PACK_DIRECTORY.size)
        if d[0]!=4:continue
        for j in range(d[4]):
            sem,sid,body,_=struct.unpack_from('<4I',mm,d[2]+j*64+16)
            if int(sem)!=51999:keys.add((int(sem),int(sid)))
    mm.close();return keys
def body_poses(receipt,group,step):
    rows=receipt.get('accepted_body_poses') or receipt.get('accepted_registered_body_poses') or []
    req(bool(rows),f'{group}/{step}: no accepted body poses')
    ids=[int(x['body_index']) for x in rows];req(len(ids)==len(set(ids)),f'{group}/{step}: duplicate body IDs')
    missing=sorted(set(REQUIRED_HEAD_CORES)-set(ids));req(not missing,f'{group}/{step}: active head cores missing {missing}')
    if group in ('current8','K1'):req(len(rows)==157,f'{group}/{step}: expected157 accepted body poses')
    if group.startswith('P011'):req(len(receipt.get('accepted_registered_body_poses',[]))==86,f'{group}/{step}: expected86 registered poses')
    return {int(x['body_index']):(np.asarray(x['position_m'],dtype='<f4'),np.asarray(x['quaternion_xyzw'],dtype='<f4')) for x in rows}
def capture_rows():
    out=[]
    for s in STEPS_CURRENT8:out.append(('current8',s,CURRENT8))
    out.append(('K1',8000,K1))
    for cohort in ('baseline','intervention'):
        folder=P011/cohort/'native-run'/'accepted-geometry'
        for s in STEPS_P011:out.append((f'P011-{cohort}',s,folder))
    return out

def main():
    req(not OUT.exists(),f'refuse existing output {OUT}');OUT.mkdir(parents=True)
    start=time.monotonic();progress=OUT/'progress.jsonl'
    for path in (C,REPAIR,C040,C040_REPORT,NATIVE164,TISS,FORWARD,W/'src/numilab_human/common_atlas_skin_clearance.py',W/'src/numilab_human/cardiac_cavity_intersections.py',W/'src/numilab_human/cardiac_cavity_partition.py'):
        req(path.is_file(),f'missing pinned input {path}')
    repair=json.loads(REPAIR.read_text());native164=json.loads(NATIVE164.read_text());old=json.loads(C040_REPORT.read_text())
    req(repair['status']=='native_capture_local_repair_passed' and repair['candidate041_sha256']==sha(C),'repair report does not pin candidate041')
    req(native164.get('complete') is True and len(native164.get('poses', []))==8 and not native164.get('all_native_geometry_checks_pass', True),'failed native164 witness schema changed')
    req(old.get('status')=='candidate_clear_all25' and old.get('all25_captures')==25,'candidate040 baseline screen changed')
    with np.load(C,allow_pickle=False) as z:
        source=np.asarray(z['positions_m'],dtype='<f4');faces=np.asarray(z['faces'],dtype=np.int64)
        binding_indices=np.asarray(z['binding_indices'],dtype='<u4');weights=np.asarray(z['weights'],dtype='<f4')
        world0=np.asarray(z['step0_world_f32'],dtype='<f4');ancestry=json.loads(str(z['face_ancestry_json'].item()))
        stable=int(np.asarray(z['stable_id'])[0]);core=int(np.asarray(z['core_body_index'])[0]);binding_sha=str(z['binding_record_sha256'][0])
    req(stable==164 and core==23 and source.shape==(5063,3) and faces.shape==(10134,3) and len(ancestry)==len(faces),'candidate041 arrays/counts changed')
    req(binding_indices.shape==(len(source),4) and np.all(binding_indices[:,0]==0) and np.all(binding_indices[:,1:]==0xffffffff),'candidate bindings are not one-hot')
    req(weights.shape==(len(source),4) and np.all(weights[:,0]==1.0) and np.all(weights[:,1:]==0.0),'candidate weights are not one-hot')
    with np.load(C040,allow_pickle=False) as z:
        base_source=np.asarray(z['positions_m'],dtype='<f4');base_faces=np.asarray(z['faces'],dtype=np.int64)
        base_ancestry=json.loads(str(z['face_ancestry_json'].item()))
    delta=source.astype(np.float64)-base_source.astype(np.float64);changed=np.flatnonzero(np.any(delta!=0,axis=1))
    req(np.array_equal(faces,base_faces) and ancestry==base_ancestry and changed.tolist()==[2119],'candidate041 differs from candidate040 beyond single authorized vertex')
    req(abs(float(np.linalg.norm(delta[2119]))-4.0e-6)<0.02e-6,'candidate041 source displacement is not pinned near4um')
    spec=importlib.util.spec_from_file_location('temporalis164_forward',FORWARD);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    tissue=helper.load_tissue(TISS);row160=helper.row_data(tissue,160);row163=helper.row_data(tissue,163)
    req(hashlib.sha256(row160['binding_bytes']).hexdigest()==binding_sha,'candidate binding record differs from live row160')
    req(int(tissue['bindings'][row160['fb']]['core'])==23 and int(tissue['bindings'][row163['fb']]['core'])==23,'row160/163 not bound to core23')
    stable_ids={int(x[6]) for x in tissue['records']};req(163 in stable_ids and 164 not in stable_ids,'screen parent TISS stable IDs changed')
    pins={str(p):sha(p) for p in (C,REPAIR,C040,C040_REPORT,NATIVE164,TISS,FORWARD,W/'src/numilab_human/common_atlas_skin_clearance.py',W/'src/numilab_human/cardiac_cavity_intersections.py',W/'src/numilab_human/cardiac_cavity_partition.py')}
    captures=[];current8_ok=True;sid163_keys=None
    for idx,(group,step,folder) in enumerate(capture_rows(),1):
        pack=folder/f'step-{step}.mrvpack';rp=folder/f'step-{step}.receipt.json'
        req(pack.is_file() and rp.is_file(),f'missing capture {group}/{step}')
        pins[str(pack)]=sha(pack);pins[str(rp)]=sha(rp)
        receipt=json.loads(rp.read_text());req(int(receipt.get('accepted_step',-1))==step and receipt.get('physical_endpoint')=='accepted',f'{group}/{step} not accepted')
        poses=body_poses(receipt,group,step)
        world=helper.forward(source,binding_indices,weights,row160,tissue,poses).astype('<f4')
        rec,unique,qf=mesh_records(world,faces)
        self_audit=ci._audit_pair(rec,rec,same_surface=True);top=cp._validate_output_topology(rec)
        keys=parse_pack_keys(pack);xyz,target_data,pack_info=ca._pack_surfaces(pack,keys)
        target_faces={k:np.asarray(v['faces'],dtype=np.int64) for k,v in target_data.items()}
        audits=ca._target_intersection_audit(rec,target_faces,xyz)
        nonzero={k:int(v['count']) for k,v in audits.items() if int(v['count'])>0}
        if group=='current8':
            req((51005,163) in keys,f'{group}/{step}: passive163 surface absent')
            req(len(target_data[(51005,163)]['faces'])==len(row163['faces']),f'{group}/{step}: passive163 face count changed')
            sid163_keys=sorted(f'{s}:{i}' for s,i in keys if i==163)
        world163=helper.forward(row163['positions'],row163['local'],row163['weights'],row163,tissue,poses).astype('<f4')
        rec163,_,_=mesh_records(world163,row163['faces']);pair163=ci._audit_pair(rec163,rec,same_surface=False)
        row={'capture':idx,'group':group,'step':step,'pack_sha256':pins[str(pack)],'receipt_sha256':pins[str(rp)],
            'accepted_body_pose_count':len(receipt.get('accepted_body_poses',[])),'registered_pose_count':len(receipt.get('accepted_registered_body_poses',[])),
            'required_head_cores_present':True,'all_target_surface_count':len(audits),'pack_info':pack_info,
            'world_f32_sha256':hashlib.sha256(world.tobytes()).hexdigest(),'self_intersection_count':int(self_audit['count']),
            'closed_oriented_2_manifold':bool(top['closed_oriented_2_manifold']),'component_count':int(top['component_count']),
            'signed_volume_m3':volume(unique,qf),'nonzero_target_counts':nonzero,'explicit_163_164_pair_triangle_count':int(pair163['count']),
            'explicit_163_164_pair_aabb_candidate_count':int(pair163['aabb_candidate_pairs']),
            'clear':bool(self_audit['count']==0 and top['closed_oriented_2_manifold'] and top['component_count']==1 and not nonzero and pair163['count']==0),
            'rss_bytes_after_audit':rss_bytes()}
        captures.append(row);emit(progress,{'event':'capture_audited',**row})
        if group=='current8' and not row['clear']:current8_ok=False
        if row['rss_bytes_after_audit']>MAX_RSS_BYTES:raise RuntimeError(f'RSS ceiling exceeded at {group}/{step}')
        del world,rec,unique,qf,self_audit,top,xyz,target_data,target_faces,audits,poses,world163,rec163,pair163;gc.collect()
        print(json.dumps({'event':'capture_audited','capture':idx,'group':group,'step':step,'clear':row['clear'],
            'self_pairs':row['self_intersection_count'],'target_hits':sum(nonzero.values()),'pair163_164':row['explicit_163_164_pair_triangle_count'],
            'rss_bytes':row['rss_bytes_after_audit'],'elapsed_s':time.monotonic()-start},sort_keys=True),flush=True)
    allclear=len(captures)==25 and all(x['clear'] for x in captures)
    status='candidate_clear_all25' if allclear else ('candidate_clear_current8' if len(captures)==8 and current8_ok else 'trial_not_admitted')
    report={'schema':'offline.temporalis.left-skin-clip-clearance-admission.v1','status':status,'candidate_npz':str(C),'candidate_npz_sha256':sha(C),
        'all25_captures':len(captures),'all25_captures_clear':bool(allclear),'current8_passed':bool(current8_ok),
        'capture_groups':{'current8':8,'K1':1,'P011-baseline':8,'P011-intervention':8},'stable_id':164,'native_tissue_binding_stable_id':160,
        'core_body_index':23,'active_head_core_ids_required':list(REQUIRED_HEAD_CORES),'current8_passive_163_target_keys':sid163_keys,
        'repair_receipt_path':str(REPAIR),'repair_receipt_sha256':sha(REPAIR),'prior_native_failure_report_path':str(NATIVE164),
        'prior_native_failure_report_sha256':sha(NATIVE164),'prior_native_failure_pairs_at_step8000':repair['prior_native_failure_pairs_at_step8000'],
        'repair_operation':repair['repair']['operation'],'repair_source_vertex_index':repair['repair']['vertex_index'],
        'repair_source_displacement_norm_m':repair['repair']['source_displacement_norm_m'],
        'all25_explicit_targets_scanned':True,'runtime_skin_modified':False,'runtime_ocular_openings_changed':False,
        'temporary_authoring_target_used_at_runtime':False,'maximum_rss_bytes':MAX_RSS_BYTES,'input_sha256':pins,
        'captures':captures,'elapsed_seconds':time.monotonic()-start,'max_rss_bytes':rss_bytes()}
    atomic_json(OUT/'report.json',report)
    print(json.dumps({'event':'finished','status':status,'captures':len(captures),'report':str(OUT/'report.json'),
        'report_sha256':sha(OUT/'report.json'),'elapsed_seconds':report['elapsed_seconds'],'max_rss_bytes':report['max_rss_bytes']},sort_keys=True),flush=True)
if __name__=='__main__':main()

