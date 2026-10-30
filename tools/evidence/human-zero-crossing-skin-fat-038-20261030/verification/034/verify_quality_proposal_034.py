"""Verify fixed-coordinate remesh proposals against complete current surfaces and bones."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
H=ROOT/'fine-physical-volume-706'
PINNED={
'O':(H/'microedge-csr-canonical-011/O/candidate-O-csr-canonical.npz','578cee97a3bbecb73c0e1a6d7c26f45ac6cc111a552996f3910866a4a3652a9b'),
'I':(H/'local-microedge-saved-operation-replay-011/I/candidate-I-saved-op-replay.npz','9721102247157eeb42c59633fda59d63fd5ba48ad11ba4e63987834136711e31'),
'D':(H/'current-D-proof-015/candidate-D-saved-op-replay-source-mapped.npz','15afa8eecdaeff456cda25ef26890da60d60dd06e0412ed2035c77a8cbeb460f')}
OWNER=ROOT/'deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py'
OWNER_SHA='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
BONE=ROOT/'bone-barrier-remeshing-609/bone_barrier.py'
BONE_SHA='734527f060dcae186ae1b9609eca1ab95768d952aa0e9e8c7af76cae882b236b'
BASE_PROOFS=[
(H/'local-microedge-condition-001/exact-gates-010/report.json','810a2f07d589eb3646155c277840ce0e98455b965cfe1d935e01f36d7e29acbc','Complete baseline self and interlayer geometry only; old bone subsection invalid.'),
(H/'local-microedge-condition-001/exact-gates-014-current-all185-F32-F64/report.json','48f5e8d76cb123f4ac890061163641331403ed342f1adf01821a0b8ff73c26bc','Complete baseline all185 bones in both coordinate domains.')]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def same_bytes(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def directed_boundary(f):
    e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
    ordered=np.sort(e,axis=1)
    _,inv,cnt=np.unique(ordered,axis=0,return_inverse=True,return_counts=True)
    if np.any(cnt>2):raise ValueError('nonmanifold edges')
    signs=np.where(e[:,0]==ordered[:,0],1,-1)
    balance=np.bincount(inv,weights=signs,minlength=len(cnt))
    if np.any((cnt==2)&(balance!=0)):raise ValueError('same-direction interior edge')
    return sorted(map(tuple,e[cnt[inv]==1].tolist()))

def check_geometry(source, proposal):
    p=source['points_world_m']; f=np.asarray(source['faces'],np.int64)
    q=proposal['points_world_m']; g=np.asarray(proposal['faces'],np.int64)
    keep=np.asarray(proposal['point_output_to_input_rows'],np.int64).reshape(-1)
    inverse=np.asarray(proposal['point_input_to_output_row'],np.int64).reshape(-1)
    fm=np.asarray(proposal['face_output_to_input_rows'],np.int64).reshape(-1)
    retired=np.asarray(proposal['retired_input_face_rows'],np.int64).reshape(-1)
    if p.ndim!=2 or p.shape[1]!=3 or q.ndim!=2 or q.shape[1]!=3:raise ValueError('point shape')
    if len(q)!=len(keep) or len(keep)!=len(np.unique(keep)) or np.any(keep<0) or np.any(keep>=len(p)):raise ValueError('point subset')
    if not same_bytes(q,p[keep]):raise ValueError('retained coordinates or dtype changed')
    if g.ndim!=2 or g.shape[1]!=3 or not len(g) or g.min()<0 or g.max()>=len(q):raise ValueError('face indices')
    if not np.array_equal(np.unique(g),np.arange(len(q))):raise ValueError('unused point rows')
    wanted=np.full(len(p),-1,np.int64);wanted[keep]=np.arange(len(keep))
    if not np.array_equal(inverse,wanted):raise ValueError('point maps not reciprocal')
    if len(fm)!=len(g) or np.any(fm < -1) or np.any(fm>=len(f)):raise ValueError('face map domain')
    old=fm[fm>=0]
    if len(old)!=len(np.unique(old)):raise ValueError('duplicate retained face IDs')
    missing=np.setdiff1d(np.arange(len(f)),old)
    if len(retired)!=len(np.unique(retired)) or not np.array_equal(np.sort(retired),missing):raise ValueError('retired face partition')
    geom=keep[g]
    if not np.array_equal(geom[fm>=0],f[old]):raise ValueError('retained connectivity changed')
    if 'new_faces_input_point_rows' in proposal and not np.array_equal(geom[fm<0],proposal['new_faces_input_point_rows']):raise ValueError('new patch records do not match output')
    if directed_boundary(f)!=directed_boundary(geom):raise ValueError('oriented boundary changed')
    if len(np.unique(np.sort(g,axis=1),axis=0))!=len(g):raise ValueError('duplicate unordered face')
    if not np.isfinite(q).all():raise ValueError('nonfinite coordinate')
    xf=np.asarray(q,np.float32)
    if 'points_world_m_f32' in proposal and not same_bytes(xf,proposal['points_world_m_f32']):raise ValueError('Float32 view changed')
    t=xf[g].astype(np.float64)
    area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    if not np.isfinite(area).all() or np.any(area<=0):raise ValueError('degenerate Float32 face')
    return {'source_points':len(p),'output_points':len(q),'retired_points':int(np.count_nonzero(inverse<0)),
            'source_faces':len(f),'output_faces':len(g),'retired_faces':len(retired),'new_faces':int(np.count_nonzero(fm<0)),
            'boundary_edges':len(directed_boundary(geom)),'retained_coordinates_exact':True}, np.flatnonzero(fm<0)


def check_targets(role, source, proposal, target_routes):
    keep=np.asarray(proposal['point_input_to_output_row'],np.int64)
    q=np.asarray(proposal['points_world_m'],np.float32)
    checked=[]
    for row in target_routes:
        if row['role']!=role:continue
        old=int(row['surface_vertex_row'])
        if old<0 or old>=len(keep):raise ValueError('protected target outside source domain')
        wanted=np.asarray(row['target_world_m_f32'],np.float32)
        if np.asarray(source['points_world_m'][old],np.float32).tobytes()!=wanted.tobytes():raise ValueError('protected source coordinate mismatch')
        new=int(keep[old])
        if new<0:raise ValueError('protected target retired')
        if q[new].tobytes()!=wanted.tobytes():raise ValueError('protected output coordinate mismatch')
        checked.append({'canonical_protected_row':int(row['canonical_protected_row']),'source_row':old,'output_row':new})
    return checked

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser()
    for role in ('O','I','D'):ap.add_argument('--'+role.lower(),required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report={'status':'STARTED','physical_admission':False,'accepted_steps':0,'runtime':0,'baseline_proofs':[],'roles':{}}
    (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    for path,digest,scope in BASE_PROOFS:
        if sha(path)!=digest:raise ValueError('baseline proof digest '+str(path))
        report['baseline_proofs'].append({'path':str(path),'sha256':digest,'scope':scope})
    if sha(OWNER)!=OWNER_SHA or sha(BONE)!=BONE_SHA:raise ValueError('frozen exact owner digest')
    sys.path.insert(0,str(OWNER.parent.parent))
    import numilab_human.cardiac_cavity_intersections as pred
    if sha(Path(pred.__file__))!=OWNER_SHA:raise ValueError('loaded exact owner differs')
    bone=import_file('quality_verify_bone_034',BONE)
    target_receipt=H/'microedge-collar-014-saved-op-replay/receipt.json'
    target_digest='a7eaac55c3cacf88aa5d31c2f2844b6da957a3348c98e76951bfc96c32968a01'
    if sha(target_receipt)!=target_digest:raise ValueError('protected target receipt digest')
    target_routes=json.loads(target_receipt.read_text())['protected149']['surface_routes']
    if len(target_routes)!=149 or len({int(r['canonical_protected_row']) for r in target_routes})!=149:raise ValueError('149 protected identity coverage')
    report['protected149_source']={'path':str(target_receipt),'sha256':target_digest}
    proposals={};changed={};total_points=0;all_records=[];new_records=[];facebase=0
    for role in ('O','I','D'):
        baseline,digest=PINNED[role];path=Path(getattr(args,role.lower()))
        if sha(baseline)!=digest:raise ValueError('baseline candidate hash')
        source=load(baseline);a=load(path)
        if str(a['source_candidate_sha256'].reshape(-1)[0])!=digest:raise ValueError('proposal source link')
        checks,ids=check_geometry(source,a)
        checks['protected_targets']=check_targets(role,source,a,target_routes)
        checks.update({'input_path':str(baseline),'input_sha256':digest,'proposal_path':str(path),'proposal_sha256':sha(path)})
        report['roles'][role]=checks
        rr,_=pred.float32_surface_records(np.asarray(a['points_world_m'],np.float32).astype(np.float64),a['faces'],source_seam_groups=())
        if len(rr)!=len(a['faces']):raise ValueError('exact record count changed')
        records=[(r[0],r[1],r[2],facebase+i,tuple(total_points+int(v)for v in r[4]))for i,r in enumerate(rr)]
        selected=set(map(int,ids+facebase));all_records.extend(records)
        new_records.extend(r for r in records if int(r[3])in selected)
        facebase+=len(records);total_points+=len(a['points_world_m'])
        del source,a,rr
    report['changed_face_records']=len(new_records);report['complete_current_surface_records']=len(all_records)
    report['status']='EXACT_CURRENT_SURFACE_CHECK'
    (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['changed_vs_complete_current_surfaces']=pred._audit_self_pairs_touching_faces(all_records,{int(r[3])for r in new_records})
    if report['changed_vs_complete_current_surfaces']['count']:raise ValueError('current geometry crossing')
    report['status']='EXACT_ALL185_BOTH_DOMAINS'
    (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    bones,breceipt=bone.load_bone_barrier_records(pred,root=ROOT,vertex_offset=total_points,
        expected_native_sha256='e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5',
        expected_manifest_sha256='0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c')
    if len(bones)!=938272:raise ValueError('bone record coverage')
    report['bones']=breceipt
    report['changed_vs_all185_both_domains']=pred._audit_pair_prepared_first(pred._prepare_surface_aabb(bones),new_records,same_surface=False)
    if report['changed_vs_all185_both_domains']['count']:raise ValueError('bone crossing')
    report.update(status='PASS_FIXED_COORDINATE_GEOMETRY_DELTA',elapsed_seconds=time.monotonic()-started,
        interpretation='Unchanged retained triangles inherit pinned complete baseline proofs; every new triangle is checked against complete current surfaces and all185 bone domains. No volume, mass or motion admission.')
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'roles':report['roles'],'elapsed_seconds':report['elapsed_seconds']},indent=2))
if __name__=='__main__':main()
