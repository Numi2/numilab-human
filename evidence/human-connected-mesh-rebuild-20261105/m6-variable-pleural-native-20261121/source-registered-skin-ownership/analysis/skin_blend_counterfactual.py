"""Compare loaded NHSKIN interpolation against native coordinates in source vertex order."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def quat(a,b):
    v=a[...,:3]*b[...,3:]+b[...,:3]*a[...,3:]+np.cross(a[...,:3],b[...,:3])
    w=a[...,3]*b[...,3]-np.einsum('...i,...i->...',a[...,:3],b[...,:3])
    return np.concatenate((v,w[...,None]),axis=-1)
def stats(a):
    if not len(a):return {'count':0}
    return dict(zip(['min','median','p95','max'],map(float,np.quantile(a,[0,.5,.95,1]))))|{'count':len(a)}
def model(p,poses):
    transforms={r['body_index']:r for r in poses}
    records=p['binds'];pose=[transforms[int(r)] for r in records['body']]
    q=Rotation.from_quat([r['quaternion_xyzw'] for r in pose])
    bind=Rotation.from_quat(records['v'][:,3:7]);Q=(q*bind).as_quat()
    t=q.apply(records['v'][:,:3])+np.array([r['position_m'] for r in pose])
    scales=records['v'][:,7]
    if np.ptp(scales)!=0:raise ValueError('The rigid comparison requires a single common scale')
    x=p['verts'][:,:3].astype(float)*scales[0];w=p['weights'].astype(float)
    mean=(w@Rotation.from_quat(Q).as_matrix().reshape(-1,9)).reshape(-1,3,3)
    lbs=np.einsum('nij,nj->ni',mean,x)+w@t
    dual=.5*quat(np.column_stack((t,np.zeros(len(t)))),Q)
    dominant=w.argmax(1);out=np.empty_like(x);ambiguity=np.zeros(len(x));denominator=np.zeros(len(x))
    for j in np.unique(dominant):
        use=dominant==j
        dots=Q@Q[j];signs=np.where(dots<0.,-1.,1.)
        joint=w[use]*signs
        qr=joint@Q;qd=joint@dual
        length=np.linalg.norm(qr,axis=1);denominator[use]=length
        if np.any(length<1e-6):raise ValueError('Degenerate weighted orientation')
        qr/=length[:,None];qd/=length[:,None]
        qd-=np.einsum('ij,ij->i',qr,qd)[:,None]*qr
        out[use]=Rotation.from_quat(qr).apply(x[use])+2*quat(qd,qr*[-1,-1,-1,1])[:,:3]
        ambiguity[use]=w[use][:,np.abs(dots)<.01].sum(axis=1)
    return lbs,out,dict(minimum_blend_norm=float(denominator.min()),max_weight_with_abs_quaternion_dot_under_0_01=float(ambiguity.max()))
def compare(workspace,capture,receipt,report,output):
    sys.path.insert(0,str(workspace/'resting-clearance-and-atrium-002/publication/scripts'))
    from rebind_human_skin_reference import decode
    r=json.loads(receipt.read_text());s=json.loads(report.read_text())
    cfg=json.loads((receipt.parent/'launch.json').read_text());file=Path(cfg['argv'][cfg['argv'].index('--skin-payload')+1])
    p=decode(file.read_bytes());faces=p['inds'];a=np.load(capture);obs=a['positions'].astype(float)
    if (sha(file)!=str(a['source_sha256']) or str(a['pack_sha256'])!=r['pack_file_sha256']
            or a['faces_sha256']!=hashlib.sha256(faces.astype('<u4').tobytes()).hexdigest()
            or s['accepted_step']!=r['accepted_step'] or s['accepted_pack_sha256']!=r['pack_file_sha256']):raise ValueError('Source/capture mismatch')
    used=np.unique(faces);selfids=np.unique(faces[np.array(s['self_intersections']['triangle_pairs'],dtype=int).ravel()])
    W=p['weights']
    metrics={}
    for k in ['accepted_body_poses','accepted_registered_body_poses']:
        x,y,d=model(p,r[k]);err=1e3*np.linalg.norm(x-obs,axis=1)
        metrics[k]={'referenced_LBS_native_error_mm':stats(err[used]),'self_witness_LBS_native_error_mm':stats(err[selfids]),'vertices_within_2um':int((err[used]<=.002).sum())}
    key=max(metrics,key=lambda k:metrics[k]['vertices_within_2um'])
    x,y,d=model(p,r[key]);dist=np.linalg.norm(x-obs,axis=1)
    ok=dist<=2e-6;mask=np.zeros(len(x),bool);mask[used]=True
    view=y.copy()
    view[~ok]=obs[~ok]
    ids=np.unique(faces);native=np.searchsorted(ids,faces)
    sys.path[:0]=[str(workspace/'resting-surface-clearance-001/src'),str(workspace/'resting-clearance-and-atrium-005'),str(workspace/'resting-surface-clearance-001')]
    import audit
    from fast_skin_intersection_audit import install_for_float32
    install_for_float32()
    def audit_self(x):
        rec=audit.clearance._exact_surface_records(np.ascontiguousarray(x[ids],dtype='<f4'),native)
        return audit.intersections._audit_pair_prepared_first(audit.intersections._prepare_surface_aabb(rec),rec,same_surface=True)
    check=audit_self(obs)
    if sorted(check['triangle_pairs'])!=sorted(s['self_intersections']['triangle_pairs']):raise ValueError('Replay changed source face witnesses')
    counter=audit_self(view);before={tuple(x) for x in check['triangle_pairs']};after={tuple(x) for x in counter['triangle_pairs']}
    summary={'accepted_step':r['accepted_step'],'pack_sha256':r['pack_file_sha256'],'payload_sha256':sha(file),'frame_comparisons':metrics,
        'chosen_frame':key,'referenced':len(used),'eligible':int((ok&mask).sum()),'witness_vertices':len(selfids),'eligible_witness_vertices':int(ok[selfids].sum()),
        'sign_alignment':d,'source_weight_sum_error':float(np.max(np.abs(W.sum(axis=1,dtype=float)-1))),'delta_mm':stats(np.linalg.norm((view-obs)[ok&mask],axis=1)*1000),
        'native_self_pairs':len(before),'counterfactual_self_pairs':len(after),'self_pairs_removed':len(before-after),'self_pairs_introduced':len(after-before),
        'first_new_pairs':sorted(after-before)[:16],'first_retained_pairs':sorted(after&before)[:16],
        'scope':'Offline rigid skin interpolation only, with observed positions kept where the rigid decode misses 2 micrometres. Requires physical mass/volume ownership before promotion.'}
    output.write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':compare(*map(Path,sys.argv[1:]))
