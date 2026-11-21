"""Check both registered proximal phalanx surfaces at each measured native input pose."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_digit_contacts import sha

def read(path):
    return json.loads(path.read_text())

def test(work,control,run,out):
    sys.path[:0]=[str(work/'resting-surface-clearance-001/src'),str(work/'resting-surface-clearance-001'),str(work/'resting-clearance-and-atrium-005')]
    import audit
    from fast_skin_intersection_audit import install_for_float32
    install_for_float32()
    evidence=run/'digit-material-frames.csv'
    report=read(control/'initial-cortical.json')
    source=read(control/'step-0.receipt.json')
    if report['pack_sha256']!=source['pack_file_sha256']:raise ValueError('Source capture mismatch')
    cfg=read(run/'launch.json')
    bone=Path(next(a for a in cfg['argv'] if a.endswith('.nhbones')))
    if sha(bone)!=report['cortical_payload_sha256'] or cfg['pins'][str(bone)]!=sha(bone):raise ValueError('Bone source differs')
    n=read(run/'failure.json')['first_failure_latch']['step'] if (run/'failure.json').exists() else read(run/'terminal.receipt.json')['accepted_step']
    poses={p['body_index']:p for p in source['accepted_body_poses']}
    src=np.load(control/'initial-cortical.npz')
    local={};body={};faces={}
    for name,index,_,_ in report['member_by_stable_id'].values():
        if name.startswith(('4proxph','5proxph')):
            p=poses[index];body[index]=name
            local[name]=Rotation.from_quat(p['quaternion_xyzw']).inv().apply(src[name+'_v'].astype(float)-p['position_m'])
            faces[name]=src[name+'_f']
    frames={}
    for row in csv.DictReader(evidence.open()):
        k=int(row['input_step']);name=body[int(row['body'])]
        if k>n:continue
        t=np.array([float(row[i])+float(row['low_'+i]) for i in 'xyz'])
        q=[float(row['q'+i]) for i in 'xyzw']
        if name in frames.setdefault(k,{}):raise ValueError('Duplicate pose')
        frames[k][name]=(t,q)
    terminal=run/'terminal.receipt.json'
    if terminal.exists():
        rec=read(terminal)
        if rec['accepted_step']!=n:raise ValueError('Terminal time mismatch')
        if n not in frames:
            frames[n]={body[p['body_index']]:(p['position_m'],p['quaternion_xyzw']) for p in rec['accepted_body_poses'] if p['body_index'] in body}
    expected=range(n+int(n in frames))
    if sorted(frames)!=list(expected):raise ValueError('Missing accepted poses')
    rows=[];largest=0.;initialerr=0.
    observed=np.load(run/'terminal-cortical.npz') if (run/'terminal-cortical.npz').exists() else None
    for k,group in sorted(frames.items()):
        if set(group)!=set(local):raise ValueError('Incomplete body set')
        vertices={s:(Rotation.from_quat(q).apply(local[s])+t).astype('<f4') for s,(t,q) in group.items()}
        for s,v in vertices.items():
            if k==0:initialerr=max(initialerr,float(np.linalg.norm(v-src[s+'_v'],axis=1).max()))
            if k==n and observed is not None:largest=max(largest,float(np.linalg.norm(v-observed[s+'_v'],axis=1).max()))
        counts={}
        for side in ['r','l']:
            a,b=['%dproxph_%s'%(i,side) for i in [4,5]]
            records=[audit.clearance._exact_surface_records(vertices[s],faces[s]) for s in [a,b]]
            count=audit.intersections._audit_pair_prepared_first(audit.intersections._prepare_surface_aabb(records[0]),records[1])['count']
            counts[side]=count
        rows.append(dict(step=k,**counts))
    if initialerr>2e-6 or largest>2e-6:raise ValueError(('Rigid source reconstruction differs',initialerr,largest))
    if observed is not None:
        late=read(run/'terminal-cortical.json')
        for side in ['r','l']:
            c=next(p['count'] for p in late['pairs'] if p['first']=='4proxph_'+side and p['second']=='5proxph_'+side)
            if c!=rows[-1][side]:raise ValueError(('Direct terminal predicates differ',side,c,rows[-1][side]))
    summary=dict(accepted_steps=n,first_pose=rows[0]['step'],last_pose=rows[-1]['step'],poses=len(rows),rows=rows,
        peak_pairs={k:max(r[k] for r in rows) for k in ['l','r']},
        first_crossing={k:next((r['step'] for r in rows if r[k]),None) for k in ['l','r']},
        measured_frame_sha256=sha(evidence),native_source_sha256=report['pack_sha256'],
        maximum_initial_reconstruction_error_m=initialerr,maximum_terminal_reconstruction_error_m=largest if observed is not None else None,
        scope='Native beginning-of-step phalanx frames, plus the committed endpoint when available. Exact intersection of reconstructed Float32 bank surfaces. No swept-volume or skin clearance assertion.')
    out.write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':test(*map(Path,sys.argv[1:]))
