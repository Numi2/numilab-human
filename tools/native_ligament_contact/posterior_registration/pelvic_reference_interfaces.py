"""Separate intersections present in the source atlas from changes under the native registration."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from numilab_human.capsular_material_registration import read_material_banks,material_bank,emitted_body_vertices
from compare_hand_interfaces import query
from material_context import sha,applied
from pose_glutes import parts
def obj(z,member,pin):
    raw=z.read(next(n for n in z.namelist() if n.endswith('/'+member+'.obj')))
    if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('Wrong atlas member')
    v=[];f=[]
    for l in raw.splitlines():
        if l.startswith(b'v '):v.append([float(s)/1000 for s in l.split()[1:4]])
        elif l.startswith(b'f '):
            seq=[int(s.split(b'/')[0])-1 for s in l.split()[1:]]
            if len(seq)!=3:raise ValueError('Unexpected atlas polygon')
            f.append(seq)
    return np.array(v,dtype='f4'),np.array(f,int)
def count(a,b,exe):
    x,t=a;y,u=b;f=np.vstack((t,u+len(x)))
    return len([j for i,j in query(exe,np.vstack((x,y)),f,np.arange(len(f))<len(t)) if i<len(t)<=j])
def run(cfg,receipt,cap,static,archive,exe,out):
    args=json.loads(cfg.read_text())['argv'];bone=Path(args[3]);bank=read_material_banks(bone)
    tissue=Path(args[args.index('--soft-tissue-payload')+1]);tm=json.loads(tissue.with_suffix('.manifest.json').read_text())
    wanted=json.loads((static/'source.json').read_text())['candidate_sha256'];base=static/tissue.name;assert sha(base)==wanted
    ref=json.loads(receipt.read_text());frames={r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in ref['initial_anatomical_registration']['body_poses']}
    by_id={r['stable_id']:r for r in tm['source']['surfaces']};data={};outrows={}
    shared=next(r['sha256'] for r in tm['source']['bodyparts']['archives'] if r['hierarchy']=='is_a');assert sha(archive)==shared
    with zipfile.ZipFile(archive) as z:
        for s in (29,30):
            r=by_id[s];raw=obj(z,r['member_id'],r['member_sha256']);mesh=parts(base,s)
            cap0=np.load(cap/('entrance-%d.npz'%s));cap1=np.load(cap/('accepted-8-%d.npz'%s))
            data[s]=dict(atlas=raw,edited_source=(mesh[0].astype('f4'),mesh[3]),entrance=(cap0['xyz'],cap0['faces']),endpoint=(cap1['xyz'],cap1['faces']))
        for sid in (1,2,3,18,19):
            body=bank[2][sid][0];an,rec,x,f=material_bank(bank,sid,body)
            sv,sf=obj(z,an['member_id'],an['member_sha256'])
            if body not in frames:raise ValueError(('No actual body frame',body))
            world=applied(frames[body],emitted_body_vertices(x,rec)).astype('f4')
            ef=next(r for r in ref['accepted_body_poses'] if r['body_index']==body)
            terminal=applied(np.array(ef['position_m']+ef['quaternion_xyzw']),emitted_body_vertices(x,rec)).astype('f4')
            record=dict(anatomical_body=an['myosim_body'],stable_id=sid,source_member=an['member_id'],member_sha256=an['member_sha256'],vertices_in_archive=len(sv),vertices_in_native=len(world))
            for s,b in data.items():
                result=dict(original_atlas=count(b['atlas'],(sv,sf),exe),registered_source_before_body_transforms=count(b['edited_source'],(sv,sf),exe),actual_registered_entrance=count(b['entrance'],(world,f),exe),actual_registered_endpoint=count(b['endpoint'],(terminal,f),exe))
                record[str(s)]=result
            outrows[sid]=record
    result=dict(raw_archive=shared,bone=sha(bone),tissue=sha(tissue),static_registration_parent=sha(base),native_pose=sha(receipt),rows=outrows)
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
