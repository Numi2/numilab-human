"""Intersect the registered anatomy between the deep fat layer and the first underlying cortical surface."""
import json,mmap,runpy,sys
from pathlib import Path
import numpy as np
from prepare_bed_columns import hits,tables
from material_context import sha

def union(parts):
    old=None;length=0.
    for a,b in sorted(parts):
        if old is None or a>old:length+=b-a;old=b
        elif b>old:length+=b-old;old=b
    return length

def run(w,cap,prev,shapes,source,out):
    geo=np.load(shapes);xy=geo['xyz'];top=xy[:,2]+geo['layer_heights_m'].sum(1)
    receipt=w/'qualified/motion-008/step-8.receipt.json';pack=w/'qualified/native-coupled-006/native/resting-human.mrvpack'
    rc=json.loads(receipt.read_text())
    assert sha(pack)==rc['base_pack_file_sha256']
    cfg=json.loads((prev/'full-motion/launch.json').read_text())
    m=Path(cfg['argv'][cfg['argv'].index('--soft-tissue-payload')+1]);manifest=json.loads(m.with_suffix('.manifest.json').read_text())
    assert sha(m)==manifest['payload']['sha256']
    names={int(r['stable_id']):r.get('label',r.get('member','unnamed')) for r in manifest['source']['surfaces']}
    native=next(json.loads(row.partition('=')[2]) for row in (cap/'emitted-native/stderr.log').open() if row.startswith('registered_ligament_actual_pose_json='))
    if native['bonePayloadSha256']!=sha(Path(cfg['argv'][3])):raise ValueError('Skeletal identity differs')
    observed={p['body_index']:p for p in rc['initial_anatomical_registration']['body_poses']}
    poses={}
    for i,pose in enumerate(native['bodyPoses']):
        if i in observed:
            p=observed[i];x=p['position_m']+p['quaternion_xyzw']
            if max(abs(np.array(x)-pose))>1e-7:raise ValueError('Inconsistent frames')
        else:p=dict(position_m=pose[:3],quaternion_xyzw=pose[3:])
        poses[i]=p
    utilities=runpy.run_path(str(source/'audit.py'))
    events=[[] for _ in xy];bank_names={};details={}
    with pack.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        for (sem,sid),index,v,faces in utilities['primitives'](data,poses):
            if sem not in (51004,51005,51006):continue
            lo,hi=v.min(0),v.max(0)
            sel=np.flatnonzero(np.all((xy[:,:2]>=lo[:2]-1e-9)&(xy[:,:2]<=hi[:2]+1e-9),axis=1)&(top<hi[2]+1e-8))
            if not len(sel):continue
            raw=hits(xy[sel],v,faces)
            key=f'{sem}/{sid}';bank_names[key]=names.get(sid,key) if sem!=51004 else key
            good=0;odd=0
            for k,row in enumerate(tables(raw,len(sel))):
                if not row:continue
                j=sel[k]
                if len(row)%2:
                    odd+=1
                    events[j].append((key,'open',[(a,b) for a,b in row]))
                else:
                    good+=1;events[j].append((key,'even',[(a,b) for a,b in row]))
            if good or odd:details[key]=[good,odd]
    results=[]
    for j,entry in enumerate(events):
        bone=[];interior=False;others=[];odd=[]
        for key,status,zs in entry:
            if status=='open':odd.extend([(key,z) for z,i in zs if z>=top[j]]);continue
            for (a,fa),(b,fb) in zip(zs[::2],zs[1::2],strict=True):
                if b<top[j]:continue
                if key.startswith('51004/'):
                    interior|=bool(a<top[j]<b)
                    bone.append((max(a,top[j]),key,fa))
                else:others.append((a,b,key))
        b=min(bone,default=None)
        if b is None:results.append(dict(bone=None));continue
        z,owner,face=b; length=z-top[j];seg=[];bank={}
        for a,b,key in others:
            a=max(a,top[j]);b=min(b,z)
            if a<b:seg.append((a,b));bank[key]=bank.get(key,0.)+b-a
        union_length=union(seg);odd=[k for k,c in odd if c<=z]
        results.append(dict(bone=owner,face=face,deep_inside_bone=interior,deep_to_bone_mm=1000*length,muscle_or_tendon_union_mm=1000*union_length,unaccounted_mm=1000*max(0,length-union_length),odd_surface_witnesses=odd,contributing_bank_lengths_mm={k:v*1000 for k,v in bank.items()}))
    report=dict(source_pack=sha(pack),source_receipt=sha(receipt),muscles=sha(m),scope='Vertical probe of the registered anatomical surfaces at release. An even ray is an occupancy estimate; the intersection length is not evidence of force transfer.',bank_names=bank_names,queries=len(xy),bank_hit_counts=details,traces=results)
    out.write_text(json.dumps(report,separators=(',',':'))+'\n')
    print(len(xy),len(details),sum(r['bone'] is not None for r in results),flush=True)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
