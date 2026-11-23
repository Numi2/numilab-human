"""Intersect vertical lines at native contact witnesses with the registered entrance surfaces."""
import argparse,json,runpy,sys
from pathlib import Path
import numpy as np
from material_context import sha
from verify_native_extensor_pair import loaded

def intersections(v,faces,point):
    if np.any(point[:2]<v[:,:2].min(0)) or np.any(point[:2]>v[:,:2].max(0)):return []
    t=v[faces].astype(float);xy=t[:,:,:2]
    rows=np.flatnonzero(np.all((xy.min(1)<=point[:2])&(xy.max(1)>=point[:2]),axis=1))
    t=t[rows];a=t[:,0];b=t[:,1]-a;c=t[:,2]-a
    det=b[:,0]*c[:,1]-b[:,1]*c[:,0];good=abs(det)>1e-16
    rows=rows[good];a,b,c,det=a[good],b[good],c[good],det[good]
    x,y=(point[:2]-a[:,:2]).T
    u=(x*c[:,1]-y*c[:,0])/det;w=(b[:,0]*y-b[:,1]*x)/det
    inside=(u>=-1e-9)&(w>=-1e-9)&(u+w<=1+1e-9)
    z=a[:,2]+u*b[:,2]+w*c[:,2]
    return sorted([(float(h-point[2]),int(row)) for h,row in zip(z[inside],rows[inside])])

def inspect(pack_case,case,failure,reader):
    read=lambda p:json.loads(p.read_text())
    old,all_pose,raw=loaded(pack_case,True,False);rc=read(case/'terminal.receipt.json')
    if rc['base_pack_file_sha256']!=old['base_pack_file_sha256']:
        raise ValueError('Native entrance identity differs')
    pose={b['body_index']:b for b in rc['initial_anatomical_registration']['body_poses']}
    if any(all_pose.get(k)!=b for k,b in pose.items()):raise ValueError('Different opening skin donors')
    class PoseRead(dict):
        def __getitem__(self,k):
            self.last=k
            return super().__getitem__(k)
    track=PoseRead(all_pose);track.last=None
    skipped=0
    cfg=read(case/'launch.json');args=cfg['argv'];payload=Path(args[args.index('--soft-tissue-payload')+1])
    manifest=read(payload.with_suffix('.manifest.json'))
    if sha(payload)!=manifest['payload']['sha256']:raise ValueError('Anatomical source differs')
    labels={s['stable_id']:s for s in manifest['source']['surfaces']}
    columns=np.genfromtxt(case/'registered-bed-columns.csv',delimiter=',',names=True)
    cert=read(failure/'terminal-bed-certificate.json')
    sample={}
    for w in cert['witnesses']:
        col,=columns[columns['vertex']==w['vertex']]
        sample[w['vertex']]=dict(row=int(col['row']),point=[float(col[k]) for k in ('x_m','y_m','z_m')],skin_m=float(col['skin_m']),fat_m=float(col['fat_m']),boundaries=[])
    sys.path.insert(0,str(reader.parent/'src'));api=runpy.run_path(str(reader))
    for key,index,v,faces in api['primitives'](raw,track):
        used_body=track.last;track.last=None
        if used_body is not None and used_body not in pose:
            skipped+=1
            continue
        for item in sample.values():
            hits=intersections(v,faces,np.array(item['point']))
            if not hits:continue
            name=None
            if key[0] in (51005,51006):
                label=labels.get(key[1],{})
                name=label.get('label')
            item['boundaries'].append(dict(semantic=key[0],stable_id=key[1],primitive=index,name=name,heights_above_skin_m=[h for h,_ in hits],faces=[i for _,i in hits]))
    for item in sample.values():
        skin,=[b for b in item['boundaries'] if (b['semantic'],b['stable_id'])==(51007,1)]
        # The support quadrature uses source vertices. A vertex can be shared by
        # several incident faces and can move a few Float32 ulps on emission.
        if min(abs(h) for h in skin['heights_above_skin_m'])>1e-5:raise ValueError('Contact source does not match emitted skin')
        item['boundaries'].sort(key=lambda b:min(abs(h) for h in b['heights_above_skin_m']))
    return dict(rigid_banks_without_current_opening_donor=skipped,confirmed_opening_donors=len(pose),pack=rc['base_pack_file_sha256'],columns=sha(case/'registered-bed-columns.csv'),failure=sha(failure/'terminal-bed-certificate.json'),source=sha(payload),
        scope='Vertical boundary intersections in the registered initial pose. These are geometry and stock lengths. They do not establish bonded interfaces, force transmission or material stiffness.',witnesses=sample)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pack_case','case','failure','reader'):p.add_argument(key,type=Path)
    print(json.dumps(inspect(**vars(p.parse_args())),indent=2,allow_nan=False))
