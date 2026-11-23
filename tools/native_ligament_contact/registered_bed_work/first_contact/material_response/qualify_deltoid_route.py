"""Compare the named wrapped posterior-deltoid routes at fixed, captured native body poses."""
from pathlib import Path
import copy
import json
import struct
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from measure_forearm_enclosure import Enclosure
from inspect_shoulder_source import nearest
from recount_registered_crossings import digest,read


def program(data, side):
    a=read(data/'bed-unscaled-four/launch.json')['argv']
    return program_file(Path(a[2]), side)


def program_file(path, side):
    raw=path.read_bytes()
    h=struct.unpack_from('<8s9I32s',raw);wo=76+16*h[4];ro=wo+64*h[5];mo=ro+16*h[6]
    mi={'r':212,'l':275}[side]
    _,start,n,_=struct.unpack_from('<4I',raw,mo+164*mi)
    sites={};wrap={};t=[]
    for j in range(start,start+n):
        kind,idx,ss,_=struct.unpack_from('<4I',raw,ro+16*j)
        ids=[idx] if kind==1 else [ss]
        for k in ids:
            b,*v=struct.unpack_from('<I3f',raw,76+16*k);sites[k]=[b,np.array(v)]
        if kind!=1:
            b,kind2,r,_,*x=struct.unpack_from('<2I14f',raw,wo+idx*64)
            if kind!=2 or kind2!=2 or sites[ss][0]!=b:raise ValueError('Unexpected wrapping program')
            wrap[idx]=[b,np.array(x[:3]),float(r),ss]
        t.append((kind,idx))
    if [kind for kind,idx in t]!=[1,2,1,2,1]:raise ValueError('Route grammar changed')
    return dict(sites=sites,wrap=wrap,seq=t)


def fmt(x):return ' '.join(format(float(v),'.17g') for v in x)


class SourceRoute:
    def __init__(self,P,bs,bh):
        def body(b):
            sites=''.join(f'<site name="s{k}" pos="{fmt(s)}"/>'for k,(i,s)in P['sites'].items() if i==b)
            geoms=''.join(f'<geom name="g{k}" size="{r}" pos="{fmt(c)}"/>'for k,(i,c,r,_)in P['wrap'].items() if i==b)
            return f'<body name="b{b}"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>{sites}{geoms}</body>'
        spec=''.join(f'<site site="s{i}"/>' if typ==1 else f'<geom geom="g{i}" sidesite="s{P["wrap"][i][3]}"/>' for typ,i in P['seq'])
        self.model=mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{body(bs)}{body(bh)}</worldbody><tendon><spatial>{spec}</spatial></tendon></mujoco>')
        self.data=mujoco.MjData(self.model)

    def geometry(self,poses):
        d=self.data
        for i,(p,r) in enumerate(poses):
            d.qpos[7*i:7*i+7]=np.r_[p,np.roll(r.as_quat(),1)]
        mujoco.mj_forward(self.model,d)
        count=int(d.ten_wrapnum[0])
        return float(d.ten_length[0]),d.wrap_xpos.reshape(-1,3)[:count].copy(),d.wrap_obj.flatten()[:count].copy()

    def measure(self,poses,center):
        length,points,tags=self.geometry(poses)
        # Equivalent rotational work about the source glenohumeral center, in
        # the humerus frame. Evaluate all three axes and halve epsilon as a control.
        torque=[];ph,rh=poses[1];world=ph+rh.apply(center)
        for step in (1e-4,5e-5):
            cols=[]
            for axis in np.eye(3):
                qs=[]
                for sign in (-1,1):
                    r=rh*R.from_rotvec(axis*(step*sign))
                    qs.append(self.geometry([poses[0],(world-r.apply(center),r)])[0])
                cols.append((qs[0]-qs[1])/(2*step))
            torque.append(np.array(cols))
        if np.linalg.norm(torque[0]-torque[1])>1e-6:raise ValueError('Moment arm derivative unresolved')
        return dict(length_m=length,moment_arm_m=torque[1].tolist(),wrap_points=len(points))


def poses(data,bs,bh):
    j=read(data/'bed-unscaled-four/native-initial-bodies.json')['all_bodies']
    samples={}
    for k in ('rest','current'):
        samples[k]=[(np.asarray(j[b][k]['position_world_m']),R.from_quat(j[b][k]['orientation_world_xyzw']))for b in (bs,bh)]
    for run in ('common-release-16s','supported-terminal-16s'):
        for f in ('step-3999.receipt.json','terminal.receipt.json'):
            r=read(data.parent/run/f)['accepted_body_poses']
            samples[run+'/'+f]=[(np.asarray(r[b]['position_m']),R.from_quat(r[b]['quaternion_xyzw']))for b in (bs,bh)]
    return samples


def metrics(prog,bs,bh,samples,center,skin):
    engine=SourceRoute(prog,bs,bh);q={}
    for k,frame in samples.items():
        q[k]=engine.measure(frame,center)
    frame=samples['current'];W=lambda b,p:frame[int(b==bh)][0]+frame[int(b==bh)][1].apply(p)
    route=[W(*prog['sites'][i])for t,i in prog['seq'] if t==1]
    labels,d=skin.query(route)
    q['release_sites']=[dict(enclosure=int(v),distance_mm=1000*r) for v,r in zip(labels,d)]
    return q


def main(folder,data,prior,lib):
    reg=read(folder/'source-comparison.json')
    foot=read(folder/'source-footprint.json')
    z=np.load(folder/'correspondence.npz');src=np.load(prior/'source-bones.npz')
    old=read(data/'current-native-forearm/native-import.json');fp=data/'current-native-forearm/mesh.node'
    if digest(fp)!=old['node_sha256']:raise ValueError('Unbound skin')
    geo=np.load(data.parent/'native-input.npz')
    skin=Enclosure(lib,np.loadtxt(fp,skiprows=1,usecols=(1,2,3),dtype='f4')[geo['node_O']],geo['face_O'])
    reports=[];regions={};configs={}
    try:
        for rec,spot in zip(reg['fit'],foot['regions']):
            side=rec['side'];sc='scapula_'+side;hu='humerus_'+side
            if sc!=spot['name']:raise ValueError('Wrong side')
            P=program(data,side);bs=rec['body'];bh=rec['route'][-1]['body']
            plus=next(a for a in read(prior/'rigid.json') if a['bone']==sc)
            g=R.from_quat(plus['rotation_xyzw']).inv();t=np.array(plus['translation_m'])
            target=rec['native_row'];pr=R.from_quat(target[9:13])
            origin=pr.apply(np.array(spot['atlas_foot'])*target[13])+target[6:9]
            originid=P['seq'][0][1];shift=origin-g.apply(P['sites'][originid][1]-t)
            cand=copy.deepcopy(P)
            for idx,(b,v)in P['sites'].items():
                if b==bs:cand['sites'][idx][1]=g.apply(v-t)+shift
            for idx,(b,x,r,ss)in P['wrap'].items():
                if b==bs:cand['wrap'][idx][1]=g.apply(x-t)+shift
            # Recover the source center without assuming the source/world axis convention.
            pts=src[hu+'_world'];local=src[hu+'_local'];u,_,vt=np.linalg.svd((pts-pts.mean(0)).T@(local-local.mean(0)))
            M=u@vt
            if np.max(np.abs((pts-pts.mean(0))@M+local.mean(0)-local))>1e-8 or np.linalg.det(M)<0:raise ValueError('Source bone frame mismatch')
            center=(src['shoulder_rot_'+side+'_center']-pts.mean(0))@M+local.mean(0)
            sample=poses(data,bs,bh)
            base=metrics(P,bs,bh,sample,center,skin);trial=metrics(cand,bs,bh,sample,center,skin)
            regions[side]=dict(base=base,mapped=trial)
            opt=dict(origin=origin,rigid_quaternion=g.as_quat(),translation=shift-g.apply(t),via=cand['sites'][P['seq'][2][1]][1],center=center)
            configs[side]={k:v.tolist()for k,v in opt.items()}
            cases=[]
            for k in sample:
                x,y=base[k],trial[k];v,w=np.array(x['moment_arm_m']),np.array(y['moment_arm_m'])
                cases.append((k,round(x['length_m']*1000,2),round(y['length_m']*1000,2),round(np.linalg.norm(w-v)*1000,2),round(v@w/(np.linalg.norm(v)*np.linalg.norm(w)),3)))
            print(side,'release',trial['release_sites'],'base/mapped length mm, moment arm difference mm and direction cosine',cases,flush=True)
    finally:skin.close()
    (folder/'route-fit.json').write_text(json.dumps(dict(fit=regions,transform=configs,mujoco=mujoco.__version__,physical_skin=old['node_sha256']),indent=2)+'\n')
if __name__=='__main__':main(*(Path(p).resolve()for p in sys.argv[1:]))
