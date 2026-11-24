"""Measure occupied and unassigned segments of the actual dominant bed normal in the coupled frame."""
import csv,json,mmap,runpy,sys
from pathlib import Path
import numpy as np
from assemble_material import frames
from registration_stages import parts
from repair_registered_surface import state
from prepare_shoulder import MOVING,read
from trace_loaded_columns import hits
from compile_connected_joint import digest

def trace(case,vol,registry,fit,out):
    cfg=read(case/'launch.json');audit=read(case/'material-check.json');support=read(case/'support-check.json')
    if audit['physical_frame']!=digest(case/'physical-accepted.node') or audit['full_scene_launch']!=digest(case/'launch.json'):raise ValueError('Mismatched physical state')
    ticket=case/'native-traces/accepted-geometry'/f"step-{audit['accepted_step']}.receipt.json";cap=read(ticket)
    pose={b['body_index']:b for b in cap['accepted_body_poses']};pack=Path(cap['accepted_pack_path'])
    if digest(pack)!=cap['pack_file_sha256']:raise ValueError('Mismatched pose geometry')
    history=case/'physical-posterior-bed-history.csv'
    with history.open()as f:imp={int(r['registered_skin_node']):float(r['terminal_microstep_normal_impulse_Ns'])for r in csv.DictReader(f)}
    n=np.asarray(support['bed_normal']);x=np.cross([0,1.,0],n);x/=np.linalg.norm(x);axes=np.column_stack((x,np.cross(n,x),n))
    xyz=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3));node=max(imp,key=imp.get);q=xyz[[node]]@axes
    q=np.repeat(q,5,axis=0);q[1:,0:2]+=np.array([[1,0],[-1,0],[0,1],[0,-1]])*25e-6
    w=np.load(vol/'material-walls.npz');wall={}
    for tag in 'ID':
        hh=hits(xyz@axes,w[tag],tag,q)
        if any(not h for h in hh):raise ValueError('Missing physical skin/fat wall')
        wall[tag]=[min(l for l,_ in h)for h in hh]
    args=cfg['argv'];raw=Path(args[args.index('--soft-tissue-payload')+1]);api=runpy.run_path(str(registry))
    exams={k:[{}for _ in q]for k in ('native','source_candidate')};other=set()
    def classify(pts,tri,key,slot):
        pos=pts@axes
        if np.any(pos[:,:2].max(0)<q[:,:2].min(0))or np.any(pos[:,:2].min(0)>q[:,:2].max(0)):return
        start=q.copy();start[:,2]=pos[:,2].min()-0.01
        for k,hs in enumerate(hits(pos,tri,key[1],start)):
            depths=sorted(z+start[k,2]-q[k,2]for z,_ in hs)
            uniq=[v for i,v in enumerate(depths) if i==0 or v-depths[i-1]>1e-9]
            if len(uniq)%2:
                if any(0.<d<.05 for d in uniq):raise ValueError(f'Unresolved parity in {key}')
                continue
            intervals=[(lo,hi)for lo,hi in zip(uniq[::2],uniq[1::2])if hi>0 and lo<.18]
            if intervals:slot[k][key]=intervals
    with pack.open('rb') as h,mmap.mmap(h.fileno(),0,access=mmap.ACCESS_READ)as mm:
        for key,_,p,f in api['primitives'](mm,pose):
            if key[0] not in (51004,51005):other.add(key[0]);continue
            classify(p,f,key,exams['native'])
            if key[0]==51005 and key[1]in MOVING:
                z=np.load(fit/f'bank-{key[1]}.npz');_,_,ids,_,a=parts(raw,key[1])
                p=state(z['x'],z['w'],a,ids,pose).astype('f4').astype(float);f=z['f']
                classify(p,f,key,exams['source_candidate'])
            else:
                for k,row in enumerate(exams['native']):
                    if key in row:exams['source_candidate'][k][key]=row[key]
    result={}
    for mode,samples in exams.items():
        data=[]
        for k,s in enumerate(samples):
            bones=[(max(0.,lo),bank)for (kind,bank),intervals in s.items()if kind==51004 for lo,hi in intervals if hi>0]
            if not bones:raise ValueError('No bone on load normal')
            stop,bank=min(bones);spans=[(0.,wall['I'][k],'skin'),(wall['I'][k],wall['D'][k],'fat')]
            spans.extend((max(0.,lo),min(hi,stop),str(b))for (kind,b),vals in s.items()if kind==51005 for lo,hi in vals if hi>0 and lo<stop)
            edges=sorted({0.,stop,*[max(0.,min(stop,t))for lo,hi,name in spans for t in (lo,hi)]})
            segments=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                mid=(lo+hi)/2;materials=[name for a,b,name in spans if a<=mid<b]
                segments.append(dict(start_mm=float(lo*1000),end_mm=float(hi*1000),registered_occupants=materials))
            data.append(dict(bone=bank,depth_mm=stop*1000,unassigned_mm=sum(s['end_mm']-s['start_mm']for s in segments if not s['registered_occupants']),segments=segments))
        result[mode]=data
    record=dict(native=digest(case/'launch.json'),registered_source=digest(fit/'results.json'),pose=digest(ticket),material_endpoint=audit['physical_frame'],tendon_payload=digest(Path(args[args.index('--tendon-payload')+1])),registry=digest(registry),fraction=imp[node]/sum(imp.values()),node=node,skin_bed_N=support['accepted_normal_force_N'],samples=result,
        exclusions=sorted(other),scope='Straight bed-normal coverage by registered skin, fat, muscle and bone. Four lateral probes at 0.025 mm. Unassigned intervals are outside these modeled solids and do not establish absence of fascia, other tissue or oblique support.')
    out.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    for k,r in result.items():print(k,node,record['fraction'],r[0],flush=True)
if __name__=='__main__':trace(*(Path(s).resolve()for s in sys.argv[1:]))
