"""Check native point and distributed force attachment locations in accepted whole-body frames."""
from pathlib import Path
import csv,hashlib,json,re,runpy,struct,sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from measure_forearm_enclosure import Enclosure
from recount_registered_crossings import read,digest


def verify(data,control,case,lib,out):
    a=read(case/'launch.json')['argv'];t=Path(a[a.index('--tendon-payload')+1])
    tm=read(t.with_suffix('.manifest.json'));tb=t.read_bytes();mb=Path(a[2]).read_bytes()
    witness=read(t.parent/'registration.json')
    if digest(t)!=witness['tendon'] or hashlib.sha256(mb).hexdigest()!=witness['muscle']:
        raise ValueError('Native physical inputs changed')
    h=struct.unpack_from('<8s10I32s32s32s',tb);rows={}
    for r in witness['bilateral']:
        k=next(i for i,e in enumerate(tm['endpoints']) if e['muscle']==r['muscle'] and e['endpoint']=='origin')
        ep=struct.unpack_from('<8I8f',tb,144+k*64)
        if ep[5]!=2 or ep[3]!=r['site'] or ep[4]!=r['body']:raise ValueError('Wrong force owner')
        patch=struct.unpack_from('<4I68f',tb,144+64*h[5]+288*ep[6]);orig=np.asarray(ep[8:11]);node=np.array(patch[4:20]).reshape(4,4)[:,:3]
        if patch[0]!=ep[4] or patch[3]!=4:raise ValueError('Wrong physical node count')
        via=struct.unpack_from('<I3f',mb,76+16*r['via'])
        if via[0]!=ep[4]:raise ValueError('Cross-body proximal segment')
        points=np.vstack((r['reference_origin_m'],orig,via[1:],node))
        rows[r['muscle']]=(ep[4],points)
    root=case/'native-traces';end=max(root.glob('accepted-geometry/step-*.receipt.json'),key=lambda p:int(p.name.split('-')[1].split('.')[0]))
    final=read(end);n=final['accepted_step']
    if n!=int(a[a.index('--muscle-step-count')+1]) or final['physical_endpoint']!='accepted':raise ValueError('Incomplete native horizon')
    rp=next(p/'resting-surface-clearance-001/audit.py' for p in data.parents if (p/'resting-surface-clearance-001/audit.py').is_file())
    sys.path.insert(0,str(rp.parent/'src'));api=runpy.run_path(str(rp))
    summaries=[]
    for file in [None,*sorted(end.parent.glob('step-*.receipt.json'),key=lambda p:int(p.name.split('-')[1].split('.')[0]))]:
        record=read(file) if file else final
        pack=file.with_name(file.name.replace('receipt.json','mrvpack')) if file else root/'resting-human.mrvpack'
        hash_key='pack_file_sha256' if file else 'base_pack_file_sha256'
        if record[hash_key]!=digest(pack):raise ValueError('Wrong native frame')
        pose={p['body_index']:p for p in (record['accepted_body_poses'] if file else record['initial_anatomical_registration']['body_poses'])}
        if not file and any(b not in pose for b,p in rows.values()):
            k=case/'native-initial-bodies.json'
            if not k.is_file():raise ValueError('No complete release body capture')
            for p in read(k)['all_bodies']:
                rr=dict(position_m=p['current']['position_world_m'],quaternion_xyzw=p['current']['orientation_world_xyzw'])
                pose.setdefault(p['body_index'],rr)
        (_,_,v,f),=api['primitives'](pack.read_bytes(),pose,skin_only=True)
        shell=Enclosure(lib,v,f);detail={}
        try:
            for name,(b,points) in rows.items():
                def depth(P):
                    label,d=shell.query(R.from_quat(pose[b]['quaternion_xyzw']).apply(P)+pose[b]['position_m'])
                    return [dict(enclosed=int(q),distance_mm=float(1000*dv)) for q,dv in zip(label,d)]
                sample=depth(points)
                line=depth(np.linspace(points[1],points[2],81))
                detail[name]=dict(previous_origin=sample[0],registered_origin=sample[1],unchanged_intermediate=sample[2],traction=sample[3:],
                    proximal_line_samples=len(line),outside_proximal_samples=sum(q['enclosed']==0 for q in line))
        finally:shell.close()
        summaries.append(dict(step=record['accepted_step'] if file else 0,pack=record[hash_key],sides=detail))
    base=control/'resting-coupled.csv' if (control/'resting-coupled.csv').is_file() else control/'native-traces/resting-coupled.csv'
    def table(p):
        with p.open() as f:return list(csv.DictReader(f))
    C,Y=table(base),table(root/'resting-coupled.csv')
    if len(C)!=len(Y) or any(x['time_s']!=y['time_s'] for x,y in zip(C,Y)):raise ValueError('Different physical clocks')
    cols=['lung_volume_ml','pleural_pa','PaO2_mmhg','PaCO2_mmhg','blood_ml','lv_ml','rv_ml','complete_filling_ejection_cycles','breaths']
    values={}
    for k in cols:
        if not Y:continue
        x=np.array([float(r[k])for r in C]);y=np.array([float(r[k])for r in Y])
        if not np.isfinite(np.r_[x,y]).all():raise ValueError('Nonfinite physiology')
        values[k]=dict(minimum=float(y.min()),maximum=float(y.max()),terminal=float(y[-1]),maximum_control_delta=float(np.abs(y-x).max()))
    stdout=(case/'stdout.log').read_text();timing={}
    for prefix,k in [('resting_integrated_throughput','warm'),('resting_integrated_body=completed','body')]:
        line=next((v for v in stdout.splitlines() if v.startswith(prefix)),None)
        if line:timing[k]=float(re.search(r'real_time_factor=([\d.eE+-]+)',line).group(1))
    ex=read(case/'execution.json');wall=ex['wall_s'];timing.update(cold=final['accepted_time_s']/wall,cold_wall_s=wall)
    report=dict(launch=digest(case/'launch.json'),control=digest(control/'launch.json'),variable_pleural=read(case/'launch.json')['environment'].get('NM_VASCULAR_STEP_EXTERNAL_PRESSURE'),
                accepted_s=final['accepted_time_s'],accepted_steps=n,samples=len(Y),timing=timing,poses=summaries,physiology=values)
    out.write_text(json.dumps(report,indent=2)+'\n');print('timing',timing,flush=True)
    for p in summaries:print(p['step'],[(name,round(r['registered_origin']['distance_mm'],2),[x['enclosed'] for x in r['traction']],r['unchanged_intermediate']['enclosed'])for name,r in p['sides'].items()])
if __name__=='__main__':verify(*(Path(p).resolve()for p in sys.argv[1:]))
