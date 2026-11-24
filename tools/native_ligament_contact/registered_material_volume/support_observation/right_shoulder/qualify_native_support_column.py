"""Intersect the native observer's physical tetrahedra with the bed normal at the largest reaction."""
import csv,json,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def read(p):return json.loads(p.read_bytes())
def rows(p):
    with p.open()as f:return list(csv.DictReader(f))
def interval(t,pt,n,depth):
    A=(t[1:]-t[0]).T
    if np.linalg.det(A)<=0:raise ValueError('Nonpositive native tissue in support bounds')
    pair=np.linalg.solve(A,np.column_stack((pt-t[0],n)));v=np.vstack((np.array([1.,0.])-pair.sum(0),pair));lo,hi=0.,depth
    for q,vv in v:
        if abs(vv)<1e-12:
            if q< -1e-12:return None
        elif vv>0:lo=max(lo,-q/vv)
        else:hi=min(hi,-q/vv)
    return (lo*1000,hi*1000) if hi-lo>1e-10 else None

def qualify(case,parent,corridor,out):
    e=read(case/'execution.json');s=read(case/'launch.json');c=read(corridor)
    if e['exit'] or s['observer_control']!=digest(parent/'launch.json') or s['column_target']!=digest(corridor):raise ValueError('Unmatched coupled observation')
    new=rows(case/'native-traces/resting-coupled.csv');control=rows(parent/'native-traces/resting-coupled.csv')
    if len(new)!=len(control) or not new:raise ValueError('Changed accepted integrator history')
    physical=['time_s','lung_volume_ml','pleural_pa','PaO2_mmhg','PaCO2_mmhg','root_assistance_n'];d={k:max(abs(float(a[k])-float(b[k]))for a,b in zip(new,control))for k in physical}
    xyz=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3));old=np.loadtxt(parent/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    coorddiff=np.linalg.norm(xyz-old,axis=1);step=int(new[-1]['step']);f=read(case/'native-traces/accepted-geometry'/f'step-{step}.receipt.json')
    if f['accepted_step']!=step or f['physical_endpoint']!='accepted' or abs(f['accepted_time_s']-float(new[-1]['time_s']))>1e-11:raise ValueError('Unconfirmed native state')
    n=np.array(read(parent/'support-check.json')['bed_normal']);p=xyz[c['node']];x=np.cross([0.,1.,0.],n);x/=np.linalg.norm(x);y=np.cross(n,x)
    probes=[p,p+x*25e-6,p-x*25e-6,p+y*25e-6,p-y*25e-6]
    obj=rows(case/'physical-posterior-tissue-objects.csv');cells=rows(case/'physical-posterior-tissue-cells.csv')
    if sum(int(r['overlapping_bounds']) for r in obj)!=len(cells) or len({r['global_cell']for r in cells})!=len(cells):raise ValueError('Incomplete native AABB scan')
    raw=[]
    for row in cells:
        points=np.array([[float(row[k+str(i)])for k in 'xyz']for i in range(4)]);raw.append((row['object'],points))
    label={o['object']:o['name'] for o in obj};coverage=[]
    for i,pt in enumerate(probes):
        base=c['samples']['native'][i];depth=base['depth_mm']*.001
        # The registered skin/fat owner is omitted from the native observer dump.
        # Its complete physical tetrahedra and separate O/I/D walls are checked independently.
        found=[(j,*seg)for j,points in raw if (seg:=interval(points,pt,n,depth))]
        bins=base['segments'];edges=sorted({0.,base['depth_mm'],*[d for j,lo,hi in found for d in (lo,hi)],*[d for b in bins for d in (b['start_mm'],b['end_mm'])]})
        pieces=[]
        for a,b in zip(edges[:-1],edges[1:]):
            if b-a<1e-7:continue
            m=(a+b)/2;named=next((r['registered_occupants']for r in bins if r['start_mm']<m<r['end_mm']),[])
            live=sorted({label[k]for k,lo,hi in found if lo<m<hi})
            pieces.append(dict(start_mm=a,end_mm=b,registered=named,other_live_continuum=live))
        coverage.append(dict(segments=pieces,unassigned_mm=sum(b['end_mm']-b['start_mm']for b in pieces if not(b['registered'] or b['other_live_continuum'])),cells=len(found)))
    target=[r for r in rows(case/'physical-posterior-bed-history.csv') if int(r['registered_skin_node'])==c['node']]
    if len(target)!=1:raise ValueError('Critical receiver changed')
    result=dict(launch=digest(case/'launch.json'),accepted_frame=digest(case/'physical-accepted.node'),physics_delta=d,max_material_node_delta_m=coorddiff.max(),observer_objects=obj,roi_tets=len(raw),coverage=coverage,receiver=c['node'],time_s=float(new[-1]['time_s']),cold_wall_s=e['wall_s'],accepted_receipt=digest(case/'native-traces/accepted-geometry'/f'step-{step}.receipt.json'))
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('all native objects',len(obj),'nearby tets',len(raw),'intersected',[q['cells']for q in coverage],'unassigned mm',[q['unassigned_mm']for q in coverage])
if __name__=='__main__':qualify(*(Path(v).resolve()for v in sys.argv[1:]))
