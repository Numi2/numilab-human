"""Compare neighboring solids using the geometry actually emitted at the matched accepted native clock."""
import json,mmap,runpy,sys
from pathlib import Path
import numpy as np
from manifold3d import Manifold,Mesh64,Error,OpType
from compile_connected_joint import digest
from prepare_shoulder import read
from measure_assembly_contacts import canonical
from build_carpal_cavity import boundary
from prepare_infraspinatus import dets


def load(case,api):
    report=read(next((case/'native-traces/accepted-geometry').glob('*.receipt.json')));pack=Path(report['accepted_pack_path'])
    if digest(pack)!=report['pack_file_sha256']:raise ValueError('Invalid native replay hash')
    body={q['body_index']:q for q in report['accepted_body_poses']}
    with pack.open('rb')as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)as m:
        shapes={key:(v.copy(),tri.copy())for key,_,v,tri in api['primitives'](m,body)}
    return shapes,report


def layer_parts(points,tets,labels,low,high):
    selected=np.concatenate([i[np.all((points[tets[i]].max(1)>=low-4)&(points[tets[i]].min(1)<=high+4),axis=1)]for i in np.array_split(np.arange(len(tets)),64)])
    def solid(ids):
        t=tets[ids];f=boundary(t);idx,inv=np.unique(f,return_inverse=True);f=inv.reshape(-1,3)
        y=np.ascontiguousarray(points[idx]);f=np.ascontiguousarray(f,dtype='u8');s=Manifold(Mesh64(y,f));expected=dets(points,t).sum()/6
        if s.volume()<0:s=Manifold(Mesh64(y,np.ascontiguousarray(f[:,[0,2,1]])))
        if s.status()==Error.NoError and abs(s.volume()-expected)<max(1e-8,expected*1e-9):return [s]
        if len(ids)<2:raise ValueError('Unresolvable material in close proximity')
        centers=points[t].mean(1);order=np.argsort(centers[:,np.ptp(centers,axis=0).argmax()],kind='stable')
        return [v for g in np.array_split(order,2)for v in solid(ids[g])]
    return {k:solid(selected[labels[selected]==k])for k in (15,16)}


def run(w,new,ids,out):
    old=w/'native-cavity-bound';api=runpy.run_path((w/'primitive-registry.path').read_text().strip());A,a=load(old,api);B,b=load(new,api)
    if a['accepted_step']!=b['accepted_step']:raise ValueError('Different accepted endpoint')
    selected=[(51005,int(z))for z in ids.split(',')];org=np.vstack([A[k][0]for k in selected]).mean(0)
    details={};dep={};v=w/'material-import/volume';raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    for label,case,shape in [('control',old,A),('changed',new,B)]:
        items=[];invalid=[];active={};lims={};bounds=[]
        for k in selected:
            y=(shape[k][0]-org)*1000;s,meta=canonical(y,shape[k][1]);active[k]=s;lims[k]=(y.min(0),y.max(0));bounds+=list(lims[k])
            if s is None:raise ValueError('Edited native surface is not a solid')
        for k,(vertices,faces)in shape.items():
            if k in selected or k==(51007,1)or k[0] not in api['BANK_CLASSES']:continue
            y=(vertices-org)*1000;lo,hi=y.min(0),y.max(0)
            relevant=[s for s,(l,h)in lims.items()if np.all(hi>=l)&np.all(lo<=h)]
            if not relevant:continue
            other,meta=canonical(y,faces)
            if other is None:invalid.append([list(k),meta]);continue
            for s in relevant:
                intersection=active[s]^other
                if intersection.status()!=Error.NoError:raise ValueError('Unresolved solid intersection')
                items.append([list(s),list(k),float(intersection.volume())])
        for i,k in enumerate(selected):
            for j in selected[i+1:]:items.append([list(k),list(j),float((active[k]^active[j]).volume())])
        frame=case/'physical-accepted.node';p=(np.loadtxt(frame,skiprows=1,usecols=(1,2,3))-org)*1000
        parts=layer_parts(p,raw[:,:4]-1,raw[:,4],np.min(bounds,axis=0),np.max(bounds,axis=0))
        for k in selected:
            for layer,solids in parts.items():
                terms=[active[k]^s for s in solids]
                if any(x.status()!=Error.NoError for x in terms):raise ValueError('Unresolved skin or adipose intersection')
                items.append([list(k),[0,layer],float(sum(x.volume()for x in terms))])
        dep[label]=dict(material=digest(frame),capture=digest(next((case/'native-traces/accepted-geometry').glob('*.receipt.json'))),pack=(a if label=='control'else b)['pack_file_sha256'])
        details[label]=dict(pairs=items,unresolved=invalid)
    def keyed(k):return {(tuple(i),tuple(j)):v for i,j,v in details[k]['pairs']}
    c=keyed('control');n=keyed('changed');diff=[dict(bank=i[1],target=list(j),old_mm3=c.get((i,j),0),new_mm3=n.get((i,j),0),delta_mm3=n.get((i,j),0)-c.get((i,j),0))for i,j in sorted(set(c)|set(n))]
    rec=dict(poses=dep,checks=details,changes=diff,scope='Pairwise intersection volume on closed neighboring solids; normalized existing folded comparison surfaces are diagnostic. Volumes for different pairs may overlap spatially.')
    out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
    print('largest increases',sorted(diff,key=lambda d:-d['delta_mm3'])[:4],flush=True)
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]))
