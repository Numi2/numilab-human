"""Install qualified source material vertices and transport normal directions with the repaired surface."""
import json,sys
from pathlib import Path
import numpy as np
from registered_forearms import Muscle,read,records,sha
from stage_bound_assets import stage
from reconcile_forearm_parent import transfer

def write(path,data):
    path.write_text(json.dumps(data,separators=(',',':'))+'\n')

def emit(m,candidate,out):
    buf=bytearray(m.owner.read_bytes());tables,vo,nv=records(buf)
    row,=tables[tables[:,6]==m.bank];_,_,vi,n,fi,ni,_,_=map(int,row)
    vertices=np.frombuffer(buf,'<f4',14*nv,vo).reshape(nv,14)[vi:vi+n]
    faces=np.frombuffer(buf,'<u4',ni,vo+56*nv+4*fi).reshape(-1,3).astype(int)-vi
    if faces.min()<0 or faces.max()>=n:raise ValueError('Foreign vertex in material')
    with np.load(candidate) as z:new=z['positions'];replace=z['changed']
    current=vertices[:,:3].copy()
    if not np.array_equal(current,m.x) or new.shape!=current.shape or not np.array_equal(replace,np.any(current!=new,axis=1)):
        raise ValueError('Mismatched complete source coordinates')
    unique,group=np.unique(current,axis=0,return_inverse=True)
    grouped=np.zeros_like(unique);grouped[group]=new
    if not np.array_equal(grouped[group],new):raise ValueError('Previously coincident source seam opened')
    f=group[faces];used=np.any(replace[faces],axis=1);nbr=np.unique(f[used])
    def vectors(x):
        tri=x.astype(float)[f];area=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        nrm=np.zeros((len(x),3))
        for k in range(3):np.add.at(nrm,f[:,k],area)
        return nrm
    before=vectors(unique);after=vectors(grouped)
    num=np.linalg.norm(before,axis=1);den=np.linalg.norm(after,axis=1)
    active=np.isin(group,nbr)&(num[group]>1e-16)&(den[group]>1e-16)
    a=before[group[active]]/num[group[active],None];b=after[group[active]]/den[group[active],None]
    v=np.cross(a,b);c=(a*b).sum(1)
    if np.any(c<0):raise ValueError('Source normal flipped')
    N=vertices[active,3:6].copy();rotated=N+np.cross(v,N)+np.cross(v,np.cross(v,N))/(1.+c[:,None])
    if not np.isfinite(rotated).all() or np.max(abs(np.linalg.norm(rotated,axis=1)-np.linalg.norm(N,axis=1)),initial=0)>1e-6:raise ValueError('Normal length changed')
    vertices[active,3:6]=rotated;vertices[:,:3]=new
    out.write_bytes(buf)
    return dict(start_sha256=sha(m.owner),end_sha256=sha(out),source_member=m.bank,registered_vertices=int(replace.sum()),
        source_vertices=n,normals_rotated=int(active.sum()),max_coordinate_change_m=float(np.linalg.norm(new-current,axis=1).max()))

def install(control,selected,out,rest_pose):
    out.mkdir(exist_ok=False)
    job=read(control/'launch.json');args=job['argv'];soft=args.index('--soft-tissue-payload')+1;ar=args.index('--resting-anatomy-receipt')+1
    old=Path(args[ar]);receipt=read(old);ext=receipt['passive_tissue_parent_bindings'];child=Path(args[soft])
    meta=read(child.with_suffix('.manifest.json'));notes={};direct=None;derived=[]
    for k in (109,110):
        info=read(selected/f'{k}-selected.json')
        if not info['admitted']:continue
        m=Muscle(job,k);point=selected/f'{k}-selected.npz'
        if sha(point)!=info['selected_sha256'] or info['source_sha256']!=sha(m.owner):raise ValueError('Source fit differs')
        if m.parent_index is not None:
            path=out/('parent-'+str(m.parent_index)+'.nhtissue');tag=ext['source_payloads'][m.parent_index]
            # Enumerate any other child that can depend on the changed bank.
            deps=[r['stable_id'] for key in ('surfaces','anatomy_surfaces') for r in ext[key]
                if any(t[0]==m.parent_index and t[1]==m.bank for v in r['vertices'] for t in v)]
            if deps!=[k] or tag['sha256']!=sha(m.owner):raise ValueError(('Unexpected cross-bank dependency',deps))
            change=emit(m,point,path)
            derived.append((m,path))
            tag.update(path=str(path),sha256=sha(path));job['pins'][str(path)]=sha(path)
        else:
            if direct is not None:raise ValueError('Overlapping direct child update')
            direct=out/child.name;change=emit(m,point,direct)
        notes[k]=dict(**change,owning_payload=m.parent_index,source_route=m.sites,proposal=sha(point),qualification=sha(selected/f'{k}-selected.json'))
    if not notes:raise ValueError('No registered geometry passed')
    for muscle,parent_path in derived:
        target=out/child.name;source=direct or child
        resident=out/(target.stem+'.reconciled.nhtissue')
        notes[muscle.sid]['registered_child']=transfer(muscle,source,parent_path,resident,rest_pose,emit)
        resident.replace(target);direct=target
    if direct is not None:
        ext['child_payload_sha256']=sha(direct);args[soft]=str(direct)
        meta['payload']['sha256']=sha(direct);meta['source']['supported_forearm_material_fit']=dict(parent_sha256=sha(child),regions=notes)
        mp=direct.with_suffix('.manifest.json');write(mp,meta)
        pr=receipt['provenance']['native_muscle_surfaces']
        if pr['sha256']!=sha(child):raise ValueError('Original surface provenance differs')
        pr.update(sha256=sha(direct),payload_path=str(direct),manifest_path=str(mp),manifest_sha256=sha(mp))
        job['pins'].update({str(direct):sha(direct),str(mp):sha(mp)})
    receipt['provenance']['supported_forearm_material_fit']=dict(parent_receipt=sha(old),regions=notes,kind='local source registration of passive muscle envelopes with unchanged named MyoSim routes')
    new=out/'resting-anatomy-receipt.json';write(new,receipt);args[ar]=str(new)
    job['pins'][str(new)]=sha(new)
    for name,h in stage(old,new).items():job['pins'][str(out/name)]=h
    job['parent_launch']=sha(control/'launch.json')
    write(out/'launch.json',job);write(out/'registration.json',notes)
if __name__=='__main__':install(*(Path(v).resolve() for v in sys.argv[1:]))
