"""Build cortex contact samples and energy-normalized material modes for the native articulated transaction."""
import json,sys
from pathlib import Path
import numpy as np
from pose_glutes import parts
from scipy.spatial.transform import Rotation
from prepare_material_interface import read_inputs
from qualify_material_motion import frame,local
from prepare_posterior_muscles import mesh_volume,closed
from material_context import sha
from numilab_human.capsular_material_registration import read_material_banks,material_bank,emitted_body_vertices

def directions(x,f):
    e=x[f[:,1]]-x[f[:,0]];g=x[f[:,2]]-x[f[:,0]];n=np.cross(e,g);n/=np.linalg.norm(n,axis=1)[:,None]
    vertex=np.zeros_like(x);lines=[]
    for j in range(3):
        a=x[f[:,(j+1)%3]]-x[f[:,j]];b=x[f[:,(j+2)%3]]-x[f[:,j]]
        angle=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b))
        np.add.at(vertex,f[:,j],n*angle[:,None]);lines.append(np.sort(f[:,[j,(j+1)%3]],axis=1))
    vertex/=np.linalg.norm(vertex,axis=1)[:,None];pairs,rev,count=np.unique(np.concatenate(lines),return_inverse=True,return_counts=True,axis=0)
    if np.any(count!=2):raise ValueError('Unclosed bone edge')
    en=np.zeros((len(pairs),3));np.add.at(en,rev,np.tile(n,(3,1)))
    en/=np.linalg.norm(en,axis=1)[:,None]
    return np.concatenate((x[f],n[:,None],vertex[f],en[rev.reshape(3,-1).T]),axis=1)
def run(p47,prev,base,response,out):
    out.mkdir(exist_ok=True);cfg=prev/'release-posterior/config.json';rc=prev/'native-release-motion/step-8.receipt.json'
    full={r['body_index']:frame(r) for r in json.loads(rc.read_text())['initial_anatomical_registration']['body_poses']};args=json.loads(cfg.read_text())['argv']
    asset=Path(args[args.index('--soft-tissue-payload')+1]);surfs,_=read_inputs(cfg,rc,prev/'native-release-motion',base)
    tris=[];banks={};offset=0
    for oid,b in ((1,1),(2,131),(3,145),(18,128),(19,128)):
        v,f=surfs[f'bone_{oid}'];v,iv=np.unique(v,axis=0,return_inverse=True);f=iv[f]
        if any(closed(v,f).values()):raise ValueError(oid)
        if mesh_volume(v,f)<0:f=f[:,[0,2,1]]
        x=local(v,full[b]);rows=directions(x,f);tris.append(rows);banks[oid]=(offset,len(rows),b,rows[:,:3].min(1),rows[:,:3].max(1));offset+=len(rows)
    dtype=np.dtype([('pelvis','<f4',4),('femur','<f4',4),('reference','<f4',4),('domain','<u4',4),('bank','<u4',4)])
    records=np.zeros((2,256),dtype=dtype);m=np.zeros((2,256,96,4),dtype='f4');lookup=[];meta=[]
    for si,sid in enumerate((29,30)):
        p=np.load(p47/'projected'/f'{sid}.npz');pt=np.load(p47/'ports'/f'{sid}.npz');red=np.load(response/f'{sid}.npz')
        data=parts(asset,sid);verts,weights,donors,_,aff=data
        if list(donors)!=[128,131 if sid==29 else 145]:raise ValueError(donors)
        rot=Rotation.from_quat(full[128][3:]).inv().as_matrix();modes=np.einsum('ij,njk->nik',rot,red['modes'])
        nk=modes.shape[-1];patch=p['port_vertices'];world=patch.mean(1);area=np.linalg.norm(np.cross(patch[:,1]-patch[:,0],patch[:,2]-patch[:,0]),axis=1)/2
        chosen=[]
        for group in np.unique(red['pressure_region'][pt['target_body']>=0]):
            ix=np.flatnonzero(red['pressure_region']==group);cent=np.average(world[ix],axis=0,weights=area[ix])
            first=ix[np.argmin(np.linalg.norm(world[ix]-cent,axis=1))]
            # Include two widely separated material samples per source pressure patch.
            second=ix[np.argmax(np.linalg.norm(world[ix]-world[first],axis=1))]
            chosen.extend(sorted({int(first),int(second)}))
        if len(chosen)>256 or nk>96:raise ValueError('Native local workgroup capacity')
        for j,i in enumerate(chosen):
            bary=p['port_shape'][i].mean(0);node=p['cells'][p['port_owner_cell'][i]];row=records[si,j]
            for k,name in enumerate(('pelvis','femur')):
                w=bary*weights[node,k];total=w.sum()
                local_nodes=verts[node]@aff[k][0].T+aff[k][1]
                row[name][:3]=(w@local_nodes)/total if total>1e-15 else 0;row[name][3]=total
            oid=int(pt['target_bank'][i][5:]);start,n,owner,lo,hi=banks[oid];ref=local(world[i],full[owner]);radius=.004
            nearby=np.flatnonzero((np.maximum(lo-ref,np.maximum(ref-hi,0))**2).sum(1)<=radius**2)
            if not len(nearby):raise ValueError('Lost registered cortical facet')
            row['reference']=np.r_[ref,radius];row['domain']=[128,donors[1],owner,nk];row['bank']=[start,n,len(lookup),len(nearby)]
            lookup.extend((nearby+start).tolist());m[si,j,:nk,:3]=(np.einsum('i,ijk->jk',bary,modes[node])).T
        meta.append(dict(muscle=sid,rows=len(chosen),modes=nk,quadrature_indices=chosen))
    geometry=np.concatenate(tris);packed=np.zeros((len(geometry),10,4),dtype='f4');packed[:,:,:3]=geometry
    files={}
    for name,table in [('cortical-triangles',packed),('cortical-neighborhood',np.array(lookup,'<u4')),('cortical-samples',records),('cortical-modes',m)]:
        p=out/(name+'.bin');table.tofile(p);files[p.name]=sha(p)
    supported=json.loads((prev/'supported-release/config.json').read_text())['argv'];paths=[Path(args[3]),Path(supported[3])]
    surgeons=[read_material_banks(path) for path in paths];evidence={}
    import hashlib
    for oid in banks:
        records=[]
        for bank in surgeons:
            b=bank[2][oid][0];_,r,v,f=material_bank(bank,oid,b)
            records.append((b,emitted_body_vertices(v,r).astype('<f4'),f.astype('<u4')))
        if records[0][0]!=records[1][0] or not all(np.array_equal(a,b) for a,b in zip(records[0][1:],records[1][1:])):raise ValueError(('Different live cortical bank',oid))
        evidence[oid]=dict(owner=records[0][0],points_sha256=hashlib.sha256(records[0][1].tobytes()).hexdigest(),faces_sha256=hashlib.sha256(records[0][2].tobytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(dict(sides=meta,files=files,source=sha(asset),bones=sha(Path(args[3])),qualified_bone_payloads=[sha(q) for q in paths],matched_cortical_banks=evidence,model='Quasistatic cut-muscle P1 compliance with unilateral cortical collocation. Muscle inertia remains in articulated donors.'),indent=2)+'\n')
    print([(a['rows'],a['modes']) for a in meta])
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
