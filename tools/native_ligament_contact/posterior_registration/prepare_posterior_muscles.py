"""Measure the registered bilateral gluteal interfaces and tetrahedralize their exact source surfaces."""
import collections
import json
import mmap
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
from material_context import get_layers, sha
from compare_hand_interfaces import query

def load_registered(w, cap, package, audit):
    receipt=w/'qualified/motion-008/step-8.receipt.json'
    pack=w/'qualified/native-coupled-006/native/resting-human.mrvpack'
    rc=json.loads(receipt.read_text())
    if sha(pack)!=rc['base_pack_file_sha256']:raise ValueError('Entrance pack mismatch')
    cfg=json.loads(package.read_text())['argv']
    payload=Path(cfg[cfg.index('--soft-tissue-payload')+1])
    manifest=json.loads(payload.with_suffix('.manifest.json').read_text())
    if sha(payload)!=manifest['payload']['sha256']:raise ValueError('Payload mismatch')
    native=next(json.loads(row.partition('=')[2]) for row in (cap/'emitted-native/stderr.log').open() if row.startswith('registered_ligament_actual_pose_json='))
    if native['bonePayloadSha256']!=sha(Path(cfg[3])):raise ValueError('Different skeletal owner')
    poses={i:dict(position_m=p[:3],quaternion_xyzw=p[3:]) for i,p in enumerate(native['bodyPoses'])}
    for p in rc['initial_anatomical_registration']['body_poses']:
        q=poses[p['body_index']]
        if max(abs(np.array(p['position_m']+p['quaternion_xyzw'])-np.array(q['position_m']+q['quaternion_xyzw'])))>1e-7:raise ValueError('Frame mismatch')
    names={s['stable_id']:s for s in manifest['source']['surfaces']}
    with pack.open('rb') as fh,mmap.mmap(fh.fileno(),0,access=mmap.ACCESS_READ) as data:
        meshes={key:(v.copy(),f.copy()) for key,i,v,f in runpy.run_path(str(audit/'audit.py'))['primitives'](data,poses)}
    layers=get_layers(pack,receipt)
    for key in ((51007,1),(51007,4294966545),(51007,4294966546)):meshes.pop(key,None)
    meshes.update({(51007,k):layers[name] for k,name in [(1,'outer'),(4294966545,'inner'),(4294966546,'deep')]})
    return meshes,names,dict(pack=sha(pack),receipt=sha(receipt),soft_payload=sha(payload),bone_payload=native['bonePayloadSha256'])
def interface_audit(key,meshes,exe):
    v,f=meshes[key]
    lo,hi=v.min(0),v.max(0)
    X=[v];F=[f];ranges=[(key,np.arange(len(f)))];nv=len(v)
    for other,(x,t) in meshes.items():
        if key==other or (x.max(0)<lo).any() or (x.min(0)>hi).any():continue
        tr=x[t];keep=np.flatnonzero(((tr.max(1)>=lo)&(tr.min(1)<=hi)).all(1))
        if not len(keep):continue
        ids,compact=np.unique(t[keep].ravel(),return_inverse=True)
        X.append(x[ids]);F.append(nv+compact.reshape(-1,3));nv+=len(ids)
        ranges.append((other,keep))
    allf=np.concatenate(F);hits=query(exe,np.concatenate(X),allf,np.arange(len(allf))<len(f))
    starts=np.cumsum([0]+[len(ids) for _,ids in ranges]);banks=collections.defaultdict(list)
    for a,b in sorted(hits):
        k=np.searchsorted(starts,b,side='right')-1
        bank,ids=ranges[k];banks['%d/%d'%bank].append([a,int(ids[b-starts[k]])])
    return banks
def mesh_volume(v,f):
    tri=v[f].astype(float)-v.mean(0)
    return float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
def closed(v,f):
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    un,ix,cnt=np.unique(np.sort(edges,1),axis=0,return_inverse=True,return_counts=True)
    orientation=np.bincount(ix,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    return dict(nonmanifold_or_open=int(np.count_nonzero(cnt!=2)),inconsistent_winding=int(np.count_nonzero(orientation)),zero_area=int(np.count_nonzero(np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)==0)))
def prepare(w,cap,package,audit,exe,tetgen,output):
    output.mkdir(exist_ok=True)
    meshes,names,identity=load_registered(w,cap,package,audit)
    results={}
    for sid in (29,30):
        key=(51005,sid);v,f=meshes[key];bank=names[sid]
        if bank['label'] != {29:'right gluteus maximus',30:'left gluteus maximus'}[sid]:raise ValueError('Wrong muscle identity')
        banks=interface_audit(key,meshes,exe)
        folder=output/str(sid);folder.mkdir(exist_ok=True)
        (folder/'intersections.json').write_text(json.dumps(banks,separators=(',',':'))+'\n')
        topo=closed(v,f)
        row=dict(label=bank['label'],source_member=bank['member'],member_sha256=bank['member_sha256'],donors=bank['body_bindings'],myosim_actuators=[a['name'] for a in bank['matched_muscles']],nodes=len(v),triangles=len(f),topology=topo,signed_volume_m3=mesh_volume(v,f),crossing_counts={k:len(p) for k,p in banks.items()})
        row['intersection_witness_sha256']=sha(folder/'intersections.json')
        results[sid]=row
        if any(topo.values()) or len(banks.get('51005/%d'%sid,[])):continue
        surface=folder/'source.poly'
        with surface.open('w') as fp:
            print(len(v),3,0,0,file=fp)
            for i,p in enumerate(v):print(i,*map(lambda z:format(z,'.17g'),p),file=fp)
            print(len(f),1,file=fp)
            for i,t in enumerate(f):
                print(1,0,i+1,file=fp);print(3,*t,file=fp)
            print(0,file=fp);print(0,file=fp)
        args=[str(tetgen),'-pzYQf',str(surface)]
        p=subprocess.run(args,capture_output=True,text=True)
        (folder/'mesher.log').write_text(p.stdout+p.stderr)
        row['mesher_arguments']=args;row['mesher_exit_code']=p.returncode
        if p.returncode:continue
        def rows(path):return [a.split('#')[0].split() for a in path.read_text().splitlines() if a.split('#')[0].strip()]
        def read(ext,dtype,cols):
            rr=rows(folder/('source.1.'+ext));a=np.array(rr[1:],dtype=dtype)
            if len(a)!=int(rr[0][0]) or not np.array_equal(a[:,0],np.arange(len(a))):raise ValueError(ext+' indices')
            return a[:,1:1+cols]
        x=read('node',float,3);t=read('ele',int,4)
        faces=read('face',int,4);boundary=faces[faces[:,3]>0];ancestry=boundary[:,3]-1
        if not np.array_equal(x[:len(v)],v) or not np.array_equal(np.sort(ancestry),np.arange(len(f))):raise ValueError('Changed source boundary ancestry')
        if not np.array_equal(np.sort(boundary[:,:3],axis=1),np.sort(f[ancestry],axis=1)):raise ValueError('Changed source facet')
        def volumes(nodes):
            a,b,c,d=nodes[t].transpose(1,0,2)
            return np.einsum('ij,ij->i',b-a,np.cross(c-a,d-a))/6
        vols=volumes(x);v32=volumes(x.astype('f4').astype(float))
        if vols.min()<=0 or v32.min()<=0:raise ValueError('Inverted volume after source emission')
        if abs(vols.sum()-abs(mesh_volume(v,f)))>abs(mesh_volume(v,f))*1e-6:raise ValueError('Volume closure')
        row.update(volume_nodes=len(x),tetrahedra=len(t),positive_volume_m3=vols.sum(),min_float32_cell_m3=v32.min(),boundary_facets_exact=len(f),mesh_sha256={p.name:sha(p) for p in [surface,*sorted(folder.glob('source.1.*'))]})
    report=dict(source=identity,mesher_sha256=sha(tetgen),predicate_sha256=sha(exe),scope='Signed source surface and facet-preserving volume in registered entrance coordinates. Exact triangle-pair predicates. Contact class, interior containment, material reference, attachments and donor debit require separate qualification.',muscles=results)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:{'cells':r.get('tetrahedra'), 'topology':r['topology'],'crossings':r['crossing_counts']} for k,r in results.items()})
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
