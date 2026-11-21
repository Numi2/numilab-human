"""Read exact native hand material and authored force-site identities at the registered entrance."""
import hashlib,json,mmap,struct
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from numilab_human import common_atlas_skin_clearance as clearance
from numilab_human.capsular_material_registration import read_material_banks,material_bank,emitted_body_vertices

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def applied(pose,xyz):
    return Rotation.from_quat(pose[3:]).apply(xyz)+pose[:3]
def unit(v):
    return v/np.maximum(1e-30,np.linalg.norm(v,axis=-1,keepdims=True))
def get_layers(pack,receipt):
    rc=json.loads(receipt.read_text())
    if sha(pack)!=rc['base_pack_file_sha256']:raise ValueError('Wrong native entrance geometry')
    labels={1:'outer',4294966545:'inner',4294966546:'deep'}
    with pack.open('rb') as fo,mmap.mmap(fo.fileno(),0,access=mmap.ACCESS_READ) as m:
        h,d=clearance._PACK_HEADER,clearance._PACK_DIRECTORY
        magic,version,n,*_=h.unpack_from(m)
        assert magic==b'MRVPACK2' and version==2
        sec={row[0]:row for i in range(n) if (row:=d.unpack_from(m,h.size+i*d.size))}
        vs,ix,pr,ins=[sec[k] for k in (2,3,4,5)];xyz=np.ndarray((vs[4],20),'<f4',m,vs[2])[:,:3]
        inds=np.ndarray(ix[4],'<u4',m,ix[2]);out={}
        for i in range(pr[4]):
            start,count,_,inst,sem,_,_,sid=struct.unpack_from('<8I',m,pr[2]+64*i)
            if sem!=51007 or sid not in labels:continue
            t=struct.unpack_from('<8f12I',m,ins[2]+ins[5]*inst)
            if t[10]!=0 or t[:8]!=(0,0,0,1,0,0,0,1):raise ValueError(f'Layer {sid} is not already in world coordinates')
            faces=inds[start:start+count].reshape(-1,3);used=np.unique(faces)
            out[labels[sid]]=(xyz[used].copy(),np.searchsorted(used,faces))
        if set(out)!=set(labels.values()):raise ValueError('Missing physical layer')
        return out

def read_inputs(package,manifest,entry_log):
    cfg=json.loads(package.read_text());argv=cfg['argv'];muscle=Path(argv[2]);bones=Path(argv[3])
    t=Path(argv[argv.index('--tendon-payload')+1]);endpoints=json.loads(t.with_suffix('.manifest.json').read_text())
    if endpoints['payload']['sha256']!=sha(t) or endpoints['source']['myosim_muscle_payload_sha256']!=sha(muscle):
        raise ValueError('Different native tendon or muscle payload')
    a=json.loads(manifest.read_text())
    if a['payloads']['muscles']['sha256']!=sha(muscle):raise ValueError('MyoSim roster differs from the native payload')
    native=next(json.loads(r.partition('=')[2]) for r in entry_log.open() if r.startswith('registered_ligament_actual_pose_json='))
    if native['bonePayloadSha256']!=sha(bones):raise ValueError('Wrong anatomical entrance')
    poses=np.array(native['bodyPoses']);bank=read_material_banks(bones);solids={};bybody={}
    for sid,rec in bank[2].items():
        b=rec[0];asset,_,attrs,faces=material_bank(bank,sid,b)
        xyz=applied(poses[b],emitted_body_vertices(attrs,rec))
        solids[sid]=(xyz,faces);bybody.setdefault(b,[]).append((sid,asset['myosim_body']))
    raw=muscle.read_bytes();magic,abi,nb,nm,ns,nw,nr,nt,na,size,digest=struct.unpack_from('<8s9I32s',raw)
    assert (magic,abi,nb,na,size)==(b'NHMYO2\0\0',2,len(poses),nm,32)
    si=76;wi=si+16*ns;ri=wi+64*nw;mi=ri+16*nr
    assert len(raw)==mi+nm*(164+32)
    lookup={(r['muscle'],r['endpoint']):r for r in endpoints['endpoints']}
    routes={};users={}
    for i,entry in enumerate(a['muscles']):
        name=entry['name'];tendon,start,count,res=struct.unpack_from('<4I',raw,mi+i*164)
        assert entry['source_tendon_index']==tendon and res==0 and start+count<=nr
        rows=[]
        for off in range(start,start+count):
            typ,idx,side,_=struct.unpack_from('<4I',raw,ri+16*off)
            if typ!=1:continue
            body,*point=struct.unpack_from('<I3f',raw,si+16*idx);users.setdefault(idx,[]).append(name)
            local=np.array(point)
            ep=lookup[name,'origin'] if off==start else lookup[name,'insertion'] if off==start+count-1 else None
            if ep:
                assert ep['body_index']==body and ep['source_site_index']==idx
                assert np.linalg.norm(local-ep['source_local_point_m'])<1e-10
                local=np.array(ep['resolved_local_point_m'])
            rows.append(dict(order=len(rows),site=idx,body=body,local=local,world=applied(poses[body],local),endpoint=ep))
        if name in {f'EDC{d}{s}' for d in (2,3,4,5) for s in ('','_l')}:routes[name]=rows
    assert len(routes)==8
    return routes,users,solids,bybody,poses
