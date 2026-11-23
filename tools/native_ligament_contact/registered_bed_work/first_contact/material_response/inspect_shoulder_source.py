from pathlib import Path
import ctypes
import hashlib
import json
import struct
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from measure_forearm_enclosure import Enclosure
from recount_registered_crossings import read,digest
def source_surface(payload, member):
    magic, abi, nr, nb, nv, ni, _ = struct.unpack_from('<8s6I', payload)
    if (magic, abi) != (b'NHTISS4\x00', 5):
        raise ValueError('Unexpected muscle payload ABI')
    records = np.frombuffer(payload,'<u4',8*nr,64).reshape(nr,8)
    row, = records[records[:,6] == member]
    bi, count, vi, n, fi, nf = map(int,row[:6])
    vo = 64+32*nr+36*nb
    f = np.frombuffer(payload,'<u4',nf,vo+56*nv+4*fi).reshape(-1,3)
    if f.min()<vi or f.max()>=vi+n:
        raise ValueError('Invalid source face')
    ids=np.unique(f)
    attr=np.frombuffer(payload,'<f4',14*nv,vo).reshape(nv,14)[ids]
    slots=attr[:,6:10].copy().view('<u4')
    w=attr[:,10:14]
    if ((slots>=count)&(w>0)).any() or np.any(w<0) or np.max(np.abs(w.sum(1)-1))>1e-5:
        raise ValueError('Invalid source body blend')
    return attr[:,:3].astype(float), np.searchsorted(ids,f)



def nearest(mesh,points,lib):
    v,f=mesh
    surf=Enclosure(lib,v,f)
    try:
        p=np.ascontiguousarray(points,'f8');R=np.empty((len(p),7));ids=np.empty(len(p),'i4')
        ptr=np.ctypeslib.ndpointer(dtype='float64',flags='C_CONTIGUOUS')
        ip=np.ctypeslib.ndpointer(dtype='int32',flags='C_CONTIGUOUS')
        q=surf.lib.cortex_query;q.argtypes=[ctypes.c_void_p,ptr,ctypes.c_int,ptr,ip];q.restype=None
        q(surf.handle,p,len(p),R,ids)
        if not np.allclose(np.abs(R[:,0]),np.linalg.norm(R[:,4:7]-p,axis=1),atol=2e-8):
            raise ValueError('Nearest material point mismatch')
        return R[:,4:7].copy(),np.abs(R[:,0]),ids
    finally:surf.close()


def measure(data,prior,lib):
    j=read(data/'bed-unscaled-four/launch.json');a=j['argv']
    bone=Path(a[3]);soft=Path(a[a.index('--soft-tissue-payload')+1]);myo=Path(a[2]);tendon=Path(a[a.index('--tendon-payload')+1])
    for p in (bone,soft,myo,tendon):
        if digest(p)!=j['pins'][str(p)]:raise ValueError('Stale native asset')
    bc=bone.read_bytes();t=soft.read_bytes();m=myo.read_bytes()
    h=struct.unpack_from('<8s5I32s',bc);n,v_count,i_count=h[2:5]
    vo=60+60*n;io=vo+v_count*24
    if len(bc)!=io+4*i_count:raise ValueError(('Unexpected bone layout',len(bc),io+4*i_count))
    tmeta=read(tendon.with_suffix('.manifest.json'));bmeta=read(bone.with_suffix('.manifest.json'))
    src=np.load(prior/'source-bones.npz');by={r['bone']:r for r in read(prior/'rigid.json')}
    ms=read(prior/'model-source.json')
    if ms['source_archive_sha256']!=tmeta['source']['myosim_archive_sha256']:raise ValueError('Source model differs')
    endpoints={r['muscle']:r for r in tmeta['endpoints'] if r['endpoint']=='origin'}
    mh=struct.unpack_from('<8s9I32s',m);ro=76+16*mh[4]+64*mh[5];mo=ro+16*mh[6]
    pose=read(data/'bed-unscaled-four/native-initial-bodies.json')['all_bodies']
    local_pose=lambda b:Rotation.from_quat(pose[b]['current']['orientation_world_xyzw'])
    p0=lambda b:np.array(pose[b]['current']['position_world_m'])
    outputs=[];arrays={}
    for side,sid,mid,mus,bid in [('r',10,79,'DELT3',34),('l',11,80,'DELT3_l',84)]:
        owner='scapula_'+side
        row=next(struct.unpack_from('<6I8fI',bc,60+60*k)for k in range(n) if struct.unpack_from('<6I',bc,60+60*k)[5]==sid)
        if row[0]!=bid:raise ValueError('Mismatched cortical ownership')
        A=np.frombuffer(bc,dtype='<f4',count=6*row[2],offset=vo+24*row[1]).reshape(-1,6)[:,:3].astype(float)
        F=np.frombuffer(bc,dtype='<u4',count=row[4],offset=io+4*row[3]).reshape(-1,3).astype('i4')-row[1]
        rot=Rotation.from_quat(row[9:13]);scale=row[13]
        bone_local=rot.apply(scale*A)+row[6:9]
        p=endpoints[mus];sourcepoint=np.array(p['source_local_point_m'])
        src_v=src[owner+'_local'];src_f=src[owner+'_faces']
        pinned=next(u for u in ms['bones'] if u['body']==owner)
        if hashlib.sha256(src[owner+'_world'].tobytes()).hexdigest()!=pinned['sha256_world_float64']:raise ValueError('Changed source bone')
        tissue_v,tissue_f=source_surface(t,mid)
        raw_s=by[owner];mapping=Rotation.from_quat(raw_s['rotation_xyzw'])
        def to_bone(q):return mapping.inv().apply(q-np.array(raw_s['translation_m']))/raw_s['scale']
        q0,sd,si=nearest((src_v,src_f),[sourcepoint],lib)
        q1,bd,bi=nearest((bone_local,F),[sourcepoint],lib)
        q2,rd,ri=nearest((bone_local,F),to_bone(q0),lib)
        expected=next(z for z in read(data/'current-registration/unbound-cortex.json')['cases'] if z['name']==mus and z['role']=='origin')['nearest']['mm']
        if abs(bd[0]*1000-expected)>.01:raise ValueError(('Bone-local transform differs from live release',side,bd,expected,row))
        _,start,count,_=struct.unpack_from('<4I',m,mo+164*p['muscle_index'])
        route=[]
        for k in range(count):
            kind,i,*_=struct.unpack_from('<4I',m,ro+16*(start+k))
            if kind!=1:route.append(dict(kind=kind,index=i));continue
            b,*xyz=struct.unpack_from('<I3f',m,76+16*i)
            route.append(dict(kind=kind,index=i,body=b,local=xyz,world=(p0(b)+local_pose(b).apply(xyz)).tolist()))
        arrays.update({owner+'_v':bone_local,owner+'_f':F,owner+'_atlas':A,owner+'_reference_surface':q0,
            owner+'_cortical_prediction':q2,owner+'_spinal_muscle':tissue_v,owner+'_spinal_muscle_f':tissue_f})
        outputs.append(dict(side=side,body=bid,source_distance_mm=1000*sd[0],baseline_mm=1000*bd[0],
            predicted_distance_mm=1000*rd[0],source_face=int(si[0]),candidate_face=int(ri[0]),
            registered_movement_mm=1000*np.linalg.norm(q2[0]-sourcepoint),native_row=list(row),
            muscle_xyz_extent=np.ptp(tissue_v,axis=0).tolist(),bone_xyz_extent=np.ptp(A,axis=0).tolist(),route=route))
    np.savez_compressed('correspondence.npz',**arrays)
    Path('source-comparison.json').write_text(json.dumps(dict(bone=digest(bone),muscle=digest(soft),routes=digest(myo),tendon=digest(tendon),
        source=digest(prior/'source-bones.npz'),rigid=digest(prior/'rigid.json'),fit=outputs),indent=2)+'\n')
    for e in outputs:print(e['side'],e['source_distance_mm'],e['baseline_mm'],e['predicted_distance_mm'],e['registered_movement_mm'],[(r.get('body'),r['kind'])for r in e['route']])
if __name__=='__main__':measure(*(Path(x).resolve()for x in sys.argv[1:]))
