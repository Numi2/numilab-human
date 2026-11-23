"""Carry native muscle sites, wrapping geometry and distributed enthesis wrenches with their cortical registration."""
import json,struct,sys
from pathlib import Path

import numpy as np
from audit_registered_skeleton import HEADER,RECORD,digest
from scipy.spatial.transform import Rotation


def run(job,bones,out):
    cfg=json.loads((job/'launch.json').read_bytes());argv=cfg['argv']
    src_m=Path(argv[2]);src_t=Path(argv[argv.index('--tendon-payload')+1]);old_bone=Path(argv[3])
    new_bone=next(bones.glob('*.nhbones'));bm=json.loads(new_bone.with_suffix('.manifest.json').read_bytes())
    if any(digest(p)!=cfg['pins'][str(p)] for p in (src_t,src_m,old_bone)):raise ValueError('Unpinned muscle or skeletal input')
    reg=json.loads((bones/bm['source']['registration']['file']).read_bytes())
    receipt=reg['regional_derivation']['connected_shoulder_registration']
    changes={int(i):(int(r['body']),np.array(r['body_local_transform'])) for i,r in receipt['changed'].items()}
    if set(changes)!={8,9,10,11}:raise ValueError('Unrecognized cortical scope')
    owners={b:(sid,A[:3,:3],A[:3,3]) for sid,(b,A) in changes.items()}
    newp=new_bone.read_bytes();oldp=old_bone.read_bytes()
    _,_,count,nvert,*_=HEADER.unpack_from(newp);vo=HEADER.size+RECORD.size*count
    for sid,(b,A) in changes.items():
        o=HEADER.size+RECORD.size*(sid-1);r0=RECORD.unpack_from(oldp,o);r1=RECORD.unpack_from(newp,o)
        def xyz(raw,row):
            v=np.frombuffer(raw,'<f4',row[2]*6,vo+24*row[1]).reshape(-1,6)[:,:3]
            return Rotation.from_quat(row[9:13]).apply(v.astype(float)*row[13])+row[6:9]
        a=xyz(oldp,r0);c=xyz(newp,r1);res=np.linalg.norm(c-(a@A[:3,:3].T+A[:3,3]),axis=1)
        if res.max()>2.e-6:raise ValueError('Cortical frame does not carry original attachment surface')
    myo=bytearray(src_m.read_bytes());ten=bytearray(src_t.read_bytes())
    meta=json.loads(src_t.with_suffix('.manifest.json').read_bytes())
    mh=struct.unpack_from('<8s9I32s',myo);th=list(struct.unpack_from('<8s10I32s32s32s',ten));w0=76+16*mh[4]
    if mh[0]!=b'NHMYO2\0\0' or th[0]!=b'NHTEND3\0' or th[-2:]!=[bytes.fromhex(digest(src_m)),bytes.fromhex(digest(old_bone))]:
        raise ValueError('Mismatched mechanical owners')
    sites=[];wraps=[];material=[];tests=[]
    for i in range(mh[4]):
        offset=76+16*i;b,*p=struct.unpack_from('<I3f',myo,offset)
        if b in owners:
            _,R,t=owners[b];struct.pack_into('<3f',myo,offset+4,*(R@p+t));sites.append(i)
    for i in range(mh[5]):
        off=w0+64*i;rec=struct.unpack_from('<2I14f',myo,off);b=rec[0]
        if b in owners:
            _,R,t=owners[b];p=R@rec[4:7]+t;axes=R@np.array(rec[7:]).reshape(3,3)
            struct.pack_into('<12f',myo,off+16,*p,*axes.ravel());wraps.append(i)
    forces=np.vstack([np.eye(3),np.random.default_rng(169).normal(size=(32,3))])
    forces/=np.linalg.norm(forces,axis=1)[:,None]
    for i,e in enumerate(meta['endpoints']):
        off=144+64*i;row=struct.unpack_from('<8I8f',ten,off);b=row[4]
        if b!=e['body_index']:raise ValueError('Mismatched tendon body')
        if b not in owners:continue
        sid,R,t=owners[b]
        x=np.array(row[8:11]);y=R@x+t;struct.pack_into('<3f',ten,off+32,*y)
        for key in ('source_local_point_m','resolved_local_point_m'):
            e[key]=(R@e[key]+t).tolist()
        site=e['source_site_index']
        native_site=np.array(struct.unpack_from('<3f',myo,76+16*site+4))
        if np.linalg.norm(native_site-e['source_local_point_m'])>2e-6 or np.linalg.norm(y-e['resolved_local_point_m'])>2e-6:
            raise ValueError('Tendon site and route point disagree')
        surface=e['surface']
        if surface is None:
            if row[6]!=0xffffffff:raise ValueError('Missing source attachment envelope')
            continue
        offset=144+64*th[5]+288*row[6]
        if struct.unpack_from('<2I',ten,offset)!=(b,sid):raise ValueError('Wrong cortical owner')
        xyz=np.frombuffer(ten,'<f4',16,offset+16).reshape(4,4)[:,:3].copy()
        matrices=np.frombuffer(ten,'<f4',48,offset+80).reshape(4,3,4)[:,:,:3].copy()
        mapped=(xyz@R.T+t).astype('f4')
        maps=(R@matrices@R.T).astype('f4')
        np.frombuffer(ten,'<f4',16,offset+16).reshape(4,4)[:,:3]=mapped
        np.frombuffer(ten,'<f4',48,offset+80).reshape(4,3,4)[:,:,:3]=maps
        surface['nearest_local_point_m']=(R@surface['nearest_local_point_m']+t).tolist()
        surface['node_local_points_m']=(np.array(surface['node_local_points_m'])@R.T+t).tolist()
        local=np.einsum('nab,kb->kna',matrices,forces)
        after=np.einsum('nab,kb->kna',maps,forces@R.T)
        ferr=float(np.linalg.norm(after-local@R.T,axis=2).max())
        torque=np.cross(xyz-x,local).sum(1)
        merr=float(np.linalg.norm(np.cross(mapped-y,after).sum(1)-torque@R.T,axis=1).max())
        if ferr>2e-6 or merr>1e-7:raise ValueError('Distributed enthesis changed its force or moment law')
        material.append(i);tests.append((ferr,merr))
    out.mkdir()
    m=out/src_m.name;m.write_bytes(myo)
    th[-2]=bytes.fromhex(digest(m));th[-1]=bytes.fromhex(digest(new_bone));th[8]=int(bm['payload']['registration_fingerprint32'],16)
    struct.pack_into('<8s10I32s32s32s',ten,0,*th)
    target=out/src_t.name;target.write_bytes(ten)
    src=meta['source']['bodyparts3d_bone_payload'];src.update(file=str(new_bone.resolve()),sha256=digest(new_bone),registration_fingerprint32=bm['payload']['registration_fingerprint32'])
    src['manifest']=dict(file=str(new_bone.with_suffix('.manifest.json').resolve()),sha256=digest(new_bone.with_suffix('.manifest.json')))
    meta['source']['myosim_muscle_payload_sha256']=digest(m);meta['payload']['sha256']=digest(target)
    report=dict(parent_muscle=digest(src_m),parent_tendon=digest(src_t),parent_bone=digest(old_bone),rigid=receipt['rigid_sha256'],
          bones=digest(new_bone),muscle=digest(m),tendon=digest(target),changed_source_sites=sites,changed_wraps=wraps,changed_envelopes=material,
          transformed_endpoints=sum(e['body_index'] in owners for e in meta['endpoints']),counted_force_tests=len(tests)*len(forces),
          max_covariant_force_error=max(t[0] for t in tests),max_covariant_moment_error_m=max(t[1] for t in tests))
    meta['source']['coupled_girdle_material_transport']=report
    target.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('sites',len(sites),'wraps',len(wraps),'envelopes',len(material),'wrenches',len(tests)*len(forces))


if __name__=='__main__':run(*map(Path,sys.argv[1:]))
