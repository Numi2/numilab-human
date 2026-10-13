#!/usr/bin/env python3
"""Prepare selected passive vessel source registrations in the existing NHA ABI5 payload.
CPU asset preparation only; native/anatomical admission requires posed audits.
"""
from pathlib import Path
import argparse, hashlib, json, struct
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v: raise ValueError(m)
def main():
    ap=argparse.ArgumentParser()
    for k in ('parent','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--geometry',type=Path,action='append',required=True)
    a=ap.parse_args()
    require(not a.output.exists(),'preserve existing preparation')
    receipt=json.loads((a.parent/'resting-anatomy-receipt.json').read_text())
    source=Path(receipt['payload']['path'])
    pins={str(p.resolve()):sha(p) for p in
        (Path(__file__),a.parent/'resting-anatomy-receipt.json',source,*a.geometry)}
    require(sha(source)==receipt['payload']['sha256'],'source payload drift')
    raw=source.read_bytes(); out=bytearray(raw)
    header=struct.unpack_from('<8s5I32s',raw)
    require(header[0]==b'NHANAT1\0',
            'unrecognized anatomy magic')
    require(header[1]==5 and header[2]==523,'unexpected source ABI/inventory')
    records=[struct.unpack_from('<8I',raw,60+32*i) for i in range(header[2])]
    vo=60+32*header[2]; io=vo+24*header[3]
    updates={}
    for path in a.geometry:
        geometry=np.load(path,allow_pickle=False)
        ids=list(map(int,geometry['source_stable_ids']))
        bodies=list(map(int,geometry['source_bodies']))
        require(len(ids)==len(bodies) and len(set(ids))==len(ids),'invalid source identities')
        for sid,body in zip(ids,bodies):
            require(sid not in updates,'duplicate source update')
            require(469<=sid<=523 and sid%2==1,'unsupported passive vessel identity')
            updates[sid]=(body,geometry,path)
    require(updates,'empty source updates')
    changed=[]
    no_op=[]
    for sid,(body,geometry,path) in updates.items():
        r=next(r for r in records if r[5]==sid)
        require(r[0]==body and r[6]==2,'anatomical registration differs')
        vertices=np.frombuffer(raw,dtype='<f4',count=6*r[2],offset=vo+24*r[1]).reshape((-1,6))
        faces=np.frombuffer(raw,dtype='<u4',count=r[4],offset=io+4*r[3]).reshape((-1,3))-r[1]
        points=geometry[f'source_local_positions_{sid}_after']
        require(np.array_equal(vertices[:,:3],geometry[f'source_local_positions_{sid}_before']) and
                np.array_equal(faces,geometry[f'source_faces_{sid}']),'source geometry differs')
        require(points.shape==vertices[:,:3].shape and points.dtype==np.dtype('<f4') and
                np.isfinite(points).all(),'invalid source coordinates')
        if np.array_equal(points,vertices[:,:3]):
            no_op.append(sid)
            continue
        p=points.astype(float)
        face_area=np.cross(p[faces[:,1]]-p[faces[:,0]],p[faces[:,2]]-p[faces[:,0]])
        require(np.all(np.linalg.norm(face_area,axis=1)>0),'degenerate source face')
        normals=np.zeros(p.shape)
        for lane in range(3):np.add.at(normals,faces[:,lane],face_area)
        length=np.linalg.norm(normals,axis=1)
        require(np.all(length>0) and np.isfinite(length).all(),'undefined source normals')
        new=np.column_stack((points,normals/length[:,None])).astype('<f4')
        out[vo+24*r[1]:vo+24*(r[1]+r[2])]=new.tobytes()
        changed.append({'stable_id':sid,'body_index':body,'vertices':r[2],
                        'faces':r[4]//3,'maximum_source_offset_m':
                        float(np.linalg.norm(p-vertices[:,:3].astype(float),axis=1).max()),
                        'geometry_path':str(path.resolve()),
                        'geometry_sha256':sha(path),
                        'source_row_before_sha256':hashlib.sha256(vertices.tobytes()).hexdigest(),
                        'source_row_after_sha256':hashlib.sha256(new.tobytes()).hexdigest()})
    require(out[:vo]==raw[:vo] and out[io:]==raw[io:],'identity, registration or face inventory changed')
    unchanged=[]
    for r in records:
        if r[5] in {x['stable_id'] for x in changed}:continue
        begin,end=vo+24*r[1],vo+24*(r[1]+r[2])
        require(out[begin:end]==raw[begin:end],'unrelated anatomy changed')
        unchanged.append(r[5])
    require(len(unchanged)+len(changed)==523,'unchanged source inventory differs')
    a.output.mkdir()
    payload=a.output/'resting-thorax.nhanatomy'
    if out==raw:
        import os
        os.link(source,payload)
    else:payload.write_bytes(out)
    report={'input_sha256':pins,'payload_sha256':sha(payload),'changed_rows':changed,
        'unchanged_source_rows':unchanged,'no_op_requested_rows':no_op,'records_indices_registration_byte_exact':True,
        'interpretation':'Declared mixed-source reference registration. The exact displacement construction and its interpretation are retained in each pinned geometry input. Source faces, identities, laterality, registration records, blood-state ownership and physical mass are preserved; area-weighted normals are recomputed only on coordinate-changed rows. CPU serialization proof does not admit posed anatomical validity.',
        'native_admission':False,'physiology_or_force_changes':False}
    require(all(sha(p)==h for p,h in pins.items()),'input drift')
    (a.output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'payload_sha256':sha(payload),'changed_rows':changed}))
if __name__=='__main__':main()
