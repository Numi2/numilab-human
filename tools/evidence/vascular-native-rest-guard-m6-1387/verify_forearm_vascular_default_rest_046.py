#!/usr/bin/env python3
"""Verify compiled vascular fields against the retained rigid default poses."""
from pathlib import Path
import hashlib,json,struct
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
C=A/'forearm-vascular-carrier-cpu-044-001'
O=A/'forearm-vascular-default-rest-046-001'
rigid=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid')
archive=Path('/Users/n/numi-human-retained-delivery-20261009/skin-source-binding-1202/source-inputs/myosim/myo_sim-33c89c2b.tar.gz')
src=A/'vascular-source-no-op-control-038-001/prepared/resting-thorax.nhanatomy'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
rep=json.loads((C/'report.json').read_text())
fields=[Path(p) for p in rep['field_sha256']]
files=[rigid,archive,src,C/'report.json',C/'prepared/exact-carrier-map-proof.json',Path(__file__),*fields]
pins={str(p.resolve()):sha(p) for p in files}
assert all(sha(p)==h for p,h in rep['field_sha256'].items())
assert not O.exists()
raw=rigid.read_bytes()
magic,pabi,eabi,sc,bc,jc,nq,nv,root,virtual,reserved,embedded=struct.unpack_from('<8s10I32s',raw)
assert magic==b'NHRIGID2' and pabi==1 and nq==nv+1 and embedded.hex()==sha(archive)
mo=224+160*bc+144*jc+64*nv+4*nq+4*nv
mapping=np.frombuffer(raw,dtype='<u4',offset=mo,count=sc)
po=mo+4*sc
assert po+28*sc==len(raw)
poses={int(core):(np.array(struct.unpack_from('<7f',raw,po+28*i)[:3],dtype='<f4'),np.array(struct.unpack_from('<7f',raw,po+28*i)[3:],dtype='<f4')) for i,core in enumerate(mapping)}
nraw=src.read_bytes();h=struct.unpack_from('<8s5I32s',nraw);vo=60+32*h[2];io=vo+24*h[3]
records={r[5]:r for r in (struct.unpack_from('<8I',nraw,60+32*i) for i in range(h[2]))}
def rotate(q,x):
 t=np.multiply(np.float32(2),np.cross(q[:3],x),dtype='<f4')
 return np.add(np.add(x,np.multiply(q[3],t,dtype='<f4'),dtype='<f4'),np.cross(q[:3],t),dtype='<f4')
def transform(body,x):
 p,q=poses[body];return np.add(p,rotate(q,x),dtype='<f4')
rows=[]
for path in fields:
 z=np.load(path,allow_pickle=False);sid=int(z['stable_surface_id'][0]);body=int(z['source_body'][0]);rr=records[sid]
 v=np.frombuffer(nraw,dtype='<f4',offset=vo+24*rr[1],count=6*rr[2]).reshape(-1,6)
 f=np.frombuffer(nraw,dtype='<u4',offset=io+4*rr[3],count=rr[4]).reshape(-1,3)-rr[1]
 assert body==rr[0] and np.array_equal(v[:,:3],z['source_local_positions']) and np.array_equal(v[:,3:],z['source_local_normals']) and np.array_equal(f,z['faces'])
 expected=transform(body,v[:,:3]);pred=np.zeros_like(expected)
 for b in z['target_body_ids']:
  b=int(b);pred=np.add(pred,np.multiply(z[f'weight_body{b}'][:,None],transform(b,z[f'body{b}_local_positions']),dtype='<f4'),dtype='<f4')
 error=np.linalg.norm(pred.astype(float)-expected.astype(float),axis=1)
 cacheerror=np.linalg.norm(pred.astype(float)-z['registered_blended_rest_positions'].astype(float),axis=1)
 sourcecache=np.linalg.norm(expected.astype(float)-z['registered_rest_world_positions'].astype(float),axis=1)
 assert error.max()<=2e-6 and cacheerror.max()<=2e-6 and sourcecache.max()<=2e-6
 rows.append({'stable_id':sid,'source_body':body,'parent_bodies':z['target_body_ids'].tolist(),'vertices':len(pred),'ordered_float32_rest_max_um':float(error.max()*1e6),'recomputed_vs_cached_blend_max_um':float(cacheerror.max()*1e6),'source_pose_vs_cached_rest_max_um':float(sourcecache.max()*1e6),'source_geometry_and_normals_faces_exact':True,'native_2um_gate_failures':int(np.count_nonzero(error>2e-6))})
assert all(sha(p)==h for p,h in pins.items())
O.mkdir()
out={'status':'registered_rest_ordered_float32_preflight_pass','input_sha256':pins,'rows':rows,'gpu_used':False,'native_run_performed':False,'anatomical_admission':False,'qualification':'Load-time rest consistency only; actual surrounding anatomy, late poses and native execution still require qualification.'}
(O/'report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'rows':rows,'report_sha256':sha(O/'report.json')}))
