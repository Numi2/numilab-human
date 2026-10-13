from pathlib import Path
import argparse,hashlib,json,struct,numpy as np
ap=argparse.ArgumentParser()
ap.add_argument('--preparation',type=Path,required=True)
ap.add_argument('--field',type=Path,action='append',required=True)
a=ap.parse_args();out=a.preparation
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((out/'resting-anatomy-receipt.json').read_text())
d=json.loads((out/'preparation.json').read_text())
base=json.loads((Path(d['parent'])/'resting-anatomy-receipt.json').read_text())
e=r['passive_tissue_parent_bindings'];b=base['passive_tissue_parent_bindings']
assert e['source_payloads'][:4]==b['source_payloads'][:4] and e['surfaces']==b['surfaces']
assert len(e['source_payloads'])==5
p=out/e['source_payloads'][4]['path'];assert sha(p)==e['source_payloads'][4]['sha256']
raw=p.read_bytes();h=struct.unpack_from('<8s6I32s',raw)
assert h[0]==b'NHTISS4\0' and h[1]==5
records=[struct.unpack_from('<8I',raw,64+32*i) for i in range(h[2])]
bo=64+32*h[2];vo=bo+36*h[3];io=vo+56*h[4];rows={}
for rr in records:
 assert rr[1]==1
 binding=struct.unpack_from('<I8f',raw,bo+36*rr[0]);body=binding[0]
 assert binding==(body,0,0,0,0,0,0,1,1)
 n=rr[3];v=np.frombuffer(raw,dtype=np.uint8,count=n*56,offset=vo+56*rr[2]).reshape(n,56)
 points=np.ndarray((n,3),dtype='<f4',buffer=v,strides=(56,4))
 normals=np.ndarray((n,3),dtype='<f4',buffer=v,offset=12,strides=(56,4))
 slots=np.ndarray((n,4),dtype='<u4',buffer=v,offset=24,strides=(56,4))
 weights=np.ndarray((n,4),dtype='<f4',buffer=v,offset=40,strides=(56,4))
 faces=np.frombuffer(raw,dtype='<u4',count=rr[5],offset=io+4*rr[4]).reshape(-1,3)-rr[2]
 assert not slots.any() and np.array_equal(weights,np.tile(np.array([1,0,0,0],np.float32),(n,1)))
 assert rr[6] not in rows
 rows[rr[6]]=(body,points,normals,faces)
children={int(x['stable_id']):x for x in e['anatomy_surfaces']}
proof=[]
for field in a.field:
 z=np.load(field);sid=int(z['stable_surface_id'][0]);child=children[sid]
 bodies=list(map(int,z['target_body_ids']));seen=set()
 for i,ts in enumerate(child['vertices']):
  active={}
  for source,stable,vi,c in ts:
   assert source==4 and vi==i and 0<c<=1
   body,points,normals,faces=rows[stable]
   if stable not in seen:
    assert np.array_equal(points,z['body'+str(body)+'_local_positions'])
    assert np.array_equal(normals,z['body'+str(body)+'_local_normals'])
    assert np.array_equal(faces,z['faces']);seen.add(stable)
   assert body not in active;active[body]=np.float32(c)
  assert abs(sum(t[3] for t in ts)-1)<=1e-12
  for body in bodies:assert active.get(body,np.float32(0))==z['weight_body'+str(body)][i]
 proof.append({'stable_id':sid,'vertices':len(child['vertices']),'parent_bodies':bodies,
               'field_sha256':sha(field),'all_source_points_normals_faces_binding_owners_exact':True,
               'all_native_float32_coefficients_exact':True,'positive_binary64_partition_exact':True})
assert {p['stable_id'] for p in proof}==set(children)
report={'status':'multi_vessel_existing_GPU_carrier_serialization_pass',
 'native_admission':False,'anatomical_admission':False,'native_run_performed':False,
 'qualification':'Load-time constructor test only. Fields347 remain unadmitted pending surrounding anatomy, late poses, and native replay. Source477 is the unchanged native body91 compatibility control. No accepted anatomy claim.',
 'rows':proof,'carrier_rows':len(rows),'source_payloads':5,
 'other_parent_sources_and_muscle_maps_unchanged':True,
 'source_payload_sha256':sha(p),'receipt_sha256':sha(out/'resting-anatomy-receipt.json'),
 'helper_sha256':sha(__file__)}
(out/'exact-carrier-map-proof.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
