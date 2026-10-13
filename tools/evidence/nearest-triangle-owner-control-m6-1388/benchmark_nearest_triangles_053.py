#!/usr/bin/env python3
"""Compare bounded nearest queries with the owner full scan on captured anatomy."""
from pathlib import Path
import ast,gc,hashlib,importlib.util,json,mmap,resource,struct,sys,time
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
O=A/'nearest-triangle-owner-control-053-001'
helper=A/'bounded_triangle_nearest_052.py'
reader=A/'full-anatomy-delivery-audit-1378/arterial-venous-route-field-073/build_route073.py'
pack=A/'back412-skin-native-022-001-60s/baseline/native-run/accepted-geometry/step-14975.mrvpack'
receipt=pack.with_suffix('.receipt.json')
owner=Path('/Users/n/numi-human-left-pq-publication-1344/src')
sys.path.insert(0,str(owner))
from numilab_human import myosim_bone_proximity as bp
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def req(value,message):
 if not value:raise ValueError(message)
spec=importlib.util.spec_from_file_location('bounded_nearest050',helper)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
pins={str(p.resolve()):sha(p) for p in [helper,reader,Path(bp.__file__),pack,receipt,Path(__file__)]}
assert not O.exists();O.mkdir()
capture=json.loads(receipt.read_text());assert capture['accepted_step']==14975 and capture['physical_endpoint']=='accepted' and capture['surface_audit_endpoint']=='passed'
tree=ast.parse(reader.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Pack')
scope={'Path':Path,'mmap':mmap,'struct':struct,'np':np,'req':req,'PH':struct.Struct('<8sIIQQ32s24s'),'PD':struct.Struct('<IIQQQII32s')}
exec(compile(ast.Module(body=[cls],type_ignores=[]),str(reader),'exec'),scope)
mp=scope['Pack'](pack)
rng=np.random.default_rng(51050)
targets=[(51004,9),(51005,92),(51005,72),(51010,313),(51007,1)]
reports=[]
distance=lambda point,tris:bp._point_triangle_distances_squared(point,tris,np)
for key in targets:
 assert key in mp.surfaces,key
 faces=mp.faces(key);triangles=np.ascontiguousarray(mp.pos[faces],dtype=np.float64)
 build0=time.perf_counter();index=mod.TriangleNearestIndex(triangles,distance);build_s=time.perf_counter()-build0
 count=24 if len(triangles)>100000 else 64
 choices=rng.integers(0,len(triangles),count)
 tri=triangles[choices];centers=tri.mean(axis=1)
 normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal,axis=1)
 normal=np.divide(normal,length[:,None],out=np.zeros_like(normal),where=length[:,None]>0)
 points=centers+normal*rng.uniform(-.014,.014,(count,1))
 # Include vertices/edge midpoints and equal-distance duplicate face cases.
 points=np.vstack([points,triangles[:4,0],(triangles[:4,0]+triangles[:4,1])*.5])
 full0=time.perf_counter();full=[]
 for point in points:
  values=distance(point,triangles);i=int(np.argmin(values));full.append((i,float(values[i])))
 full_s=time.perf_counter()-full0
 near0=time.perf_counter();bounded=[];sizes=[]
 for point in points:
  bounded.append(index.query(point));sizes.append(index.last_stats['exact_candidate_faces'])
 bounded_s=time.perf_counter()-near0
 mismatches=[]
 for j,(f,b) in enumerate(zip(full,bounded)):
  if f!=b:mismatches.append({'point':j,'full':f,'bounded':b})
 assert not mismatches,mismatches[:3]
 report={'target':list(key),'faces':len(triangles),'queries':len(points),'exact_face_and_squared_distance_mismatches':0,'full_scan_s':full_s,'bounded_query_s':bounded_s,'index_build_s':build_s,'query_speedup':full_s/max(bounded_s,1e-20),'candidate_face_count_quantiles':np.quantile(sizes,[0,.5,.95,1]).tolist()}
 reports.append(report);print(json.dumps(report),flush=True)
 del index,triangles,faces,tri,centers,normal,length,points;gc.collect()
mp.close()
assert all(sha(p)==h for p,h in pins.items())
report={'status':'nearest_triangle_broadphase_owner_equivalence_pass','input_sha256':pins,'captured_accepted_step':14975,'rows':reports,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gpu_used':False,'simulation_real_time_factor_claimed':False,'qualification':'Offline static-target nearest-distance acceleration only. Exact inside/outside classification and physical simulation owners unchanged.'}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'report_sha256':sha(O/'report.json')}),flush=True)
