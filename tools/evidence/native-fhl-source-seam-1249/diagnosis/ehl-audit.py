"""Locate EHL self-intersections before registration using unchanged exact predicates."""
from pathlib import Path
import hashlib, importlib.util, json, sys, time, zipfile
import numpy as np
ROOT=Path('/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247')
OUT=ROOT/'ehl-source-self-diagnosis-001'
MODEL=Path('/Users/n/numi-human-source-seam-connectivity-1247/src/numilab_human/model.py')
HIST=Path('/Users/n/numi-human-common-skin-multipose-001/src')
ARCHIVE=Path('/Users/n/numi-human-source-cache/isa_BP3D_4.0_obj_99.zip')
MANIFEST=ROOT/'regen-4rows-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'
REFERENCE=ROOT/'regen-4rows-001/regenerated-row-exact-f32-self-audit-001.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def main():
 start=time.monotonic(); need(not (OUT/'report.json').exists(),'refuse overwrite')
 sys.path.insert(0,str(HIST))
 from numilab_human import cardiac_cavity_intersections as ci
 need(sha(ci.__file__)=='11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb','predicate owner')
 spec=importlib.util.spec_from_file_location('source_model_seam1247',MODEL)
 model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)
 paths=[Path(__file__),MODEL,ARCHIVE,MANIFEST,REFERENCE]
 paths += [Path(v.__file__) for k,v in tuple(sys.modules.items()) if k.startswith('numilab_human') and getattr(v,'__file__',None)]
 pins={str(p):sha(p) for p in paths}
 need(pins[str(ARCHIVE)]=='40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e','archive')
 reference={r['stable_id']:r for r in json.loads(REFERENCE.read_text())['regenerated_rows']}
 rows=[r for r in json.loads(MANIFEST.read_text())['source']['surfaces'] if r['stable_id'] in (23,24,27,28)]
 reports=[]
 with zipfile.ZipFile(ARCHIVE) as z:
  for r in rows:
   blob=z.read(r['member']);need(hashlib.sha256(blob).hexdigest()==r['member_sha256'],'member')
   v,f=model._bodyparts_obj_triangles(blob,r['member'])
   v,f,selection=model._bodyparts_largest_connected_surface_component(v,f,r['member'])
   v,f,cancellation=model._bodyparts_cancel_opposite_surface_faces(v,f,r['member'])
   need(not cancellation['cancelled_opposite_face_pairs'],'unexpected source face cancellation')
   modes=[]
   for mode,dtype in [('source_binary64_mm','<f8'),('source_binary32_mm','<f4')]:
    unique,inverse=np.unique(np.asarray(v,dtype=dtype),axis=0,return_inverse=True)
    faces=inverse[np.asarray(f,dtype=np.int64)]
    ratios=[[float(x).as_integer_ratio() for x in p] for p in unique]
    denominator=max(d for p in ratios for n,d in p)
    lattice=[tuple(n*(denominator//d) for n,d in p) for p in ratios]
    records=ci._records(lattice,faces)
    result=ci._audit_pair(records,records,same_surface=True)
    modes.append({'coordinate_mode':mode,'coordinate_unit':'mm','integer_lattice_denominator':denominator,'source_vertex_records':len(v),'quotient_vertices':len(unique),'face_count':len(f),**result})
   reports.append({'stable_id':r['stable_id'],'member':r['member'],'member_sha256':r['member_sha256'],'source_audits':modes,'registered_binary32_m_count':reference[r['stable_id']]['exact_predicate']['unallowed_self_intersection_pair_count'],'registered_reference_report_sha256':pins[str(REFERENCE)]})
 for p,h in pins.items():need(sha(p)==h,'input changed '+p)
 result={'scope':'Exact topology-aware source self-intersection audit before registration. Binary64 is parsed source decimal represented as Python Float64; binary32 source-mm comparison is separately labeled. Registered count is retained evidence only; no correspondence, repair, or native qualification is inferred.','inputs':pins,'inputs_unchanged':True,'predicate_actual_path':ci.__file__,'python':sys.version,'numpy':np.__version__,'rows':reports,'wall_seconds':time.monotonic()-start}
 p=OUT/'report.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'report':str(p),'sha256':sha(p),'wall_seconds':result['wall_seconds'],'rows':[{'stable_id':r['stable_id'],'source_counts':[m['count'] for m in r['source_audits']],'registered_count':r['registered_binary32_m_count']} for r in reports]},indent=2),flush=True)
if __name__=='__main__':main()
