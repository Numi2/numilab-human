#!/usr/bin/env python3
"""Compare old and corrected component selection on the six exact source members."""
from pathlib import Path
import ast, collections, hashlib, importlib.util, json, sys, time, zipfile
import numpy as np
HERE=Path(__file__).resolve().parent
CURRENT=Path('/Users/n/numi-human-source-seam-connectivity-1247')
PARENT=Path('/Users/n/numi-human-target-broadphase-1241')
MANIFEST=Path('/Users/n/numi-human-retained-delivery-20261009/passive-biceps-micro-overlap-1225/compose-current-1cd-attempt003/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json')
ARCHIVE=Path('/Users/n/numi-human-source-cache/isa_BP3D_4.0_obj_99.zip')
AUDITOR=Path('/Users/n/numi-human-retained-delivery-20261009/passive-muscle-self-audit-1213/audit_muscle_self_1213.py')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def main():
 start=time.monotonic()
 output=HERE/'source-selection-comparison.json'
 need(not output.exists(),'refuse overwrite')
 paths=[Path(__file__),CURRENT/'src/numilab_human/model.py',PARENT/'src/numilab_human/model.py',MANIFEST,ARCHIVE,AUDITOR]
 pins={str(p):sha(p) for p in paths}
 need(pins[str(ARCHIVE)]=='40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e','source archive identity')
 sys.path.insert(0,str(CURRENT/'src'))
 from numilab_human import model
 tree=ast.parse((PARENT/'src/numilab_human/model.py').read_text())
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_bodyparts_largest_connected_surface_component')
 env={'ImportError':model.ImportError}
 exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_parent_helper','exec'),env)
 previous=env[function.name]
 spec=importlib.util.spec_from_file_location('retained_audit',AUDITOR)
 audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
 def topology(v,f):
  unique,inverse=np.unique(np.asarray(v,dtype=np.float64),axis=0,return_inverse=True)
  return audit.edge_diagnostics(inverse[np.asarray(f)],len(unique))
 def support(v,f):return collections.Counter(tuple(tuple(v[i]) for i in face) for face in f)
 manifest=json.loads(MANIFEST.read_text())
 rows=[r for r in manifest['source']['surfaces'] if r['stable_id'] in [23,24,27,28] or r['layer']=='tendon']
 need(len(rows)==6,'actual selected source inventory')
 result=[]
 with zipfile.ZipFile(ARCHIVE) as archive:
  for row in rows:
   member=row['member'];obj=archive.read(member)
   need(hashlib.sha256(obj).hexdigest()==row['member_sha256'],'source member identity')
   v,f=model._bodyparts_obj_triangles(obj,member)
   beforev,beforef,beforeinfo=previous(v,f,member)
   afterv,afterf,afterinfo=model._bodyparts_largest_connected_surface_component(v,f,member)
   cv,cf,cancel=model._bodyparts_cancel_opposite_surface_faces(afterv,afterf,member)
   original_support=support(v,f);after_support=support(afterv,afterf)
   need(not(after_support-original_support),'non-source geometry emitted')
   need(not(support(beforev,beforef)-after_support),'prior retained source faces lost')
   result.append({'stable_id':row['stable_id'],'member_id':row['member_id'],'member':member,'member_sha256':row['member_sha256'],'label':row['label'],
    'source':topology(v,f),'before':{'selection':beforeinfo,'topology':topology(beforev,beforef)},
    'after':{'selection':afterinfo,'topology':topology(afterv,afterf)},
    'after_existing_opposite_pair_cancellation':{'topology':topology(cv,cf),'cancellation':cancel},
    'restored_source_face_count':len(afterf)-len(beforef),'new_inferred_faces':0,'source_points_moved':False,
    'prior_retained_oriented_source_support_preserved':True,
    'selected_geometry_identical':beforev==afterv and beforef==afterf})
 for p,h in pins.items():need(sha(p)==h,'input changed: '+p)
 report={'scope':'Exact source-coordinate topology and source-face preservation only. Does not regenerate registration, admit anatomical surfaces, or run native physics.',
  'inputs':pins,'input_pins_unchanged':True,'surfaces':result,'wall_seconds':time.monotonic()-start}
 output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report':str(output),'sha256':sha(output),'surfaces':[{'id':r['stable_id'],'restored_faces':r['restored_source_face_count'],'boundary_edges_before':r['before']['topology']['boundary_edge_count'],'boundary_edges_after':r['after']['topology']['boundary_edge_count'],'identical':r['selected_geometry_identical']} for r in result]},indent=2),flush=True)
if __name__=='__main__':main()
