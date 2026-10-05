from pathlib import Path
import sys,json,hashlib,struct,itertools
import numpy as np
root=Path('/Users/n/numi-human-resting-evidence-20261005')
out=Path(__file__).parent
sys.path.insert(0,'/Users/n/numi-human-resting-conforming-source-009/matter/tools')
import accepted_mrvpack_surface_audit as a
predicate=Path('/Users/n/numilab-human/src/numilab_human/cardiac_cavity_intersections.py')
pred=a.predicate_module(predicate)
run=root/'thorax-short-edge-native-001'
old_audit=Path('/tmp/candidate005-parent-child-audit.json')
old=json.loads(old_audit.read_text())['frames'][0]['parent_pairs']
map_paths={sid:root/f'thorax-conforming-field-009/surface-{sid}-parent-faces.json' for sid in (305,311)}
maps={sid:json.loads(p.read_text()) for sid,p in map_paths.items()}
pairs=set()
for row in old:
 left=[i for old in row['candidate_lung_children'] for i in maps[305][str(old)]]
 right=[i for old in row['candidate_diaphragm_children'] for i in maps[311][str(old)]]
 pairs.update(itertools.product(left,right))
# Include the separate old patch-versus-nonpatch witness.
pairs.update(itertools.product(maps[305]['8043'],maps[311]['27058']))
receipt_path=root/'cardiac-wall-binding-005/resting-anatomy-receipt.json'
receipt=json.loads(receipt_path.read_text())
patch=next(x for x in receipt['provenance']['diaphragm_lung_interface']['interface_rows'] if x['lung_stable_id']==305)
lpatch={i for begin,end in patch['registered_lung_face_index_ranges'] for i in range(begin,end)}
dpatch=set(range(patch['diaphragm_patch_face_start'],patch['diaphragm_patch_face_start']+patch['diaphragm_patch_face_count']))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'driver_sha256':sha(Path(__file__)),'predicate_sha256':sha(predicate),'source_pair_audit_sha256':sha(old_audit),'mapping_sha256':{str(sid):sha(p) for sid,p in map_paths.items()},'input_anatomy_receipt_sha256':sha(receipt_path),'qualification':'targeted regression of all31 retained parent pairs and one separate patch/nonpatch witness; not whole-thorax clearance','candidate_pair_count':len(pairs),'frames':[]}
for step in (0,639,2783,2999):
 pack=run/f'accepted-geometry/step-{step}.mrvpack';cap=pack.with_suffix('.receipt.json')
 mm,stream,off,surfaces=a.read_pack(pack)
 provenance=a.validate_accepted_receipt(pack,cap,step,mm,off,surfaces)
 vertices=np.ndarray(((len(mm)-off)//80,20),dtype='<f4',buffer=mm,offset=off)
 zero={};triangles={}
 for sid in range(305,312):
  sem=51023 if sid<310 else 51024 if sid==310 else 51010
  f=np.asarray(surfaces[sem,sid]['faces']);tri=vertices[f,:3].astype(float)
  n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);zero[str(sid)]=np.flatnonzero(np.all(n==0,axis=1)).tolist()
  if sid in (305,311):triangles[sid]=tri
 if any(zero.values()):
  report['frames'].append({'step':step,'receipt':provenance,'zero_area_face_indices':zero,'pass':False,'reason':'degenerate triangles rejected before interface audit'})
 else:
  chosen={sid:sorted({index for pair in pairs for index in [pair[0] if sid==305 else pair[1]]}) for sid in (305,311)}
  scale=a.coordinate_lattice_scale(float(x) for sid in chosen for idx in chosen[sid] for p in triangles[sid][idx] for x in p)
  tri={sid:{idx:tuple(tuple(a.lattice_integer(float(x),scale) for x in p) for p in triangles[sid][idx]) for idx in chosen[sid]} for sid in chosen}
  allowed=0;disjoint=0;bad=[]
  for fi,fj in sorted(pairs):
   first,second=tri[305][fi],tri[311][fj]
   if any(max(p[k] for p in first)<min(p[k] for p in second) or max(p[k] for p in second)<min(p[k] for p in first) for k in range(3)):
    disjoint+=1;continue
   pts=set(pred.triangle_intersection_points(first,second))
   if not pts:disjoint+=1;continue
   common=set(first)&set(second)
   ok=(len(common)<=2 and all(pred._allowed_shared_point(p,common) for p in pts)) or (len(common)==3 and fi in lpatch and fj in dpatch)
   if ok:allowed+=1;continue
   spans=max([sum(float((p[k]-q[k])/scale)**2 for k in range(3))**.5 for p in pts for q in pts] or [0])
   bad.append({'lung_face':fi,'diaphragm_face':fj,'common_vertices':len(common),'unique_intersection_points_m':[[float(x/scale) for x in p] for p in sorted(pts)],'segment_span_m':spans})
  report['frames'].append({'step':step,'receipt':provenance,'zero_area_face_indices':zero,'pair_count':len(pairs),'disjoint_pairs':disjoint,'allowed_declared_patch_or_common_boundary_pairs':allowed,'forbidden_pairs':bad,'pass':not bad})
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'step':step,'pass':report['frames'][-1]['pass'],'zeros':{k:len(v) for k,v in zero.items()},'bad':len(report['frames'][-1].get('forbidden_pairs',[]))}),flush=True)
 del vertices;mm.close();stream.close()
