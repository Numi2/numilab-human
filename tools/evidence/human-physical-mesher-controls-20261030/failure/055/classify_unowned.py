from pathlib import Path
import numpy as np,json,hashlib,time,importlib.util,sys
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706';out=H/'tetgen-material-ownership-diagnostic-055';path=H/'native-domain-217-source-004/build_native_domain_217.py'
sys.path[:0]=[str(path.parent),str(P.parent/'authoring-deps')]
spec=importlib.util.spec_from_file_location('owner_classify055',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
t0=time.monotonic();z=np.load(H/'geometry-feasibility-008-warning037/skin-shell.npz');d={k:z[k]for k in z.files};whole=m._manifold(d['points_world_m'],d['faces'],d['face_marker'])
records=json.loads((out/'cell-role-witnesses-001.json').read_text())['cells'];recs=[]
for r in records:
 if r['attribute']==15:continue
 c=np.array(r['centroid_world_m'])
 box=np.asarray(whole.bounding_box());span=max(float(np.max(box[3:]-box[:3])),.01);rays=[]
 for axis in range(3):
  start=c.copy();start[axis]=box[axis]-span-.25;other=[j for j in range(3)if j!=axis];start[other[0]]+=.0137*span;start[other[1]]-=.0211*span
  hit=whole.ray_cast(start.tolist(),c.tolist());rays.append(len(hit))
 recs.append({'element_row':r['element_row'],'centroid_world_m':c.tolist(),'existing_owner_contains':m._point_inside(whole,c),'ray_hit_counts':rays})
r={'status':'UNOWNED_CELL_CENTROID_DIAGNOSTIC_ONLY','source_cell_witness_sha256':hashlib.sha256((out/'cell-role-witnesses-001.json').read_bytes()).hexdigest(),'shell_volume_m3':float(whole.volume()),'inside_count':sum(x['existing_owner_contains']for x in recs),'outside_count':sum(not x['existing_owner_contains']for x in recs),'records':recs,'wall_s':time.monotonic()-t0,'physical_admission':False}
p=out/'unowned-cell-containment-001.json';p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in r.items()if k!='records'}))

