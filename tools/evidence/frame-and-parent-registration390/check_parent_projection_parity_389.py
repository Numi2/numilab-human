from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
P=Path(__file__).parent
OUT=P/'parent-registration-filter-parity-389';OUT.mkdir(exist_ok=False)
def load(name,path):
 sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
old=load('parent314',P/'bounded-parent-registration-314/export_authoring_registered_surface_sidecars_230.py')
new=load('parent388',P/'bounded-parent-registration-388/export_authoring_registered_surface_sidecars_230.py')
base=P/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
parent,index,indices,meta=old.load_parent_current(base)
started=time.perf_counter();out=[]
for material in (15,16):
 mi=indices[material]
 for local in (0,len(mi['globalRows'])//2,len(mi['globalRows'])-1):
  row=int(mi['globalRows'][local]);point=parent['points'][parent['tets'][row]].mean(0)
  # Include an off-surface point while keeping each query tied to actual parent geometry.
  if local==0:point=point+np.array([.00001,-.00002,.00003])
  reference=None;timings=[]
  for owner in (old,new):
   tick=time.perf_counter();value=owner._exact_nearest_same_material(parent,mi,material,point,row);timings.append(time.perf_counter()-tick)
   key=(value['row'],value['q'],value['bary'],value['d2'],value['ties'])
   if reference is None:reference=key
   elif key!=reference:raise AssertionError('real parent exact projection/tie result differs')
  out.append({'material':material,'seed_parent_tet':row,'point_world_m':point.tolist(),'selected_parent_tet':value['row'],
   'exact_projection_parity':True,'exact_tie_count':len(value['ties'])-1,'baseline_wall_s':timings[0],'filtered_wall_s':timings[1],
   'parent_material_tets':len(mi['globalRows']),'conservative_survivors':value['diagnostic']['conservativeAABBPrefilterSurvivors'],
   'exact_aabb_candidates':value['diagnostic']['exactAABBCandidateCount']})
receipt={'status':'PASS_REAL_PARENT_EXACT_PROJECTION_AND_TIE_PARITY','samples':out,'parent':meta,
 'source_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [P/'conservative_parent_aabb_388.py',P/'bounded-parent-registration-314/export_authoring_registered_surface_sidecars_230.py',P/'bounded-parent-registration-388/export_authoring_registered_surface_sidecars_230.py']},
 'elapsed_s':time.perf_counter()-started,'physical_admission':False,'surface_export_completed':False}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
