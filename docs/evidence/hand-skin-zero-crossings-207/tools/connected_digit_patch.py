from pathlib import Path
import sys,numpy as np,json,time
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'authoring-deps'));import manifold3d as mf
from cycle_fill_resume import close_for_boolean,from_mesh,export,outer
r=R/'envelope-028-digit-patch';r.mkdir(exist_ok=False);z=np.load(R/'envelope-014-connected-cycles/candidate.raw.npz');P=z['points_world_m_f64'];F=z['source_O_faces'];V,C,_,_=close_for_boolean(P,F[:,[0,2,1]]);body=from_mesh(V,C,np.arange(len(C)));sid=int(np.asarray(body.to_mesh64().run_original_id)[0]);base=np.load(R/'envelope-006-local-lattice/candidate.npz')['points_world_m'];Q=np.concatenate([np.load(R/('envelope-005-all18-stable/offset-'+key+'.npz'))['points_world_m_f64'] for key in ['59-95','60-96','61-97']]);reports=[]
for pad in [.002,.004,.006]:
 lo=Q.min(axis=0)-pad;hi=Q.max(axis=0)+pad;clip=mf.Manifold.cube(hi-lo).translate(lo);patch=body^clip;mesh=patch.to_mesh64();HP=np.array(mesh.vert_properties)[:,:3];hull=mf.Manifold.hull_points(np.vstack([HP,Q])).as_original();trial,voids=outer(body+hull);other=[]
 for path in (R/'envelope-005-all18-stable').glob('offset-*.npz'):
  if path.stem in ['offset-59-95','offset-60-96','offset-61-97','offset-57-89']:continue
  zz=np.load(path);ob=from_mesh(zz['points_world_m_f64'],zz['source_O_faces']);iv=(hull^ob).volume()*1e9
  if iv>1e-7:other.append({'bank':path.stem,'overlap_mm3':iv})
 report={'pad_mm':pad*1000,'bounds_mm':[(lo*1000).tolist(),(hi*1000).tolist()],'before_genus':body.genus(),'after_genus':trial.genus(),'added_volume_mm3':(trial.volume()-body.volume())*1e9,'other_ray_offset_overlaps':other,'cropped_parts':[(p.volume()*1e9,p.genus())for p in patch.decompose()]};reports.append(report);print(json.dumps(report),flush=True)
 if trial.genus()==0 and not other:
  body=trial;np.savez_compressed(r/'addition.npz',points_world_m_f64=np.asarray(hull.to_mesh64().vert_properties)[:,:3],source_O_faces=np.asarray(hull.to_mesh64().tri_verts));break
body,voids=outer(body.simplify(1e-6));result={'trials':reports,'final_genus':body.genus()};(r/'report.json').write_text(json.dumps(result,indent=2));result['emission']=export(body,r/'candidate.npz',sid,len(F),base);(r/'report.json').write_text(json.dumps(result,indent=2));print('done',flush=True)
