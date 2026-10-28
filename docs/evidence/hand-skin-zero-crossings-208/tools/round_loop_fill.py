from pathlib import Path
import json,sys,time,numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/"authoring-deps"));import manifold3d as mf
from cycle_fill_resume import close_for_boolean,from_mesh,outer,export
from short_cycles import loops
out=R/"envelope-031-rounded-cycle";out.mkdir(exist_ok=False);z=np.load(R/"envelope-014-connected-cycles/candidate.raw.npz");P=z["points_world_m_f64"];F=z["source_O_faces"];V,C,_,_=close_for_boolean(P,F[:,[0,2,1]]);body=from_mesh(V,C,np.arange(len(C)));sid=int(np.asarray(body.to_mesh64().run_original_id)[0]);base=np.load(R.parent/"publication-206-stage-001/inputs/baseline-806.npz")["points_world_m"];cs=loops(V,C,40);start=time.monotonic();trials=[]
for radius in [.0003,.0008,.0015]:
 for length,span,ids in cs[:20]:
  if time.monotonic()-start>90:break
  Q=V[ids];ball=mf.Manifold.sphere(radius,8);hull=mf.Manifold.hull_points(np.array(Q,copy=True));patch=hull.minkowski_sum(ball).as_original();trial,voids=outer(body+patch);g=trial.genus();row={"radius_mm":radius*1000,"span_mm":span*1000,"length_mm":length*1000,"genus":g,"added_mm3":(trial.volume()-body.volume())*1e9,"voids":[p.volume()*1e9 for p in voids]};trials.append(row);print(json.dumps(row),flush=True)
  if g==0:
   np.savez_compressed(out/"patch.npz",cycle_points=Q,points=np.asarray(patch.to_mesh64().vert_properties)[:,:3],faces=np.asarray(patch.to_mesh64().tri_verts));row["emission"]=export(trial,out/"candidate.npz",sid,len(F),base);(out/"report.json").write_text(json.dumps({"trials":trials,"native_admission":False,"physical_steps":0},indent=2));sys.exit(0)
(out/"report.json").write_text(json.dumps({"trials":trials,"native_admission":False,"physical_steps":0},indent=2))
