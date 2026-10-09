from pathlib import Path
import ast, hashlib, importlib.util, json, sys, time
import numpy as np
H=Path("/Users/n/numi-human-free-apex-two-family-1178")
F=Path("/Users/n/numi-human-conforming-composition-source-1216")
B=Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218")
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(H/"src"))
from numilab_human import common_atlas_skin_clearance as current
from numilab_human import cardiac_cavity_intersections as ci
old_path=F/"src/numilab_human/common_atlas_skin_clearance.py"
old_pred=F/"src/numilab_human/cardiac_cavity_intersections.py"
pack=B/"native-baseline-310s-preparation/native-run/accepted-geometry/step-155000.mrvpack"
receipt=pack.with_suffix(".receipt.json")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
    if not ok: raise RuntimeError(msg)
need(not (OUT/"report.json").exists(),"refusing output overwrite")
paths=[old_path,old_pred,Path(current.__file__),Path(ci.__file__),Path(__file__),pack,receipt]
before={str(p):sha(p) for p in paths}
need(before[str(pack)]=="965518a6cacf8e2ff7c48638c8eed8c8d989b8b32f68a1359451f7584a41e62a","pack changed")
need(before[str(receipt)]=="c119e4c24f8e85f374d89ffc4d4422102e5b1588d94f22c07358dc5bea99f3cb","receipt changed")
need(before[str(old_path)]=="ec198693fce562db27c4b761e3dbbc4b1bff4f0b9458ae67ab7ba596a97a000f","legacy owner changed")
need(before[str(old_pred)]=="934c64fa6a64a06044aee6db44cdcd71dc497fcec97a20a80029575faaaff4b4","legacy predicate changed")
def function_ast(p,name):
    return ast.dump(next(n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
need(function_ast(old_pred,"point_location")==function_ast(ci.__file__,"point_location"),"legacy reference function changed")
spec=importlib.util.spec_from_file_location("numilab_human._legacy_clearance_full_inside_1228",old_path)
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
r=json.loads(receipt.read_text())
need(r.get("accepted_step")==155000 and r.get("accepted_pack_path")==str(pack) and r.get("physical_endpoint")=="accepted" and r.get("surface_audit_endpoint")=="passed","accepted capture binding mismatch")
xyz,surfaces,_=current._pack_surfaces(pack,{(51005,64)})
target_faces=surfaces[(51005,64)]["faces"]; target_ids=np.unique(target_faces)
t=time.perf_counter()
target=current._prepare_closed_clearance_target(xyz[target_ids],np.searchsorted(target_ids,target_faces),allow_nested_enclosure=True)
prepare_s=time.perf_counter()-t
skin_ids=np.unique(surfaces[(51007,1)]["faces"])
points=np.asarray(xyz[skin_ids],dtype="<f4")
need(len(points)==54663 and len(target_faces)==6690,"geometry coverage differs")
t=time.perf_counter(); old=legacy._closed_target_inside_vertices(points,target); old_s=time.perf_counter()-t
t=time.perf_counter(); new=current._closed_target_inside_vertices(points,target); new_s=time.perf_counter()-t
need(np.array_equal(old,new),"inside vertex classifications differ")
after={str(p):sha(p) for p in paths};need(before==after,"inputs changed")
report={"status":"pass","scope":"offline full referenced-skin containment query equivalence on one accepted late pose; not anatomy or simulation qualification","accepted_step":155000,"target":[51005,64],"all_skin_vertex_count":len(points),"target_face_count":len(target_faces),"outer_face_count":len(target["_interior_records"]),"inside_vertex_indices_legacy":old.tolist(),"inside_vertex_indices_prepared":new.tolist(),"identical_inside_vertex_sets":True,"query_points_f32_sha256":hashlib.sha256(points.tobytes()).hexdigest(),"prepared_record_identity_sha256":target["_interior_point_locator"].record_identity_sha256,"target_preparation_wall_s":prepare_s,"legacy_query_wall_s":old_s,"prepared_query_wall_s":new_s,"timing_scope":"one shared-host sequential legacy-then-prepared comparison; no host exclusivity or simulation performance claim","legacy_point_location_function_ast_unchanged":True,"inputs_before_sha256":before,"inputs_after_sha256":after,"command":[sys.executable,str(Path(__file__).resolve())]}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:report[k] for k in ("status","all_skin_vertex_count","outer_face_count","legacy_query_wall_s","prepared_query_wall_s","identical_inside_vertex_sets")}))
