from pathlib import Path
from fractions import Fraction
from itertools import product
import hashlib,importlib.util,sys,json,numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=H/"fat614-yy-volume-replay-004/run-007-comp7-current-plc"
D=OUT/"run-008-interior-steiner-ulp-search";D.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def table(path):
 lines=[x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
 n=int(lines[0].split()[0]);a=np.asarray([[float(v) for v in q.split()] for q in lines[1:1+n]],np.float64)
 if len(a)!=n:raise ValueError("row count mismatch")
 return a
pins={"node":"14df56f8f0774537b936e82d4599b8b942b3b92be9e312f8b4da1b02ddba26c9","ele":"e7ef87fd5b5956c2833e349271751bb620c728db708066b2dc1a2a1e83559b09","face":"c38706db6285972ebba4258703b4108d6bf09074248e1a86c032113a09d09bcc","poly":"2383b6e5a12fcda84a46b7c0bb009d53c407c54a45203d243b8de9c33ca07149","stdout":"86c521129bbcc1c66febab92e010bd435dba296388d8d34479075042179eb930","v4":"a3864e53c431034e6dc1f55a672025b5dd6709f5b041dc86735450b454d7f698"}
paths={"node":OUT/"cavity.1.node","ele":OUT/"cavity.1.ele","face":OUT/"cavity.1.face","poly":OUT/"cavity.poly","stdout":OUT/"tetgen.stdout.log","v4":OUT/"rebuild_and_run_comp7_current_D_plc_v4.py"}
for k,pth in paths.items():
 got=sha(pth)
 if got!=pins[k]:raise ValueError(f"{k} hash changed: {got}")
m=load("comp7_quality_198_ulpsearch_v3",H/"fixed_boundary_volume_flip_198_root.py")
node=table(paths["node"]);ele=table(paths["ele"]);ids=node[:,0].astype(np.int64);lut={int(x):i for i,x in enumerate(ids)}
t=np.asarray([[lut[int(x)] for x in rr] for rr in ele[:,1:5]],np.int64);raw64=node[:,1:4];p=raw64.astype(np.float32);target_id=156
idx=lut[target_id];inc=np.flatnonzero(np.any(t==idx,axis=1))
if len(inc)!=16 or any(int(ele[i,5])!=16 for i in inc):raise ValueError("unexpected node star/attributes")
ip=m.integer_points(p);base_exact=[int(m.exact_det(ip,t[i])) for i in inc];base_det,base_q=m.measurements(p,t[inc]);base_pass=m.passes(base_det,base_q)
def shift(x,n):
 bits=int(np.asarray(np.float32(x)).view(np.uint32))
 if n:bits+=n if (bits>>31)==0 else -n
 return np.asarray([bits],np.uint32).view(np.float32)[0]
def f64sign(x):
 a=[[Fraction.from_float(float(x[j,k]))-Fraction.from_float(float(x[0,k])) for k in range(3)] for j in (1,2,3)]
 d=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
 return (d>0)-(d<0)
raw64sign=[f64sign(raw64[[lut[int(v)] for v in ele[i,1:5].astype(np.int64)]]) for i in inc]
orig=p[idx].copy();best=None;tested=0
for radius in range(17):
 for off in product(range(-radius,radius+1),repeat=3):
  if max(map(abs,off))!=radius:continue
  cand=np.asarray([shift(orig[j],off[j]) for j in range(3)],np.float32)
  ip[idx]=tuple(int(v) for v in m.integer_points(cand.reshape(1,3))[0]);tested+=1
  ds=[int(m.exact_det(ip,t[i])) for i in inc]
  if min(ds)<=0:continue
  pp=p.copy();pp[idx]=cand;det,q=m.measurements(pp,t[inc]);ps=m.passes(det,q)
  if np.any(base_pass&~ps):continue
  key=(int(np.sum(~ps)),sum(abs(x) for x in off),sum(x*x for x in off),tuple(off))
  if best is None or key<best[0]:best=(key,off,cand.copy(),ds,q.copy(),ps.copy())
ip[idx]=tuple(int(v) for v in m.integer_points(orig.reshape(1,3))[0])
if best is None:raise SystemExit("no exact-positive nondegrading point in bounded ULP search")
poly=table(paths["poly"]);poly_ids={int(row[0]) for row in poly[:int(poly[0,0])]}
_,off,cand,ds,q,ps=best
source=Path(__file__)
record={"schema":"numi.fat614.comp7.interior-steiner-f32-ulp-search.v3","status":"PASS_EXACT_POSITIVE_NO_NEW_NATIVE_DEFECT","physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","source_path":str(source),"source_sha256":sha(source),"pinned_inputs":{k:{"path":str(paths[k]),"sha256":pins[k]} for k in paths},"raw_tetgen_return_code":"NOT_CAPTURED_BY_V4_VALIDATION_NAMEERROR","target_output_node_id":target_id,"target_node_row":int(idx),"incident_output_tet_rows":inc.tolist(),"incident_ele_ids":ele[inc,0].astype(np.int64).tolist(),"incident_material_ids":ele[inc,5].astype(np.int64).tolist(),"raw_float64_exact_det_signs":raw64sign,"baseline_float32_exact_det_signs":[(x>0)-(x<0) for x in base_exact],"candidate_float32_exact_det_signs":[(x>0)-(x<0) for x in ds],"baseline_native_pass":[bool(x) for x in base_pass],"candidate_native_pass":[bool(x) for x in ps],"baseline_normalized_dets":base_q.tolist(),"candidate_normalized_dets":q.tolist(),"search_max_radius_ulps":16,"tested_candidates":tested,"selected_componentwise_ulp_offsets":list(map(int,off)),"original_f32_coordinate":orig.tolist(),"candidate_f32_coordinate":cand.tolist(),"input_PLC_node":bool(target_id in poly_ids),"previously_good_incident_cells_preserved":True}
(D/"coordinate-correction-v3.json").write_text(json.dumps({"schema":"numi.fat614.comp7.interior-steiner-coordinate-correction.v3","tetgen_output_node_id":target_id,"node_row":int(idx),"original_f32_coordinate":orig.tolist(),"corrected_f32_coordinate":cand.tolist(),"componentwise_ulp_offsets":list(map(int,off)),"incident_tet_rows":inc.tolist(),"input_PLC_node":bool(target_id in poly_ids),"reason":"exact F32 correction of an inferred interior TetGen Steiner node; raw Float64 output star was exact-positive"},indent=2)+"\n")
(D/"search-v3.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({"status":record["status"],"candidate":cand.tolist(),"offsets":off,"native_fail_count":int(np.sum(~ps)),"tested":tested}))
