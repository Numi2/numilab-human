#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import hashlib, importlib.util, json, sys, time
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
RUN=H/"d-shortedge-quotient-414-batch001-volume-transfer/run002"
BASE=H/"fat-comp0-connected-warning-repair-412/run-012-final-D-current-target-rebind/fat-current-typed-target-rebound.npz"
BASE_SHA="72c851692580a639f89a4916bc4bf5c38331be9b6def89322eccf02ee66ea6ec"
SURF=H/"d-shortedge-quotient-414-batch001-surface-proposal/candidate-D-short-edge-collapse-raw.npz"
SURF_SHA="60f89988727fd00d272aaf2bed798e0f19e9f91f98067c308408a935c183cacc"
SREPORT=H/"d-shortedge-quotient-414-batch001-surface-proposal/report.json"
SREPORT_SHA="4d4be7ee3d713f8cc7a4a4e52e0adff63e1c021d710a74015aa24da223899653"
CAND=RUN/"fat-D414-surface-quotient-volume-proposal.npz"
CAND_SHA="0cfd2d78372597e5ff0bad547b10321d2b3e953ee7a20b899017bb197d146c0a"
MAP=RUN/"operation-cell-star-map.npz"
MAP_SHA="f3a757fb951f01fb15ce86e87cb93494ad3efd966c05ca762e2041af4b91c5bc"
REPORT=RUN/"report.json"
REPORT_SHA="5126e5195c05673e118896f1c5ee84f4fb52a98f4bfc6ad55018a647f14b3d96"
V414=H/"d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/fat-D411-379-plus-comp1-run041-composed.npz"
V414_SHA="14f06beaf20a2f78396585e7868df831cff4b9f27e3401e271d7643175073a77"
OWNER=H/"transfer_surface_contractions_indexed_379_root.py"
OWNER_SHA="ea9135fe6c19efc644bd8429bd01e5f36547f304e86da9214aa1bb594441da37"
OUT=RUN/"independent-replay-417.json"
def sha(p): return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def require(ok,msg):
 if not ok:raise ValueError(msg)
start=time.monotonic()
for p,h in ((BASE,BASE_SHA),(SURF,SURF_SHA),(SREPORT,SREPORT_SHA),(CAND,CAND_SHA),(MAP,MAP_SHA),(REPORT,REPORT_SHA),(V414,V414_SHA),(OWNER,OWNER_SHA)):
 require(sha(p)==h,"sha mismatch: "+str(p))
r=json.loads(REPORT.read_text());sr=json.loads(SREPORT.read_text())
require(r["status"]=="PASS_D414_VOLUME_QUOTIENT_PROPOSAL","transfer status")
require(r["physical_admission"] is False and r["mass_mapping"]=="NOT_RUN" and r["native_steps"]==0,"scope flags")
require(r["volume_boundary_equals_mapped_current_boundary"] is True,"driver boundary flag")
base=load(BASE);vol414=load(V414);surf=load(SURF);cand=load(CAND);mp=load(MAP)
for k in ("points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces"):
 require(np.array_equal(base[k],vol414[k]),"414 qualification geometry differs from target source: "+k)
p=np.asarray(base["points_world_m_f32"],np.float32);t=np.asarray(base["tetrahedra"],np.int64);mat=np.asarray(base["material_indices"],np.int32)
require(np.array_equal(base["points_world_m"],p),"input point F32 readback mismatch")
require(np.array_equal(cand["points_world_m_f32"],p) and np.array_equal(cand["points_world_m"],p),"output changed point coordinates")
ops=sr["accepted_operations"]
require(len(ops)==2363 and sr["accepted_collapse_count"]==2363,"surface operation count")
dq=np.arange(len(surf["points_world_m_f32"]),dtype=np.int64)
seen_drop=set();seen_pair=set()
for oi,o in enumerate(ops):
 require(o["role"]=="D" and o["input_sha256"]=="fba94572f52f51e58902063da181d24fcdf47473d0b322d68370277d9ab32c03","surface operation provenance")
 d=int(o["collapsed_input_point_row"]);k=int(o["retained_input_point_row"])
 require(0<=d<len(dq) and 0<=k<len(dq) and d!=k,"surface endpoint bounds")
 require(d not in seen_drop and (d,k) not in seen_pair,"duplicate contraction endpoint")
 require(dq[d]==d and dq[k]==k,"surface contraction chain dependency")
 dq[d]=k;seen_drop.add(d);seen_pair.add((d,k))
require(np.array_equal(dq,np.asarray(cand["source_D414_surface_quotient"],np.int64)),"surface quotient map differs from operation replay")
require(np.array_equal(surf["points_world_m_f32"],np.asarray(surf["points_world_m"],np.float32)),"raw D f32 coordinates differ")
# Derive D→volume mapping by exact source point coordinates, then compose the quotient.
d=load(H/"d-shortedge-quotient-414-next-001-preflight-attempt004/identity-seed-D.npz")
base_dv=np.asarray(base["fat_current_target_rebind_D_point_to_volume_rows"],np.int64)
require(len(base_dv)==len(dq),"D point crosswalk row count")
require(np.array_equal(p[base_dv],np.asarray(d["points_world_m_f32"],np.float32)),"D→volume coordinate crosswalk")
node=np.arange(len(p),dtype=np.int64);node[base_dv]=base_dv[dq]
require(np.array_equal(node,np.asarray(cand["input_node_to_output_node"],np.int64)),"volume point map differs from independently composed D quotient")
require(np.array_equal(node[base_dv],np.asarray(cand["fat_current_target_rebind_D_point_to_volume_rows"],np.int64)),"candidate D target crosswalk")
require(np.array_equal(node[np.asarray(base["fat_current_target_rebind_I_point_to_volume_rows"],np.int64)],np.asarray(cand["fat_current_target_rebind_I_point_to_volume_rows"],np.int64)),"candidate I point map")
require(np.array_equal(node[np.asarray(base["fat_current_target_rebind_wall_faces_volume_rows"],np.int64)],np.asarray(cand["fat_current_target_rebind_wall_faces_volume_rows"],np.int64)),"candidate wall point map")
# Independently derive retained source rows by applying the point quotient to all input tets.
mt=node[t];smt=np.sort(mt,axis=1);deleted=np.any(smt[:,1:]==smt[:,:-1],axis=1);retained=np.flatnonzero(~deleted)
require(np.array_equal(retained,np.asarray(cand["retained_input_cell_rows"],np.int64)),"retained input cell rows do not match quotient degeneracies")
require(np.array_equal(mt[retained],np.asarray(cand["tetrahedra"],np.int64)),"output tets do not equal independently remapped source tets")
require(np.array_equal(mat[retained],np.asarray(cand["material_indices"],np.int32)),"output materials not retained rowwise")
ct=np.asarray(cand["tetrahedra"],np.int64);cbd=np.asarray(cand["boundary_faces"],np.int64)
require(len(ct)==len(retained)==811807 and len(mat)==819309,"cell count")
require(len(np.unique(np.sort(ct,axis=1),axis=0))==len(ct),"duplicate candidate tets")
require(np.all(np.asarray(cand["material_indices"])==16),"Fat material field")
# Independent oriented boundary chain, including multiplicity and parity.
PAT=np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),np.int64)
def chain(tets):
 arr=np.asarray(tets,np.int64)
 fs=(arr[:,PAT].reshape(-1,3) if arr.ndim==2 and arr.shape[1]==4 else arr.reshape(-1,3))
 s=np.sort(fs,axis=1)
 inv=((fs[:,0]>fs[:,1]).astype(np.int8)+(fs[:,0]>fs[:,2]).astype(np.int8)+(fs[:,1]>fs[:,2]).astype(np.int8))
 sign=1-2*(inv%2)
 order=np.lexsort((s[:,2],s[:,1],s[:,0]));s=s[order];sign=sign[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(s[1:]!=s[:-1],axis=1))]
 vals=np.add.reduceat(sign,starts)
 return {tuple(map(int,s[starts[i]])):int(vals[i]) for i in range(len(starts)) if vals[i]}
cc=chain(ct);bc=chain(cbd)
require(cc==bc,"candidate listed boundary differs from independently derived oriented chain")
# Independently establish exact current-boundary quotient and target residual.
base_chain=chain(t);expected={}
for f,v in base_chain.items():
 mf=tuple(int(node[x]) for x in f)
 if len(set(mf))==3:
  sf=tuple(sorted(mf));inv=((mf[0]>mf[1])+(mf[0]>mf[2])+(mf[1]>mf[2]))%2;sg=(1 if inv==0 else -1)*v
  expected[sf]=expected.get(sf,0)+sg
expected={k:v for k,v in expected.items() if v}
require(expected==cc,"output boundary does not equal mapped input boundary chain")
# Recompute native metrics on all cells and rowwise no-old-good-regression.
owner=module("independent379_validation417",OWNER);m=owner.m
bd,qb=m.measurements(p,t);ad,qa=m.measurements(p[0:0],np.empty((0,4),np.int64)) if False else (None,None)
nd,nq=m.measurements(p,ct)
require(np.isfinite(nd).all() and np.isfinite(nq).all() and np.all(nd>0),"float64 determinant positivity")
passes=lambda det,q:np.isfinite(det)&np.isfinite(q)&(det/6>1e-18)&(q>=1e-5)
bg=passes(bd,qb)[retained];ng=passes(nd,nq)
regress=int(np.count_nonzero(bg&~ng));bad_before=int(np.count_nonzero(~passes(bd,qb)));bad_after=int(np.count_nonzero(~ng))
require(regress==0,"surviving previously good cells regressed")
require(bad_before==24174 and bad_after==20620 and bad_before-bad_after==3554,"native quality counts mismatch")
# Independent exact integer-lattice replay for all changed cells and exact volume delta.
volmod=module("exact_volume_delta_validation417",H/"interior-quality-conditioning-309-banks/condition_registered_interior_309.py")
mpoints={int(row):volmod.integer_point(p[int(row)]) for row in np.unique(np.asarray(mp["operation_old_changed_tetrahedra"],np.int64))}
oldc=np.asarray(mp["operation_old_changed_tetrahedra"],np.int64);newc=np.asarray(mp["operation_new_changed_tetrahedra"],np.int64);delc=np.asarray(mp["operation_changed_deleted"],bool)
require(len(oldc)==len(newc)==len(delc)==22356,"changed-cell CSR count")
old_det=[volmod.m.exact_det([mpoints[int(v)] for v in tet],[0,1,2,3]) for tet in oldc]
new_det=[volmod.m.exact_det([mpoints[int(v)] for v in tet],[0,1,2,3]) for tet,dead in zip(newc,delc) if not dead]
require(min(new_det)>0,"changed output tets fail exact F32 positivity")
exact_delta=sum(new_det)-sum(old_det)
require(str(exact_delta)==r["exact_F32_local_det6_delta_integer"],"exact integer-lattice volume delta mismatch")
changed_rows=np.asarray(mp["operation_changed_cell_rows"],np.int64)
require(len(np.unique(changed_rows))==len(changed_rows)==len(oldc),"changed rows not unique")
star_rows=np.asarray(mp["operation_full_star_cell_rows"],np.int64)
star_offsets=np.asarray(mp["operation_full_star_offsets"],np.int64)
require(len(star_rows)==22356 and len(np.unique(star_rows))==len(star_rows),"full operation stars overlap")
require(star_offsets.shape==(2364,) and star_offsets[0]==0 and star_offsets[-1]==len(star_rows),"full star CSR malformed")
protected=np.asarray(mp["comp0_comp1_protected_cell_rows"],np.int64)
require(not np.intersect1d(star_rows,protected,assume_unique=False).size,"D transfer overlaps protected comp0/comp1 cells")
# Cell maps are authoritative; point map intentionally retains unused historical rows.
in2out=np.asarray(cand["input_cell_to_output_cell"],np.int64);out2in=np.asarray(cand["output_cell_to_input_cell"],np.int64)
require(np.array_equal(in2out[retained],np.arange(len(retained))) and np.all(in2out[deleted]==-1),"input cell map")
require(np.array_equal(out2in,retained),"output-to-input retained cell map")
# Target residual as an oriented face-counter difference.
targ=node[np.asarray(base["desired_boundary_faces"],np.int64)];st=np.sort(targ,axis=1);tdeg=np.any(st[:,1:]==st[:,:-1],axis=1)
tc=chain(targ[~tdeg])
t_tokens=Counter((k,int(v)) for k,v in tc.items())
a_tokens=Counter((k,int(v)) for k,v in cc.items())
missing=sum((t_tokens-a_tokens).values())
extra=sum((a_tokens-t_tokens).values())
require(missing==123 and extra==187,"mapped target residual mismatch")
result={"schema":"h706.d414-independent-transfer-replay.v1","status":"PASS_INDEPENDENT_D414_CELL_POINT_BOUNDARY_REPLAY","script_path":str(Path(__file__)),"script_sha256":sha(Path(__file__)),"input_candidate_sha256":BASE_SHA,"surface_candidate_sha256":SURF_SHA,"surface_report_sha256":SREPORT_SHA,"transfer_candidate_sha256":CAND_SHA,"transfer_report_sha256":REPORT_SHA,"cell_star_map_sha256":MAP_SHA,"qualified_414_sha256":V414_SHA,"indexed379_owner_sha256":OWNER_SHA,"operation_count":len(ops),"point_quotient_matches_operation_replay":True,"all_input_points_retained_bitwise":True,"retained_cell_rows_match_derived_quotient":True,"output_tetrahedra_materials_match_rowwise_replay":True,"output_boundary_matches_independent_oriented_chain":True,"mapped_input_boundary_chain_matches_output":True,"changed_cell_exact_signs_positive":True,"changed_cell_count":len(oldc),"exact_det6_delta_integer":str(exact_delta),"native_bad_cells_before":bad_before,"native_bad_cells_after":bad_after,"surviving_good_cell_regressions":regress,"protected_cell_overlap":0,"mapped_current_target_residual":{"missing":missing,"extra":extra},"mass_mapping":"NOT_RUN","native_steps":0,"physical_admission":False,"elapsed_s":time.monotonic()-start,"limitations":["This independent replay verifies the indexed D quotient on the existing 414 volume only.","Current target remains incomplete by 123 missing and 187 extra oriented faces.","The later I466 branch is not included; typed D binding/full current 354 audit, mass mapping, and native execution remain separate gates."]}
OUT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
