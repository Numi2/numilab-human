from pathlib import Path
import numpy as np,sys,importlib.util,json,time,hashlib
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
OWNER=Path("/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src")
BASE=OWNER.parents[1]/"incremental-exact-audit-owner-253/src"
sys.path.insert(0,str(BASE))
spec_exact=importlib.util.spec_from_file_location("numilab_human.coarse_physical_boundary_exact479",OWNER/"numilab_human/cardiac_cavity_intersections.py");exact=importlib.util.module_from_spec(spec_exact);sys.modules[spec_exact.name]=exact;spec_exact.loader.exec_module(exact)
sp=importlib.util.spec_from_file_location("gate427",P/"audit_coarse_material_bone_volume_427.py");gate=importlib.util.module_from_spec(sp);sp.loader.exec_module(gate)
candidate=P/"connected-feasibility-restoration-504/half-step-001/candidate.npz";out=P/"connected-feasibility-restoration-504/half-step-001/exact-self-505";out.mkdir(exist_ok=False)
with np.load(candidate) as z:p=z["points_reference_world_m_f64"].astype(np.float32);t=z["volume_tetrahedra"];mat=z["material_index"]
start=time.monotonic();layers={}
for material,name in [(15,"skin"),(16,"fat")]:
 f=gate.exterior(t[mat==material])
 records,_=exact.float32_surface_records(p,f,source_seam_groups=())
 result=exact._audit_pair(records,records,same_surface=True)
 layers[name]={"faces":len(f),"self_intersection_pairs":len(result["triangle_pairs"]),"audit":result}
 print(json.dumps({"layer":name,"faces":len(f),"self_intersection_pairs":len(result["triangle_pairs"]),"elapsed_s":time.monotonic()-start}),flush=True)
r={"status":"exact_coarse_material_boundary_self_audit","candidate_sha256":gate.sha(candidate),"query_owner":str(exact.__file__),"query_owner_sha256":gate.sha(exact.__file__),"coordinate_contract":"canonical Float32 metre coordinates; exact integer predicates; no geometric welding or seam exemptions","layers":layers,"elapsed_s":time.monotonic()-start,"physical_admission":False}
(out/"receipt.json").write_text(json.dumps(r,indent=2)+"\n")

sp=importlib.util.spec_from_file_location("cw505",P/"extract_contact_construction_constraints_475.py");cw=importlib.util.module_from_spec(sp);sp.loader.exec_module(cw)
manifest=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
_,coverage=cw.extract(candidate,manifest)
(out/"full-185-coverage.json").write_text(json.dumps(coverage,indent=2)+"\n")
print(json.dumps({"positive_material_bank_pairs":coverage["positive_pairs"],"unlocalized":coverage["unlocalized_containments_or_contact_rows"],"overlap_m3_summed_per_bank_not_union":sum(z["intersection_m3"] for z in coverage["banks"])}))
