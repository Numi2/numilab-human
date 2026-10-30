from pathlib import Path
import importlib.util,json,time,numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
D=P/"fine-physical-volume-706";OUT=D/"microedge-full-PLC-013";OUT.mkdir()
H=D/"handoff/prepare_tetgen_handoff_697_typedD.py";spec=importlib.util.spec_from_file_location("h706plc",H);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
manifest=json.loads((D/"plan-003/manifest.json").read_text());inputs=manifest["inputs"]
reindex_path=D/"microedge-collar-012/receipt.json";reindex=json.loads(reindex_path.read_text())
if reindex["status"]!="PASS_COLLAR_AND_PROTECTED149_REINDEX_SOURCE_ONLY":raise ValueError("current collar/149 not ready")
exact_path=D/"local-microedge-condition-001/exact-gates-006/report.json"
f64_path=D/"local-microedge-condition-001/exact-f64-gates-003/report.json"
if h.sha(exact_path)!="590eb970e76be3cdcb719678215f729e85a37ec9a87d2e2cd337e2208698065c" or h.sha(f64_path)!="95a8005e4bc654b696d4ad4000f22afd8af1a9d99fbc3cffa1ffadd10c97409c":raise ValueError("current exact report pin mismatch")
exact=json.loads(exact_path.read_text());f64=json.loads(f64_path.read_text())
if exact["status"]!="PASS_CHANGED_FACES_SELF_CROSS_ALL185" or f64["status"]!="PASS_CURRENT_OI_AND_CHANGED_D_VS_ALL185_F64":raise ValueError("current exact gates failed")
cert_path=D/"microedge-csr-canonical-011/certificate.json";cert=json.loads(cert_path.read_text())
if cert["status"]!="PASS_SCALAR_CSR_SHAPE_CANONICALIZATION_GEOMETRY_EQUIVALENT_ONLY":raise ValueError("CSR encoding not ready")
for role,eq in cert["roles"].items():
 if h.sha(eq["input_path"])!=eq["input_sha256"] or h.sha(eq["output_path"])!=eq["output_sha256"]:raise ValueError("encoding certificate input changed")
 with np.load(eq["input_path"],allow_pickle=False) as za,np.load(eq["output_path"],allow_pickle=False) as zb:
  for key in ("points_world_m","faces"):
   aa=np.asarray(za[key]);bb=np.asarray(zb[key])
   if aa.dtype!=bb.dtype or aa.shape!=bb.shape or aa.tobytes()!=bb.tobytes():raise ValueError("encoding certificate changed geometry")
for role,key in [("O","outer"),("I","inner"),("D","deep")]:
 item=reindex["inputs"][role];ev=exact["candidate_readback"][role]
 eq=cert["roles"][role]
 if ev["candidate_path"]!=eq["input_path"] or ev["candidate_sha256"]!=eq["input_sha256"] or item["path"]!=eq["output_path"] or item["sha256"]!=eq["output_sha256"]:raise ValueError("crossing report/encoding proof binds different candidate")
 inputs[key]=dict(item)
inputs["baseline_canonical_d_proof"]=inputs.pop("canonical_d_proof")
inputs["collar"]=reindex["outputs"]["collar"];inputs["wall_csv"]=reindex["outputs"]["wall_csv"]
for role,key in [("I","inner_hole_core_0"),("D","deep_hole_core_0")]:
 p=D/"microedge-cores-009"/role/"cavity-core.npz";r=json.loads((p.parent/"report.json").read_text())
 if r["surface_sha256"]!=cert["roles"][role]["input_sha256"] or r["core_sha256"]!=h.sha(p):raise ValueError("core binds different surface")
 inputs[key]={"path":str(p),"sha256":h.sha(p)}
for key,p in [("csr_encoding_certificate",cert_path),("current_149_reindex",reindex_path),("current_exact_f32",exact_path),("current_exact_f64",f64_path)]:
 inputs[key]={"path":str(p),"sha256":h.sha(p)}

for k,item in inputs.items():
 if h.sha(item["path"])!=item["sha256"]:raise RuntimeError("input changed: "+k)
m=h.load_owner();m.configure_capacity_profile("full-reference-500k-nodes-2m-cells")
paths={k:Path(v["path"]) for k,v in inputs.items()};start=time.monotonic()
with np.load(paths["collar"],allow_pickle=False) as z:collar={k:np.asarray(z[k]) for k in z.files}
o=h.load_surface_217(m,paths["outer"],"O",collar);i=h.load_surface_217(m,paths["inner"],"I");d=h.load_surface_217(m,paths["deep"],"D")
walls=m.read_wall_provenance(collar,paths["wall_csv"],paths["wall_parent_cells"],paths["wall_parent_nodes"],"registered-native-current")
skin,fat,assembly=m.assemble_shells(o,i,d,collar,walls)
holes={"skin":m._core_holes([paths["inner_hole_core_0"]]),"fat":m._core_holes([paths["deep_hole_core_0"]])}
views=OUT/"source330-views";views.mkdir()
viewinfo={"nodes":h.make_217_parent_view(paths["parent_nodes"],views/"nodes.csv.gz","nodes"),"cells":h.make_217_parent_view(paths["parent_cells"],views/"cells.csv.gz","cells")}
centroids=m._parent_centroids_by_material(views/"nodes.csv.gz",views/"cells.csv.gz")
regions={};seeds={}
for name,material,shell in [("skin",15,skin),("fat",16,fat)]:
 regions[name],seeds[name]=m._region_points(shell,material,centroids)
 print(json.dumps({"prepared_material":name,"region_count":len(regions[name]),"holes":len(holes[name]),"elapsed_seconds":time.monotonic()-start}),flush=True)
a,b=m.export_shells_and_plcs(skin,fat,OUT/"paired-domains",holes["skin"],holes["fat"],regions)
r={"schema":"numi.paired-fullbody-tetgen-readiness-697.v1","status":"PASS_PREPARED_SOURCE_DOMAINS_AND_POSITIVE_CORE_SEEDS_ONLY",
 "gates":{"positive_core_seeds":"PASS"},"physical_admission":False,"runtime":0,"accepted_steps":0,"inputs":inputs,
 "owner_sha256":h.BUILD_SHA,"source330_views":viewinfo,"region_seeds":seeds,"hole_counts":{k:len(v) for k,v in holes.items()},
 "outputs":{"skin_poly":{"path":str(a),"sha256":h.sha(a)},"fat_poly":{"path":str(b),"sha256":h.sha(b)}},"wall_seconds":time.monotonic()-start,
 "limitations":["Not tetrahedral cells, mass transfer, physical admission or motion","Current conditioned volume construction has not run; older source trials remain failed evidence"]}
(OUT/"receipt.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({"status":r["status"],"region_counts":{k:len(v) for k,v in regions.items()},"seconds":r["wall_seconds"]}),flush=True)
