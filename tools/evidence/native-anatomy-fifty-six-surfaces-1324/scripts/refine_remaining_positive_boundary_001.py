from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,importlib.util,inspect,numpy as np,subprocess
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");sid=int(sys.argv[1]);D=A/f"remaining-original-boundary-batch-001/stable-{sid}";O=D/"first-boundary-intermediate-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-positive-winding-1279/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_anatomy_interface_patch import normals
H=A.parent/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ppath=D/"original-quotient.npz";pp=np.load(ppath);partition=D/"source-separation/rejected-partition.json";pd=json.loads(partition.read_text());sep=json.loads((D/"source-separation/report.json").read_text())
src=(A/"remaining-original-boundary-batch-001/stable-135/source-lift-001/parameterized_existing_attribute_calculation.py").read_text()
# Existing canonical exact barycentric attribute lift, without a geometry-admission claim.
script=O/"canonical_attribute_calculation.py";script.write_text(src);exec(compile(src,str(script),"exec"),h.__dict__)
h.OUT=O/"attribute-intermediate";h.OUT.mkdir();h.UNION_DIR=h.OUT
doc={"source_tissue_sha256":sha(h.TISS),"source_manifest_sha256":sha(h.MAN),"construction":{"output_mesh":pd["mesh"]},"ancestry_by_output_face":pd["face_ancestry"]}
(h.OUT/f"stable-{sid}-exact-union.json").write_text(json.dumps(doc,sort_keys=True)+"\n")
t=h.load_tissue(h.TISS);original=h.row_data(t,sid);row=dict(original);row.update(positions=pp["vertices6"][:,:3],faces=pp["faces"],local=pp["binding_indices"],weights=pp["weights"],vc=len(pp["vertices6"]))
geometry={"manifest_rows":{int(x["stable_id"]):x for x in json.loads(h.MAN.read_text())["source"]["surfaces"]},"analyze_topology":analyze_topology}
attrs=h.lift_partition_attributes_only(sid,t,row,doc,ci,geometry)
v=np.asarray([[float(Fraction(int(c[0],16),int(c[1],16))) for c in p] for p in pd["mesh"]["vertices_m"]]);f=np.asarray(pd["mesh"]["triangles"],int);nv=v.tolist();mapping=list(range(len(v)))
for split in sep["splits"]:
 for k,comp in enumerate(split["components"]):
  index=split["vertex"] if k==0 else len(nv)
  if k:nv.append(v[split["vertex"]].tolist());mapping.append(split["vertex"])
  nv[index]=(v[split["vertex"]]+1e-6*np.asarray(comp["direction"])).tolist()
  for fi in comp["faces"]:f[fi,f[fi]==split["vertex"]]=index
p=np.asarray(nv,np.float32);mapping=np.asarray(mapping);local=attrs["local"][mapping];weights=attrs["weights"][mapping]
outpath=O/f"stable-{sid}-unadmitted-row-patch.npz";np.savez(outpath,vertices6=np.column_stack((p,normals(p.astype(float),f))).astype(np.float32),faces=f,face_origins=pp["face_origins"][attrs["parents"]],binding_indices=local,weights=weights)
rec,dg=h.exact_rows(p,f,ci);ar=ci._audit_pair(rec,rec,same_surface=True)
pins={str(x):sha(x) for x in [Path(__file__),H,h.TISS,h.MAN,ppath,partition,D/"source-separation/report.json",script,Path(ci.__file__)]}
report={"scope":"Unadmitted intermediate after exact boundary and1micrometre point-fan separation; self failures explicitly retained. Canonical source-parent attributes and face origins lifted unchanged for a second exact-boundary operation.","stable_id":sid,"pins":pins,"candidate_path":str(outpath),"candidate_sha256":sha(outpath),"self_pairs":ar["triangle_pairs"],"degenerate_faces":dg,"partition_vertex_ancestry":mapping.tolist(),"canonical_attribute_lift":attrs["lift_rows"],"complete":True,"candidate_admitted":False}
(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
template=(D/"separate_source_pinches.py").read_text()
template=template.replace(str(D/"source-separation"),str(D/"second-boundary-001")).replace(str(ppath),str(outpath))
dst=D/"separate_second_boundary_001.py";dst.write_text(template);compile(template,str(dst),"exec")
with (D/"second-boundary-001.log").open("w") as log:r=subprocess.run([sys.executable,str(dst)],stdout=log,stderr=subprocess.STDOUT,timeout=180)
print(json.dumps({"stable_id":sid,"first_self_pairs":ar["triangle_pairs"],"second_returncode":r.returncode}),flush=True)
