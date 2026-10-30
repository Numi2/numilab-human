from pathlib import Path
import hashlib,json,subprocess,sys,importlib.util
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=P/"fine-physical-volume-706"
VAR=H/"tetgen-explicit-no-improvement-054"
OWNER=H/"native-domain-217-source-004/build_native_domain_217.py"
sys.path[:0]=[str(OWNER.parent),str(P.parent/"authoring-deps")]
spec=importlib.util.spec_from_file_location("tetgen_control_owner054",OWNER)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
m.configure_capacity_profile("full-reference-500k-nodes-2m-cells")
manifest=json.loads((H/"current-D-proof-015/plan-source004-020/manifest.json").read_text())
stock=Path(manifest["inputs"]["tetgen_binary"]["path"])
expected_stock=manifest["inputs"]["tetgen_binary"]["sha256"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(stock)!=expected_stock: raise SystemExit("stock binary hash mismatch")
compile_receipt=json.loads((VAR/"compile.json").read_text())
binary=VAR/"tetgen"
if sha(binary)!=compile_receipt["binary_sha256"]: raise SystemExit("054 binary hash mismatch")
outbase=VAR/"controls-001";outbase.mkdir(exist_ok=False)
points=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
faces=np.asarray([[0,2,1],[0,1,3],[1,2,3],[0,3,2]],np.int64)
cases=[("stock-default",stock,"-pYq1.2/10AS512"),
       ("patched-default",binary,"-pYq1.2/10AS512"),
       ("patched-no-improvement",binary,"-pYq1.2/10AS512O0/0/0"),
       ("patched-no-improvement-trailing-Q",binary,"-pYq1.2/10AS512O0/0/0Q")]
meshes={};records=[]
for name,exe,flag in cases:
    d=outbase/name;d.mkdir();poly=d/"control.poly"
    m.write_poly(poly,points,faces,np.ones(4,np.int32),[],[(np.asarray([.1,.1,.1]),15)])
    cmd=[str(exe),flag,str(poly)]
    r=subprocess.run(cmd,capture_output=True,text=True)
    (d/"stdout.log").write_text(r.stdout);(d/"stderr.log").write_text(r.stderr)
    if r.returncode or "Unknown switch" in r.stdout: raise RuntimeError(name+" failed "+r.stdout[-1000:])
    mesh=m.read_tetgen_mesh(d/"control.1.node",d/"control.1.ele",15);meshes[name]=mesh
    if name in ("stock-default","patched-default") and "Improving mesh..." not in r.stdout: raise ValueError("default improvement path changed")
    if name.startswith("patched-no-improvement") and "Improving mesh..." in r.stdout: raise ValueError("zero-improvement path unexpectedly invoked")
    records.append({"name":name,"argv":cmd,"nodes":len(mesh["points"]),"cells":len(mesh["tetrahedra"]),
                    "all_cells_positive":bool(np.all(mesh["volumes"]>0))})
for name,mesh in meshes.items():
    for key in ("node_ids","points","tetrahedra","material_indices","volumes"):
        if not np.array_equal(mesh[key],meshes["stock-default"][key]): raise ValueError(name+" differs "+key)

receipt={"status":"PASS_DEFAULT_PARITY_ZERO_OPTION_AND_PARSER_CONTROLS",
         "records":records,"all_mesh_arrays_bitwise_equal":True,
         "stock_binary_sha256":expected_stock,"patched_binary_sha256":compile_receipt["binary_sha256"],
         "degenerate_fixture":{"status":"NOT_RUN_IN_THIS_CONTROL"},
         "physical_admission":False,"runtime":0,"accepted_steps":0}
path=outbase/"control-receipt.json";path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+chr(10))
print(json.dumps(receipt,sort_keys=True))
print("receipt_sha256",sha(path))
