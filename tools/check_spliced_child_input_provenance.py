from pathlib import Path
import argparse,csv,gzip,hashlib,importlib.util,json,subprocess,sys,time
import numpy as np
ap=argparse.ArgumentParser()
ap.add_argument("--root",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
a=ap.parse_args();P=a.root;d=Path(__file__).resolve().parent;o=a.output;o.mkdir(exist_ok=False,parents=True)
base=P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz"
geom=P/"regional-identity-splice-001/run-004/attempt-003/current006-external-only-regional-splice-before-gates.npz"
ancestry=P/"regional-identity-splice-001/run-004/attempt-004/retet-cell-parent-ancestry.csv"
cap=P/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
common=[sys.executable,str(d/"prepare_spliced_child_inputs.py"),"--base-npz",str(base),"--geometry-npz",str(geom),
"--cell-ancestry-csv",str(ancestry),"--parent-labels-csv",str(cap)+".labels.csv","--parent-nodes-csv",str(cap)+".nodes.csv","--parent-tets-csv",str(cap)+".tets.csv"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
runs=[]
def run(name,cmd,expected):
 out=o/name;out.mkdir()
 cmd=cmd+["--output-dir",str(out),"--receipt-json",str(out/"receipt.json")]
 t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True)
 (out/"stdout.txt").write_text(r.stdout);(out/"stderr.txt").write_text(r.stderr)
 check=(r.returncode==0) if expected is None else (r.returncode!=0 and expected in r.stderr and not (out/"receipt.json").exists())
 runs.append({"name":name,"argv":cmd,"exit_code":r.returncode,"expected_error":expected,"passed":check,"wall_seconds":time.monotonic()-t})
 if not check:raise RuntimeError(f"{name}: unexpected result: {r.stderr[-1000:]}")
run("identity",common+["--compare-cli-dir",str(d/"identity-control-run-005")],None)
with ancestry.open(newline="") as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
with Path(str(cap)+".labels.csv").open(newline="") as f:labels=list(csv.DictReader(f))
first=rows[0]
correct=int(first["source330_parent_tet_row"])
same=next(int(x["parent_tet_row"]) for x in labels if int(x["parent_tet_row"])!=correct and
 (x["material_index"],x["geometric_donor_body"],x["anatomical_region_id"])==
 (labels[correct]["material_index"],labels[correct]["geometric_donor_body"],labels[correct]["anatomical_region_id"]))
for name,mutate,expected in [
 ("wrong_same_label_source",{"source330_parent_tet_row":str(same)},"ancestry is not the retained current006 composition"),
 ("forged_registration_endpoint",{"registered_x_m":str(float(first["registered_x_m"])+0.001)},"does not reconstruct registered point"),
 ("forged_registration_residual",{"residual_norm_m":str(float(first["residual_norm_m"])+0.0001)},"actual registration residual does not match")]:
 changed=[dict(x) for x in rows];changed[0].update(mutate);path=o/(name+".csv")
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(changed)
 cmd=common.copy();cmd[cmd.index("--cell-ancestry-csv")+1]=str(path);run(name,cmd,expected)
 # Preserve exact failing input without a second uncompressed copy.
 with path.open("rb") as src,gzip.open(str(path)+".gz","wb") as dst:dst.write(src.read())
 path.unlink()
# A bijective node reordering must retain geometry, 149 pins and exact cut collars.
with np.load(base,allow_pickle=False) as z:b={k:z[k] for k in z.files}
with np.load(geom,allow_pickle=False) as z:g={k:z[k] for k in z.files}
n=len(g["points_current_world_m_f32"]);order=np.arange(n-1,-1,-1);old_to_new=np.argsort(order)
g["points_current_world_m_f32"]=g["points_current_world_m_f32"][order]
g["points_reference_world_m_f64"]=g["points_reference_world_m_f64"][order]
g["volume_tetrahedra"]=old_to_new[g["volume_tetrahedra"]]
# Force fallback to mapped base boundaries, not stale surface rows in the packet.
for k in list(g):
 if k.startswith("coarse_boundary_faces_"):del g[k]
side={"base_node_output_rows":old_to_new}
for k in ("points_donor_local_m_f64","parent_volume_node_row","full86_weights","target_current_world_m"):side[k]=b[k][order]
offs=[0];src=[];weights=[]
for row in order:
 aa,bb=b["refinement_node_source_offsets"][row:row+2]
 src.extend(b["refinement_node_source_rows"][aa:bb]);weights.extend(b["refinement_node_source_weights"][aa:bb]);offs.append(len(src))
side.update(refinement_node_source_offsets=np.asarray(offs,dtype=np.int64),refinement_node_source_rows=np.asarray(src,dtype=np.int64),refinement_node_source_weights=np.asarray(weights,dtype=np.float64))
gp=o/"reordered-geometry.npz";sp=o/"reordered-node-sidecars.npz";np.savez_compressed(gp,**g);np.savez_compressed(sp,**side)
cmd=common.copy();cmd[cmd.index("--geometry-npz")+1]=str(gp)
run("bijective_node_reordering",cmd+["--node-sidecars-npz",str(sp)],None)
receipt={"status":"PASS","source_sha256":sha(d/"prepare_spliced_child_inputs.py"),"test_source_sha256":sha(__file__),"cases":runs,"inputs":{"base_sha256":sha(base),"geometry_sha256":sha(geom),"ancestry_sha256":sha(ancestry)},"physical_admission":False,"native_steps":0}
(o/"report.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n");print(json.dumps(receipt,sort_keys=True))
