import pathlib,json,numpy as np,hashlib
from scipy.optimize import linprog
h=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
rp=h/"current-native-1024-root/current-433-1044/mass-group-audit-011/zero_receiver_source_tet_nearest_011.json"
npz=h/"current-source-carrier-bridge-1046c-r5/fat-current-source-quadrature-1046c.npz"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
assert sha(rp)=="c3b83550037bd42e81611fb38d7335e0e5085badebb468f3a9751717e74ff15f"
assert sha(npz)=="33aaf568d056248788d17d8e5b4f36f0c31a4dc3b362ff52c3339170d7f9546f"
r=json.loads(rp.read_text())
with np.load(npz) as z:d={k:z[k].copy() for k in z.files}
rows=[]
for g in (19252,19253):
 p=int(np.flatnonzero(d["global_source330_tet_rows"]==g)[0]);a,b=map(int,d["parent_offsets"][p:p+2]);w=d["carrier_weights"][a:b];bar=d["carrier_barycentric"][a:b]
 rr=next(x for x in r["rows"] if x["nearest_source_tetrahedron_point"]["global_source330_parent_row"]==g)
 q=np.array(rr["nearest_source_tetrahedron_point"]["tet_barycentric"]);v=np.vstack([bar,q]);c=np.r_[np.zeros(len(w)),-1.]
 sol=linprog(c,A_eq=v.T,b_eq=np.full(4,.25),bounds=[(float(x*.5),None) for x in w]+[(0,None)],method="highs",options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
 assert sol.success
 mixed=.99*np.r_[w,0]+.01*sol.x
 assert mixed.min()>0 and np.max(np.abs(mixed@v-.25))<1e-12 and abs(mixed.sum()-1)<1e-12
 assert np.all(mixed[:-1]>=.995*w-1e-15)
 rows.append({"global_source330_parent_row":g,"target_global_child_row":rr["child_row"],"old_packet_slice":[a,b],"barycentric":v.tolist(),"old_weights":w.tolist(),"max_target_weight":float(sol.x[-1]),"lp_weights":sol.x.tolist(),"conservative_mixture_theta":.01,"mixed_weights":mixed.tolist(),"new_target_weight":float(mixed[-1]),"moment_error":float(np.max(np.abs(mixed@v-.25)))})
out=h/"current-native-1024-root/current-433-1044/mass-support-runtime-012-root/root_anchor_refinement_lp_014.json"
assert not out.exists()
receipt={"status":"PASS_CONSTRUCTIVE_MOMENT_REWEIGHT_ONLY","input_pins":[{"path":str(p),"sha256":sha(p)} for p in (rp,npz)],"rows":rows,"scope":"Two affected parent quadratures; preserves at least 99.5 percent of every old anchor mass and exact parent barycentric moment. Does not establish full mass allocation.","native_steps":0,"physical_admission":False}
out.write_text(json.dumps(receipt,indent=2)+"\n");print(str(out),sha(out))

