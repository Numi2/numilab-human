from pathlib import Path
import hashlib, json, sys
import numpy as np
H=Path(__file__).resolve().parents[3]
P=H/"fat-comp0-connected-warning-repair-412/run-036-reused484-volume-splice-488/result-004/fat-comp0-reused484-spliced-volume-proposal.npz"
P_SHA="ed7d84228d5b1d11e9fbf0c39af296c6ace54ce0add867c5cc7b527c46820a08"
OUT=Path(__file__).resolve().parents[1]/"input-comp0-replacement.npz"
RECEIPT=OUT.with_suffix(".receipt.json")
def sha(p):
 with open(p,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
sys.path.insert(0,str(H))
import importlib.util
s=importlib.util.spec_from_file_location("fb198",H/"fixed_boundary_volume_flip_198_root.py")
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert sha(P)==P_SHA
with np.load(P) as z:
 v={k:z[k] for k in z.files}
p=v["points_world_m"]
rows=v["comp0_replacement_output_cell_rows"].astype(np.int64)
t=v["comp0_replacement_output_tetrahedra_global_node_rows"].astype(np.int64)
mat=v["comp0_replacement_output_material_indices"].astype(np.int32)
assert len(rows)==len(t)==13675 and mat.shape==(len(t),) and np.all(mat==16)
assert np.array_equal(v["tetrahedra"][rows],t)
assert np.array_equal(v["material_indices"][rows],mat)
assert len(np.unique(rows))==len(rows)
d,q=m.measurements(p,t)
assert np.all(np.isfinite(d)) and np.all(d>0)
faces=t[:,m.PAT].reshape(-1,3)
sortedf=np.sort(faces,axis=1)
order=np.lexsort((sortedf[:,2],sortedf[:,1],sortedf[:,0]))
sf=sortedf[order]
starts=np.r_[0,1+np.flatnonzero(np.any(sf[1:]!=sf[:-1],axis=1))]
counts=np.diff(np.r_[starts,len(sf)])
assert np.all(counts<=2)
signs=1-2*(((faces[:,0]>faces[:,1]).astype(np.int8)+(faces[:,0]>faces[:,2]).astype(np.int8)+(faces[:,1]>faces[:,2]).astype(np.int8))%2)
summed=np.add.reduceat(signs[order],starts)
assert np.all(summed[counts==2]==0)
boundary=faces[order[starts[counts==1]]]
expected=v["comp0_refined_selected_shell_faces_global"]
def chain(fs):
 out={}
 for f in fs:
  a,b,c=map(int,f);k=tuple(sorted((a,b,c)));sgn=1 if ((a>b)+(a>c)+(b>c))%2==0 else -1
  out[k]=out.get(k,0)+sgn
 return {k:x for k,x in out.items() if x}
assert chain(boundary)==chain(expected)
np.savez_compressed(OUT,points_world_m=p,tetrahedra=t,material_indices=mat,desired_boundary_faces=boundary,source_global_cell_rows=rows)
receipt={"status":"PASS_LOCAL_COMP0_REPLACEMENT_INPUT_PREPARED","source_candidate":str(P),"source_candidate_sha256":P_SHA,"output":str(OUT),"output_sha256":sha(OUT),"points":len(p),"cells":len(t),"material":16,"exact_F32_positive":True,"oriented_boundary_faces":len(boundary),"boundary_equals_pinned_refined_shell":True,"normalized_det_failures":int(np.sum(q<1e-5)),"absolute_volume_failures":int(np.sum(d/6<=1e-18)),"source_replacement_output_rows_sha256":hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),"replacement_tet_array_sha256":hashlib.sha256(t.astype("<i8").tobytes()).hexdigest(),"coordinate_dtype":str(p.dtype),"physical_admission":False,"native_steps":0}
RECEIPT.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
