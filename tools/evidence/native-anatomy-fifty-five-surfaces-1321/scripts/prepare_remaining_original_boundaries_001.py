from pathlib import Path
import sys,importlib.util,json,hashlib,numpy as np,subprocess,time
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"remaining-original-boundary-batch-001";O.mkdir(exist_ok=False)
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"fifty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
template=(A/"separate_stable127_original_pinches_008.py").read_text()
summary={"pins":{str(p):sha(p) for p in [Path(__file__),H,T,A/"separate_stable127_original_pinches_008.py"]},"rows":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(summary,indent=2)+"\n")
for sid in (129,130,131,132,135,136,137,138,121,122,144,21,22):
 d=O/f"stable-{sid}";d.mkdir();r=h.row_data(t,sid);row={"stable_id":sid};summary["rows"].append(row);save()
 try:
  p=r["positions"];f=r["faces"];u,first,inv=np.unique(p,axis=0,return_index=True,return_inverse=True);used=np.unique(inv[f]);qm=np.full(len(u),-1);qm[used]=np.arange(len(used));ff=qm[inv[f]];xx=u[used];ll=r["local"][first[used]];ww=r["weights"][first[used]]
  def canon(l,w):
   a=np.zeros(r["bc"])
   for k,v in zip(l,w):
    if v>0:a[int(k)]+=v
   return a
  for j in np.unique(f):
   k=qm[inv[j]];assert np.array_equal(canon(r["local"][j],r["weights"][j]),canon(ll[k],ww[k]))
  parent=d/"original-quotient.npz";np.savez(parent,vertices6=np.column_stack([xx,h.vertex_normals(xx,ff)]).astype("<f4"),faces=ff,face_origins=np.arange(len(ff)),binding_indices=ll,weights=ww)
  row["parent"]={"path":str(parent),"sha256":sha(parent),"coordinate_and_map_quotient_only":True}
  code=template.replace('O=A/"stable-127-original-point-separation-008"',f'O=Path({str(d/"source-separation")!r})').replace('p=A/"stable-127-original-quotient-parent-004/stable-127-original-quotient.npz"',f'p=Path({str(parent)!r})')
  code=code.replace('try:part.construct_positive_winding_self_union({"vertices":exact,"triangles":f.tolist(),"source_sha256":sha(p)});raise RuntimeError("expected diagnostic rejection")','try:\n union=part.construct_positive_winding_self_union({"vertices":exact,"triangles":f.tolist(),"source_sha256":sha(p)})\n mesh=union["output_mesh"];ancestry=union["output_face_ancestry"];error=None')
  code=code.replace('for epsilon in [1e-6,5e-6,1e-5,2.5e-5,5e-5]:','for epsilon in ([0.0] if not splits else [1e-6,5e-6,1e-5,2.5e-5,5e-5]):')
  script=d/"separate_source_pinches.py";script.write_text(code);compile(code,str(script),"exec");row["script_sha256"]=sha(script);save()
  with (d/"stdout.log").open("w") as log:
   result=subprocess.run([sys.executable,str(script)],stdout=log,stderr=subprocess.STDOUT,timeout=180)
  row["returncode"]=result.returncode
  rp=d/"source-separation/report.json"
  if rp.exists():
   doc=json.loads(rp.read_text());row["report_sha256"]=sha(rp);row["source_clear_candidates"]=[{"path":r["candidate_path"],"sha256":r["candidate_sha256"],"epsilon_m":r["epsilon_m"]} for r in doc["trials"] if r.get("candidate_path")];row["split_count"]=len(doc["splits"])
 except Exception as e:row["error"]=type(e).__name__+": "+str(e)
 save();print(json.dumps(row),flush=True)
summary["complete"]=True;summary["inputs_unchanged"]=all(sha(p)==d for p,d in summary["pins"].items());save()
