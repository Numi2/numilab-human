from pathlib import Path
import sys,importlib.util,json,hashlib,numpy as np
from fractions import Fraction
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pectoral-left-f32-separation-014";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_partition as part,cardiac_cavity_intersections as ci,passive_attachment_composition as pc
from numilab_human.physiology import canonical
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/"thirty-four-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,74);p=row["positions"].copy();f=row["faces"]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,T,T.with_suffix(".manifest.json"),Path(part.__file__),Path(ci.__file__),Path(pc.__file__)]}
# Exactly diagnosed contact: the low edge of source face5210 touches face5202.
tri=p[f[5210]];edgepts=tri[tri[:,2]==tri[:,2].min()];assert len(edgepts)==2
ids=np.flatnonzero(np.any(np.all(p[:,None,:]==edgepts[None,:,:],axis=2),axis=1));q=p.copy();q[ids,2]=np.nextafter(q[ids,2],np.float32(np.inf))
cp=O/"stable-74-source-contact-separation.npz";np.savez(cp,vertices6=np.column_stack([q,row["normals"]]).astype("<f4"),faces=f.astype("<u4"),weights=row["weights"],binding_indices=row["local"],face_origins=np.arange(len(f),dtype="<u4"))
report={"scope":"Unadmitted explicit reference source proposal: separate one exact point-only contact by moving the two coordinates on face5210 lower edge upward oneFloat32ULP, including all coincident seam rows. Binding maps/physical routes unchanged; originalsource retained. Intermediate input normals retained byteexact, because unused source vertices have undefined recomputed area normals; this unadmitted construction asset never enters the viewer. Any repaired boundary will regenerate normals. Positive-winding and source/posed/native gates still required.","pins":pins,"moved_source_vertex_ids":ids.tolist(),"max_source_displacement_m":float(np.linalg.norm(q.astype(float)-p,axis=1).max()),"candidate_path":str(cp),"candidate_sha256":sha(cp),"source_points_changed":True}
rp=O/"source-proposal.json";rp.write_text(json.dumps(report,indent=2)+"\n")
comp=pc.compose(T,O/"source-assets",[(74,cp,rp)],reference_surface_rows=(74,));P=O/"source-assets"/T.name
result={"source_tissue":str(P),"source_tissue_sha256":sha(P),"composition":comp,"status":"started"}
u,iv=np.unique(q,axis=0,return_inverse=True);ff=iv[f]
try:
 union=part.construct_positive_winding_self_union({"vertices":[tuple(Fraction.from_float(float(c)) for c in x) for x in u],"triangles":ff.tolist(),"source_sha256":sha(P)})
 doc={"schema":"HumanPack.cardiac-cavity-exact-self-union-result.v1","status":"offline_exact_rational_candidate","stable_id":74,"source_tissue_sha256":sha(P),"source_manifest_sha256":sha(P.with_suffix(".manifest.json")),"construction":union,"ancestry_by_output_face":[{**r,"candidate_source_face_origin":int(r["source_face"])} for r in union["output_face_ancestry"]]}
 dst=O/"stable-74-exact-union.json";dst.write_bytes(canonical(part.encode_rational(doc))+b"\n");result.update(status="source_union_constructed",path=str(dst),sha256=sha(dst))
except Exception as e:
 result.update(status="rejected",error=str(e));tb=e.__traceback__
 while tb:
  loc=tb.tb_frame.f_locals
  if tb.tb_frame.f_code.co_name=="_construct_winding_self_union" and "output_mesh" in loc:
   w={"scope":"Rejected exact boundary, not admission","output_mesh":loc["output_mesh"],"output_face_ancestry":loc["output_ancestry"],"error":str(e)}
   dst=O/"rejected-boundary-witness.json";dst.write_bytes(canonical(part.encode_rational(w))+b"\n");result["witness_path"]=str(dst)
  tb=tb.tb_next
result["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());(O/"report.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:v for k,v in result.items() if k!="composition"}))
