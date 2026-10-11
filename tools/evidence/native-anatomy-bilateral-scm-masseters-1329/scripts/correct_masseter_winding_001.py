from pathlib import Path
import sys,json,hashlib,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");P=A/"left-scm-and-masseters-composition-002";O=A/"left-scm-masseters-positive-winding-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import passive_attachment_composition as pc
T=P/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";d=pc._read_nhtiss4(T);row=next(r for r in d["records"] if int(r[6])==160);old=pc._biceps_row_arrays(pc._row_slices(d,row));new={k:v.copy() for k,v in old.items()};new["faces"]=new["faces"][:,[0,2,1]].copy();new["vertices6"][:,3:]*=-1
def volume(p,f):
 p=p.astype(float)-p.astype(float).mean(0);tr=p[f];return float(np.einsum("ij,ij->i",tr[:,0],np.cross(tr[:,1],tr[:,2])).sum()/6)
before=volume(old["vertices6"][:,:3],old["faces"]);after=volume(new["vertices6"][:,:3],new["faces"]);assert before<0 and after>0 and before==-after
cp=O/"stable-160-outward-winding.npz";np.savez(cp,**new)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof={"scope":"Correct mirrored left superficial masseter inward winding. Reverses each face's final two indices and negates vertex normals. All positions, weights, binding indices, body transforms and triangle face identity are unchanged. This fixes sign, not source shape or physiology.","stable_id":160,"source_tissue":str(T),"source_tissue_sha256":sha(T),"patch_sha256":sha(cp),"signed_volume_source_before_m3":before,"signed_volume_source_after_m3":after,"positions_bindings_weights_exact":all(np.array_equal(old[k],new[k]) for k in ("binding_indices","weights")) and np.array_equal(old["vertices6"][:,:3],new["vertices6"][:,:3]),"triangle_vertex_sets_exact":np.array_equal(np.sort(old["faces"],1),np.sort(new["faces"],1)),"native_verification_pending":True}
rp=O/"winding-proof.json";rp.write_text(json.dumps(proof,indent=2)+"\n")
composition=pc.compose(T,O/"assets",[(160,cp,rp)],reference_surface_rows=(160,));payload=O/"assets"/T.name;bound=pc.bind_anatomy_receipt(P/"resting-anatomy-receipt.json",payload,O/"resting-anatomy-receipt.json")
child=pc._read_nhtiss4(payload)
for r,cr in zip(d["records"],child["records"]):
 if int(r[6])!=160:assert pc._row_local_equal(pc._row_slices(d,r),pc._row_slices(child,cr))
assert bound["mass_geometry_accounting"]==json.loads((P/"resting-anatomy-receipt.json").read_text())["mass_geometry_accounting"]
report={"complete":True,"payload_path":str(payload),"payload_sha256":sha(payload),"receipt_sha256":sha(O/"resting-anatomy-receipt.json"),"all153_other_rows_exact":True,"mass_accounting_exact":True,"proof":proof,"composition":composition}
(O/"composition-result.json").write_text(json.dumps(composition,indent=2)+"\n");(O/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k not in ("proof","composition")}))
