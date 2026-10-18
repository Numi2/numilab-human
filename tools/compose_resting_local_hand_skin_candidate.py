"""Compose a scoped local hand skin candidate into the existing anatomy receipt.

This is offline candidate preparation, not anatomical admission. No physical
state, masses, force owners, tissue bindings or NHA rows are edited.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from numilab_human import skin_source_payload_preflight as sp,resting_scene as rs
from numilab_human.resting_skin_boundaries import boundary_loops
from numilab_human.composed_bone_body_frame import _preserve_receipt_common_field_sidecars

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_local_rows(before,after,changed):
    a=sp.decode_payload(before);b=sp.decode_payload(after)
    if a["vertex_count"]!=b["vertex_count"] or a["binding_count"]!=86 or b["binding_count"]!=86:
        raise ValueError("local repair must preserve allocated vertices and 86 atlas owners")
    be=60+36*86
    if before[:20]!=after[:20] or before[24:be]!=after[24:be]:
        raise ValueError("source identity or common atlas records changed")
    changed=np.asarray(changed,dtype=np.int64)
    if len(np.unique(changed))!=len(changed) or changed.min()<0 or changed.max()>=54949:
        raise ValueError("local repair must declare distinct original source hand slots")
    fixed=np.setdiff1d(np.arange(a["vertex_count"]),changed)
    if not np.array_equal(a["vertices_u"][fixed],b["vertices_u"][fixed]) or not np.array_equal(a["full_weights"][fixed],b["full_weights"][fixed]):
        raise ValueError("an unrelated source vertex or weight changed")
    if not np.isfinite(b["vertices_f"][:,:6]).all() or not np.isfinite(b["full_weights"]).all() or (b["full_weights"]<0).any() or np.max(np.abs(b["full_weights"].sum(1)-1))>1e-5:
        raise ValueError("invalid candidate coordinates or weight partition")
    if b["indices"].max()>=b["vertex_count"]:raise ValueError("invalid candidate triangle index")
    f=b["indices"].reshape(-1,3);af=a["indices"].reshape(-1,3)
    oldloops,oldedges=boundary_loops(af);loops,edges=boundary_loops(f)
    edge_set=lambda loops:{tuple(sorted((u,v))) for ring in loops for u,v in zip(ring,ring[1:]+ring[:1])}
    if edge_set(loops)!=edge_set(oldloops):raise ValueError("local repair changes retained anatomical apertures")
    return a,b,dict(boundary_edges=edges,boundary_loops=len(loops),nonmanifold_edges=0,orientation_errors=0,unchanged_outside_vertex_count=len(fixed),changed_source_vertex_count=len(changed))

def compose(base_receipt,skin_path,preparation,original_source_npz,rigid_path,output):
    raw=Path(base_receipt).read_bytes();rec=json.loads(raw)
    old=Path(rec["mass_geometry_accounting"]["skin_payload_path"])
    if sha(old)!=rec["mass_geometry_accounting"]["skin_payload_sha256"]:raise ValueError("stale base skin")
    report=json.loads(Path(preparation).read_text())
    if report["output_payload_sha256"]!=sha(skin_path):raise ValueError("stale preparation")
    changed=np.array(report["changed_source_vertex_ids"],dtype=int)
    a,b,audit=validate_local_rows(old.read_bytes(),Path(skin_path).read_bytes(),changed)
    original=np.load(original_source_npz);p=original["source"]
    if p.shape!=(54949,3):raise ValueError("unexpected original outer-sheet dimensions")
    # The input is also pinned in the composition report; this candidate is not
    # promoted to source authentication or whole-anatomy admission.
    local=p[changed]
    if not ((np.abs(local[:,0])>.19)&(np.abs(local[:,0])<.35)&(local[:,2]>.63)&(local[:,2]<.901)).all():
        raise ValueError("edit escapes the bilateral source hand/wrist region")
    aset={tuple(x) for x in a["indices"].reshape(-1,3)}
    bset={tuple(x) for x in b["indices"].reshape(-1,3)}
    sourcefaces={tuple(x) for x in original["faces"]}
    restored=bset-aset;removed=aset-bset
    if not restored<=sourcefaces:raise ValueError("restored triangle lacks exact original source ancestry")
    handslots=set(map(int,changed))
    if any(not(any(int(v)>=54949 or int(v) in handslots for v in face)) for face in removed):
        raise ValueError("triangle removal escapes hand source or inferred envelope slots")
    rigid=rs.load_rigid(Path(rigid_path));skin=rs.load_skin(Path(skin_path),rigid)
    world,_,_=rs.source_shell_world(rigid,skin);f=b["indices"].reshape(-1,3);tri=world[f]
    integral=float(np.sum(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])))/6)
    output=Path(output).resolve();output.mkdir(exist_ok=False,parents=True)
    result=copy.deepcopy(rec)
    sidecars=_preserve_receipt_common_field_sidecars(result,Path(base_receipt).parent,output)
    accounting=result["mass_geometry_accounting"]
    audit.update(restored_original_triangles=len(restored),removed_inferred_or_modified_hand_triangles=len(removed))
    previous_accounting=copy.deepcopy(accounting)
    accounting.update(skin_payload_path=str(Path(skin_path).resolve()),skin_payload_sha256=sha(skin_path),open_surface_signed_integral_m3_diagnostic_only=integral,skin_boundary_edge_count=audit["boundary_edges"],skin_boundary_loop_count=audit["boundary_loops"],skin_nonmanifold_edge_count=0,skin_orientation_error_count=0,skin_self_intersection="not established by local receipt composition",closed_skin_volume_m3=None,apparent_density_kg_m3=None)
    provenance=result["provenance"]
    if "skin_visual_binding_candidate" in provenance:
        provenance["historical_skin_visual_binding_candidate"]=provenance.pop("skin_visual_binding_candidate")
    proof=dict(source_receipt_path=str(Path(base_receipt).resolve()),source_receipt_sha256=hashlib.sha256(raw).hexdigest(),source_skin_path=str(old.resolve()),source_skin_sha256=sha(old),output_skin_path=str(Path(skin_path).resolve()),output_skin_sha256=sha(skin_path),preparation_path=str(Path(preparation).resolve()),preparation_sha256=sha(preparation),original_source_npz_path=str(Path(original_source_npz).resolve()),original_source_npz_sha256=sha(original_source_npz),previous_mass_geometry_accounting=previous_accounting,scope="Bilateral passive hand skin registration and local source topology restoration; existing rigid-digit runtime.",audit=audit,common_field_sidecars=sidecars,physical_mass_and_forces_unchanged=True,native_admission=False,anatomical_admission=False)
    provenance["local_hand_skin_candidate"]=proof
    receipt=output/"resting-anatomy-receipt.json";receipt.write_text(json.dumps(result,indent=2)+"\n")
    proof["output_receipt_path"]=str(receipt);proof["output_receipt_sha256"]=sha(receipt)
    (output/"rebind-report.json").write_text(json.dumps(proof,indent=2)+"\n")
    return proof

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["base-receipt","skin","preparation","original-source-npz","rigid","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();r=compose(a.base_receipt,a.skin,a.preparation,a.original_source_npz,a.rigid,a.output)
    print(json.dumps(r["audit"]))
