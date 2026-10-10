"""Bounded source-only muscle repair using the existing exact winding owner."""
from pathlib import Path
from fractions import Fraction
import argparse, hashlib, json, signal, sys, time
import numpy as np
ROOT=Path("/Users/n/numi-human-retained-delivery-20261009")
OWNER=Path("/Users/n/numi-human-positive-winding-1279/src")
sys.path.insert(0,str(OWNER))
from numilab_human import passive_attachment_composition as pac, cardiac_cavity_partition as part, cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.physiology import canonical
T=ROOT/"source-seam-connectivity-1247/fhl-current-7b23-count-reconciliation-1258/attempt-002/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rounded(x):
    x=Fraction(x)
    a=np.float32(float(x))
    return min((a,np.nextafter(a,np.float32(-np.inf)),np.nextafter(a,np.float32(np.inf))),
               key=lambda y:(abs(Fraction.from_float(float(y))-x),int(np.asarray(y).view(np.uint32))&1))
def timeout(*_):raise TimeoutError("bounded 180 second source union")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stable-id",type=int,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--parent-patch",type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(exist_ok=False)
    start=time.monotonic()
    pins={str(p):sha(p) for p in (T,T.with_suffix(".manifest.json"),Path(__file__),Path(pac.__file__),Path(part.__file__),Path(ci.__file__),OWNER/"numilab_human/cardiac_cavity_geometry.py")}
    result={"stable_id":args.stable_id,"scope":"Explicit strictly-positive winding reference inference, excluding negative exterior folds. Same exact partition owner. Geometry preparation only. No binding lift, accepted-pose or native admission.","input_sha256":pins,"status":"started"}
    def save(): (args.output/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    save()
    try:
        assert pins[str(T)]=="6579d06d358bfcf51f964feb57425fe568c467fd361930bce46d376ec8aedaa4"
        data=pac._read_nhtiss4(T);record=next(r for r in data["records"] if int(r[6])==args.stable_id)
        row=pac._row_slices(data,record);arr=pac._biceps_row_arrays(row)
        parent=args.parent_patch;parent_hash=sha(parent);pins[str(parent)]=parent_hash
        pp=np.load(parent);arr={k:pp[k] for k in ["vertices6","binding_indices","weights","faces"]}
        result["reference_parent_npz"]={"path":str(parent),"sha256":parent_hash,"original_face_ancestry":True}
        result["scope"]+=" Input geometry is the explicitly reconstructed parent NPZ, not unchanged original source coordinates."
        man=json.loads(T.with_suffix(".manifest.json").read_text())
        member=next(r for r in man["source"]["surfaces"] if r["stable_id"]==args.stable_id)
        result.update(label=member["label"],member_id=member["member_id"],binding_table_sha256=hashlib.sha256(row["binding_bytes"]).hexdigest())
        v,inv=np.unique(arr["vertices6"][:,:3],axis=0,return_inverse=True);f=inv[arr["faces"]]
        exact=[tuple(Fraction.from_float(float(c)) for c in p) for p in v]
        before=ci._audit_pair(ci._records(exact,f.tolist()),ci._records(exact,f.tolist()),same_surface=True)
        result["source"]={"vertex_count":len(v),"face_count":len(f),"self_audit":before,"topology":analyze_topology(v.astype(float).tolist(),f.tolist())}
        result["status"]="constructing";save()
        union=part.construct_positive_winding_self_union({"vertices":exact,"triangles":f.tolist(),"source_sha256":parent_hash})
        mesh=union["output_mesh"];outv=np.asarray([[rounded(c) for c in p] for p in mesh["vertices_m"]],dtype="<f4")
        outf=np.asarray(mesh["triangles"],dtype="<u4")
        uv,iv=np.unique(outv,axis=0,return_inverse=True);qf=iv[outf]
        er=ci._records([tuple(Fraction.from_float(float(c)) for c in p) for p in uv],qf.tolist())
        audit=ci._audit_pair(er,er,same_surface=True)
        topo=analyze_topology(uv.astype(float).tolist(),qf.tolist())
        doc={"schema":"HumanPack.cardiac-cavity-exact-self-union-result.v1","status":"offline_exact_rational_candidate","stable_id":args.stable_id,"member_id":member["member_id"],"source_tissue_sha256":pins[str(T)],"source_manifest_sha256":pins[str(T.with_suffix(".manifest.json"))],"reference_parent_npz_sha256":parent_hash,"construction":union,"ancestry_by_output_face":[{**r,"candidate_source_face_origin":int(pp["face_origins"][r["source_face"]])} for r in union["output_face_ancestry"]]}
        dst=args.output/f"stable-{args.stable_id}-exact-union.json";dst.write_bytes(canonical(part.encode_rational(doc))+b"\n")
        result.update(status="source_geometry_evaluated",candidate={"vertex_count":len(outv),"face_count":len(outf),"self_audit":audit,"topology":topo,"exact_union_path":str(dst),"exact_union_sha256":sha(dst)},source_geometry_clear=(audit["count"]==0 and topo["closed_oriented_manifold_candidate"] and not topo["degenerate_face_ids"]),whole_body_admitted=False)
    except Exception as e:
        result.update(status="failed_retained",error=type(e).__name__+": "+str(e))
    finally:
        result.update(wall_seconds=time.monotonic()-start,inputs_unchanged=all(sha(p)==h for p,h in pins.items()));save()
    print(json.dumps({k:result.get(k) for k in ("stable_id","label","status","source_geometry_clear","wall_seconds","error")}),flush=True)
    return 0 if result["status"]=="source_geometry_evaluated" and result["inputs_unchanged"] else 2
if __name__=="__main__":
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    raise SystemExit(main())
