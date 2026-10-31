"""Source-pinned exact bone barriers for offline connected tissue remeshing."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def exact_common_lattice_key(point):
    denominator=1<<149
    out=[]
    for x in point:
        if not math.isfinite(float(x)):raise ValueError("nonfinite bone point")
        n,d=float(x).as_integer_ratio()
        if denominator%d:raise ValueError("bone coordinate does not fit exact common lattice")
        out.append(n*(denominator//d))
    return tuple(out)

def reidentify_records(records,*,vertex_offset,face_index_offset=0):
    if vertex_offset<0 or face_index_offset<0:raise ValueError("negative record namespace offset")
    return [(r[0],r[1],r[2],-1-face_index_offset-int(r[3]),
             tuple(vertex_offset+int(v)for v in r[4]))for r in records]

def load_bone_barrier_records(pred,*,root,vertex_offset,
                              expected_native_sha256,expected_manifest_sha256):
    started=time.monotonic();root=Path(root)
    native=root/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
    manifest=root/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
    if sha(native)!=expected_native_sha256 or sha(manifest)!=expected_manifest_sha256:
        raise ValueError("bone barrier source pins changed")
    records=[];ranges=[]
    with np.load(native,allow_pickle=False) as z:
        points=z["points_world_f32"].astype(np.float64);faces=z["faces"].astype(np.int64)
        offsets=z["bank_face_offsets"];body=z["bank_body"];bone=z["bank_bone"]
        if len(body)!=185 or len(bone)!=185 or len(offsets)!=186:raise ValueError("incomplete native bone bank inventory")
        native_raw,_=pred.float32_surface_records(points,faces,source_seam_groups=())
        records.extend(reidentify_records(native_raw,vertex_offset=vertex_offset))
        for i in range(185):
            ranges.append({"domain":"source_order_native_f32_replay","body":int(body[i]),"bone":int(bone[i]),
                           "negative_face_index_start":int(offsets[i]),"negative_face_index_end":int(offsets[i+1])})
        next_vertex=vertex_offset+len(points)
    source=json.loads(manifest.read_text())
    operands=source["operands"]
    if len(operands)!=185 or len({(int(b["body"]),int(b["bone"]))for b in operands})!=185:
        raise ValueError("incomplete pinned Float64 bone inventory")
    for bank in operands:
        path=Path(bank["operand_path"])
        if sha(path)!=bank["operand_sha256"]:raise ValueError("Float64 bone operand changed")
        with np.load(path,allow_pickle=False) as z:
            points=z["points_world_m_f64"];faces=z["faces_f64"]
            keys=tuple(exact_common_lattice_key(point)for point in points)
            raw=pred._records(keys,faces)
            start=len(records)
            records.extend(reidentify_records(raw,vertex_offset=next_vertex,face_index_offset=start))
            ranges.append({"domain":"pinned_f64_exact","body":int(bank["body"]),"bone":int(bank["bone"]),
                           "negative_face_index_start":start,"negative_face_index_end":len(records)})
            next_vertex+=len(points)
    if not records or any(r[3]>=0 for r in records):
        raise ValueError("bone barrier negative face namespace is invalid")
    receipt={"schema":"existing-bone-record-remesh-barrier.v1","native_sha256":expected_native_sha256,
             "pinned_manifest_sha256":expected_manifest_sha256,"record_count":len(records),
             "bank_coordinate_domains":370,"vertex_namespace_start":vertex_offset,"vertex_namespace_end":next_vertex,
             "negative_face_index":"global index is -1 minus face_id; query faces must have nonnegative IDs",
             "ranges":ranges,"seconds":time.monotonic()-started,
             "scope":"offline exact intersection barriers; no volume containment or motion admission"}
    return records,receipt
