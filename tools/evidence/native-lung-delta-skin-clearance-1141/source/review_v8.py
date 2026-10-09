#!/usr/bin/env python3
"""Compact independent integrity check for frozen lung composition v8.

Uses only Python stdlib. It verifies pinned payload/receipt/manifest hashes,
source row identity/changes, and row-310 face ancestry by exact Float32 xyz
bytes. It does not perform mesh intersection predicates or native qualification.
"""
from __future__ import annotations
import ast, hashlib, json, mmap, struct, sys
from collections import Counter
from pathlib import Path

E = Path("/Users/n/numi-human-resting-evidence-20261005")
V8 = E / "native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8"
REPORT = V8 / "composition-report.json"
FINAL = V8 / "final/resting-thorax.nhanatomy"
RECEIPT = V8 / "final/resting-anatomy-receipt.json"
MANIFEST = V8 / "final/resting-anatomy-manifest.json"
LINEAGE = V8 / "row310-face-lineage.npy"
EXPECTED = {
    str(REPORT): "f2fd49d0486e20bdca5ea8638215466f4a59ff59d94f1dffa53c8caa5018b460",
    str(FINAL): "1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc",
    str(RECEIPT): "118788f2db039f805cff15c0bff7efa8c62ec68595c997564358d14044ee53e8",
    str(MANIFEST): "0b59c2eb417e339c67ed548b766bbd30b81bec5dedd9de5bea30a7a3c96ddadc",
    str(LINEAGE): "2e2c82e96faaafd9f0c6ef3245bc865adeabfcccf3ce5b3a0366c1369fc9d3ba",
}
HEADER = struct.Struct("<8s5I32s")
RECORD = struct.Struct("<8I")

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

class NHA:
    def __init__(self,path:Path):
        self.path=path; self.file=path.open("rb"); self.mm=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        h=HEADER.unpack_from(self.mm,0)
        if h[0]!=b"NHANAT1\0" or h[1]!=5: raise ValueError(f"bad NHANAT ABI {h[:2]}")
        self.abi,self.nrows,self.nv,self.ni=h[1:5]
        self.vo=HEADER.size+self.nrows*RECORD.size; self.io=self.vo+self.nv*24
        if len(self.mm)!=self.io+self.ni*4: raise ValueError("NHANAT size/header mismatch")
        self.rows={}
        for i in range(self.nrows):
            body,fv,nv,fi,ni,sid,layer,flags=RECORD.unpack_from(self.mm,HEADER.size+i*RECORD.size)
            if sid in self.rows: raise ValueError(f"duplicate stable id {sid}")
            self.rows[sid]={"body":body,"fv":fv,"nv":nv,"fi":fi,"ni":ni,"layer":layer,"flags":flags}
            if fv+nv>self.nv or fi+ni>self.ni or ni%3: raise ValueError(f"invalid row ranges {sid}")
    def vertices_raw(self,sid):
        r=self.rows[sid]; a=self.vo+r["fv"]*24; return self.mm[a:a+r["nv"]*24]
    def faces(self,sid):
        r=self.rows[sid]; return [struct.unpack_from("<I",self.mm,self.io+(r["fi"]+k)*4)[0]-r["fv"] for k in range(r["ni"])]
    def point_key(self,sid,local):
        r=self.rows[sid]; off=self.vo+(r["fv"]+local)*24
        return self.mm[off:off+12]
    def close(self): self.mm.close(); self.file.close()

def load_npy_i32_2(path:Path):
    raw=path.read_bytes()
    if raw[:6]!=b"\x93NUMPY": raise ValueError("bad NPY magic")
    major,minor=raw[6],raw[7]
    if major==1: n=struct.unpack_from("<H",raw,8)[0]; off=10+n
    elif major in (2,3): n=struct.unpack_from("<I",raw,8)[0]; off=12+n
    else: raise ValueError(f"unsupported NPY version {major}.{minor}")
    header_bytes=raw[10:off] if major==1 else raw[12:off]
    header=ast.literal_eval(header_bytes.decode("latin1"))
    if header.get("shape")!=(303656,2) or header.get("fortran_order") or header.get("descr") not in ("<i4","=i4"):
        raise ValueError(f"unexpected lineage NPY header {header}")
    if len(raw)-off!=303656*2*4: raise ValueError("lineage data byte count mismatch")
    rows=list(struct.iter_unpack("<ii",raw[off:]))
    return rows, hashlib.sha256(raw).hexdigest()

def triangle_key(nha,sid,face):
    f=nha.faces(sid); ids=f[face*3:face*3+3]
    return tuple(sorted(nha.point_key(sid,i) for i in ids))

def main(out_path):
    for p,want in EXPECTED.items():
        got=sha(Path(p))
        if got!=want: raise ValueError(f"pin mismatch {p}: {got} != {want}")
    report=json.loads(REPORT.read_text()); receipt=json.loads(RECEIPT.read_text()); manifest=json.loads(MANIFEST.read_text())
    if report["outputs"]["final_nha"]["sha256"]!=EXPECTED[str(FINAL)]: raise ValueError("composition report NHA pin mismatch")
    if report["outputs"]["final_receipt"]["sha256"]!=EXPECTED[str(RECEIPT)]: raise ValueError("composition report receipt pin mismatch")
    if receipt["payload"]["sha256"]!=EXPECTED[str(FINAL)]: raise ValueError("receipt does not bind final payload")
    if receipt["payload"]["path"]!=str(FINAL): raise ValueError("receipt payload path mismatch")
    if manifest.get("functional_bindings",{}).get("anatomy_payload_sha256")!=EXPECTED[str(FINAL)]: raise ValueError("manifest does not bind final payload")
    base_receipt_path=Path("/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/skin-927-lung-1078-viewer-018-v015-attempt1/anatomy/resting-anatomy-receipt.json")
    base_receipt=json.loads(base_receipt_path.read_text()); base=Path(base_receipt["payload"]["path"])
    if sha(base)!=base_receipt["payload"]["sha256"]: raise ValueError("1078 base payload hash mismatch")
    b=NHA(base); c=NHA(FINAL)
    if set(b.rows)!=set(c.rows) or len(c.rows)!=524: raise ValueError("stable ID set/count differs")
    changed=[]; metadata_changed=[]
    for sid in sorted(b.rows):
        rb,rc=b.rows[sid],c.rows[sid]
        if (rb["body"],rb["layer"],rb["flags"])!=(rc["body"],rc["layer"],rc["flags"]): metadata_changed.append(sid)
        same_geom=(rb["nv"]==rc["nv"] and rb["ni"]==rc["ni"] and
                   b.vertices_raw(sid)==c.vertices_raw(sid) and b.faces(sid)==c.faces(sid))
        if not same_geom: changed.append(sid)
    if metadata_changed: raise ValueError(f"metadata changed on rows {metadata_changed}")
    if changed != [306,307,308,309,310]: raise ValueError(f"unexpected changed geometry IDs {changed}")
    lineage,lh=load_npy_i32_2(LINEAGE)
    cached_faces={sid:c.faces(sid) for sid in range(305,311)}
    allowed=set(range(305,310)); parent_counts=Counter(); mismatches=[]
    for i,(sid,parent_face) in enumerate(lineage):
        if sid not in allowed or parent_face<0 or parent_face>=c.rows[sid]["ni"]//3:
            mismatches.append([i,sid,parent_face,"bad parent range"]); continue
        row310_key=tuple(sorted(c.point_key(310,x) for x in cached_faces[310][i*3:i*3+3]))
        parent_ids=cached_faces[sid][parent_face*3:parent_face*3+3]
        parent_key=tuple(sorted(c.point_key(sid,x) for x in parent_ids))
        if parent_key!=row310_key:
            mismatches.append([i,sid,parent_face,"xyz triangle mismatch"])
        parent_counts[(sid,parent_face)]+=1
    if mismatches: raise ValueError(f"row310 lineage mismatches {len(mismatches)}: {mismatches[:3]}")
    if len(lineage)!=c.rows[310]["ni"]//3: raise ValueError("row310 lineage face count mismatch")
    stage=Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-305-308-sequential-collapse-trial-1105")
    stage_pins={x:sha(stage/x) for x in ("stage-1.npz","stage-2.npz","report.json")}
    stage_report=json.loads((stage/"report.json").read_text())
    if stage_pins["stage-2.npz"]!=stage_report["final_candidate"]["sha256"]: raise ValueError("1105 stage2 pin mismatch")
    topology=report.get("final_topology")
    summary={
        "schema":"numi.human.v8-independent-integrity-review.v1",
        "passed":True,
        "scope":"source payload, receipt/manifest bindings, stable-ID geometry diff, exact row310 parent-triangle xyz replay; no intersection scan or native qualification",
        "inputs":{p:{"sha256":sha(Path(p)),"size_bytes":Path(p).stat().st_size} for p in EXPECTED},
        "base_nha":{"path":str(base),"sha256":sha(base),"surface_count":len(b.rows),"vertex_count":b.nv,"index_count":b.ni},
        "final_nha":{"path":str(FINAL),"sha256":sha(FINAL),"surface_count":len(c.rows),"vertex_count":c.nv,"index_count":c.ni},
        "source_row_review":{"stable_ids":524,"metadata_changed":metadata_changed,"geometry_changed_ids":changed,"all_other_rows_vertex6_and_local_face_order_byte_identical":True},
        "row310_lineage":{"path":str(LINEAGE),"sha256":lh,"faces":len(lineage),"parent_stable_ids":sorted({sid for sid,_ in lineage}),"unique_parent_face_keys":len(parent_counts),"face_xyz_matches":len(lineage)-len(mismatches),"mismatches":len(mismatches),"synthetic_stage_ids_in_final_lineage":False,"same_winding_order_matches":sum(1 for i,(sid,pf) in enumerate(lineage) if tuple(c.point_key(310,x) for x in cached_faces[310][i*3:i*3+3])==tuple(c.point_key(sid,x) for x in cached_faces[sid][pf*3:pf*3+3]))},
        "recorded_independent_checks_from_prior_review":{"stage_replay":"both ordered 1105 308/310 collapse stages plus pinned shared-star operation replayed from their correct stage inputs; output XYZ/faces/normals matched final arrays bitwise","runtime_area_volume":"per-lobe geometry and basal effective areas matched 1078 at Float32; aggregate diaphragm effective area 0.018687047064304352 m2; aggregate lobe volume 0.0030022754799574614 m3","topology":"305-311 closed/oriented, zero zero-area/repeated/duplicate/unused defects; row310 two components Euler -32"},
        "stage_checkpoints":{"1105":stage_pins,"1105_final_metrics":stage_report["final_candidate"]["metrics"]},
        "historical_metadata_caveat":"final receipt contains operation_308_first_cluster.parent_coverage.310.synthetic_stage_parent_count=104975 from the superseded pre-pleura intermediate; row310-face-lineage.npy independently maps every final row310 triangle to current source lobe rows 305-309 and contains no synthetic IDs. Label the receipt count historical in a metadata successor; do not mutate this frozen asset."
    }
    out=Path(out_path); out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":True,"report":str(out),"changed_geometry_ids":changed,"row310_faces":len(lineage),"row310_xyz_mismatches":len(mismatches)},sort_keys=True))
    b.close();c.close()
if __name__=="__main__":
    if len(sys.argv)!=2: raise SystemExit("usage: review_v8.py OUTPUT.json")
    main(sys.argv[1])
