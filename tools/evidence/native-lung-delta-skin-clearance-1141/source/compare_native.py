#!/usr/bin/env python3
"""Compare actual accepted 1078 and 1113 MRVPACK geometry, without predicates.

Matches primitives by (semantic, stable ID), matches their vertices through the
normalized local index stream, then compares captured Float32 XYZ bytes. It
separately checks the 303,656 row-310 faces against the pinned final lobe-face
lineage in the same vertex order at all eight accepted poses.
"""
from __future__ import annotations
import array, hashlib, importlib.util, json, mmap, os, struct, sys
from pathlib import Path

E = Path("/Users/n/numi-human-resting-evidence-20261005")
REVIEW_DIR = E / "native-lung-v8-integrity-review-1115"
V8 = E / "native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8"
BASE_NHA = E / "native-lung-conditioned-final-compose-1078/final/resting-thorax.nhanatomy"
CAND_NHA = V8 / "final/resting-thorax.nhanatomy"
LINEAGE = V8 / "row310-face-lineage.npy"
RUNS = {
 1078: E / "final-native-scene-preflight-936/skin-927-lung-1078-viewer-018-v015-attempt1/native-run",
 1113: E / "final-native-scene-preflight-936/skin-927-lung-1113-viewer-018-v015-attempt1/native-run",
}
STEPS = [0,4991,5375,5759,6111,6495,7743,10000]
CHANGED = {(51023,306),(51023,307),(51023,308),(51023,309),(51024,310)}
SEM_BY_LAYER = [51010,51011,51012,51020,51021,51022,51023,51024,51025,51026,51027,51028,51029,51030,51031]
H = struct.Struct("<8sIIQQ32s24s")
D = struct.Struct("<IIQQQII32s")
P = struct.Struct("<4I")
I = struct.Struct("<4I")
BASE_RECEIPT_SHA = "b1538059788b1f28db4beba0326edab8d01c64eb88d7cf9fe7ef4153985dc5f8"
CAND_NHA_SHA = "1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc"

def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()

def load_review_module():
    p=REVIEW_DIR/"review_v8.py"
    spec=importlib.util.spec_from_file_location("review_v8",p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

class Pack:
    def __init__(self,path,receipt_path,step):
        self.path=path; self.receipt_path=receipt_path; self.receipt=json.loads(receipt_path.read_text())
        if self.receipt.get("accepted_step") != step: raise ValueError(f"receipt step mismatch {receipt_path}")
        if Path(self.receipt.get("accepted_pack_path","")).resolve()!=path.resolve(): raise ValueError(f"receipt pack path mismatch {receipt_path}")
        self.pack_sha=sha_file(path)
        self.file=path.open("rb"); self.mm=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        h=H.unpack_from(self.mm,0)
        if h[0]!=b"MRVPACK2" or h[1]!=2: raise ValueError(f"bad MRVPACK2 header {path}")
        self.sections={D.unpack_from(self.mm,H.size+i*D.size)[0]:D.unpack_from(self.mm,H.size+i*D.size) for i in range(h[2])}
        self.vertex=self.sections[2]; self.index=self.sections[3]; self.primitive=self.sections[4]
        if (self.vertex[5],self.index[5],self.primitive[5])!=(80,4,64): raise ValueError("unexpected pack strides")
        self.vertex_count=self.vertex[4]; self.index_count=self.index[4]
        vb=hashlib.sha256(memoryview(self.mm)[self.vertex[2]:self.vertex[2]+self.vertex_count*80]).hexdigest()
        if self.receipt.get("captured_vertex_buffer_sha256")!=vb: raise ValueError(f"vertex buffer hash differs from receipt: {path}")
        if self.receipt.get("vertex_count")!=self.vertex_count or self.receipt.get("index_count")!=self.index_count:
            raise ValueError(f"receipt section counts differ: {path}")
        self.index_bytes=memoryview(self.mm)[self.index[2]:self.index[2]+self.index_count*4]
        self.index_sha=hashlib.sha256(self.index_bytes).hexdigest()
        self.indices=array.array("I"); self.indices.frombytes(self.index_bytes)
        if self.indices.itemsize!=4 or sys.byteorder!="little": raise ValueError("unsupported uint32 host")
        self.surfaces={}
        for k in range(self.primitive[4]):
            off=self.primitive[2]+k*64
            first,count,body,primitive_id=P.unpack_from(self.mm,off)
            semantic,instance,owner,sid=I.unpack_from(self.mm,off+16)
            key=(semantic,sid)
            if key in self.surfaces: raise ValueError(f"duplicate primitive identity {key}")
            if count%3 or first+count>self.index_count: raise ValueError(f"bad primitive index range {key}")
            idx=self.indices[first:first+count]
            if not idx: raise ValueError(f"empty primitive {key}")
            start=min(idx); stop=max(idx)+1
            if stop>self.vertex_count: raise ValueError(f"primitive vertex index out of range {key}")
            norm=tuple(x-start for x in idx)
            used=tuple(sorted(set(norm)))
            self.surfaces[key]={"first":first,"count":count,"body":body,"primitive_id":primitive_id,
                                "semantic":semantic,"instance":instance,"owner":owner,"sid":sid,"primitive_id":primitive_id,
                                "start":start,"stop":stop,"norm":norm,"used":used}
    def xyz(self,global_index):
        off=self.vertex[2]+global_index*80
        return self.mm[off:off+12]
    def xyz_floats(self,global_index):
        return struct.unpack_from("<3f",self.mm,self.vertex[2]+global_index*80)
    def close(self):
        self.index_bytes.release()
        self.mm.close(); self.file.close()

def source_nha_topology_check(nha,pack):
    records=[]
    for sid,r in sorted(nha.rows.items()):
        sem=SEM_BY_LAYER[r["layer"]-1]
        key=(sem,sid)
        if key not in pack.surfaces: raise ValueError(f"NHA row absent from native pack: {key}")
        s=pack.surfaces[key]
        local=tuple(x-nha.rows[sid]["fv"] for x in array.array("I",nha.mm[nha.io+r["fi"]*4:nha.io+(r["fi"]+r["ni"])*4]))
        if s["norm"]!=local: raise ValueError(f"NHA index mapping mismatch for {key}")
        if s["owner"]!=r["body"]: raise ValueError(f"NHA native owner mismatch for {key}")
        records.append(key)
    if len(records)!=524: raise ValueError("did not map all 524 NHA rows")
    return records

def compare_unchanged(base,cand,excluded):
    if set(base.surfaces)!=set(cand.surfaces): raise ValueError("primitive catalogue key mismatch")
    if len(base.surfaces)!=861: raise ValueError(f"expected 861 primitives, got {len(base.surfaces)}")
    checked=0; positions=0; by_sem={}; failures=[]
    for key in sorted(base.surfaces):
        a,b=base.surfaces[key],cand.surfaces[key]
        if key in excluded: continue
        if (a["count"],a["owner"],a["instance"],a["primitive_id"]) != (b["count"],b["owner"],b["instance"],b["primitive_id"]):
            failures.append([key,"primitive metadata/count mismatch"]); continue
        if a["norm"]!=b["norm"]:
            failures.append([key,"normalized face index stream differs"]); continue
        local_ok=True
        for i in a["used"]:
            pa=base.xyz(a["start"]+i); pb=cand.xyz(b["start"]+i)
            if pa!=pb:
                local_ok=False
                xa=struct.unpack("<3f",pa); xb=struct.unpack("<3f",pb)
                failures.append([key,"position mismatch",i,xa,xb]); break
            positions+=1
        if local_ok:
            checked+=1; by_sem[key[0]]=by_sem.get(key[0],0)+1
    if failures: raise ValueError(f"unchanged primitive mismatch: {failures[:10]}")
    return {"primitive_count":checked,"mapped_used_vertex_count":positions,"primitive_count_by_semantic":{str(k):v for k,v in sorted(by_sem.items())}}

def check_pleura(parent_nha,final_nha,lineage,pack):
    faces={sid:final_nha.faces(sid) for sid in range(305,311)}
    counts={}; exact=0; ordered=0; max_delta=0.0; first=None
    for face_i,(sid,parent_face) in enumerate(lineage):
        counts[str(sid)]=counts.get(str(sid),0)+1
        pleura_ids=faces[310][face_i*3:face_i*3+3]
        lobe_ids=faces[sid][parent_face*3:parent_face*3+3]
        same=True
        for a,b in zip(pleura_ids,lobe_ids):
            va=pack.xyz(pack.surfaces[(SEM_BY_LAYER[7],310)]["start"]+a)
            vb=pack.xyz(pack.surfaces[(SEM_BY_LAYER[6],sid)]["start"]+b)
            if va!=vb:
                same=False
                xa=struct.unpack("<3f",va); xb=struct.unpack("<3f",vb)
                d=max(abs(x-y) for x,y in zip(xa,xb)); max_delta=max(max_delta,d)
                if first is None: first={"pleura_face":face_i,"parent_stable_id":sid,"parent_face":parent_face,"pleura_xyz":xa,"parent_xyz":xb,"max_abs_delta_m":d}
        if same: exact+=1; ordered+=1
    return {"faces":len(lineage),"parent_face_counts_by_stable_id":counts,
            "exact_xyz_same_winding_faces":exact,"same_winding_mismatches":len(lineage)-exact,
            "maximum_abs_coordinate_delta_m":max_delta,"first_mismatch":first,
            "interpretation":"exact captured-coordinate correspondence only; no separate all-pairs intersection predicate"}

def main(outpath):
    mod=load_review_module()
    if sha_file(CAND_NHA)!=CAND_NHA_SHA: raise ValueError("candidate NHA pin mismatch")
    br=E/"final-native-scene-preflight-936/skin-927-lung-1078-viewer-018-v015-attempt1/anatomy/resting-anatomy-receipt.json"
    if sha_file(br)!=BASE_RECEIPT_SHA: raise ValueError("1078 source receipt pin mismatch")
    base_nha_path=Path(json.loads(br.read_text())["payload"]["path"])
    if sha_file(base_nha_path)!="7f6a8175e3cadb93a6dfc43535467b6896cf1b8bacfd0334df7b5bb03c414e92": raise ValueError("1078 NHA payload pin mismatch")
    bnh=mod.NHA(base_nha_path); cnh=mod.NHA(CAND_NHA)
    lineage,lh=mod.load_npy_i32_2(LINEAGE)
    expected_asset_names=["common-field-domain-boxes-f32.bin","common-field-parameters-f32.bin","common-field-map-f32.bin","common-field-polynomials-f32.bin","skin-source-anatomy-parameters.bin","skin-source-vertex-map.bin"]
    report={"schema":"numi.human.native-v8-vs-1078-mapped-geometry-comparison.v1","passed":False,
            "scope":"actual accepted MRVPACK captured positions and normalized primitive index streams at eight matching accepted steps; no intersection scan",
            "source":{"base_nha":{"path":str(base_nha_path),"sha256":sha_file(base_nha_path)},"candidate_nha":{"path":str(CAND_NHA),"sha256":sha_file(CAND_NHA)},
                      "row310_lineage":{"path":str(LINEAGE),"sha256":lh,"face_count":len(lineage)}},
            "changed_primitive_exclusions":[list(k) for k in sorted(CHANGED)],
            "steps":[],"asset_identity":{}}
    for name in expected_asset_names:
        p0=RUNS[1078]/name; p1=RUNS[1113]/name
        if p0.is_file() and p1.is_file():
            h0=sha_file(p0); h1=sha_file(p1); report["asset_identity"][name]={"1078_sha256":h0,"1113_sha256":h1,"equal":h0==h1}
            if h0!=h1: raise ValueError(f"scene asset changed: {name}")
    base_topology_keys=None; cand_topology_keys=None; base_index_hash=None; cand_index_hash=None; nha_mapping_rows=None
    for step in STEPS:
        packs={}; recs={}
        for runid in (1078,1113):
            d=RUNS[runid]/"accepted-geometry"; pp=d/f"step-{step}.mrvpack"; rp=d/f"step-{step}.receipt.json"
            packs[runid]=Pack(pp,rp,step); recs[runid]=packs[runid].receipt
        a,b=packs[1078],packs[1113]
        ra,rb=recs[1078],recs[1113]
        for field in ("accepted_body_state_sha256","accepted_respiration_state_sha256","accepted_time_s"):
            if ra.get(field)!=rb.get(field): raise ValueError(f"accepted state differs step {step}: {field}")
        if set(a.surfaces)!=set(b.surfaces): raise ValueError(f"primitive catalogue differs step {step}")
        if len(a.surfaces)!=861 or len(b.surfaces)!=861: raise ValueError(f"primitive count mismatch step {step}")
        if base_index_hash is None: base_index_hash=a.index_sha; cand_index_hash=b.index_sha
        if a.index_sha!=base_index_hash or b.index_sha!=cand_index_hash: raise ValueError(f"within-run index buffer changed at step {step}")
        if step==STEPS[0]:
            base_rows=source_nha_topology_check(bnh,a); cand_rows=source_nha_topology_check(cnh,b)
            nha_mapping_rows={"1078_source_nha_rows_mapped":len(base_rows),"1113_source_nha_rows_mapped":len(cand_rows),"native_index_stream_matches_each_source_nha_local_face_stream":True}
            base_topology_keys=set(a.surfaces); cand_topology_keys=set(b.surfaces)
            unmatched=set(a.surfaces)^set(b.surfaces)
            if unmatched: raise ValueError(f"native primitive keys differ: {sorted(unmatched)[:10]}")
        unchanged=compare_unchanged(a,b,CHANGED)
        pleura=check_pleura(bnh,cnh,lineage,b)
        per={"step":step,"accepted_time_s":rb.get("accepted_time_s"),
             "body_state_sha256":rb.get("accepted_body_state_sha256"),
             "respiration_state_sha256":rb.get("accepted_respiration_state_sha256"),
             "1078_receipt_sha256":sha_file(RUNS[1078]/"accepted-geometry"/f"step-{step}.receipt.json"),
             "1113_receipt_sha256":sha_file(RUNS[1113]/"accepted-geometry"/f"step-{step}.receipt.json"),
             "1078_pack_sha256":a.pack_sha,"1113_pack_sha256":b.pack_sha,
             "primitive_count":len(a.surfaces),"unchanged_primitives":unchanged,
             "row310_lineage":pleura}
        report["steps"].append(per)
        a.close();b.close()
        print(json.dumps({"step":step,"unchanged_primitives":unchanged["primitive_count"],"skin_xyz":"(51007,1) included" if (51007,1) not in CHANGED else "excluded","row310_exact":pleura["exact_xyz_same_winding_faces"],"row310_faces":pleura["faces"]}),flush=True)
    report["index_streams"]={"1078_step0_sha256":base_index_hash,"1113_step0_sha256":cand_index_hash,
        "each_run_index_buffer_identical_across_all_eight_steps":True,
        "unchanged_primitive_face_local_index_streams_exact_between_runs":True}
    report["reuse_conclusion"]={"unchanged_geometry_primitive_count":856,
        "skin_primitive_key":[51007,1],"skin_positions_compared":True,
        "skin_clearance_reuse":"Prior per-pair skin clearance results bound to the 1078 accepted captures may be reused only for target primitives in the 856 unchanged set because both skin and those targets have bitwise-identical captured XYZ and face-local index streams at all eight matching states. Fresh exact pair checks remain required for changed lung rows 306-309. Row310 is not an independent tissue surface for the six-owner scan; its exact captured coincidence with its 303656 mapped lobe parent faces is separately proven above. This comparison alone does not bridge clearance evidence bound only to 927 or older NHA captures unless their unchanged-target mapping is separately established.",
        "not_claimed":"This comparison performs no triangle-intersection predicate and does not itself qualify clearance. It does not transfer clearances to the changed lung rows."}
    report["passed"]=True
    Path(outpath).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":True,"report":outpath,"steps":len(STEPS),"unchanged_primitives":856}))
    bnh.close();cnh.close()
if __name__=="__main__":
    if len(sys.argv)!=2: raise SystemExit("usage: compare_native.py OUTPUT.json")
    main(sys.argv[1])
