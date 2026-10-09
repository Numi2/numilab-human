#!/usr/bin/env python3
from __future__ import annotations
import collections, hashlib, importlib, json, math, os, sys, time
from fractions import Fraction
from pathlib import Path

OUT = Path("/Users/n/numi-human-retained-delivery-20261009/passive-muscle-self-witness-classification-1214")
IN_DIR = Path("/Users/n/numi-human-retained-delivery-20261009/passive-muscle-self-audit-1213")
REPORT_IN = IN_DIR / "report.json"
LEDGER_IN = IN_DIR / "unallowed-self-pairs.jsonl"
FRAME_IN = IN_DIR / "coordinate-frame-clarification.json"
HIST_SRC = Path("/Users/n/numi-human-common-skin-multipose-001/src")
CI_PATH = HIST_SRC / "numilab_human/cardiac_cavity_intersections.py"
MODEL_PATH = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/src/numilab_human/model.py")
MANIFEST = Path("/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json")
TISSUE = Path("/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
PACK = Path("/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/native-run/accepted-geometry/step-0.mrvpack")
RECEIPT = PACK.with_suffix(".receipt.json")
DEN = 1 << 149
PINS = {
    REPORT_IN: "58da57b10bf89bc9436bfd993425cb725ec9759d587cbad6589db54c73ff392a",
    LEDGER_IN: "4f865f756c9fe08d9930490bc07db9942aa45cbe2d062228b1ca3d2c77493a26",
    FRAME_IN: "e5de244f4fe2b986f76917f16dbfaceebbce3cf406a3d9ca41b651fe01e37aaa",
    CI_PATH: "11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb",
    MODEL_PATH: "b12b059c7de919c1f6047353497ba49d89bbd0ddcb008084bd6e223f59a02729",
    MANIFEST: "82cdd937e0f2daf0a8704fb21246353c38148527f8602d674ac865f935264dde",
    TISSUE: "b3d0381f73e05b04ed7ea1ba23eb6559073ad20aa0d9e2ad0f97a18fab3359bd",
    PACK: "68dd2aecebd5febb561738e586cb9093813aee61309ccf1c0ceae5a535751bef",
}

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()

def req(c,m):
    if not c: raise RuntimeError(m)

def add(a,b): return tuple(x+y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def affine_rank(points):
    if len(points)<=1: return 0
    p0=points[0]
    v=next((sub(p,p0) for p in points[1:] if p!=p0),None)
    if v is None: return 0
    return 1 if all(not any(cross(v,sub(p,p0))) for p in points) else 2

def dist2(a,b):
    return sum((x-y)*(x-y) for x,y in zip(a,b))

def max_diameter2(points):
    return max((dist2(a,b) for i,a in enumerate(points) for b in points[i+1:]), default=Fraction(0))

def triangle_normal(t):
    return cross(sub(t[1],t[0]),sub(t[2],t[0]))

def coplanar(first, second):
    n=triangle_normal(first)
    return all(sum(n[k]*(p[k]-first[0][k]) for k in range(3))==0 for p in second)

def polygon_area_m2(points, normal):
    # Convex hull in a dominant-axis projection; exact shoelace, then lift
    # projected area to the original triangle plane.
    axis=max(range(3),key=lambda k: abs(normal[k]))
    axes=[k for k in range(3) if k!=axis]
    pts=sorted(set((p[axes[0]],p[axes[1]]) for p in points))
    if len(pts)<3: return 0.0
    def orient(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[]
    for p in pts:
        while len(lo)>=2 and orient(lo[-2],lo[-1],p)<=0: lo.pop()
        lo.append(p)
    hi=[]
    for p in reversed(pts):
        while len(hi)>=2 and orient(hi[-2],hi[-1],p)<=0: hi.pop()
        hi.append(p)
    hull=lo[:-1]+hi[:-1]
    twice=sum(hull[i][0]*hull[(i+1)%len(hull)][1]-hull[(i+1)%len(hull)][0]*hull[i][1] for i in range(len(hull)))
    projected=abs(twice)/2
    nlen=math.sqrt(sum(float(x*x) for x in normal))
    # lattice coordinates to metres: divide by DEN on each axis.
    return float(projected) * (nlen/abs(float(normal[axis]))) / (DEN*DEN)

def classify(ci, rec):
    tris=rec["triangle_xyz_f32_m"]
    t=[tuple(ci.float32_point_lattice_key(tuple(float(c) for c in p)) for p in tri) for tri in tris]
    pts=ci.triangle_intersection_points(t[0],t[1])
    pts=sorted(set(tuple(Fraction(c) for c in p) for p in pts))
    req(bool(pts), "retained unallowed witness has no exact intersection: "+str(rec["stable_id"])+" "+str(rec["face_rows"]))
    rank=affine_rank(pts)
    is_coplanar=coplanar(t[0],t[1])
    if rank==0:
        kind="isolated_point_contact"
    elif rank==1:
        kind="coplanar_segment_contact" if is_coplanar else "transverse_segment_contact"
    else:
        req(is_coplanar, "rank-2 intersection is not coplanar")
        kind="coplanar_area_overlap"
    d2=max_diameter2(pts)
    extent_m=math.sqrt(float(d2))/DEN
    thresholds={"1um":Fraction("1e-6"),"0.1mm":Fraction("1e-4"),"1mm":Fraction("1e-3")}
    below={name:bool(d2 < (threshold*DEN)**2) for name,threshold in thresholds.items()}
    area_m2=polygon_area_m2(pts,triangle_normal(t[0])) if kind=="coplanar_area_overlap" else 0.0
    return {
        "stable_id":int(rec["stable_id"]),"member_id":rec["member_id"],"label":rec["label"],
        "coordinate_space":rec["coordinate_space"],"face_rows":[int(x) for x in rec["face_rows"]],
        "classification":kind,"intersection_point_count":len(pts),"affine_rank":rank,
        "extent_diameter_m":extent_m,"below_threshold":below,"coplanar_overlap_area_m2":area_m2,
        "exact_lattice_points":[[[str(x.numerator),str(x.denominator)] for x in p] for p in pts],
        "exact_lattice_denominator":DEN,
    }

def distribution(rows):
    vals=sorted(x["extent_diameter_m"] for x in rows)
    def quantile(p):
        if not vals: return None
        i=(len(vals)-1)*p
        lo=int(math.floor(i)); hi=int(math.ceil(i))
        return vals[lo] + (vals[hi]-vals[lo])*(i-lo)
    return {
        "count":len(vals),"min_m":vals[0] if vals else None,"p50_m":quantile(.5),
        "p95_m":quantile(.95),"max_m":vals[-1] if vals else None,
        "count_below_1um":sum(x["below_threshold"]["1um"] for x in rows),
        "count_below_0_1mm":sum(x["below_threshold"]["0.1mm"] for x in rows),
        "count_below_1mm":sum(x["below_threshold"]["1mm"] for x in rows),
    }

def main():
    req(not (OUT/"report.json").exists(), "refusing to overwrite existing report")
    for p,h in PINS.items(): req(Path(p).is_file() and sha(p)==h,f"pinned input changed: {p}")
    sys.path.insert(0,str(HIST_SRC))
    from numilab_human import cardiac_cavity_intersections as ci
    req(Path(ci.__file__).resolve()==CI_PATH.resolve(),"predicate module resolved outside pinned historical source")
    start=time.monotonic()
    rows=[json.loads(line) for line in LEDGER_IN.read_text().splitlines() if line]
    req(len(rows)==2964,f"expected 2964 retained witness rows, got {len(rows)}")
    classified=[classify(ci,r) for r in rows]
    keys={}
    for x in classified:
        key=(x["coordinate_space"],x["stable_id"],tuple(x["face_rows"]))
        req(key not in keys,f"duplicate witness identity {key}")
        keys[key]=x
    spaces=sorted({x["coordinate_space"] for x in classified})
    expected={"authored_nhtiss4_float32_owner_local","accepted_mrvpack2_step0_float32_world"}
    req(set(spaces)==expected,f"unexpected coordinate spaces: {spaces}")
    by_space={s:[x for x in classified if x["coordinate_space"]==s] for s in spaces}
    cats={s:dict(collections.Counter(x["classification"] for x in rs)) for s,rs in by_space.items()}
    per_surface={}
    manifest=json.loads(MANIFEST.read_text())
    muscle_rows=[r for r in manifest["source"]["surfaces"] if r.get("layer")=="muscle"]
    labels={int(r["stable_id"]):(r["member_id"],r["label"]) for r in muscle_rows}
    for x in classified:
        req(labels.get(x["stable_id"])==(x["member_id"],x["label"]),"witness label disagrees with pinned manifest")
        entry=per_surface.setdefault(x["stable_id"],{"stable_id":x["stable_id"],"member_id":x["member_id"],"label":x["label"]})
        s=x["coordinate_space"]
        entry.setdefault(s,[]).append(x)
    req(len(labels)==148,"expected exact 148 stable muscle IDs in witnesses")
    src_key={(x["stable_id"],tuple(x["face_rows"])):x for x in by_space["authored_nhtiss4_float32_owner_local"]}
    pose_key={(x["stable_id"],tuple(x["face_rows"])):x for x in by_space["accepted_mrvpack2_step0_float32_world"]}
    common=set(src_key)&set(pose_key)
    per_surface_delta=[]
    for sid in sorted(labels):
        src=[x for x in by_space["authored_nhtiss4_float32_owner_local"] if x["stable_id"]==sid]
        pos=[x for x in by_space["accepted_mrvpack2_step0_float32_world"] if x["stable_id"]==sid]
        sk={(x["stable_id"],tuple(x["face_rows"])) for x in src}
        pk={(x["stable_id"],tuple(x["face_rows"])) for x in pos}
        retained=sk&pk
        per_surface_delta.append({
            "stable_id":sid,"member_id":labels[sid][0],"label":labels[sid][1],
            "source_witness_count":len(sk),"accepted_step0_witness_count":len(pk),
            "retained_pair_count":len(retained),"newly_posed_pair_count":len(pk-sk),
            "source_only_pair_count":len(sk-pk),
            "newly_posed_face_pairs":[list(k[1]) for k in sorted(pk-sk)],
            "source_only_face_pairs":[list(k[1]) for k in sorted(sk-pk)],
            "retained_classification_changes":[
                {"face_rows":list(k[1]),"source":src_key[k]["classification"],"accepted_step0":pose_key[k]["classification"],
                 "source_extent_m":src_key[k]["extent_diameter_m"],"accepted_step0_extent_m":pose_key[k]["extent_diameter_m"]}
                for k in sorted(retained)
                if src_key[k]["classification"]!=pose_key[k]["classification"]
            ],
        })
    top={}
    for space,rs in by_space.items():
        top[space]={
            "by_witness_count":[{"stable_id":sid,"member_id":labels[sid][0],"label":labels[sid][1],
               "count":sum(x["stable_id"]==sid for x in rs)} for sid in sorted(labels,key=lambda s:(-sum(x["stable_id"]==s for x in rs),s))[:12]],
            "by_max_extent":[{"stable_id":sid,"member_id":labels[sid][0],"label":labels[sid][1],
               "max_extent_m":max((x["extent_diameter_m"] for x in rs if x["stable_id"]==sid),default=0.0),
               "count":sum(x["stable_id"]==sid for x in rs)} for sid in sorted(labels,key=lambda s:(-max((x["extent_diameter_m"] for x in rs if x["stable_id"]==s),default=0.0),s))[:12]],
        }
    report={
        "schema":"numi.human.passive-muscle-self-witness-classification.v1",
        "status":"complete",
        "scope":"Reclassified only the 2,964 face-pair witness records retained by audit 1213. Recomputed each pair with the pinned exact Float32 lattice triangle_intersection_points predicate. No full mesh scan, native execution, GPU work, geometry edits, or added tolerance.",
        "inputs":{"1213_report":{"path":str(REPORT_IN),"sha256":sha(REPORT_IN)},
                  "1213_pair_ledger":{"path":str(LEDGER_IN),"sha256":sha(LEDGER_IN),"rows":len(rows)},
                  "1213_coordinate_frame_clarification":{"path":str(FRAME_IN),"sha256":sha(FRAME_IN)},
                  "nhtiss_payload":{"path":str(TISSUE),"sha256":sha(TISSUE)},
                  "nhtiss_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST)},
                  "accepted_step0_pack":{"path":str(PACK),"sha256":sha(PACK),"receipt_path":str(RECEIPT),"receipt_sha256":sha(RECEIPT)},
                  "exact_intersection_owner":{"path":str(CI_PATH),"sha256":sha(CI_PATH)},
                  "source_unit_conversion_owner":{"path":str(MODEL_PATH),"sha256":sha(MODEL_PATH)}},
        "coordinate_interpretation":{
            "stored_numeric_unit":"metres",
            "source":"NHTISS4 Float32 xyz values are consumed as stored; the model compiler converts raw BodyParts3D source millimetres to stored_vertices_m before Float32 packing (model.py 11704, 11733). The prior clarification says this audit does not establish the payload coordinate-frame tag; no transform was applied.",
            "accepted_step0":"Actual accepted MRVPACK2 Float32 world-coordinate values; treated as metres.",
            "not_claimed":"The source mesh is not a person-specific measurement; it is BodyParts3D-derived geometry stored in metres. Source-vs-pose comparisons are by stable ID and face row, and compare extents (translation/rotation invariant), not absolute positions.",
            "extent_definition":"Maximum Euclidean distance among exact intersection-polygon vertices for each pair; isolated point extent is zero. Exact classification and threshold counts use rational lattice arithmetic; distribution lengths are meter values converted from the common Float32 lattice."
        },
        "coordinate_spaces":{s:{"count":len(by_space[s]),"classification_counts":cats[s],"extent_distribution":distribution(by_space[s])} for s in spaces},
        "source_to_accepted_step0":{
            "identity_key":"(stable_id, face_rows)",
            "retained_pair_count":len(common),
            "newly_posed_pair_count":len(set(pose_key)-set(src_key)),
            "source_only_pair_count":len(set(src_key)-set(pose_key)),
            "class_changed_retained_pair_count":sum(src_key[k]["classification"]!=pose_key[k]["classification"] for k in common),
            "extent_changed_retained_pair_count":sum(src_key[k]["extent_diameter_m"]!=pose_key[k]["extent_diameter_m"] for k in common),
            "per_148_surface":per_surface_delta,
            "classification":"Only 1213 authored-source and accepted step-0 pose records exist; no later native poses are represented by this ledger."
        },
        "top_contributors":top,
        "method":{"predicate":"Pinned cardiac_cavity_intersections.triangle_intersection_points on integer 2**149 Float32 lattice keys; affine rank gives point/segment/area type; no approximate intersections.","classes":["isolated_point_contact","coplanar_segment_contact","transverse_segment_contact","coplanar_area_overlap"],"quantiles":"linear interpolation at p50 and p95 over pairwise maximum extents","thresholds":"strictly below each named threshold; exact squared-rational comparison"},
        "outputs":{"classified_pair_ledger":{"path":str(OUT/"classified-pairs.jsonl"),"rows":len(classified)}},
        "timing":{"wall_seconds":time.monotonic()-start},
        "qualification":"Diagnostic classification of retained self-witnesses only. It does not revalidate omitted pairs, classify biological significance, or establish that contacts are harmless."
    }
    (OUT/"classified-pairs.jsonl").write_text("".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in classified))
    (OUT/"per-surface-deltas.json").write_text(json.dumps(per_surface_delta,sort_keys=True,indent=2)+"\n")
    (OUT/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")

if __name__=="__main__": main()
