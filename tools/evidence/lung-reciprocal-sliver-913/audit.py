from pathlib import Path
from fractions import Fraction
from collections import Counter, defaultdict
import json, sys
import numpy as np

REPO=Path("/Users/n/numi-human-common-skin-multipose-001")
sys.path.insert(0,str(REPO/"src"))
from numilab_human import resting_lung_edge_repair as repair
from numilab_human import cardiac_cavity_intersections as exact

BASE=Path("/Users/n/numi-human-resting-evidence-20261005/airway-sibling-overlap-partition-001/resting-thorax.nhanatomy")
OUT=Path("/Users/n/numi-human-resting-evidence-20261005/lung-307-309-shared-edge-changed-star-audit-913")
EXPECTED="c6adae6522a2f7e5a8bd661035d464d686c843e96ab04ffc3ccd9aa1371569c1"
if repair.sha256(BASE)!=EXPECTED or EXPECTED!=repair.PRECISION_PAYLOAD_SHA256:
    raise RuntimeError("pinned source hash mismatch")
header, source_rows=repair.parse_payload(BASE)
# Reproduce only the source-owner stages immediately preceding the targeted operation.
rows1,flip1=repair._condition_precision_lobes(source_rows,minimum_altitude_m=1e-6,selection_altitude_m=1e-6)
rows2,flip2=repair._condition_precision_lobes(rows1,minimum_altitude_m=512e-9,selection_altitude_m=1e-6)
after,operation,ancestry=repair._collapse_reciprocal_shared_edge(
    rows2,owner_pair=(307,309),endpoint_coordinates_m=repair.SHARED_SLIVER_307_309_ENDPOINTS_M,
    max_endpoint_displacement_m=1e-6,max_abs_volume_delta_m3=1e-12)

# Exact binary32 coordinates are promoted to binary64 exactly and then represented as rationals.
def records(row):
    xyz=np.asarray(row["vertices6"],dtype=np.float32)[:,:3]
    exact_xyz=[tuple(Fraction.from_float(float(c)) for c in p) for p in xyz]
    return exact._records(exact_xyz,np.asarray(row["faces"],dtype=np.int64).tolist())

def fkey(rec): return tuple(sorted(rec[0]))
def edgekey(a,b): return tuple(sorted((a,b)))
def on_segment(p,e):
    a,b=e
    return exact._cross(exact._sub(p,a),exact._sub(b,a))==(0,0,0) and all(min(a[k],b[k])<=p[k]<=max(a[k],b[k]) for k in range(3))

def reciprocal_support(rec_a,rec_b):
    by_a=defaultdict(list); by_b=defaultdict(list)
    for r in rec_a: by_a[fkey(r)].append(r)
    for r in rec_b: by_b[fkey(r)].append(r)
    shared=set(by_a)&set(by_b); edges=set(); vertices=set()
    for face in shared:
        vertices.update(face)
        for i in range(3): edges.add(edgekey(face[i],face[(i+1)%3]))
    return shared,edges,vertices

def classify_cross(points,a,b,support):
    faces,edges,vertices=support
    if fkey(a)==fkey(b) and fkey(a) in faces: return True,"reciprocal_face"
    if any(all(p==v for p in points) for v in vertices): return True,"shared_patch_vertex"
    for e in edges:
        if all(on_segment(p,e) for p in points): return True,"shared_patch_edge"
    return False,None

def scan(state, ancestor, changed, owners):
    recs={sid:records(state[sid]) for sid in owners}
    hits=[]; candidate_counts=Counter()
    for i,a in enumerate(owners):
        for b in owners[i:]:
            sides=[a] if a==b and changed.get(a) else []
            if a!=b:
                if changed.get(a): sides.append(a)
                if changed.get(b): sides.append(b)
            if not sides: continue
            support=None if a==b else reciprocal_support(recs[a],recs[b])
            for changed_side in sides:
                other=b if changed_side==a else a
                first=[r for r in recs[changed_side] if r[3] in changed[changed_side]]
                second=recs[other]
                for ra,rb in exact._aabb_candidate_pairs(first,second,same_surface=False):
                    ia,ib=int(ra[3]),int(rb[3])
                    if a==b and ia==ib: continue
                    candidate_counts[(a,b)]+=1
                    pts=exact.triangle_intersection_points(ra[0],rb[0])
                    if not pts: continue
                    if a==b:
                        common_ids=set(ra[4])&set(rb[4])
                        common={ra[0][ra[4].index(v)] for v in common_ids}
                        allowed=(len(common_ids) in (1,2) and all(exact._allowed_shared_point(p,common) for p in pts))
                        category="topological_adjacent_contact" if allowed else None
                        owner_ra=owner_rb=a
                    else:
                        allowed,category=classify_cross(pts,ra,rb,support)
                        owner_ra,owner_rb=changed_side,other
                    face_ra=ancestor[owner_ra][ia][0]
                    face_rb=ancestor[owner_rb][ib][0]
                    if owner_ra>owner_rb:
                        owner_ra,owner_rb=owner_rb,owner_ra
                        face_ra,face_rb=face_rb,face_ra
                    if owner_ra==owner_rb and face_ra>face_rb: face_ra,face_rb=face_rb,face_ra
                    key=(owner_ra,int(face_ra),owner_rb,int(face_rb))
                    hits.append({"key":key,"allowed":bool(allowed),"classification":category,
                                 "point_count":len(pts),"points_exact_m_as_float":[[float(x) for x in p] for p in pts]})
    # Two changed sides can discover the same owner-pair witness. Keep nonallowed if disagreement.
    unique={}
    for h in hits:
        key=h["key"]
        if key not in unique or (unique[key]["allowed"] and not h["allowed"]): unique[key]=h
    return list(unique.values()),dict(candidate_counts)

owners=tuple(sid for sid in (*range(305,310),311) if sid in rows2)
identity={sid:[[i] for i in range(len(rows2[sid]["faces"]))] for sid in owners}
post_ancestry={sid:(ancestry[sid] if sid in ancestry else [[i] for i in range(len(after[sid]["faces"]))]) for sid in owners}
before_changed={sid:set(map(int,operation["owner_operations"][str(sid)]["source_face_indices_in_changed_one_ring"])) for sid in (307,309)}
after_changed={sid:set(int(x["output_face_index"]) for x in operation["owner_operations"][str(sid)]["surviving_changed_face_lineage"]) for sid in (307,309)}
before_hits,before_candidates=scan(rows2,identity,before_changed,owners)
after_hits,after_candidates=scan(after,post_ancestry,after_changed,owners)
before_bad={tuple(h["key"]) for h in before_hits if not h["allowed"]}
after_bad={tuple(h["key"]) for h in after_hits if not h["allowed"]}
new=sorted(after_bad-before_bad); removed=sorted(before_bad-after_bad)

def summary(hits,candidates):
    classes=Counter(h["classification"] or "nonallowed_exact_intersection" for h in hits)
    return {"affected_star_aabb_candidates":sum(candidates.values()),"intersecting_face_pairs":len(hits),
            "nonallowed_pairs":sum(not h["allowed"] for h in hits),"allowed_declared_contacts":sum(h["allowed"] for h in hits),
            "classes":dict(sorted(classes.items()))}

report={
 "schema":"numi.human.resting-lung-307-309-changed-star-audit.v1",
 "input":{"path":str(BASE),"sha256":repair.sha256(BASE)},
 "source_pins":{"repair_module_path":str(REPO/"src/numilab_human/resting_lung_edge_repair.py"),"repair_module_sha256":repair.sha256(REPO/"src/numilab_human/resting_lung_edge_repair.py"),"exact_predicate_path":str(REPO/"src/numilab_human/cardiac_cavity_intersections.py"),"exact_predicate_sha256":repair.sha256(REPO/"src/numilab_human/cardiac_cavity_intersections.py")},
 "scope":"Exact one-ring of the reciprocal 307/309 edge collapse, rescanned against all faces of source owners 305-309 and 311 immediately before and after the operation. Unchanged pairs outside the changed star retain identical geometry at this stage.",
 "predicate":"Existing exact AABB tree and exact triangle-intersection predicate over Fraction.from_float values promoted from packed Float32 metre coordinates; no tolerance.",
 "operation":operation,
 "changed_before_face_count_by_owner":{str(k):len(v) for k,v in before_changed.items()},
 "changed_after_face_count_by_owner":{str(k):len(v) for k,v in after_changed.items()},
 "before":summary(before_hits,before_candidates),"after":summary(after_hits,after_candidates),
 "before_nonallowed_pair_keys":[list(x) for x in sorted(before_bad)],
 "after_nonallowed_pair_keys":[list(x) for x in sorted(after_bad)],
 "new_nonallowed_pair_keys":[list(x) for x in new],
 "removed_nonallowed_pair_keys":[list(x) for x in removed],
 "new_nonallowed_witnesses":[h for h in after_hits if tuple(h["key"]) in set(new)],
 "gate":"pass" if not new else "fail",
 "limitations":["Source-stage local test only; not native transformed-pose clearance.","Conforming reciprocal patch face/edge/vertex contacts are counted and classified, not globally exempted.","Whole owner-pair counts need not be zero where exact declared shared patches are present."]
}
path=OUT/"report.json"
path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(path),"sha256":repair.sha256(path),"before":report["before"],"after":report["after"],"new":new,"removed":removed,"gate":report["gate"]},indent=2))
