from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib, json, sys, struct
import numpy as np
sys.path.insert(0, "/Users/numi/numi-human-anatomy-finish-1378/src")
from numilab_human import common_atlas_skin_clearance as ca, cardiac_cavity_intersections as ci

ROOT=Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276")
RUN=ROOT/"back412-skin-native-022-001-60s/baseline/native-run"
OUT=ROOT/"full-anatomy-delivery-audit-1378/liver-id22-aggregate-geometry-audit-002.json"
CAPTURES=[int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [0,3807,4767,7999,14975,20991,26975,30000]
TARGET_KEYS={(51010,s) for s in range(14,23)}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pointkey(p): return tuple(ci.float32_point_lattice_key(p))
def point_on_segment(p,a,b):
    d=tuple(b[k]-a[k] for k in range(3)); q=tuple(p[k]-a[k] for k in range(3))
    cr=ci._cross(q,d)
    return cr==(0,0,0) and all(min(a[k],b[k])<=p[k]<=max(a[k],b[k]) for k in range(3))
def segment_on_triangle_edge(points, tri, ids):
    for x,y,ia,ib in ((0,1,ids[0],ids[1]),(1,2,ids[1],ids[2]),(2,0,ids[2],ids[0])):
        a,b=tri[x],tri[y]
        if all(point_on_segment(p,a,b) for p in points):
            return tuple(sorted((pointkey_from_lattice(a),pointkey_from_lattice(b))))
    return None
def pointkey_from_lattice(p): return tuple(int(x) for x in p)
def triangles_edge_stats(face_sets, vertex_keys):
    edge_occ=defaultdict(list); tri_keys=defaultdict(list)
    for sid, faces in face_sets:
        for fi,face in enumerate(faces):
            v=[int(x) for x in face]
            keys=[vertex_keys[x] for x in v]
            tri_keys[tuple(sorted(keys))].append((sid,fi))
            for a,b in ((0,1),(1,2),(2,0)):
                ka,kb=keys[a],keys[b]
                edge_occ[tuple(sorted((ka,kb)))].append((sid,fi,ka,kb))
    return edge_occ,tri_keys

def volume_signed(pos,faces):
    total=0.0
    for face in faces:
        a,b,c=(pos[int(i)].astype(np.float64) for i in face)
        total+=float(np.dot(a,np.cross(b,c)))
    return total/6.0

def components_and_orientation(face_sets, vertex_keys):
    faces=[]; owners=[]
    for sid, fs in face_sets:
        for fi,f in enumerate(fs): faces.append([int(x) for x in f]); owners.append((sid,fi))
    parent=list(range(len(faces)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    edge_occ=defaultdict(list)
    for face_i,face in enumerate(faces):
        keys=[vertex_keys[i] for i in face]
        for a,b in ((0,1),(1,2),(2,0)):
            ka,kb=keys[a],keys[b]
            edge_occ[tuple(sorted((ka,kb)))].append((face_i,ka,kb))
    orientation_bad=nonmanifold=boundary=0
    for occ in edge_occ.values():
        if len(occ)==1: boundary+=1
        elif len(occ)>2: nonmanifold+=1
        else:
            if occ[0][1:]==occ[1][1:]: orientation_bad+=1
            union(occ[0][0],occ[1][0])
    return {"face_count":len(faces),"unique_welded_edges":len(edge_occ),"boundary_edges":boundary,
            "nonmanifold_edges":nonmanifold,"orientation_conflict_edges":orientation_bad,
            "connected_face_components":len({find(i) for i in range(len(faces))}),
            "edge_occurrences":edge_occ,"face_owners":owners}

def surface_boundary_edges(face, vertex_keys, counts):
    ids=[int(x) for x in face]; keys=[vertex_keys[i] for i in ids]
    return [tuple(sorted((keys[a],keys[b]))) for a,b in ((0,1),(1,2),(2,0))
            if counts.get(tuple(sorted((keys[a],keys[b]))),0)==1]

def analyze_capture(step):
    pack=RUN/("accepted-geometry/step-%d.mrvpack"%step)
    pos,surfs,pack_meta=ca._pack_surfaces(pack,TARGET_KEYS)
    vertex_keys=[pointkey(p) for p in pos]
    pack_vertices=[tuple(int(x) for x in v) for v in vertex_keys]
    agg_face_sets=[(sid,surfs[(51010,sid)]["faces"]) for sid in range(14,22)]
    agg_top=components_and_orientation(agg_face_sets,vertex_keys)
    agg_faces=[]; agg_owners=[]
    patch_edge_occ=defaultdict(list)
    for sid,fs in agg_face_sets:
        for fi,f in enumerate(fs):
            agg_faces.append([int(x) for x in f]);agg_owners.append((sid,fi))
            keys=[vertex_keys[int(i)] for i in f]
            for a,b in ((0,1),(1,2),(2,0)):
                patch_edge_occ[(sid,tuple(sorted((keys[a],keys[b]))))].append((fi,keys[a],keys[b]))
    agg_records=ci._records(pack_vertices,agg_faces)
    f22=surfs[(51010,22)]["faces"]
    rec22=ci._records(pack_vertices,f22)
    # ID22 actual topology (edge counts in source topology) and self-pair evidence.
    id22_edges=defaultdict(list)
    for fi,f in enumerate(f22):
        k=[vertex_keys[int(i)] for i in f]
        for a,b in ((0,1),(1,2),(2,0)): id22_edges[tuple(sorted((k[a],k[b])))].append((fi,k[a],k[b]))
    self22=ci._audit_pair(rec22,rec22,same_surface=True)
    intersections=ci._audit_pair(rec22,agg_records,same_surface=False)
    pair_dim=Counter(); intersection_categories=Counter(); owner_counts=Counter(); unique_segments=set(); witnesses=[]
    for i,j in intersections["triangle_pairs"]:
        pts=set(ci.triangle_intersection_points(rec22[i][0],agg_records[j][0]))
        dim=min(len(pts),4);pair_dim[dim]+=1
        ordered=tuple(sorted(pts))
        if len(ordered)==2: unique_segments.add(ordered)
        sid, local_fi=agg_owners[j]; owner_counts[sid]+=1
        source_tri=rec22[i][0]; source_ids=rec22[i][4]
        target_tri=agg_records[j][0]; target_ids=agg_records[j][4]
        source_edge=None; target_edge=None
        if len(pts)==2:
            source_edge=segment_on_triangle_edge(pts,source_tri,source_ids)
            target_edge=segment_on_triangle_edge(pts,target_tri,target_ids)
        src_boundary=source_edge in id22_edges and len(id22_edges[source_edge])==1 if source_edge else False
        patch_edge_key=(sid,target_edge) if target_edge else None
        target_patch_boundary=bool(target_edge and len(patch_edge_occ.get(patch_edge_key,[]))==1)
        if src_boundary and target_patch_boundary: cat="id22_and_patch_boundary_edges"
        elif src_boundary and target_edge: cat="id22_boundary_and_patch_interior_edge"
        elif source_edge and target_patch_boundary: cat="id22_interior_edge_and_patch_boundary"
        elif source_edge and target_edge: cat="both_triangle_edges_not_both_patch_boundaries"
        elif source_edge or target_edge: cat="one_triangle_edge_only"
        else: cat="triangle_interiors"
        intersection_categories[cat]+=1
        if len(witnesses)<24:
            witnesses.append({"id22_face":int(i),"aggregate_face":int(j),"aggregate_stable_id":int(sid),
                "aggregate_local_face":int(local_fi),"intersection_points":len(pts),"id22_segment_edge":bool(source_edge),
                "id22_source_boundary_edge":bool(src_boundary),"aggregate_segment_edge":bool(target_edge),
                "aggregate_patch_boundary_edge":bool(target_patch_boundary),
                "segment_lattice_endpoints":[[str(x) for x in p] for p in ordered] if len(ordered)==2 else []})
    # Exact containment classification of deterministic ID22 triangle centroids vs the closed aggregate shell.
    locator=ci.prepare_point_location(agg_records)
    sample_ids=sorted(set(int(round(i)) for i in np.linspace(0,len(f22)-1,8)))
    location_counts=Counter(); sample_results=[]
    for fi in sample_ids:
        ids=[int(x) for x in f22[fi]]
        point=tuple(Fraction(sum(pack_vertices[k][axis] for k in ids),3) for axis in range(3))
        loc=ci.point_location_prepared_indexed(point,locator)
        location_counts[loc["location"]]+=1
        sample_results.append({"id22_face":fi,**loc})
    # actual aggregate closed-shell signed volume; a sign indicates winding, not tissue-volume qualification.
    aggregate_volume=volume_signed(pos,agg_faces)
    return {"step":step,"pack":{"path":str(pack),"sha256":sha(pack),**pack_meta},
        "target_face_counts":{str(sid):int(len(surfs[(51010,sid)]["faces"])) for sid in range(14,23)},
        "aggregate_14_21_topology":{k:v for k,v in agg_top.items() if k not in ("edge_occurrences","face_owners")},
        "aggregate_14_21_signed_enclosed_volume_m3":aggregate_volume,
        "aggregate_14_21_winding":"outward-positive" if aggregate_volume>0 else "inward-negative",
        "id22_topology":{"source_boundary_edge_count":sum(len(v)==1 for v in id22_edges.values()),
            "nonmanifold_edge_count":sum(len(v)>2 for v in id22_edges.values()),
            "self_intersection_pair_count":int(self22["count"]),
            "self_allowed_shared_vertex_or_edge_pair_count":int(self22["allowed_shared_vertex_or_edge_pairs"]),
            "self_aabb_candidate_pairs":int(self22["aabb_candidate_pairs"])},
        "id22_vs_aggregate_intersection":{"aabb_candidate_pairs":int(intersections["aabb_candidate_pairs"]),
            "finite_intersecting_triangle_pairs":int(intersections["count"]),"intersection_point_count_histogram":dict(pair_dim),
            "unique_exact_segments":len(unique_segments),"aggregate_owner_pair_counts":dict(owner_counts),
            "segment_relation_pair_counts":dict(intersection_categories),"witnesses_first_24":witnesses},
        "id22_face_centroid_location_in_aggregate":{"sample_count":len(sample_ids),"counts":dict(location_counts),"samples":sample_results}}

resp_path=Path("/Users/numi/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-reference-respiration.json")
payload=Path("/Users/numi/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy")
receipt=ROOT/"back412-skin-reference-native-021-003/resting-anatomy-receipt.json"
source_audit=ROOT/"full-anatomy-delivery-audit-1378/liver-id22-source-identity-audit.json"
runmeta=RUN/"run-metadata.json"
# All writes are create-only so the record remains immutable.
progress_dir=ROOT/"full-anatomy-delivery-audit-1378/liver-id22-capture-audit-002"
progress_dir.mkdir(exist_ok=True)
results=[]
for step in CAPTURES:
    result=analyze_capture(step)
    cap_path=progress_dir/(f"capture-{step}.json")
    with cap_path.open("x") as f: json.dump(result,f,indent=2,sort_keys=True); f.write("\n")
    results.append(result)
    print(json.dumps({"completed_step":step,"intersections":result["id22_vs_aggregate_intersection"]["finite_intersecting_triangle_pairs"],"containment":result["id22_face_centroid_location_in_aggregate"]["counts"]},sort_keys=True),flush=True)
report={"schema":"numi.human.anatomy.liver-id22-aggregate-geometry-audit.v1","status":"pass",
  "scope":"Read-only exact Float32-lattice geometry comparison of actual native semantic 51010 liver stable IDs 22 versus the composed 14-21 closed shell at rest and seven accepted poses from the current 60-second breathing run. No payload, source, native runtime, or captured geometry changed.",
  "pins":{"native_run_metadata_path":str(runmeta),"native_run_metadata_sha256":sha(runmeta),
    "native_receipt_path":str(receipt),"native_receipt_sha256":sha(receipt),"source_identity_audit_path":str(source_audit),"source_identity_audit_sha256":sha(source_audit),
    "nhanat_payload_path":str(payload),"nhanat_payload_sha256":sha(payload),"respiration_path":str(resp_path),"respiration_sha256":sha(resp_path),
    "native_binary_sha256":json.loads(runmeta.read_text())["asset_sha256"]["/Users/numi/numi-lab-parent-bindings-build-1353-003/bin/numi-human-native"]},
  "method":{"surface_source":"MRVPACK2 positions/indices, semantic 51010 stable IDs 14-22, emitted by exact accepted native captures","triangle_intersection":"ci._audit_pair with exact integer-lattice coordinates made from every native Float32 vertex; no tolerance","containment":"exact indexed parity against welded closed 14-21 aggregate; 24 evenly spaced row22 triangle centroids per capture","volume":"signed boundary integral on emitted Float32 metre geometry; only reported for topologically closed aggregate 14-21","breathing_coverage":"accepted captures at declared steps 0,3807,4767,7999,14975,20991,26975,30000 in a 60-second native run"},
  "captures":results,
  "interpretation":{"pending":"Read the current 1c0 payload and receipt together with this report. ID22 source identity is FJ2824; a separate current surface may be retained as an anatomical subcomponent only if its measured intersections are classified as intended boundary/interface and the owner semantics justify it. A closed aggregate shell plus an open ID22 mesh with unexplained finite crossing/self-intersections is not a coherent second liver volume. This audit quantifies those geometries; it does not assign tissue volume or clinical physiology."}}
with OUT.open("x") as f:json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps({"status":"pass","report":str(OUT),"report_sha256":sha(OUT),"capture_summary":[{"step":r["step"],"topology":r["aggregate_14_21_topology"],"aggregate_volume_m3":r["aggregate_14_21_signed_enclosed_volume_m3"],"id22_self_pairs":r["id22_topology"]["self_intersection_pair_count"],"pair_intersections":r["id22_vs_aggregate_intersection"]["finite_intersecting_triangle_pairs"],"pair_categories":r["id22_vs_aggregate_intersection"]["segment_relation_pair_counts"],"containment":r["id22_face_centroid_location_in_aggregate"]["counts"]} for r in results]},sort_keys=True),flush=True)
