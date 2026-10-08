from pathlib import Path
import collections, hashlib, json, sys, time, math
import numpy as np
E=Path("/Users/n/numi-human-resting-evidence-20261005")
OUT=E/"native-lung-all-source-cross-audit-1079"
NHA=E/"native-lung-d311-crossfield-registered-flip-1076/candidate-source.nhanatomy"
NPZ=E/"native-lung-d311-crossfield-registered-flip-1076/candidate-rows.npz"
DMAP=E/"native-lung-d311-crossfield-registered-flip-1076/candidate-D311-to-lobes-map.jsonl"
BASE_REPORT=E/"native-lung-d311-crossfield-registered-flip-1076/report.json"
PRED=Path("/Users/n/numi-human-lung-seam-correction-001/src/numilab_human/cardiac_cavity_intersections.py")
CONTACT=Path("/Users/n/numi-human-lung-contact-classifier-001/src/numilab_human/lung_contact_classification.py")
EXPECTED={
 str(NHA):"110ed37c7b109386658c098015ef6e75b4fff898121b7ef1f9845f8c0d4e15ad",
 str(NPZ):"2328b65bbb15c2c8567ec6a3c34ed5ad563fcc4a5c172e4cbc53a841eba8948a",
 str(DMAP):"05ab5b8f6ab02a0f3236495b385bc0002d750be58820117e7f149e64df1e398a",
 str(BASE_REPORT):"cde8d985f4cf43eb8f249015e01d0a43745ca25f2fa3d4660a72ab92f8eb6c6b",
 str(PRED):"11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb",
 str(CONTACT):"a4567b717fddbb3ab0caf11de8de6d955b0d0e8bae6efc899a41700ae9a0b6ae",
}
ROOT=Path("/Users/n/numi-human-final-lung-composition-001")
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,"/Users/n/numi-human-lung-contact-classifier-001/src")
from numilab_human.resting_anatomy_interface_patch import parse_payload
from numilab_human.cardiac_cavity_intersections import float32_point_lattice_key,_records,_audit_pair,triangle_intersection_points
from numilab_human.lung_contact_classification import (
    ALLOWED_LABELS, classify_lobe_lobe, classify_diaphragm_lobe, classification_counts,
)
LOBES=(305,306,307,308,309);ROWS=(*LOBES,311)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def edge(a,b):return (a,b) if a<=b else (b,a)
def point_on_segment(p,a,b):
 d=tuple(b[k]-a[k] for k in range(3));q=tuple(p[k]-a[k] for k in range(3))
 cr=(q[1]*d[2]-q[2]*d[1],q[2]*d[0]-q[0]*d[2],q[0]*d[1]-q[1]*d[0])
 if cr!=(0,0,0):return False
 dot=sum(q[k]*d[k] for k in range(3));den=sum(x*x for x in d)
 return 0<=dot<=den
def boundary(points,faces,selected):
 c=collections.Counter()
 for fi in selected:
  tri=[points[int(i)] for i in faces[int(fi)]]
  for k in range(3):c[edge(tri[k],tri[(k+1)%3])]+=1
 bad={e:n for e,n in c.items() if n not in (1,2)}
 return {e for e,n in c.items() if n==1},bad
def signature(points,faces,fi):
 t=tuple(points[int(i)] for i in faces[int(fi)])
 if len(set(t))!=3:raise ValueError("mapped face has repeated lattice vertex")
 ordered=sorted(t);perm=[ordered.index(x) for x in t]
 parity=sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))%2
 return tuple(ordered),(-1 if parity else 1)
def span_m(points):
 if len(points)<2:return 0.0
 m=0.0;den=float(1<<149)
 for i,p in enumerate(points):
  for q in points[i+1:]:
   d=[(p[k]-q[k])/den for k in range(3)]
   m=max(m,sum(x*x for x in d))
 return math.sqrt(m)
if OUT.exists():raise SystemExit(f"output exists: {OUT}")
for p,h in EXPECTED.items():
 if sha(p)!=h:raise SystemExit(f"input hash mismatch {p}: {sha(p)}")
OUT.mkdir()
runner=Path(__file__);runner_copy=OUT/"scan_all_pairs.py";runner_copy.write_bytes(runner.read_bytes())
t0=time.monotonic()
header,rows=parse_payload(NHA)
meshes={sid:(np.asarray(rows[sid]["vertices6"],dtype="<f4"),np.asarray(rows[sid]["faces"],dtype=np.int64)) for sid in ROWS}
points={sid:[float32_point_lattice_key(v[:3]) for v in meshes[sid][0]] for sid in ROWS}
records={sid:_records(points[sid],meshes[sid][1]) for sid in ROWS}
with NPZ.open("rb") as f:
 z=np.load(f,allow_pickle=False)
 for sid in ROWS:
  if not np.array_equal(meshes[sid][0],np.asarray(z[f"row{sid}_vertices6"],dtype="<f4")) or not np.array_equal(meshes[sid][1],np.asarray(z[f"row{sid}_faces"],dtype=np.int64)):
   raise SystemExit(f"candidate NPZ/NHA row mismatch {sid}")
map_rows=[json.loads(line) for line in DMAP.read_text().splitlines() if line.strip()]
if len(map_rows)!=47343:raise SystemExit(f"map row count changed: {len(map_rows)}")
d2l={sid:{} for sid in LOBES};l2d={sid:{} for sid in LOBES};map_meta={sid:[] for sid in LOBES}
for x in map_rows:
 sid=int(x["lobe_stable_id"]);di=int(x["d_face_row"]);li=int(x["l_face_row"])
 if sid not in d2l or di in d2l[sid] or li in l2d[sid]:raise SystemExit(f"duplicate/out-of-range map row {x}")
 if x["orientation"]!="exact_opposite_winding":raise SystemExit("unsupported map orientation")
 if signature(points[311],meshes[311][1],di)[0]!=signature(points[sid],meshes[sid][1],li)[0] or signature(points[311],meshes[311][1],di)[1]!=-signature(points[sid],meshes[sid][1],li)[1]:
  raise SystemExit(f"map geometry changed at {sid}/{di}/{li}")
 d2l[sid][di]=li;l2d[sid][li]=di;map_meta[sid].append(x)
boundaries={}
for sid in LOBES:
 db,dbad=boundary(points[311],meshes[311][1],d2l[sid].keys())
 lb,lbad=boundary(points[sid],meshes[sid][1],l2d[sid].keys())
 if dbad or lbad or db!=lb:raise SystemExit(f"mapped union boundary mismatch lobe {sid}: d_bad={len(dbad)} l_bad={len(lbad)} d={len(db)} l={len(lb)}")
 boundaries[sid]=db
# lobe-lobe interfaces are descriptive exact indexed-coordinate contacts; no intersection is removed from raw counts.
pairs=[(a,b,"lobe_lobe") for i,a in enumerate(LOBES) for b in LOBES[i+1:]]+[(sid,311,"diaphragm_lobe") for sid in LOBES]
wpath=OUT/"exact-cross-witnesses.jsonl";summaries=[]
boundary_cache={sid:{} for sid in LOBES}
with wpath.open("x") as wf:
 for a,b,kind in pairs:
  started=time.monotonic();audit=_audit_pair(records[a],records[b],same_surface=False)
  counts=collections.Counter();unclassified=[];hits=0
  for fa,fb in audit["triangle_pairs"]:
   tri_a=records[a][int(fa)][0];tri_b=records[b][int(fb)][0]
   ps=sorted(set(triangle_intersection_points(tri_a,tri_b)))
   if not ps:raise SystemExit(f"predicate candidate missing point {a}/{b}/{fa}/{fb}")
   common=set(tri_a)&set(tri_b)
   if kind=="lobe_lobe":
    cls=classify_lobe_lobe(tri_a,tri_b,ps)
   else:
    sid=a;di=int(fb);li=int(fa)
    mapped_l=d2l[sid].get(di)
    ltri=tuple(points[sid][int(i)] for i in meshes[sid][1][mapped_l]) if mapped_l is not None else ()
    candidate_ltri=tuple(points[sid][int(i)] for i in meshes[sid][1][li])
    cls=classify_diaphragm_lobe(ltri,candidate_ltri,ps,boundaries[sid],is_reciprocal_mapped_face=(mapped_l==li))
   counts[cls]+=1;hits+=1
   item={"pair":[a,b],"pair_kind":kind,"face_rows":[int(fa),int(fb)],"intersection_point_count":len(ps),
         "shared_coordinate_vertex_count":len(common),"intersection_span_m":span_m(ps),"classification":cls}
   if kind=="diaphragm_lobe":
    item["d_face_row"]=int(fb);item["l_face_row"]=int(fa)
    if mapped_l==li:
     mm=next(x for x in map_meta[sid] if int(x["d_face_row"])==di)
     item["mapped_d_source_face_id"]=int(mm["d_source_face_id"]);item["mapped_l_source_face_id"]=int(mm["l_source_face_id"])
    if cls=="unmapped_or_unclassified_diaphragm_contact": item["intersection_points_lattice"]=[list(map(int,p)) for p in ps]
   if kind=="lobe_lobe":
    item["face_rows_order"]=[int(fa),int(fb)]
   wf.write(json.dumps(item,separators=(",",":"),sort_keys=True)+"\n")
   if cls not in ALLOWED_LABELS[kind] and len(unclassified)<100:
    unclassified.append({"faces":[int(fa),int(fb)],"points_lattice":[list(map(int,p)) for p in ps],"span_m":span_m(ps)})
  wf.flush()
  summary={"pair":[a,b],"pair_kind":kind,"raw_exact_hit_count":int(audit["count"]),"aabb_candidate_pair_count":int(audit["aabb_candidate_pairs"]),
           "hit_class_counts":dict(counts),"unclassified_hit_count":int(sum(v for label,v in counts.items() if label not in ALLOWED_LABELS[kind])),
           "unclassified_examples":unclassified,"elapsed_s":time.monotonic()-started}
  summaries.append(summary)
  print(json.dumps({"progress":"pair_complete",**summary},sort_keys=True),flush=True)
report={"schema":"numi.human.lung-all-source-cross-audit.v1",
"scope":"Exact static source Float32-lattice intersections for all10 lobe-lobe pairs and all5 D311-to-lobe pairs on candidate1076. Raw exact intersection events are retained; classifications are explicit and do not alter predicates. Fail-closed classifier: every label outside the pair-kind allowlist is counted unclassified.",
"inputs":{p:{"sha256":h,"bytes":Path(p).stat().st_size} for p,h in EXPECTED.items()},
"candidate":{"nha_path":str(NHA),"nha_sha256":EXPECTED[str(NHA)],"map_path":str(DMAP),"map_sha256":EXPECTED[str(DMAP)],"map_rows":len(map_rows)},
"predicate":{"module":str(PRED),"sha256":EXPECTED[str(PRED)],"lattice":"exact integer lattice of Float32 coordinates; no tolerance"},
"pair_scan_complete":len(summaries)==15,"pair_summaries":summaries,
"full_map":{"declared_lobe_pairs":{str(s):{"mapped_face_count":len(d2l[s]),"mapped_union_boundary_edge_count":len(boundaries[s]),"d_to_l_boundary_equal":True} for s in LOBES}},
"witness_file":{"path":str(wpath),"sha256":sha(wpath),"format":"JSONL; each exact hit records pair, face rows, point count, exact shared-feature class and span; full lattice points retained for unexplained D-lobe hits"},
"elapsed_s":time.monotonic()-t0,
"limitations":["This is source-pose geometry only and does not qualify native transformed Float32 poses.","The current NHA lacks complete per-face source-origin arrays for all output faces; D-lobe mapped records include exact map source IDs, while lobe-lobe classifications are geometric and source-face lineage is separately provided by the parent-map evidence chain."]}
rp=OUT/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
(OUT/"README.md").write_text("# Exact source cross audit 1079\n\nThis future-run scanner uses the exact packed-Float32 lattice intersection predicate and a separately pinned fail-closed classifier. Every label outside an explicit pair-kind allowlist counts as unclassified. The boundary classifier requires each exact witness point to lie on at least one concrete declared union-boundary segment; empty boundaries cannot pass vacuously. Raw events remain preserved.\n\nThis source does not retroactively change the executed 1077 report. Run only with a fresh output path and preserve its exact output pins. No tolerance or native-pose qualification is introduced.\n")
print(json.dumps({"status":"complete","report":str(rp),"report_sha256":sha(rp),"witness_sha256":sha(wpath),"raw_hits":sum(x["raw_exact_hit_count"] for x in summaries),"unclassified_hits":sum(x["unclassified_hit_count"] for x in summaries)},sort_keys=True),flush=True)

