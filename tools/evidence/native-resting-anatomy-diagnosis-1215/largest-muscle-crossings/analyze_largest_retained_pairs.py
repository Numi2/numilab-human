#!/usr/bin/env python3
import hashlib,json,math,sys
from fractions import Fraction
from pathlib import Path
BASE=Path("/Users/n/numi-human-retained-delivery-20261009/passive-muscle-self-intersection-topology-1215")
AUDIT=Path("/Users/n/numi-human-retained-delivery-20261009/passive-muscle-self-audit-1213")
CLASS=Path("/Users/n/numi-human-retained-delivery-20261009/passive-muscle-self-witness-classification-1214")
HIST=Path("/Users/n/numi-human-common-skin-multipose-001/src")
PRED=HIST/"numilab_human/cardiac_cavity_intersections.py"
PAIR=AUDIT/"unallowed-self-pairs.jsonl"; CLASSIFIED=CLASS/"classified-pairs.jsonl"
PINS={
 PAIR:"4f865f756c9fe08d9930490bc07db9942aa45cbe2d062228b1ca3d2c77493a26",
 AUDIT/"report.json":"58da57b10bf89bc9436bfd993425cb725ec9759d587cbad6589db54c73ff392a",
 CLASS/"report.json":"4afb5b0f5dd8369d34b9d20e154132c7d55d3dfe0efab5c854d6e2a4c91e01cf",
 CLASSIFIED:"35bb0c50a96b8ba374f0b7392d79c1869707781a54014ccb83d5c7f268ea4a34",
 AUDIT/"coordinate-frame-clarification.json":"e5de244f4fe2b986f76917f16dbfaceebbce3cf406a3d9ca41b651fe01e37aaa",
 PRED:"11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb"}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4194304),b""):h.update(b)
 return h.hexdigest()
def req(c,m):
 if not c:raise RuntimeError(m)
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def norm(v):
 scale=max(abs(float(x)) for x in v)
 if not scale:return(0.,0.,0.)
 q=[float(x)/scale for x in v]; z=math.sqrt(sum(x*x for x in q));return tuple(x/z for x in q)
def metrics(ci,tri):
 q=tuple(ci.float32_point_lattice_key(tuple(map(float,p))) for p in tri)
 n=cross(sub(q[1],q[0]),sub(q[2],q[0]));req(any(n),"degenerate triangle")
 e=[sub(q[1],q[0]),sub(q[2],q[1]),sub(q[0],q[2])]
 D=float(ci._FLOAT32_LATTICE_DENOMINATOR)
 lengths=[math.hypot(float(v[0]),float(v[1]),float(v[2]))/D for v in e]
 area=math.hypot(*map(float,n))/(2*D*D)
 alt=[2*area/x for x in lengths]
 angles=[]
 for i in range(3):
  a,b,c=lengths[(i+1)%3],lengths[(i+2)%3],lengths[i]
  angles.append(math.degrees(math.acos(max(-1.,min(1.,(a*a+b*b-c*c)/(2*a*b))))))
 return {"xyz_m":[list(map(float,p)) for p in tri],"lattice_points":[list(map(str,p)) for p in q],"edges_m":lengths,"area_m2":area,"altitudes_m":alt,"min_angle_deg":min(angles),"aspect_ratio":max(lengths)/min(alt),"normal":n,"unit_normal":norm(n),"q":q}
def distances(ci,points,plane):
 q=[ci.float32_point_lattice_key(tuple(map(float,p))) for p in points]
 t=[ci.float32_point_lattice_key(tuple(map(float,p))) for p in plane]
 n=cross(sub(t[1],t[0]),sub(t[2],t[0]));nl=math.hypot(*map(float,n));D=float(ci._FLOAT32_LATTICE_DENOMINATOR)
 return [float(dot(n,sub(p,t[0])))/nl/D for p in q]
def detail(ci,r,c):
 a,b=metrics(ci,r["triangle_xyz_f32_m"][0]),metrics(ci,r["triangle_xyz_f32_m"][1])
 pts=sorted(set(tuple(Fraction(v) for v in p) for p in ci.triangle_intersection_points(a["q"],b["q"])))
 encoded=[[[str(v.numerator),str(v.denominator)] for v in p] for p in pts]
 req(encoded==c["exact_lattice_points"],"selected pair differs from exact 1214 replay")
 cos=abs(sum(x*y for x,y in zip(a["unit_normal"],b["unit_normal"])))
 ids1,ids2=[set(map(int,x)) for x in r["source_vertex_ids"]]
 qids1,qids2=[set(map(int,x)) for x in r["quotient_vertex_ids"]]
 ca={tuple(ci.float32_point_lattice_key(tuple(map(float,p))):int(r["source_vertex_ids"][0][i]) for i,p in enumerate(r["triangle_xyz_f32_m"][0])}
 cb={tuple(ci.float32_point_lattice_key(tuple(map(float,p))):int(r["source_vertex_ids"][1][i]) for i,p in enumerate(r["triangle_xyz_f32_m"][1])}
 shared=set(ca)&set(cb)
 return {"space":r["coordinate_space"],"classification":c["classification"],"extent_m":c["extent_diameter_m"],"points":len(pts),
 "plane_angle_deg":math.degrees(math.acos(max(-1.,min(1.,cos)))),
 "triangle_A":{k:v for k,v in a.items() if k not in ("q","normal")},"triangle_B":{k:v for k,v in b.items() if k not in ("q","normal")},
 "A_vertex_signed_distances_to_B_plane_m":distances(ci,r["triangle_xyz_f32_m"][0],r["triangle_xyz_f32_m"][1]),
 "B_vertex_signed_distances_to_A_plane_m":distances(ci,r["triangle_xyz_f32_m"][1],r["triangle_xyz_f32_m"][0]),
 "shared_source_vertex_ids":sorted(ids1&ids2),"shared_quotient_vertex_ids":sorted(qids1&qids2),"shared_exact_coordinate_count":len(shared),
 "shared_exact_coordinate_source_id_pairs":[[ca[k],cb[k]] for k in sorted(shared)],
 "face_rows":r["face_rows"],"source_vertex_ids":r["source_vertex_ids"],"quotient_vertex_ids":r["quotient_vertex_ids"]}
def main():
 req(not (BASE/"report.json").exists(),"refusing to overwrite")
 for p,h in PINS.items():req(p.is_file() and sha(p)==h,"input hash mismatch "+str(p))
 sys.path.insert(0,str(HIST))
 from numilab_human import cardiac_cavity_intersections as ci
 req(Path(ci.__file__).resolve()==PRED.resolve(),"predicate import path mismatch")
 old={}; classes={}
 for line in PAIR.read_text().splitlines():
  r=json.loads(line);old[(r["coordinate_space"],int(r["stable_id"]),tuple(r["face_rows"]))]=r
 for line in CLASSIFIED.read_text().splitlines():
  c=json.loads(line);classes[(c["coordinate_space"],int(c["stable_id"]),tuple(c["face_rows"]))]=c
 identities={}
 for k,c in classes.items():
  if c["classification"] in ("transverse_segment_contact","coplanar_segment_contact"):
   ident=(k[1],k[2]);identities[ident]=max(identities.get(ident,0.),c["extent_diameter_m"])
 chosen=sorted(identities,key=lambda x:(-identities[x],x[0],x[1]))[:20]
 out=[]
 for sid,faces in chosen:
  sides={}
  for space in ("authored_nhtiss4_float32_owner_local","accepted_mrvpack2_step0_float32_world"):
   k=(space,sid,faces)
   if k in old:sides[space]=detail(ci,old[k],classes[k])
  out.append({"stable_id":sid,"face_rows":list(faces),"max_extent_m":identities[(sid,faces)],
   "status":"retained_in_both" if len(sides)==2 else ("source_only" if "authored_nhtiss4_float32_owner_local" in sides else "newly_posed_step0"),"spaces":sides})
 report={"schema":"numi.human.passive-muscle-self-intersection-largest-pair-topology.v1","status":"complete",
  "scope":"Top20 unique source/step0 segment witness identities from retained 1213/1214 records; recomputed only these pairs with pinned exact Float32 lattice predicate; no mesh scan/native/GPU/source edits.",
  "inputs":{str(p):{"sha256":sha(p)} for p in PINS},
  "units_and_frames":"Triangle coordinates are stored Float32 metres. Source NHTISS frame tag is not established (1213 coordinate-frame clarification); source is BodyParts3D-derived, not individual measurement. Accepted step0 is captured MRVPACK world coordinates. Distances converted from lattice units to metres without raw-mm rescaling.",
  "metrics":"Acute plane angle; signed vertex-to-opposing-plane distances in metres; edge lengths, area, altitudes, minimum internal angle, longest-edge/min-altitude aspect ratio; source/quotient vertex IDs and exact coordinate equality. Long segment extent is not waived by a small plane angle.",
  "selection":"Unique (stable_id,face_rows), ranked by maximum exact intersection-vertex diameter across source and accepted step0; one accepted pose only.",
  "pairs":out,"qualification":"Metrics distinguish extended exact contact from shallow near-coplanar contact but do not alone prove visible folds or biological severity."}
 (BASE/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print("completed",len(out))
if __name__=="__main__":main()
