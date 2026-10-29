from __future__ import annotations
import collections, hashlib, importlib.util, json, pathlib, sys, traceback
import numpy as np
P=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
BASE=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz"
FLIP=P/"geometry-worker/D-connectivity-218/limiting-O-edge-audit-001/quality-failure-inventory-001/contraction-batch-001/shape-repair-batch-005/contraction-001/authoring-source-chain/local_float32_edge_flip.py"
PAIR=P/"exact_local_pair_witnesses_381.py"
ROOT=P/"reduced-physical-boundary-400/attempt-033-witness-centered-current-D-cavity"
OUT=ROOT/"plc-exact-preflight-001"
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wj(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def parse_poly(path):
 lines=path.read_text().splitlines();n=int(lines[1].split()[0])
 Pts=np.array([[float(x) for x in lines[i].split()[1:4]] for i in range(2,2+n)],np.float32).astype(float)
 j=2+n;nf=int(lines[j].split()[0]);k=j+1;F=[]
 for _ in range(nf):
  if lines[k].split()[0]!="1":raise ValueError("non-one-facet poly stream")
  q=lines[k+1].split()
  if q[0]!="3":raise ValueError("non-triangle poly facet")
  F.append([int(x)-1 for x in q[1:]]);k+=2
 return Pts,np.asarray(F,np.int64)
def oriented_edges(F):
 d=collections.defaultdict(list)
 for ri,f in enumerate(F):
  a,b,c=map(int,f)
  for u,v in ((a,b),(b,c),(c,a)):d[tuple(sorted((u,v)))].append((u,v,ri))
 return d
def categorize(i,shelln,annn,patchn):
 if i<shelln:return "shell"
 if i<shelln+annn:return "annulus"
 if i<shelln+annn+patchn:return "fine_patch"
 return "unknown"
def main():
 spec=importlib.util.spec_from_file_location("flip033",FLIP);flip=importlib.util.module_from_spec(spec);spec.loader.exec_module(flip)
 ps=importlib.util.spec_from_file_location("pairs381",PAIR);pairmod=importlib.util.module_from_spec(ps);ps.loader.exec_module(pairmod)
 base=np.load(BASE,allow_pickle=False);CF=np.asarray(base["coarse_boundary_faces_D"],np.int64)
 rows=[]
 for label,dirname in [("run001","run-001"),("run002","run-002-hit-shell-expansion"),("run003","run-003-second-hit-shell-expansion")]:
  run=ROOT/dirname;poly=run/"witness-centered-current-D-cavity.poly"
  if not poly.is_file():continue
  Pts,F=parse_poly(poly);m=run/"witness-centered-D-patch.npz";z=np.load(m,allow_pickle=False)
  ann=np.asarray(z["annulus_faces"],np.int64);patchall=np.asarray(z["candidate_D_faces"],np.int64)
  patchF=patchall[-56:];pre=np.load(run/"preannulus-witness-patch.npz",allow_pickle=False)
  cand=np.asarray(pre["candidate_points_reference_f32"],np.float32)
  key=lambda p:np.asarray(p,np.float32).tobytes()
  lut={key(x):i for i,x in enumerate(cand)}
  poly_to_candidate=[lut.get(key(x),-1) for x in Pts]
  if any(i<0 for i in poly_to_candidate) or len(set(poly_to_candidate))!=len(poly_to_candidate):
   raise ValueError(f"{label}: local PLC points do not map bijectively to candidate rows")
  candidate_to_poly={v:i for i,v in enumerate(poly_to_candidate)}
  shelln=len(F)-len(ann)-len(patchF)
  if shelln<0 or shelln+len(ann)+len(patchF)!=len(F):raise ValueError(f"{label}: facet category counts disagree")
  ann_local=np.asarray([[candidate_to_poly[int(v)] for v in t] for t in ann],np.int64)
  patch_local=np.asarray([[candidate_to_poly[int(v)] for v in t] for t in patchF],np.int64)
  ann_poly=F[shelln:shelln+len(ann)];patch_poly=F[shelln+len(ann):]
  same_ann=np.array_equal(ann_poly,ann_local);rev_ann=np.array_equal(ann_poly,ann_local[:,[0,2,1]])
  same_patch=np.array_equal(patch_poly,patch_local);rev_patch=np.array_equal(patch_poly,patch_local[:,[0,2,1]])
  if not ((same_ann and same_patch) or (rev_ann and rev_patch)):
   raise ValueError(f"{label}: local PLC rep is not exact ordered annulus+patch (ann_same={same_ann},ann_rev={rev_ann},patch_same={same_patch},patch_rev={rev_patch})")
  owner=flip.Flips(Pts,F)
  pairs=pairmod.pair_witnesses(owner,set(range(len(F))))
  if pairs is None:raise ValueError(f"{label}: exact pair query contains degenerate record")
  group=collections.Counter()
  witnesses=[]
  for a,b in pairs:
   ca=categorize(int(a),shelln,len(ann),len(patchF));cb=categorize(int(b),shelln,len(ann),len(patchF))
   group["/".join(sorted((ca,cb)))]+=1
   if len(witnesses)<50:
    witnesses.append({"local_face_rows":[int(a),int(b)],"face_classes":[ca,cb],
      "poly_vertex_ids_1based":[[int(x)+1 for x in F[int(a)]],[int(x)+1 for x in F[int(b)]]],
      "candidate_vertex_rows":[[int(poly_to_candidate[x]) for x in F[int(a)]],[int(poly_to_candidate[x]) for x in F[int(b)]]]})
  ed=oriented_edges(F);hist=collections.Counter(map(len,ed.values()))
  same=[e for e,q in ed.items() if len(q)==2 and q[0][0:2]==q[1][0:2]]
  areas=.5*np.linalg.norm(np.cross(Pts[F[:,1]]-Pts[F[:,0]],Pts[F[:,2]]-Pts[F[:,0]]),axis=1)
  f32keys=collections.defaultdict(list)
  for i,x in enumerate(Pts):f32keys[key(x)].append(i)
  dup=[q for q in f32keys.values() if len(q)>1]
  # Verify exact global-row seam identities and directed incidence at both collar loops.
  inner=[int(x) for x in z["inner_boundary_candidate_rows"]]
  outer=[6112,6348,6153]
  ao=oriented_edges(np.asarray([[candidate_to_poly[int(v)] for v in t] for t in ann]))
  po=oriented_edges(np.asarray([[candidate_to_poly[int(v)] for v in t] for t in patchF]))
  inner_checks=[]
  for a,b in zip(inner,inner[1:]+inner[:1]):
   x,y=candidate_to_poly[a],candidate_to_poly[b];e=tuple(sorted((x,y)))
   ai=ao.get(e,[]);pi=po.get(e,[])
   inner_checks.append(len(ai)==1 and len(pi)==1 and ai[0][0:2]==(y,x) and pi[0][0:2]==(x,y))
  outer_checks=[]
  ct=CF[4648]
  cedge=oriented_edges(np.asarray([ct],np.int64))
  for a,b in zip(outer,outer[1:]+outer[:1]):
   x,y=candidate_to_poly[a],candidate_to_poly[b];e=tuple(sorted((x,y)))
   ai=ao.get(e,[]);ci=cedge.get(tuple(sorted((a,b))),[])
   # compare both in candidate index domain; coarse boundary direction is ct order.
   want=next(((int(u),int(v)) for u,v,_ in ci),None)
   got=tuple(poly_to_candidate[q] for q in ai[0][:2]) if len(ai)==1 else None
   outer_checks.append(got==want)
  rows.append({"run":label,"poly_sha256":sha(poly),"facets":{"total":int(len(F)),"shell":int(shelln),"annulus":int(len(ann)),"fine_patch":int(len(patchF))},
   "points":int(len(Pts)),"exact_float32_duplicate_coordinate_classes":int(len(dup)),
   "float32_zero_area_faces":int(np.sum(areas==0)),"minimum_positive_area_m2":float(areas[areas>0].min()) if np.any(areas>0) else None,
   "edge_incidence_histogram":{str(k):int(v) for k,v in hist.items()},"same_directed_shared_edges":int(len(same)),
   "global_seam_identity":{"inner_boundary_edges":int(len(inner)),"inner_edge_identity_and_opposite_direction_pass":bool(all(inner_checks)),
     "outer_triangle_edges":3,"outer_edge_identity_and_direction_pass":bool(all(outer_checks))},
   "exact_self_triangle_pairs":int(len(pairs)),"exact_pair_classes":{k:int(v) for k,v in sorted(group.items())},
   "exact_pair_witnesses":witnesses,"I_interface_present_in_local_PLC":False,
   "note":"This local PLC contains only one fat source group and a D exterior patch; the paired skin-fat I interface is outside this cavity trial."})
 result={"schema":"numi-human.local-plc-exact-preflight.v1","physical_admission":False,
  "source":{"exact_owner":str(FLIP),"exact_owner_sha256":sha(FLIP),"witness_adapter":str(PAIR),"witness_adapter_sha256":sha(PAIR),
   "source002_sha256":sha(BASE)},"runs":rows}
 wj(OUT/"receipt.json",result);print(json.dumps({"out":str(OUT/"receipt.json"),"runs":[{k:v for k,v in r.items() if k not in ("exact_pair_witnesses",)} for r in rows]},indent=2))
if __name__=="__main__":main()
