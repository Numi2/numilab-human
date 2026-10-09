#!/usr/bin/env python3
import hashlib,importlib.util,json,struct,sys,time
from collections import Counter,defaultdict
from pathlib import Path
B=Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-accepted-geometry-audit-001"); O=B/"raw-self-seam-diagnosis-001"
R=B/"audit_native_candidate_1218_root002.py"; D=B/"early-40s-002"; G=D/"step-19999.muscle-geometry.json"; W=D/"step-19999.muscle-crossing-witnesses.jsonl"; P=B.parent/"native-baseline-310s-preparation/native-run/accepted-geometry/step-19999.mrvpack"; Q=P.with_suffix(".receipt.json")
T=Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"); TM=T.with_suffix(".manifest.json"); C=Path("/Users/n/numi-human-conforming-composition-source-1216/src/numilab_human"); CON=Path("/Users/n/numi-human-retained-delivery-20261009/passive-muscle-self-audit-1213/audit_muscle_self_1213.py")
pins={R:"c7af4ff6602e46587278eadef41fee7adbbb80152c938669a59aad8ddcbeaac9",G:"8933ccddff845332e270a817574500e3a0c90c2514d2146ab1bf1470b5861dcb",W:"2597153779cec001afbfa547448af6e6adf3b58b750fdf056502b7e39652442b",P:"252990c6490c4a5164f3200001087b3b9dc33c0a0313a10bba2c10933d22dd89",Q:"78deca302ffea6a5907bb85d2f8102187fac7095ec4509864157a3dd3edd6dc1",T:"1cd0c3d5bd1ff6d163f6544be4d0fa169e8bf696bf6c729ef0b18879b0856c48",TM:"f052eff736f040f2f6ca1a1a99d2939ca0ed5c0bead055dfb39d3141f05a3cac",C/"cardiac_cavity_intersections.py":"934c64fa6a64a06044aee6db44cdcd71dc497fcec97a20a80029575faaaff4b4",C/"common_atlas_skin_clearance.py":"ec198693fce562db27c4b761e3dbbc4b1bff4f0b9458ae67ab7ba596a97a000f",CON:"d56dff6df2286953ab6ffdcbb50ad51e794d12f2e371c1aa15efcf990ca26f46"}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def need(x,m):
 if not x: raise RuntimeError(m)
def k3(p): return struct.pack("<3f",*map(float,p))
def vrec(row,i): return struct.unpack_from("<6f4I4f",row["vertex_bytes"],int(i)*56)
def main():
 start=time.monotonic(); before={str(p):sha(p) for p in pins}
 for p,h in pins.items(): need(before[str(p)]==h,"pin mismatch "+str(p))
 spec=importlib.util.spec_from_file_location("_audit1218_root002_seams",R); mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 _,owner,ci,cap,val,core,outer,codec,_=mod.load_owners(); _,src=mod.validate_candidate(codec); accepted=owner.verify_pack(19999,P,Q,val,mod.NHA_SHA)
 xyz,surfs,packcounts=cap._pack_surfaces(P,{mod.M63,mod.M64}); geom=json.loads(G.read_text()); gr={tuple(x["surface"]):x for x in geom["surfaces"]}
 agg={mod.M63:Counter(),mod.M64:Counter()}; samples=defaultdict(list); nw=0
 for key in (mod.M63,mod.M64):
  faces=surfs[key]["faces"]; row=src[key]; g=gr[key]; base=int(g["captured_vertex_base"]); need((faces.astype("int64")-base).tolist()==row["local_faces"],"source faces differ")
  ids=__import__("numpy").unique(faces); local=faces.astype("int64")-base; records,rr,deg=core.exact_records(xyz[ids],__import__("numpy").searchsorted(ids,faces),ci); raw=ci._audit_pair(records,records,same_surface=True)
  target=outer._prepare_closed_clearance_target(xyz[ids],local,allow_nested_enclosure=True); rep=target["report"]
  need(raw["count"]==g["self_intersection_count"],"raw predicate count differs"); need(rep["self_intersection_audit"]["count"]==0 and rep["face_count"]==len(faces),"quotient self/face count differs"); need(rep["all_components_closed_oriented_unused_free"],"quotient topology fails")
 with W.open() as f:
  for line in f:
   w=json.loads(line); key=tuple(w["target_surface"]); need(key in agg,"unexpected surface"); g=gr[key]; row=src[key]; base=int(g["captured_vertex_base"])
   fa,fb=int(w["skin_source_face_row"]),int(w["target_surface_face_row"]); pa,pb=list(map(int,w["skin_pack_vertex_ids"])),list(map(int,w["target_pack_vertex_ids"])); la,lb=[x-base for x in pa],[x-base for x in pb]
   need(row["local_faces"][fa]==la and row["local_faces"][fb]==lb,"witness face/source map mismatch")
   ka,kb=[k3(p) for p in w["skin_triangle_xyz_f32_m"]],[k3(p) for p in w["target_triangle_xyz_f32_m"]]; need(ka==[k3(xyz[x]) for x in pa] and kb==[k3(xyz[x]) for x in pb],"witness does not match pack")
   shared=set(ka)&set(kb); hits={k3(p) for p in w["intersection_points_m"]}; need(shared and hits<=shared,"hit extends off common coordinate vertices")
   c=agg[key]; c["raw_indexed_pairs"]+=1; c["exact_witness_points"]+=len(hits); c["one_shared_coordinate_vertex" if len(shared)==1 else "two_shared_coordinate_vertices" if len(shared)==2 else "three_shared_coordinate_vertices"]+=1; c["all_points_on_shared_coordinate_vertices"]+=1
   matched=0
   for point in shared:
    for i,a in enumerate(ka):
     if a!=point: continue
     for j,b in enumerate(kb):
      if b!=point or la[i]==lb[j]: continue
      ra,rb=vrec(row,la[i]),vrec(row,lb[j]); pe=struct.pack("<3f",*ra[:3])==struct.pack("<3f",*rb[:3]); re=struct.pack("<4I4f",*ra[6:])==struct.pack("<4I4f",*rb[6:])
      need(pe and re,"duplicated source seam position/route differs"); matched+=1; c["distinct_source_vertex_coordinate_matches"]+=1
      if len(samples[key])<3: samples[key].append({"face_rows":[fa,fb],"source_local_vertex_ids":[la[i],lb[j]],"source_position_and_sparse_route_equal":True})
   need(matched,"shared coordinate did not bind distinct source IDs"); c["pairs_with_exact_source_duplicate_seam"]+=1; nw+=1
 for key,c in agg.items():
  g=gr[key]; need(c["raw_indexed_pairs"]==g["self_intersection_count"]==c["all_points_on_shared_coordinate_vertices"],"raw-hit classification incomplete")
  need(c["pairs_with_exact_source_duplicate_seam"]==c["raw_indexed_pairs"],"source duplicate mapping incomplete")
 after={str(p):sha(p) for p in pins}; need(before==after,"input changed")
 result={"schema":"numi.human.native-muscle-raw-self-seam-diagnosis.v1","status":"all_raw_indexed_hits_are_exact_duplicate_source_seams; exact_quotient_self_zero","accepted_step":19999,"accepted_time_s":accepted["accepted_time_s"],"pack_counts":packcounts,
 "surfaces":{str(key[1]):{"surface":list(key),"retained_face_count":gr[key]["face_count_in_prepared_target"],"raw_indexed_hit_pairs":dict(agg[key]),"exact_quotient_unallowed_self_pairs":gr[key]["quotient_self_intersection_count"],"closed_oriented_unused_free":gr[key]["all_components_closed_oriented_unused_free"],"outer_envelope_clear":gr[key]["outer_envelope_clear"],"examples":samples[key]} for key in (mod.M63,mod.M64)},
 "predicate_contract":{"authoritative":"Exact Float32 coordinate quotient; retain all face rows; require closed/oriented/unused-free; then run the same exact triangle predicate. Matches retained 1213 muscle self audit and frozen 1216 outer-envelope helper.","raw_count":"Direct indexed ci._audit_pair count is retained; these duplicate-index contacts become shared-coordinate vertex/edge adjacency under exact quotient.","limits":"Only step 19999 and stable 63/64; other target, pose, and cross audits remain independent. No threshold, face, or geometry changes."},
 "inputs_before":before,"inputs_after":after,"inputs_unchanged":before==after,"raw_witness_lines":nw,"elapsed_wall_seconds":time.monotonic()-start}
 out=O/"diagnosis.json";out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"status":result["status"],"raw_witness_lines":nw,"report":str(out)},sort_keys=True))
if __name__=="__main__":main()
