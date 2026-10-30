"""Reindex existing collar and protected149 after bounded endpoint contractions.

No physical state, historical source rows, or anatomical targets are changed.
Crossing predicates, volume quality, mass transfer and motion remain separate.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=P/"fine-physical-volume-706"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def fkey(p):return np.asarray(p,dtype="<f4").tobytes()
def validate_contraction(parent,new,parent_sha):
 if str(np.asarray(new["microedge_parent_candidate_sha256"]).reshape(-1)[0])!=parent_sha:
  raise ValueError("contraction parent SHA mismatch")
 a=np.asarray(parent["points_world_m"],np.float32);b=np.asarray(new["points_world_m"],np.float32)
 f=np.asarray(new["faces"],np.int64);m=np.asarray(new["microedge_input_point_to_output_row"],np.int64)
 off=np.asarray(new["microedge_parent_input_point_offsets"],np.int64);r=np.asarray(new["microedge_parent_input_point_rows"],np.int64)
 if (m.shape!=(len(a),) or off.shape!=(len(b)+1,) or off[0]!=0 or off[-1]!=len(r)
     or np.any(np.diff(off)<=0) or np.any(m<0) or np.any(m>=len(b))
     or not np.array_equal(np.sort(r),np.arange(len(a)))):
  raise ValueError("contraction point CSR is not a complete partition")
 if not np.array_equal(m[r],np.repeat(np.arange(len(b)),np.diff(off))):
  raise ValueError("contraction point CSR disagrees with row map")
 if f.ndim!=2 or f.shape[1]!=3 or np.any(f<0) or np.any(f>=len(b)) or not np.isfinite(b).all():
  raise ValueError("contraction output geometry is malformed")
 if not np.array_equal(np.unique(f),np.arange(len(b))):raise ValueError("unused contracted point")
 for i in range(len(b)):
  if not any(fkey(a[j])==fkey(b[i]) for j in r[off[i]:off[i+1]]):
   raise ValueError("contracted coordinate is not a preserved endpoint")
 distance=np.linalg.norm(a.astype(float)-b[m].astype(float),axis=1)
 bound=float(np.asarray(new["microedge_max_displacement_m"]).reshape(-1)[0])
 if bound>1e-6 or not np.isfinite(bound) or np.any(distance>bound):
  raise ValueError("contraction exceeds the 1 um original-point bound")
 fo=np.asarray(new["microedge_parent_input_face_offsets"],np.int64);fr=np.asarray(new["microedge_parent_input_face_rows"],np.int64)
 if (fo.shape!=(len(f)+1,) or fo[0]!=0 or fo[-1]!=len(fr) or np.any(np.diff(fo)<=0)
     or not np.array_equal(np.unique(fr),np.arange(len(parent["faces"])))):
  raise ValueError("contraction face ancestry is incomplete")
 return m,off,{"source_point_count":len(a),"candidate_point_count":len(b),
              "candidate_face_count":len(f),"maximum_original_point_displacement_m":float(distance.max(initial=0)),
              "point_parent_partition_complete":True,"face_parent_coverage_complete":True}
def preserved_target(parent,new,row_map,offsets,point):
 a=np.asarray(parent["points_world_m"],np.float32);b=np.asarray(new["points_world_m"],np.float32)
 matches=np.flatnonzero(np.all(a.view(np.uint32)==np.asarray(point,np.float32).view(np.uint32),axis=1))
 if len(matches)!=1:raise ValueError("protected target is not a unique parent point")
 old=int(matches[0]);current=int(row_map[old])
 if offsets[current+1]-offsets[current]!=1 or fkey(b[current])!=fkey(point):
  raise ValueError("protected target was merged or moved")
 return current
def reindex_collar(c,new_outer,parent_outer,row_map):
 oldno=len(c["points_world_m"])-90;newno=len(new_outer["points_world_m"])
 if oldno!=len(parent_outer["points_world_m"]):raise ValueError("collar parent O row domain mismatch")
 oldO=np.asarray(c["donor_O_to_candidate_vertex"],np.int64)
 mapped=row_map[oldO]
 oldp=np.asarray(c["points_world_m"],np.float32)
 newp=np.vstack((np.asarray(new_outer["points_world_m"],np.float32),oldp[oldno:]))
 if not np.array_equal(oldp[oldO].view(np.uint32),newp[mapped].view(np.uint32)):
  raise ValueError("collar O boundary coordinate changed")
 remap={int(a):int(b) for a,b in zip(oldO,mapped)}
 remap.update({oldno+i:newno+i for i in range(90)})
 out={k:np.array(v,copy=True) for k,v in c.items()}
 out.update(points_world_m=newp,outer_faces=np.asarray(new_outer["faces"],np.int64),
            donor_O_to_candidate_vertex=mapped.astype(np.uint32),
            exact_candidate_face_source_rows=np.full(len(new_outer["faces"]),-1,np.int64))
 for key in ("skin_sidewall_faces","fat_sidewall_faces"):
  out[key]=np.asarray([[remap[int(v)] for v in face] for face in c[key]],np.int64)
  if not np.array_equal(oldp[np.asarray(c[key],np.int64)].view(np.uint32),newp[out[key]].view(np.uint32)):
   raise ValueError("collar ordered triangle geometry changed")
 return out
def validate_parent_plan_inputs(pins):
 for key in ("outer","inner","deep","collar","wall_csv","protected_targets","canonical_d_proof","wall_parent_cells","wall_parent_nodes"):
  if sha(pins[key]["path"])!=pins[key]["sha256"]:raise ValueError("parent plan input changed: "+key)
def main():
 ap=argparse.ArgumentParser()
 for role in ("outer","inner","deep"):ap.add_argument("--"+role,type=Path,required=True)
 ap.add_argument("--parent-plan",type=Path,default=H/"plan-003/manifest.json")
 ap.add_argument("--output",type=Path,required=True)
 x=ap.parse_args()
 if x.output.exists():raise FileExistsError(x.output)
 spec=importlib.util.spec_from_file_location("h706_microedge",H/"handoff/prepare_tetgen_handoff_697_typedD.py")
 h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 manifest=json.loads(x.parent_plan.read_text());pins=manifest["inputs"]
 validate_parent_plan_inputs(pins)
 old={};new={};maps={};stats={}
 candidate_paths={"O":x.outer,"I":x.inner,"D":x.deep};labels={"O":"outer","I":"inner","D":"deep"}
 hashes={}
 for role,path in candidate_paths.items():
  key=labels[role];old[role]=arrays(pins[key]["path"]);hashes[role]=sha(path);new[role]=arrays(path)
  maps[role]=validate_contraction(old[role],new[role],pins[key]["sha256"])
  stats[role]=maps[role][2]
 owner=h.load_owner();parentD=h.load_surface_217(owner,pins["deep"]["path"],"D")
 rows=h.read_pins(pins["protected_targets"]["path"])
 canonical=h.load_696_d_attachment_map(pins["canonical_d_proof"]["path"],pins["deep"]["path"],parentD,rows)
 routes=[]
 for row in rows:
  role=row["role"];c=int(row["canonical_protected_row"])
  point=(canonical["attachment_routes"][c]["target_world_m_f32"] if row["mapping_kind"]!="head_ring_boundary_target"
         else [float(row["target_world_"+axis+"_m"]) for axis in "xyz"])
  v=preserved_target(old[role],new[role],maps[role][0],maps[role][1],point)
  item={"canonical_protected_row":c,"role":role,"mapping_kind":row["mapping_kind"],"surface_vertex_row":v,"target_world_m_f32":list(map(float,point))}
  if row["mapping_kind"]!="head_ring_boundary_target":
   f=np.asarray(new[role]["faces"],np.int64);p=np.asarray(new[role]["points_world_m"],np.float32)
   incident=np.flatnonzero(np.any(f==v,axis=1));tri=p[f[incident]].astype(float)
   double_area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
   edges=np.linalg.norm(np.roll(tri,-1,axis=1)-tri,axis=2)
   altitude=float(np.min(double_area[:,None]/np.maximum(edges,1e-300)))
   if altitude<50e-6:raise ValueError("canonical attachment star fails existing 50 um gate")
   item.update(stable_attachment_id=int(row["stable_attachment_id"]),minimum_surface_altitude_m=altitude,incident_face_count=len(incident))
  routes.append(item)
 c=arrays(pins["collar"]["path"]);collar=reindex_collar(c,new["O"],old["O"],maps["O"][0])
 with Path(pins["wall_csv"]["path"]).open(newline="") as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;walls=list(reader)
 if len(walls)!=180:raise ValueError("wall provenance must have 180 rows")
 for i,row in enumerate(walls):
  key="skin_sidewall_faces" if i<90 else "fat_sidewall_faces"
  for k in range(3):row["output_node"+str(k)]=str(collar[key][i%90,k])
 # Check current wall ancestry through the existing source003 owner.
 x.output.mkdir(parents=True)
 cp=x.output/"collar-current.npz";wp=x.output/"wall-provenance-current.csv"
 np.savez_compressed(cp,**collar)
 with wp.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(walls)
 owner.read_wall_provenance(collar,wp,Path(pins["wall_parent_cells"]["path"]),Path(pins["wall_parent_nodes"]["path"]),"registered-native-current")
 for role,path in candidate_paths.items():
  if sha(path)!=hashes[role]:raise ValueError("candidate changed during preparation")
 result={"schema":"numi.paired-fullbody-tetgen-handoff-697.v1","status":"PASS_COLLAR_AND_PROTECTED149_REINDEX_SOURCE_ONLY",
         "physical_admission":False,"runtime":0,"accepted_steps":0,"owner_sha256":sha(__file__),"parent_plan":{"path":str(x.parent_plan),"sha256":sha(x.parent_plan)},
         "inputs":{role:{"path":str(path),"sha256":hashes[role]} for role,path in candidate_paths.items()},
         "microedge_checks":stats,"protected149":{"count":149,"surface_routes":routes},"paired_rim_ordered_triangle_bits_unchanged":True,
         "wall_ancestry_existing_owner_pass":True,"outputs":{"collar":{"path":str(cp),"sha256":sha(cp)},"wall_csv":{"path":str(wp),"sha256":sha(wp)}},
         "limitations":["No crossing, volume, mass, native admission or motion claim. Historical protected CSV remains immutable."]}
 (x.output/"receipt.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":result["status"],"protected_targets":149,"outputs":result["outputs"]}),flush=True)
if __name__=="__main__":main()
