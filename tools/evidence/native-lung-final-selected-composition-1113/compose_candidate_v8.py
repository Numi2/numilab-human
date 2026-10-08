import collections,copy,hashlib,json,pathlib,sys,os,time
import numpy as np
E=pathlib.Path("/Users/n/numi-human-resting-evidence-20261005")
BASE=E/"native-lung-conditioned-final-compose-1078/final";OUT=E/"native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8"
SRC=pathlib.Path("/Users/n/numi-human-final-lung-composition-001/src");sys.path.insert(0,str(SRC))
from numilab_human.resting_anatomy_interface_patch import HEADER,RECORD,parse_payload,normals
from numilab_human.resting_lung_edge_repair import _serialize_payload,_refresh_diaphragm_interface_registration,_derive_basal_effective_area,_load_pinned_respiratory_owner,_source_volume_rows,_pleura_face_lineage
from numilab_human.resting_pleura_proxy import build_candidate as build_pleura
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.resting_respiratory_mesh_quality import open_registered_shared_vertex_star
NHA=BASE/"resting-thorax.nhanatomy"; REC=BASE/"resting-anatomy-receipt.json"; MAN=BASE/"resting-anatomy-manifest.json"; CFG=BASE/"resting-reference-respiration.json"
DMAP=BASE.parent/"final-exact-D311-to-lobes-map.jsonl"
LLMAP=E/"native-lung-lobe-source-lineage-1055-verified/reciprocal-coincident-face-map.jsonl";LLREPORT=LLMAP.parent/"source-proven-reciprocal-interface-report.json";LLNHA=E/"lung-choroid-composition-924/choroid-v2/resting-thorax.nhanatomy"
FLIP=E/"native-lung-307-sliver-reciprocal-flip-1083/candidate-307-309-rows.npz";FLIPLED=FLIP.parent/"candidate-307-309-parent-ledger.json"
C306=E/"native-lung-306-interior-seam-collapse-1099/candidate-rows.npz";L306=C306.parent/"parent-ledger.json"
C308=E/"native-lung-305-308-sequential-collapse-trial-1105/candidate-rows.npz";R308=C308.parent/"report.json"
C308BASE=E/"native-lung-305-308-shared-edge-collapse-1099/candidate-308-310-rows.npz"
STAR=E/"native-lung-left-second-cluster-star-trials-1105/trial-02.npz";STAR_REPORT=STAR.parent/"report.json"
OWNER=SRC/"numilab_human/resting_respiratory_mesh_quality.py";RESP=SRC/"numilab_human/resting_respiratory_conforming_field.py"
PINS={NHA:"7f6a8175e3cadb93a6dfc43535467b6896cf1b8bacfd0334df7b5bb03c414e92",
REC:"bea11d334575aceff2dd3151f1c38d3eafe8b825c0dbf9bef42633777fe4f9db",MAN:"01d3a744a5765cfe336ce540fc8a537823d6491c6ff260485bfbfab00ffd3500",
CFG:"3598d6402697f2bf4de9575afb5b7f30d4c4b05d7712a8c85a078c3ddd592222",DMAP:"05ab5b8f6ab02a0f3236495b385bc0002d750be58820117e7f149e64df1e398a",
LLMAP:"13b99a1de0fb0afbffe2e12f28f1194bdb193b7e610c1ea9bb26d2c22fcbe040",LLREPORT:"0b1f8dee81944f3338cc63d461706b825c963c9f1fb6be227fb875cb0877eb50",LLNHA:"3c444be7736c066a992988cc32b687917e1c4c5c3968a16b4d5f0106d5b5024e",
FLIP:"83898f5257a9c2cbb61ab567ed4af2106d761a2db20f7bd160f75fc80a8cec89",FLIPLED:"df98cf7961a3f791d16d4f7eb7d5b6d63f1ac0b01df72aa22c360b70a34e34e0",
C306:"25812e48e0f4bdeccfa13ffd6b2392b6de5008fc2f593c7c1b2a094ce3e7b325",L306:"de646ea2ffc7f015aa43a3ce4d9291c01b15276e5e56d0fb526e58a2189c4fb4",
C308:"c94ee31676bbb76d3ac697bab3af93b4225cbf10f900c6f586346a4441f8781e",R308:"a658e3abe1cf93bba9928f31a5e465d1a2c0cb42deb2370c32c7f177fa077057",C308BASE:"829409fc8206b65a3d91a1683a9e5736284be0da11a653826b02eb91a0b93da5",
STAR:"74d8f052f3847725769b571ca99b8dd9b5cf74984a5d8f1ea573a83e5511cba9",STAR_REPORT:"4f7769e02fe0665feeb0941c945673dcb96f24d041675e58533790938642f1be",
OWNER:"774b49fc5cc97518f035ad4434e1cf64b3dceb037b91dfd0a37aff89a81c8951",RESP:"980b368d8de0cc6cd7e25a93e61e5b6f304275dbd18511207da8cc9919410bb5"}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4<<20),b""):h.update(b)
 return h.hexdigest()
def meta(p):return {"path":str(p),"sha256":sha(p)}
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def pk(p):return np.ascontiguousarray(p,dtype="<f4").tobytes()
def key(points):return tuple(sorted(pk(p) for p in points))
def fkey(row,f):return key(row["vertices6"][np.asarray(f,dtype=np.int64),:3])
def ori(row,f):return tuple(pk(p) for p in row["vertices6"][np.asarray(f,dtype=np.int64),:3])
def bits(row,f):return [[f"{b:02x}" for b in pk(p)] for p in row["vertices6"][np.asarray(f,dtype=np.int64),:3]]
def opposite(a,b):return any(tuple(a[(s-i)%3] for i in range(3))==b for s in range(3))
def geom(row):return hashlib.sha256(np.ascontiguousarray(row["vertices6"],dtype="<f4").tobytes()+np.ascontiguousarray(row["faces"],dtype="<i8").tobytes()).hexdigest()
def area_vol(row):
 p=np.asarray(row["vertices6"][:,:3],dtype=np.float64);t=p[np.asarray(row["faces"],dtype=np.int64)]
 return float(.5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).sum()),float(np.einsum("ij,ij->i",t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
def verify_normals(row,sid):
 xyz=row["vertices6"][:,:3].astype(np.float64);faces=np.asarray(row["faces"],dtype=np.int64);used=np.unique(faces)
 expected=normals(xyz,faces).astype(np.float32)
 mismatch=np.flatnonzero(np.any(row["vertices6"][used,3:6].view(np.uint32)!=expected[used].view(np.uint32),axis=1))
 if len(mismatch):raise ValueError(f"serialized vertex normals differ from owner recomputation on row{sid}: {len(mismatch)} used vertices")
 return {"used_vertex_count":len(used),"normal_bit_mismatches":0}
def ixpoint(v,p,label):
 target=np.asarray(p,dtype="<f4");ids=np.flatnonzero(np.all(np.asarray(v[:,:3],dtype="<f4")==target,axis=1))
 if len(ids)!=1:raise ValueError(f"{label}: exact coordinate multiplicity {len(ids)}")
 return int(ids[0])
def replay_collapse(base,candv,candf,ledger,sid):
 op=ledger["operation"];drop=ixpoint(base["vertices6"],op["drop_point"],f"{sid}drop");keep=ixpoint(base["vertices6"],op["keep_point"],f"{sid}keep")
 ev=np.delete(base["vertices6"],drop,axis=0)
 if not np.array_equal(ev[:,:3].view(np.uint32),candv[:,:3].view(np.uint32)):raise ValueError(f"{sid} xyz is not exact vertex deletion")
 f=np.asarray(base["faces"],dtype=np.int64).copy();f[f==drop]=keep;deg=(f[:,0]==f[:,1])|(f[:,0]==f[:,2])|(f[:,1]==f[:,2]);f=f[~deg];kn=keep-(drop<keep)
 def remap(i):return kn if i==drop else i-(i>drop)
 if not np.array_equal(np.vectorize(remap,otypes=[np.int64])(f),candf):raise ValueError(f"{sid} faces do not replay collapse")
 return {"drop_old":drop,"keep_old":keep,"removed_faces":np.flatnonzero(deg).tolist(),"exact_xyz":True,"exact_faces":True}
def face_index(row,sid):
 d=collections.defaultdict(list)
 for i,f in enumerate(row["faces"]):d[fkey(row,f)].append(i)
 if any(len(v)!=1 for v in d.values()):raise ValueError(f"row {sid} has duplicate geometric faces")
 return d
def bridge_maps(base,final,ledger,flipnpz):
 old=[json.loads(x) for x in LLMAP.read_text().splitlines()]
 changed={307:set(map(int,ledger["modified_faces"]["307"]["face_rows"])),309:set(map(int,ledger["modified_faces"]["309"]["face_rows"]))}
 bi={s:face_index(base[s],s) for s in range(305,310)};fi={s:face_index(final[s],s) for s in range(305,310)}
 out=[];counts=collections.Counter();parent={}
 for source_record_ordinal,r in enumerate(old):
  a,b=map(int,r["pair"])
  k=tuple(sorted(bytes.fromhex("".join(p)) for p in r["mapped_output_triangle_vertices_bits"][0]))
  if k!=tuple(sorted(bytes.fromhex("".join(p)) for p in r["mapped_output_triangle_vertices_bits"][1])):raise ValueError("1055 paired triangles differ")
  pa,pb=bi[a].get(k,[]),bi[b].get(k,[])
  if len(pa)!=1 or len(pb)!=1:raise ValueError(f"1055 parent not unique rows={r['face_rows']} counts={[len(pa),len(pb)]}")
  pa,pb=pa[0],pb[0]
  parent[(a,pa)]=r;parent[(b,pb)]=r
  replaced=(int(r["face_rows"][0]) in changed.get(a,set()) or int(r["face_rows"][1]) in changed.get(b,set()))
  if replaced:
   if (a,b)!=(307,309):raise ValueError("only declared1083 parent pairs may be replaced")
   continue
  fa,fb=fi[a].get(k,[]),fi[b].get(k,[])
  if [len(fa),len(fb)] != [1,1]:raise ValueError(f"1055 map not uniquely preserved rows={r['face_rows']} counts={[len(fa),len(fb)]}")
  fa,fb=fa[0],fb[0]
  if not opposite(ori(final[a],final[a]["faces"][fa]),ori(final[b],final[b]["faces"][fb])):raise ValueError("current map opposite winding failed")
  x=copy.deepcopy(r);x["face_rows"]=[fa,fb];x["mapped_output_triangle_vertices_bits"]=[bits(final[a],final[a]["faces"][fa]),bits(final[b],final[b]["faces"][fb])]
  x["parent_face_rows_1078"]=[pa,pb];x["parent_face_vertices_bits_1078"]=[bits(base[a],base[a]["faces"][pa]),bits(base[b],base[b]["faces"][pb])]
  x["parent_face_rows_1055"]=[int(r["face_rows"][0]),int(r["face_rows"][1])];x["source_face_ids_1055"]=list(r["source_face_ids"]);x["source_map_record_ordinal_1055"]=source_record_ordinal;x["operation_ref"]=None
  out.append(x);counts[(a,b)]+=1
 child={}
 for sid in (307,309):
  child[sid]=[]
  for f in ledger["modified_faces"][str(sid)]["after_faces"]:
   k=fkey({"vertices6":flipnpz[f"row{sid}_vertices6"]},f);hits=fi[sid].get(k,[])
   if len(hits)!=1:raise ValueError(f"1083 child face missing row{sid}")
   child[sid].append((hits[0],k))
 parents={sid:sorted(map(int,ledger["modified_faces"][str(sid)]["face_rows"])) for sid in (307,309)}
 pi={}
 for sid in (307,309):
  for p in parents[sid]:
   r=parent.get((sid,p))
   if r is None:raise ValueError(f"1083 parent {sid}:{p} absent from 1055")
   side=0 if int(r["pair"][0])==sid else 1
   ordinal=next(i for i,x in enumerate(old) if x is r)
   pi[(sid,p)]={"source":int(r["source_face_ids"][side]),"kind":int(r["patch_kinds"][side]),"row1055":int(r["face_rows"][side]),"map_ordinal":ordinal,"sourcebits":r["source_parent_face_vertices_bits"][side]}
 ca={k:i for i,k in child[307]};cb={k:i for i,k in child[309]}
 if set(ca)!=set(cb) or len(ca)!=2:raise ValueError("1083 child pair key mismatch")
 op={"path":str(FLIPLED),"sha256":sha(FLIPLED),"npz_path":str(FLIP),"npz_sha256":sha(FLIP),"label":ledger["label"],"method":ledger["operation"]["method"]}
 for k in sorted(ca):
  fa,fb=ca[k],cb[k]
  if not opposite(ori(final[307],final[307]["faces"][fa]),ori(final[309],final[309]["faces"][fb])):raise ValueError("1083 child winding invalid")
  ids=[[pi[(307,p)]["source"] for p in parents[307]],[pi[(309,p)]["source"] for p in parents[309]]]
  kinds=[[pi[(307,p)]["kind"] for p in parents[307]],[pi[(309,p)]["kind"] for p in parents[309]]]
  map_ordinals=[[pi[(307,p)]["map_ordinal"] for p in parents[307]],[pi[(309,p)]["map_ordinal"] for p in parents[309]]]
  out.append({"pair":[307,309],"stable_names":["middle_lobe_right","superior_lobe_right"],"face_rows":[fa,fb],
   "source_face_ids":ids,"patch_kinds":kinds,"orientation_relation":"opposite_winding","source_parent_relation":"two_parent_inferred_reference_diagonal_flip",
   "mapped_output_triangle_vertices_bits":[bits(final[307],final[307]["faces"][fa]),bits(final[309],final[309]["faces"][fb])],
   "parent_face_rows_1078":[parents[307],parents[309]],"parent_face_rows_1055":[[pi[(307,p)]["row1055"] for p in parents[307]],[pi[(309,p)]["row1055"] for p in parents[309]]],
   "source_face_ids_1055":ids,"patch_kinds_1055":kinds,"source_map_record_ordinal_1055":map_ordinals,
   "source_parent_face_vertices_bits_1055":[[pi[(307,p)]["sourcebits"] for p in parents[307]],[pi[(309,p)]["sourcebits"] for p in parents[309]]],
   "parent_face_vertices_bits_1078":[[bits(base[307],base[307]["faces"][p]) for p in parents[307]],[bits(base[309],base[309]["faces"][p]) for p in parents[309]]],"operation_ref":op})
  counts[(307,309)]+=1
 out.sort(key=lambda x:(x["pair"],x["face_rows"]))
 return out,counts
def edge_map(maps,rows):
 result=[];summary={};allpairs=[(a,b) for a in range(305,310) for b in range(a+1,310)]
 for pair in allpairs:
  group=[(i,x) for i,x in enumerate(maps) if tuple(x["pair"])==pair]
  if not group:
   summary[f"{pair[0]}-{pair[1]}"]={"face_pair_count":0,"edge_count":0,"boundary_edge_count_a":0,"boundary_edge_count_b":0,"boundary_sets_equal":True,"explicit_zero_no_map":True};continue
  sides=[collections.defaultdict(list),collections.defaultdict(list)]
  for mi,m in group:
   for j,sid in enumerate(pair):
    fi=int(m["face_rows"][j]);t=ori(rows[sid],rows[sid]["faces"][fi])
    for k in range(3):
     d=(t[k],t[(k+1)%3]);ek=tuple(sorted(d))
     sides[j][ek].append({"face_map_row":mi,"face_row":fi,"source_face_ids":m["source_face_ids"][j],"patch_kinds":m["patch_kinds"][j],"directed_endpoint_bits":[[f"{b:02x}" for b in d[0]],[f"{b:02x}" for b in d[1]]]})
  if set(sides[0])!=set(sides[1]):raise ValueError(f"edge sets differ {pair}")
  ba={e for e,x in sides[0].items() if len(x)==1};bb={e for e,x in sides[1].items() if len(x)==1}
  if ba!=bb:raise ValueError(f"boundary sets differ {pair}")
  for e in sorted(sides[0]):
   a,b=sides[0][e],sides[1][e]
   if len(a)!=len(b) or len(a) not in (1,2):raise ValueError(f"edge incidence invalid {pair}")
   for inc in a:
    mate=next((x for x in b if x["face_map_row"]==inc["face_map_row"]),None)
    if mate is None:raise ValueError("edge incidence missing matching face")
    da=tuple(bytes.fromhex("".join(q)) for q in inc["directed_endpoint_bits"]);db=tuple(bytes.fromhex("".join(q)) for q in mate["directed_endpoint_bits"])
    if da!=(db[1],db[0]):raise ValueError("mapped edge direction mismatch")
   result.append({"pair":list(pair),"endpoint_coordinate_bits":[[f"{q:02x}" for q in e[0]],[f"{q:02x}" for q in e[1]]],"incidences_a":a,"incidences_b":b,"classification":"boundary" if len(a)==1 else "interior","paired_face_edge_directions_opposed":True})
  summary[f"{pair[0]}-{pair[1]}"]={"face_pair_count":len(group),"edge_count":len(sides[0]),"boundary_edge_count_a":len(ba),"boundary_edge_count_b":len(bb),"boundary_sets_equal":True,"explicit_zero_no_map":False}
 return result,summary
def copy_sidecars(rec,dest,source):
 cf=rec["provenance"]["cardiac_geometry_binding"]["common_field"]
 for k in ("map","polynomials","domain_boxes"):
  x=cf[k];src=source/x["path"]
  if sha(src)!=x["sha256"]:raise ValueError("sidecar pin mismatch")
  dst=dest/pathlib.Path(x["path"]).name
  if not dst.exists():os.link(src,dst)
  x["path"]=dst.name
def update_bind(rec,payload,rows,base_receipt):
 raw=payload.read_bytes();s=sha(payload);hd=HEADER.unpack_from(raw)
 rec["payload"].update(path=str(payload),sha256=s,vertex_count=int(hd[3]),index_count=int(hd[4]))
 rec["functional_bindings"]["anatomy_payload_sha256"]=s
 rec["provenance"]["cardiac_geometry_binding"]["common_field"]["anatomy_payload_sha256"]=s
 rec["functional_bindings"]["respiratory_geometry_binding"]["source_refinement_area_update"]["candidate_payload_sha256"]=s
 total,vols=_source_volume_rows(rows);old=base_receipt["thorax_source_volume_m3"]
 rec["thorax_source_volume_m3"]={"interpretation":"sum of five registered lung-envelope absolute signed tetrahedral volumes","five_lung_envelopes":[x["enclosed_volume_m3"] for x in vols],"sum":total,
  "candidate_geometry_derivation":{"basis":"signed tetrahedral volume on serialized Float32 lobe positions","payload_sha256":s,"geometry_byte_order":"little-endian float32 vertices6 followed by little-endian int64 local faces",
  "row_geometry_byte_sha256":{str(i):geom(rows[i]) for i in range(305,310)},"per_lobe":vols,"input_lobe_volumes_m3":old["five_lung_envelopes"],"aggregate_signed_volume_delta_m3":total-old["sum"]}}
 return total,vols
def main():
 if OUT.exists():raise FileExistsError(str(OUT))
 OUT.mkdir(parents=True)
 for p,h in PINS.items():
  if sha(p)!=h:raise ValueError("input hash mismatch "+str(p))
 (OUT/"input-pins.json").write_text(json.dumps({str(p):h for p,h in PINS.items()},indent=2,sort_keys=True)+"\n")
 raw=NHA.read_bytes();head,base=parse_payload(NHA);rows=copy.deepcopy(base);order=[int(RECORD.unpack_from(raw,HEADER.size+i*RECORD.size)[5]) for i in range(int(head[2]))]
 rec0=read(REC);rec=copy.deepcopy(rec0)
 z306=np.load(C306);l306=read(L306);ck306=replay_collapse(base[306],z306["row306_vertices6"],z306["row306_faces"],l306,306)
 rows[306]["vertices6"]=z306["row306_vertices6"].copy();rows[306]["faces"]=z306["row306_faces"].copy()
 zf=np.load(FLIP);lf=read(FLIPLED);flipchecks={}
 for sid in (307,309):
  ch=lf["modified_faces"][str(sid)];v=zf[f"row{sid}_vertices6"];f=zf[f"row{sid}_faces"];expect=base[sid]["faces"].copy()
  if not np.array_equal(v[:,:3].view(np.uint32),base[sid]["vertices6"][:,:3].view(np.uint32)):raise ValueError("1083 changed XYZ")
  if not np.array_equal(expect[ch["face_rows"]],np.asarray(ch["before_faces"],dtype=np.int64)):raise ValueError("1083 parent faces mismatch")
  expect[ch["face_rows"]]=np.asarray(ch["after_faces"],dtype=np.int64)
  if not np.array_equal(expect,f):raise ValueError("1083 changed unrelated face")
  rows[sid]["vertices6"]=v.copy();rows[sid]["faces"]=f.copy();flipchecks[str(sid)]={"xyz_exact":True,"modified_rows":ch["face_rows"],"other_faces_exact":True}
 z308=np.load(C308);r308=read(R308);z308base=np.load(C308BASE)
 # Stage input is 1078 row308 plus the selected306 collapse on row310.
 stage_rows={308:{"vertices6":base[308]["vertices6"].copy(),"faces":base[308]["faces"].copy()},
             310:{"vertices6":z306["row310_vertices6"].copy(),"faces":z306["row310_faces"].copy()}}
 def replay_named_collapse(row,drop_key,keep_key,sid,ordinal):
  v=row["vertices6"];f=row["faces"];drop=ixpoint(v,drop_key,f"1105 stage{ordinal} row{sid} drop");keep=ixpoint(v,keep_key,f"1105 stage{ordinal} row{sid} keep")
  if drop==keep:raise ValueError("collapse drop and keep alias")
  newv=np.delete(v,drop,axis=0);oldf=f.copy();oldf[oldf==drop]=keep;deg=(oldf[:,0]==oldf[:,1])|(oldf[:,0]==oldf[:,2])|(oldf[:,1]==oldf[:,2]);newf=oldf[~deg].copy();newkeep=keep-(drop<keep)
  newf=np.where(newf==keep,newkeep,np.where(newf>drop,newf-1,newf))
  newv[:,3:6]=normals(newv[:,:3].astype(np.float64),newf).astype(newv.dtype)
  return {"vertices6":newv,"faces":newf},{"drop_index":drop,"keep_index":keep,"removed_face_rows":np.flatnonzero(deg).tolist()}
 collapse_checks=[]
 for ordinal,oprow in enumerate(r308["operations"],start=1):
  checkpoint=np.load(oprow["checkpoint"]["path"])
  if sha(oprow["checkpoint"]["path"])!=oprow["checkpoint"]["sha256"]:raise ValueError(f"stage{ordinal} checkpoint hash mismatch")
  stage_check={}
  for sid in (308,310):
   stage_rows[sid],detail=replay_named_collapse(stage_rows[sid],oprow["drop_key_f32"],oprow["keep_key_f32"],sid,ordinal)
   if not np.array_equal(stage_rows[sid]["vertices6"],checkpoint[f"row{sid}_vertices6"]) or not np.array_equal(stage_rows[sid]["faces"],checkpoint[f"row{sid}_faces"]):
    raise ValueError(f"1105 exact stage{ordinal} replay differs row{sid} checkpoint")
   stage_check[str(sid)]=detail
  collapse_checks.append({"ordinal":ordinal,"checkpoint":oprow["checkpoint"],"rows":stage_check,"exact_xyz_faces_recomputed_normals":True})
 for sid in (308,310):
  if not np.array_equal(stage_rows[sid]["vertices6"],z308[f"row{sid}_vertices6"]) or not np.array_equal(stage_rows[sid]["faces"],z308[f"row{sid}_faces"]):raise ValueError(f"1105 final collapse replay mismatch row{sid}")
  rows[sid]["vertices6"]=stage_rows[sid]["vertices6"].copy();rows[sid]["faces"]=stage_rows[sid]["faces"].copy()
 # Verify the recorded CSR ancestry covers every resulting face.
 parent_coverage={}
 for sid in (308,310):
  child=np.asarray(z308[f"row{sid}_parent_children"],dtype=np.int64);n=len(rows[sid]["faces"])
  if set(map(int,child))!=set(range(n)):raise ValueError(f"1105 parent coverage incomplete row{sid}")
  parent_coverage[str(sid)]={"output_face_count":n,"covered_face_count":len(child),"synthetic_stage_parent_count":int(r308["final_candidate"]["parent_map_summary"][str(sid)]["synthetic_stage_parent_count"])}
 # Replay the selected shared-star point move against the merged rows.
 prepared={sid:(rows[sid]["vertices6"].copy(),rows[sid]["faces"].copy(),list(range(len(rows[sid]["faces"]))),{}) for sid in rows}
 point=[-0.0016818622825667262,0.04442258179187775,-0.06169665977358818];direction=[0.6257451256349068,0.7024334754842433,-0.33916109780350956]
 star_result=open_registered_shared_vertex_star(prepared,point,direction,5e-7,maximum_source_displacement_m=1.01e-6,
  minimum_altitude_m=1.2e-6,minimum_result_altitude_m=1.25e-7,allow_local_minimum_regression=False,maximum_volume_change_m3=1e-12,
  volume_bounded_surface_ids=tuple(range(305,312)),volume_bounded_surface_groups=((305,306,307,308,309),))
 star_before={sid:rows[sid]["vertices6"][:,:3].copy() for sid in rows}
 star_faces_before={sid:rows[sid]["faces"].copy() for sid in rows}
 for sid in rows:rows[sid]["vertices6"]=prepared[sid][0];rows[sid]["faces"]=prepared[sid][1]
 zstar=np.load(STAR);star_report=read(STAR_REPORT);expected_point=np.asarray(star_report["trials"][2]["operation"]["new_point"],dtype="<f4")
 if not np.array_equal(np.asarray(star_result["new_point"],dtype="<f4").view(np.uint32),expected_point.view(np.uint32)):
  raise ValueError("owner replay target differs from selected frozen 0.5um trial")
 for sid in rows:
  before=star_before[sid];after=rows[sid]["vertices6"][:,:3]
  if sid in star_result["owner_surface_ids"]:
   oldidx=ixpoint(before,star_result["old_point"],f"star old owner {sid}")
   if not np.array_equal(after[oldidx].view(np.uint32),expected_point.view(np.uint32)):
    raise ValueError(f"owner replay target coordinate differs row{sid}")
   if len(np.flatnonzero(np.all(after==expected_point,axis=1)))!=1:
    raise ValueError(f"replayed new coordinate is not unique row{sid}")
   b=np.delete(before,oldidx,axis=0);a=np.delete(after,oldidx,axis=0)
   if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
    raise ValueError(f"shared-star replay changed other XYZ row{sid}")
  else:
   if not np.array_equal(before.view(np.uint32),after.view(np.uint32)):
    raise ValueError(f"shared-star replay changed unexpected XYZ row{sid}")
  if not np.array_equal(rows[sid]["faces"],star_faces_before[sid]):
   raise ValueError(f"shared-star replay changed faces row{sid}")
 if set(star_result["owner_surface_ids"])!={308,310}:raise ValueError("shared star changed unexpected owners")
 reciprocal=star_result["reciprocal_seam_checks"]
 if len(reciprocal)!=1 or reciprocal[0]["surface_a"]!=308 or reciprocal[0]["surface_b"]!=310 or reciprocal[0]["shared_local_triangle_count"]!=6 or reciprocal[0]["same_winding_triangle_count"]!=6 or reciprocal[0]["opposite_winding_triangle_count"]!=0 or not reciprocal[0]["preserved"]:
  raise ValueError("shared-star local six-face reciprocal seam differs from selected trial")
 star_trial_point_checks={str(sid):{"trial_target_occurrences":int(np.count_nonzero(np.all(zstar[f"row{sid}_vertices6"][:,:3]==expected_point,axis=1))),"replay_target_occurrences":int(np.count_nonzero(np.all(rows[sid]["vertices6"][:,:3]==expected_point,axis=1))),"other_xyz_unchanged":True} for sid in (308,310)}
 if any(v["trial_target_occurrences"]!=1 or v["replay_target_occurrences"]!=1 for v in star_trial_point_checks.values()):raise ValueError("trial/replay star coordinate multiplicity mismatch")
 for sid in base:
  for k in ("body_index","layer","flags"):
   if rows[sid][k]!=base[sid][k]:raise ValueError(f"owner field changed row{sid}/{k}")
 normal_checks={str(sid):verify_normals(rows[sid],sid) for sid in (306,307,308,309,310)}
 avcheck={}
 for sid in range(305,310):
  a0,v0=area_vol(base[sid]);a1,v1=area_vol(rows[sid]);ae=np.float32(a0).tobytes()==np.float32(a1).tobytes();ve=np.float32(v0).tobytes()==np.float32(v1).tobytes()
  avcheck[str(sid)]={"area_f32_equal":ae,"volume_f32_equal":ve,"before_area_m2":a0,"after_area_m2":a1,"before_volume_m3":v0,"after_volume_m3":v1}
  if not ae or not ve:raise ValueError(f"per-lobe F32 area/volume changed {sid}")
 db=(base[311]["vertices6"].tobytes(),base[311]["faces"].tobytes())
 drefresh=_refresh_diaphragm_interface_registration(rec,rows)
 if (rows[311]["vertices6"].tobytes(),rows[311]["faces"].tobytes())!=db:raise ValueError("row311 geometry/order changed")
 # Rebind D faces by exact face geometry; preserve D face order.
 drows=[json.loads(x) for x in DMAP.read_text().splitlines()];ix={s:face_index(rows[s],s) for s in range(305,310)};dmap=[]
 for m in drows:
  sid=int(m["lobe_stable_id"]);oldfi=int(m["l_face_row"]);key=fkey(base[sid],base[sid]["faces"][oldfi]);hits=ix[sid].get(key,[])
  if len(hits)!=1:raise ValueError(f"D registration cannot rebind {sid}:{oldfi}")
  nf=int(hits[0]);df=int(m["d_face_row"])
  if not opposite(ori(rows[311],rows[311]["faces"][df]),ori(rows[sid],rows[sid]["faces"][nf])):raise ValueError("D interface winding changed")
  q=dict(m);q["parent_face_row_1078"]=oldfi;q["l_face_row"]=nf;dmap.append(q)
 if len(dmap)!=47343:raise ValueError("D map row count changed")
 dpath=OUT/"final-exact-D311-to-lobes-map.jsonl";dpath.write_text("".join(json.dumps(q,separators=(",",":"))+"\n" for q in dmap))
 ll,counts=bridge_maps(base,rows,lf,zf);expected={(305,308):8884,(306,307):8727,(306,309):2010,(307,309):9605}
 if dict(counts)!=expected:raise ValueError(f"current L-L map counts {dict(counts)}")
 edges,esummary=edge_map(ll,rows)
 lpath=OUT/"current-reciprocal-face-map-v2.jsonl";lpath.write_text("".join(json.dumps(q,separators=(",",":"))+"\n" for q in ll))
 epath=OUT/"current-reciprocal-edge-map-v2.jsonl";epath.write_text("".join(json.dumps(q,separators=(",",":"))+"\n" for q in edges))
 declarations=[{"pair":[a,b],"face_pair_count":int(counts.get((a,b),0)),"status":"mapped" if counts.get((a,b),0) else "explicit_zero_no_map"} for a in range(305,310) for b in range(a+1,310)]
 if sum(x["face_pair_count"]==0 for x in declarations)!=6:raise ValueError("six zero L-L pairs required")
 def topology_check(row,sid,expected_euler=None,expected_components=None):
  t=analyze_topology(row["vertices6"][:,:3].tolist(),row["faces"].tolist())
  bad={"degenerate_faces":len(t.get("degenerate_face_ids",[])),"duplicate_faces":len(t.get("duplicate_face_ids",[])),
       "nonmanifold_edges":len(t.get("nonmanifold_edges",[])),"orientation_defect_edges":len(t.get("orientation_defect_edges",[])),
       "unused_vertices":len(t.get("unused_vertex_ids",[])),"vertex_manifold_defects":len(t.get("vertex_manifold_defect_ids",[])),
       "repeated_vertex_faces":len(t.get("repeated_vertex_face_ids",[]))}
  if not t.get("closed_oriented_manifold_candidate") or t.get("boundary_edge_count")!=0 or any(bad.values()):
   raise ValueError(f"closed/oriented/topology defects row{sid}: {bad}")
  if expected_euler is not None and t.get("euler_characteristic")!=expected_euler:raise ValueError(f"unexpected Euler characteristic row{sid}: {t.get('euler_characteristic')}")
  if expected_components is not None and t.get("face_component_count")!=expected_components:raise ValueError(f"unexpected component count row{sid}: {t.get('face_component_count')}")
  return {"vertex_count":t.get("vertex_count"),"face_count":t.get("face_count"),"edge_count":t.get("edge_count"),
          "euler_characteristic":t.get("euler_characteristic"),"face_component_count":t.get("face_component_count"),
          "boundary_edge_count":t.get("boundary_edge_count"),"closed_oriented_manifold_candidate":t.get("closed_oriented_manifold_candidate"),"defects":bad}
 topo={}
 for sid in range(305,310):topo[str(sid)]=topology_check(rows[sid],sid,2,1)
 topo["311"]=topology_check(rows[311],311,-2,1)
 pre=OUT/"pre-pleura";final=OUT/"final";pre.mkdir();final.mkdir();prenha=pre/"resting-thorax.nhanatomy"
 rawout=_serialize_payload(head,order,rows);prenharaw=rawout;prenhr=prenharaw;prenharaw and prenha.write_bytes(prenharaw);preh=sha(prenha)
 rec["provenance"]["retained_1078_lung_compose"]={"status":"historical input","payload":meta(NHA),"prior_record":rec["provenance"].get("lung_final_shared_interface_rebuild")}
 rec["provenance"]["lung_final_shared_interface_rebuild"]={"status":"provisional composition dry-run; native pending","base_1078_payload":meta(NHA),
  "operation_306":{"npz":meta(C306),"ledger":meta(L306),"replay":ck306},"operation_1083":{"npz":meta(FLIP),"ledger":meta(FLIPLED),"checks":flipchecks},
  "operation_308_first_cluster":{"row308_input_1078":meta(NHA),"row310_input_selected306":meta(C306),"prior_1099_reference_trial":meta(C308BASE),"selected306_row310_exact_match":True,"npz":meta(C308),"report":meta(R308),"stages":collapse_checks,"parent_coverage":parent_coverage},
  "operation_308_second_cluster":{"npz":meta(STAR),"report":meta(STAR_REPORT),"owner":meta(OWNER),"replayed_result":star_result},
  "d_registration_refresh":drefresh,"d_map_path":str(dpath),"d_map_sha256":sha(dpath),"ll_map_path":str(lpath),"ll_map_sha256":sha(lpath),
  "ll_edge_path":str(epath),"ll_edge_sha256":sha(epath),"qualification_limit":"Provisional composition; full exact scans and native accepted-cycle qualification remain pending."}
 rec.setdefault("qualification",{})["source_geometry_candidate"]="Provisional 1078-based selected-patch dry-run. Not a final asset."
 rec["payload"].update(path=str(prenha),sha256=preh,vertex_count=int(HEADER.unpack_from(prenharaw)[3]),index_count=int(HEADER.unpack_from(prenharaw)[4]))
 rec["functional_bindings"]["anatomy_payload_sha256"]=preh;rec["provenance"]["cardiac_geometry_binding"]["common_field"]["anatomy_payload_sha256"]=preh
 rec["functional_bindings"]["respiratory_geometry_binding"]["source_refinement_area_update"]["candidate_payload_sha256"]=preh
 total,vols=_source_volume_rows(rows);ov=read(REC)["thorax_source_volume_m3"]
 rec["thorax_source_volume_m3"]={"interpretation":"sum of five registered lung-envelope absolute signed tetrahedral volumes","five_lung_envelopes":[x["enclosed_volume_m3"] for x in vols],"sum":total,
 "candidate_geometry_derivation":{"basis":"signed tetrahedral volume on serialized Float32 lobe positions","payload_sha256":preh,"geometry_byte_order":"little-endian float32 vertices6 followed by little-endian int64 local faces","row_geometry_byte_sha256":{str(s):geom(rows[s]) for s in range(305,310)},"per_lobe":vols,"input_lobe_volumes_m3":ov["five_lung_envelopes"],"aggregate_signed_volume_delta_m3":total-ov["sum"]}}
 for k in ("map","polynomials","domain_boxes"):
  q=rec["provenance"]["cardiac_geometry_binding"]["common_field"][k];src=BASE/q["path"]
  if sha(src)!=q["sha256"]:raise ValueError("cardiac sidecar hash mismatch")
  os.link(src,pre/pathlib.Path(q["path"]).name);q["path"]=pathlib.Path(q["path"]).name
 pr=pre/"resting-anatomy-receipt.json";write(pr,rec)
 pm=read(MAN);pm["payload"].update(rec["payload"]);pm["receipt"].update(path=str(pr),sha256=sha(pr));pm["functional_bindings"]=copy.deepcopy(rec["functional_bindings"]);pm["qualification"]=copy.deepcopy(rec.get("qualification",{}));pm["thorax_source_volume_m3"]=copy.deepcopy(rec["thorax_source_volume_m3"])
 write(pre/"resting-anatomy-manifest.json",pm)
 area64,arows=_derive_basal_effective_area(rows,_load_pinned_respiratory_owner(RESP).kuhn_basis);area32=float(np.float32(area64));cfg=read(CFG)
 if np.float32(area32).tobytes()!=np.float32(cfg["diaphragm_area_m2"]).tobytes():raise ValueError("runtime effective area changed; do not adjust config")
 pleura=build_pleura(prenha,pr,final);fnha=final/"resting-thorax.nhanatomy";frp=final/"resting-anatomy-receipt.json";fr=read(frp);_,frs=parse_payload(fnha)
 final_topology={str(sid):topology_check(frs[sid],sid,(-32 if sid==310 else -2 if sid==311 else 2),(2 if sid==310 else 1)) for sid in (305,306,307,308,309,310,311)}
 final_normal_checks={str(sid):verify_normals(frs[sid],sid) for sid in (305,306,307,308,309,310,311)}
 for sid in rows:
  if sid!=310 and (not np.array_equal(frs[sid]["vertices6"],rows[sid]["vertices6"]) or not np.array_equal(frs[sid]["faces"],rows[sid]["faces"])):raise ValueError(f"pleura changed row{sid}")
 for sid in range(305,310):
  a0,v0=area_vol(base[sid]);a1,v1=area_vol(frs[sid])
  if np.float32(a0).tobytes()!=np.float32(a1).tobytes() or np.float32(v0).tobytes()!=np.float32(v1).tobytes():raise ValueError(f"post-pleura lobe change {sid}")
 lp=OUT/"row310-face-lineage.npy";np.save(lp,_pleura_face_lineage(frs),allow_pickle=False)
 for k in ("map","polynomials","domain_boxes"):
  q=fr["provenance"]["cardiac_geometry_binding"]["common_field"][k];src=pre/pathlib.Path(q["path"]).name;os.link(src,final/pathlib.Path(q["path"]).name);q["path"]=pathlib.Path(q["path"]).name
 fsha=sha(fnha);fraw=fnha.read_bytes();fr["payload"].update(path=str(fnha),sha256=fsha,vertex_count=int(HEADER.unpack_from(fraw)[3]),index_count=int(HEADER.unpack_from(fraw)[4]))
 fr["functional_bindings"]["anatomy_payload_sha256"]=fsha;fr["provenance"]["cardiac_geometry_binding"]["common_field"]["anatomy_payload_sha256"]=fsha
 fr["functional_bindings"]["respiratory_geometry_binding"]["source_refinement_area_update"]["candidate_payload_sha256"]=fsha
 fr["provenance"]["lung_final_shared_interface_rebuild"].update(output_payload_sha256=fsha,pre_pleura_payload_sha256=preh,row310_face_lineage_path=str(lp),row310_face_lineage_sha256=sha(lp),pleura_owner_derivation=pleura["derivation"])
 fr["provenance"]["respiratory_configuration"]={"path":str(CFG),"sha256":sha(CFG),"all_other_physical_parameters_unchanged":True,"effective_area_f32_equal_to_parent":True}
 write(frp,fr);fm=copy.deepcopy(pm);fm["payload"].update(fr["payload"]);fm["receipt"].update(path=str(frp),sha256=sha(frp));fm["functional_bindings"]=copy.deepcopy(fr["functional_bindings"]);fm["qualification"]=copy.deepcopy(fr.get("qualification",{}));fm["thorax_source_volume_m3"]=copy.deepcopy(fr["thorax_source_volume_m3"]);write(final/"resting-anatomy-manifest.json",fm)
 bridge_report={"schema":"numi.human.lobe-lobe-source-proven-reciprocal-interface-map.v2","status":"provisional source dry-run only","final_nha":meta(fnha),"parent_1078":meta(NHA),
  "source_1055":{"nha":meta(LLNHA),"report":meta(LLREPORT),"map":meta(LLMAP)},"operation_1083":{"npz":meta(FLIP),"ledger":meta(FLIPLED)},
  "operation_306":{"npz":meta(C306),"ledger":meta(L306)},"operation_308_first":{"npz":meta(C308),"report":meta(R308)},"operation_308_second":{"npz":meta(STAR),"report":meta(STAR_REPORT),"owner":meta(OWNER)},
  "face_map":{"path":str(lpath),"sha256":sha(lpath),"rows":len(ll)},"edge_map":{"path":str(epath),"sha256":sha(epath),"rows":len(edges)},"pairs":declarations,"edge_summary":esummary,
  "limits":["Provisional composition only; no native geometry qualification."]}
 write(OUT/"current-reciprocal-map-report-v2.json",bridge_report)
 report={"schema":"numi.human.final-lung-selected-composition-dryrun-v1","status":"provisional 1105 test case; native and full scans pending",
  "inputs":{str(p):h for p,h in PINS.items()},"outputs":{k:meta(p) for k,p in [("pre_nha",prenha),("pre_receipt",pr),("pre_manifest",pre/"resting-anatomy-manifest.json"),("final_nha",fnha),("final_receipt",frp),("final_manifest",final/"resting-anatomy-manifest.json"),("d_map",dpath),("ll_map",lpath),("ll_edges",epath),("row310_lineage",lp)]},
  "306_replay":ck306,"1083_replay":flipchecks,"308_first_cluster":meta(C308),"308_first_cluster_replay":collapse_checks,"308_parent_coverage":parent_coverage,"owner_normal_checks":normal_checks,"308_second_cluster_owner_replay":star_result,"308_second_cluster_trial_point_checks":star_trial_point_checks,
  "D_registration_refresh":drefresh,"D_map_count":len(dmap),"L_L_counts":{f"{a}-{b}":int(counts.get((a,b),0)) for a in range(305,310) for b in range(a+1,310)},
  "edge_summary":esummary,"area_volume_f32_checks":avcheck,"owner_normal_checks":normal_checks,"topology":topo,"final_topology":final_topology,"final_normal_checks":final_normal_checks,"runtime_area_f32_unchanged":True,"area64":area64,"area32":area32,"pleura":pleura["derivation"]}
 write(OUT/"composition-report.json",report)
 sums=OUT/"SHA256SUMS";sums.write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in sorted(OUT.rglob("*")) if p.is_file() and p!=sums))
 print(json.dumps({"outputs":report["outputs"],"L_L_counts":report["L_L_counts"],"star":star_result,"star_trial_point_checks":star_trial_point_checks,"report_sha256":sha(OUT/"composition-report.json")},indent=2))
if __name__=="__main__":main()
