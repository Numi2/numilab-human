from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import csv, hashlib, importlib.util, json, sys, time, traceback
import numpy as np

P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
A=P/"reduced-physical-boundary-400/attempt-038-connected-minkowski-counterfactual-023mm"
IN=P/"reduced-physical-boundary-400/attempt-024-coupled-paired-thickness"
RUN=A/"run-005-protected-cap-minthickness"
OUT=RUN/"output"
BASELINE=A/"run-003-locality/baseline-delta-report.json"
CAND=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz"
MASS=P/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
O_INPUT=IN/"source002-O-closed-f32-as-f64.npz"
I_INPUT=IN/"source002-I-closed-f32-as-f64.npz"
D_INPUT=P/"geometry-worker/matched-nesting-390/run-038-run037-fat-occupancy/closed-inputs/D-closed-run037-f32.npz"
HELPER=P/"geometry-worker/fixed-physical-O-local-nested-001/build_union_only_local.py"
OWNER=P/"geometry-worker/fixed-physical-O-local-nested-001/build_union_only_local.py"
EXACT_ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
EXACT_OWNER=EXACT_ROOT/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
EXACT_BASE=EXACT_ROOT/"incremental-exact-audit-owner-253/src"
OUT_RECEIPT=OUT/"receipt.json"
R_M=0.0023
CAP_MARGIN_M=50e-6
MIN_SKIN_M=0.0009
KSEG=16
ID={"cap_guard":60100401,"cap_guard_kernel":60100402,"skin_min_kernel":60100403,
    "outer_safe":60100411,"inner_safe":60100412,"inner_clipped":60100413}
if RUN.exists(): raise FileExistsError(f"refusing to overwrite {RUN}")
RUN.mkdir(parents=True)
OUT.mkdir()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n")
def load(path):
    return np.load(path,allow_pickle=False)
def stage(m,path):
    z=m.to_mesh64();pp=np.asarray(z.vert_properties,np.float64)[:,:3];ff=np.asarray(z.tri_verts,np.int64)
    np.savez_compressed(path,points_world_m_f64=pp,faces_f64=ff,
        run_index=np.asarray(z.run_index,np.uint64),run_original_id=np.asarray(z.run_original_id,np.uint32),
        face_id=np.asarray(z.face_id,np.uint64),status=np.asarray([str(m.status())]),volume_m3=np.asarray([float(m.volume())]))
    return {"path":str(path),"sha256":sha(path),"vertices":len(pp),"faces":len(ff),"volume_m3":float(m.volume()),"status":str(m.status())}
def load_stage(path,mf):
    z=load(path);pp=np.array(z["points_world_m_f64"],np.float64,order="C",copy=True);ff=np.array(z["faces_f64"],np.uint64,order="C",copy=True)
    m=mf.Manifold(mf.Mesh64(vert_properties=pp,tri_verts=ff,run_index=np.array(z["run_index"],np.uint64),
      run_original_id=np.array(z["run_original_id"],np.uint32),face_id=np.array(z["face_id"],np.uint64)))
    if m.status()!=mf.Error.NoError:raise RuntimeError(f"invalid stage {path}: {m.status()}")
    return m,pp,np.asarray(z["faces_f64"],np.int64),z
def topology(Pp,Ff):
    Pp=np.asarray(Pp,np.float64);Ff=np.asarray(Ff,np.int64);tri=Pp[Ff]
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    E=np.concatenate((Ff[:,[0,1]],Ff[:,[1,2]],Ff[:,[2,0]]),axis=0)
    _,cnt=np.unique(np.sort(E,axis=1),axis=0,return_counts=True)
    q=tri-(tri[0,0] if len(tri) else np.zeros(3))
    sv=float(np.einsum("ij,ij->i",q[:,0],np.cross(q[:,1],q[:,2])).sum()/6) if len(tri) else 0.0
    return {"vertices":len(Pp),"faces":len(Ff),"used_vertices":int(len(np.unique(Ff))),
      "boundary_edges":int(np.count_nonzero(cnt==1)),"nonmanifold_edges":int(np.count_nonzero(cnt>2)),
      "zero_area_faces":int(np.count_nonzero(area2==0)),"min_double_area_m2":float(area2.min()) if len(area2) else None,
      "signed_volume_m3_local_origin":sv}
def source_closed(path,role,mf,h):
    z=load(path);pp=np.asarray(z["points_world_m_f64"],np.float64);ff=np.asarray(z["faces_f64"],np.int64)
    sids=np.asarray(z["run_original_id"],np.int64);fids=np.asarray(z["face_id"],np.int64)
    if len(sids)!=1:raise RuntimeError(f"{role}: expected single source ID")
    sid=int(sids[0]);m,_=h.make_mesh(pp,ff,face_ids=fids.astype(np.uint64),source_id=sid)
    if m.status()!=mf.Error.NoError:raise RuntimeError(f"{role} source import {m.status()}")
    capids=np.asarray(z["cap_face_ids"],np.int64)
    return {"role":role,"P":pp,"F":ff,"fid":fids,"sid":sid,"cap_ids":capids,
      "path":str(path),"sha256":sha(path),"mesh":m,"z":z}
def cap_components(base):
    caprows=np.flatnonzero(np.isin(base["fid"],base["cap_ids"]))
    edgefaces=defaultdict(list)
    for local,row in enumerate(caprows):
        a,b,c=base["F"][row]
        for e in ((a,b),(b,c),(c,a)):edgefaces[tuple(sorted(map(int,e)))].append(local)
    if any(len(v)>2 for v in edgefaces.values()):raise RuntimeError(f"{base['role']}: cap has nonmanifold edge")
    adj=[set() for _ in caprows]
    for inc in edgefaces.values():
        for q in inc:
            adj[q].update(x for x in inc if x!=q)
    seen=set();out=[]
    for q in range(len(caprows)):
        if q in seen:continue
        stack=[q];seen.add(q);comp=[]
        while stack:
            x=stack.pop();comp.append(x)
            for y in adj[x]:
                if y not in seen:seen.add(y);stack.append(y)
        out.append({"face_rows":caprows[np.asarray(comp,np.int64)],"vertices":np.unique(base["F"][caprows[np.asarray(comp,np.int64)]])})
    if sorted(len(x["face_rows"]) for x in out)!=[19,22]:raise RuntimeError(f"{base['role']}: unexpected cap components {[len(x['face_rows']) for x in out]}")
    return out
def exact_pairs(exact,P32,F,label):
    rec,_=exact.float32_surface_records(np.asarray(P32,np.float32),np.asarray(F,np.int64),source_seam_groups=())
    r=exact._audit_pair(rec,rec,same_surface=True);pp=np.asarray(r.get("triangle_pairs",[]),np.int64).reshape(-1,2)
    path=OUT/f"{label}-self-pairs.npy"
    if len(pp):np.save(path,pp)
    return {"count":int(r.get("count",len(pp))),"aabb_candidate_pairs":int(r.get("aabb_candidate_pairs",0)),
      "allowed_shared_vertex_or_edge_pairs":int(r.get("allowed_shared_vertex_or_edge_pairs",0)),
      "path":str(path) if len(pp) else None,"sha256":sha(path) if len(pp) else None}
def exact_cross(exact,PA,FA,PB,FB,label):
    ra,_=exact.float32_surface_records(np.asarray(PA,np.float32),np.asarray(FA,np.int64),source_seam_groups=())
    rb,_=exact.float32_surface_records(np.asarray(PB,np.float32),np.asarray(FB,np.int64),source_seam_groups=())
    r=exact._audit_pair(ra,rb,same_surface=False);pp=np.asarray(r.get("triangle_pairs",[]),np.int64).reshape(-1,2)
    path=OUT/f"{label}-pairs.npy"
    if len(pp):np.save(path,pp)
    return {"count":int(r.get("count",len(pp))),"aabb_candidate_pairs":int(r.get("aabb_candidate_pairs",0)),
      "allowed_shared_vertex_or_edge_pairs":int(r.get("allowed_shared_vertex_or_edge_pairs",0)),
      "path":str(path) if len(pp) else None,"sha256":sha(path) if len(pp) else None}
def boolop(h,mf,a,b,op,label):
    m=h.boolean(a,b,op,label)
    if m.status()!=mf.Error.NoError:raise RuntimeError(f"{label}: {m.status()}")
    return m
def output_layer(role,manifold,base,support,capmask,source32,h,mf,emit):
    # Rewrap all generated support triangles under one explicitly operation-derived source ID.
    sz=support.to_mesh64();SP=np.asarray(sz.vert_properties,np.float64)[:,:3];SF=np.asarray(sz.tri_verts,np.int64)
    sm,sid=h.make_mesh(SP,SF,face_ids=np.arange(len(SF),dtype=np.uint64))
    derived={"role":role,"source_id":int(sid),"face_count":len(SF),"identity0":np.arange(len(SF),dtype=np.int64),
      "identity1":np.full(len(SF),-1,np.int64),"identity_keys":[f"{role}_protected_support_face_row","no_unique_parent_face"],
      "path":str(RUN/f"{role}-support-derived.npz"),"sha256":None}
    source={"role":role,"source_id":base["sid"],"face_count":int(base["fid"].max(initial=-1))+1,
      "identity0":np.arange(int(base["fid"].max(initial=-1))+1,dtype=np.int64),
      "identity1":np.full(int(base["fid"].max(initial=-1))+1,-1,np.int64),
      "identity_keys":[f"source002_{role}_closed_face_id","none"],"path":base["path"],"sha256":base["sha256"]}
    sources=[source,derived]
    src32=np.asarray(source32,np.float32)
    prep=h.prepare_mesh_output(role,manifold,sources,{},src32,OUT,[],report)
    Y,F32,em=emit(prep["P64"],prep["PF64"],src32)
    if len(prep["PF64"])!=len(prep["P64"]):pass
    orig=np.asarray(prep["orig"],np.int64);fid=np.asarray(prep["fid"],np.int64)
    c_mask=(orig==base["sid"])&np.isin(fid,base["cap_ids"])
    path=OUT/f"{role}-protected-cap-f32.npz"
    np.savez_compressed(path,points_world_m_f64=prep["P64"],faces_f64=prep["F64"],
      points_world_m_f32=Y,faces_f32=F32,face_source_original_id=orig,face_source_face_id=fid,
      face_source_layer=np.asarray(prep["face_role"],np.int8),face_source_identity0=np.asarray(prep["identity0"],np.int64),
      face_source_identity1=np.asarray(prep["identity1"],np.int64),temporary_cap_descendant_face_mask=c_mask,
      face_source_identity_schema_by_original_id_json=np.asarray([json.dumps(prep["source_schema"],sort_keys=True)]),
      float32_emission_json=np.asarray([json.dumps(em,separators=(",",":"))]))
    return {"path":str(path),"sha256":sha(path),"P64":prep["P64"],"F64":prep["F64"],"P32":Y,"F32":F32,
      "orig":orig,"fid":fid,"capmask":c_mask,"emission":em,"topology":topology(Y,F32),"source_schema":prep["source_schema"]}
def source_identity_check(base,out,support_bbox):
    # Each outside-support source face must survive as the same oriented coordinate triangle under its pinned source face ID.
    tri=base["P"][base["F"]];lo,hi=support_bbox
    outside=~(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
    byid=defaultdict(list)
    for q in np.flatnonzero(out["orig"]==base["sid"]):byid[int(out["fid"][q])].append(int(q))
    counts={"outside_face_count":int(outside.sum()),"exact_oriented_descendants":0,"missing":0,"ambiguous":0}
    missing=[]
    for row in np.flatnonzero(outside):
        faceid=int(base["fid"][row]);hits=byid.get(faceid,[])
        if not hits:
            counts["missing"]+=1;missing.append(faceid);continue
        if len(hits)!=1:
            counts["ambiguous"]+=1;missing.append(faceid);continue
        q=hits[0];expected=base["P"][base["F"][row]];actual=out["P64"][out["F64"][q]]
        e=[tuple(x) for x in expected];a=[tuple(x) for x in actual]
        if set(e)==set(a) and a in (e,e[1:]+e[:1],e[2:]+e[:2]):
            counts["exact_oriented_descendants"]+=1
        else:
            counts["missing"]+=1;missing.append(faceid)
    counts["missing_face_ids_sample"]=missing[:20]
    return counts
def cap_identity_check(base,out):
    rows=np.flatnonzero(np.isin(base["fid"],base["cap_ids"]))
    byid=defaultdict(list)
    for q in np.flatnonzero(out["orig"]==base["sid"]):byid[int(out["fid"][q])].append(int(q))
    exact=0;amb=0;missing=[]
    for r in rows:
        fid=int(base["fid"][r]);hits=byid.get(fid,[])
        if len(hits)!=1:missing.append(fid);amb+=int(len(hits)>1);continue
        q=hits[0];a=[tuple(x) for x in base["P"][base["F"][r]]];b=[tuple(x) for x in out["P64"][out["F64"][q]]]
        if set(a)==set(b) and b in (a,a[1:]+a[:1],a[2:]+a[:2]):exact+=1
        else:missing.append(fid)
    capverts=np.unique(base["F"][rows]);coords=base["P"][capverts]
    present={np.asarray(v,dtype="<f8").tobytes() for v in out["P64"]}
    ring=sum(np.asarray(v,dtype="<f8").tobytes() in present for v in coords)
    return {"input_cap_faces":len(rows),"exact_oriented_source_cap_faces":exact,"missing_or_ambiguous_face_ids":missing,
      "input_cap_ring_vertices":len(coords),"exact_ring_vertices_found":int(ring),
      "pass":exact==len(rows) and ring==len(coords) and not missing}
def load_parent_mass():
    with MASS.open(newline="") as f:rows=list(csv.DictReader(f))
    return {int(x["source_tet_row"]):float(x["mass_kg"]) for x in rows}
def local_capacity(local_skin,local_fat,C,parents,mass_lookup):
    M=np.asarray(C["material_index"],np.int64);D=np.asarray(C["geometric_donor_body"],np.int64);Rgn=np.asarray(C["anatomical_region_id"],np.int64)
    pr=np.asarray(C["child_parent_tet_row"],np.int64)
    out={}
    for name,mat,solid,bounds in (("skin23",15,local_skin,(900.,1100.)),("fat23",16,local_fat,(850.,950.))):
        mask=(M==mat)&(D==23)&(Rgn==23);src=np.unique(pr[mask]);mass=float(sum(mass_lookup[int(x)] for x in src));vol=float(solid.volume())
        out[name]={"volume_m3":vol,"volume_ml":vol*1e6,"mass_kg":mass,"density_kg_m3":mass/vol if vol>0 else None,
          "density_bounds_kg_m3":list(bounds),"allowed_volume_m3":[mass/bounds[1],mass/bounds[0]],
          "pass":bool(vol>0 and mass/bounds[1]<=vol<=mass/bounds[0]),"parent_tet_rows":int(len(src))}
    return out

t0=time.monotonic()
report={"schema":"numi.source002-protected-cap-minimum-skin-counterfactual.v1","status":"RUNNING","physical_admission":False,
 "construction":{"outer":"Oold UNION (outer_dilation minus common_cap_guard)",
  "inner":"Iold UNION ((inner_dilation minus common_cap_guard) INTERSECT erode(Onew, 0.9mm sphere16))",
  "common_cap_guard":"Union of convex hulls of the four pinned O/I cap vertex components, Minkowski-dilated by sphere16 radius 2.35mm; the exact same guard is subtracted from both cached supports.",
  "minimum_skin_parameter_source":str(OWNER),"minimum_skin_radius_m":MIN_SKIN_M,"minimum_skin_kernel_segments":KSEG,
  "counterfactual_only":True},"inputs":{}}
try:
 if not BASELINE.exists():raise FileNotFoundError(f"baseline/delta report not ready: {BASELINE}")
 bdelta=json.loads(BASELINE.read_text())
 if bdelta.get("status")!="COMPLETE_BASELINE_COUNTERFACTUAL_DELTA":raise RuntimeError(f"baseline/delta gate is not complete: {bdelta.get('status')}")
 report["baseline_delta"]={"path":str(BASELINE),"sha256":sha(BASELINE),"status":bdelta["status"],
   "baseline_open_OI_pairs":bdelta["baseline_exact_open_O_I_cross"]["pair_count"],
   "counter_open_OI_pairs":bdelta["baseline_counterfactual_open_O_I_cross_delta"]["counter_pair_count"],
   "baseline_manifold":bdelta["manifold_f32_boolean_comparison"]["baseline"],
   "counter_manifold":bdelta["manifold_f32_boolean_comparison"]["counterfactual"],
   "counter_minus_baseline_volume_m3":bdelta["manifold_f32_boolean_comparison"]["volume_delta_counter_minus_baseline_m3"]}
 sys.path.insert(0,str(P.parent/"authoring-deps"));import manifold3d as mf
 sys.path.insert(0,str(P.parent));from emit_f32 import emit
 sp=importlib.util.spec_from_file_location("protected_cap_owner_038",HELPER);h=importlib.util.module_from_spec(sp);sys.modules[sp.name]=h;sp.loader.exec_module(h)
 if not EXACT_OWNER.exists():raise FileNotFoundError(EXACT_OWNER)
 sys.path.insert(0,str(EXACT_BASE));sx=importlib.util.spec_from_file_location("numilab_human.protected_cap_exact_038",EXACT_OWNER);exact=importlib.util.module_from_spec(sx);sys.modules[sx.name]=exact;sx.loader.exec_module(exact)
 report["inputs"]={k:{"path":str(path),"sha256":sha(path)} for k,path in {
  "O_source":O_INPUT,"I_source":I_INPUT,"D037":D_INPUT,"counter_O":A/"run-002/output/O-closed-counterfactual-f32.npz",
  "counter_I":A/"run-002/output/I-closed-counterfactual-f32.npz","outer_support":A/"run-002/output/outer-region-dilation-f64-stage-f64.npz",
  "inner_support":A/"run-002/output/inner-region-dilation-f64-stage-f64.npz","source002_candidate":CAND,"parent_mass_csv":MASS,
  "geometry_owner":OWNER,"exact_owner":EXACT_OWNER}.items()}
 O=source_closed(O_INPUT,"O",mf,h);I=source_closed(I_INPUT,"I",mf,h)
 outer,outerP,outerF,outerz=load_stage(A/"run-002/output/outer-region-dilation-f64-stage-f64.npz",mf)
 inner,innerP,innerF,innerz=load_stage(A/"run-002/output/inner-region-dilation-f64-stage-f64.npz",mf)
 Dz=load(D_INPUT);DP=np.asarray(Dz["points_world_m_f64"],np.float64);DF=np.asarray(Dz["faces_f64"],np.int64)
 Dmesh,_=h.make_mesh(DP,DF,face_ids=np.arange(len(DF),dtype=np.uint64))
 if Dmesh.status()!=mf.Error.NoError:raise RuntimeError(f"D037 source import {Dmesh.status()}")
 D={"mesh":Dmesh}
 # Shared cap protection: use one positive-volume convex solid for each actual cap component.
 cap_parts=[];cap_records=[]
 for base in (O,I):
  for comp_i,comp in enumerate(cap_components(base)):
   Q=base["P"][comp["vertices"]]
   hull=mf.Manifold.hull_points(np.ascontiguousarray(Q,np.float64))
   if hull.status()!=mf.Error.NoError or hull.is_empty() or hull.volume()<=0:
    raise RuntimeError(f"{base['role']} cap component {comp_i}: invalid positive-volume hull {hull.status()} vol={hull.volume()}")
   cap_parts.append(hull)
   cap_records.append({"role":base["role"],"component":comp_i,"face_rows":comp["face_rows"].tolist(),
      "vertex_rows":comp["vertices"].tolist(),"vertex_count":len(Q),"bbox_min_m":Q.min(0).tolist(),
      "bbox_max_m":Q.max(0).tolist(),"hull_volume_m3":float(hull.volume()),"hull_faces":int(hull.num_tri())})
 cap_union=h.add_all(cap_parts,"shared O/I cap protection union")
 if cap_union.status()!=mf.Error.NoError:raise RuntimeError(f"cap hull union {cap_union.status()}")
 cap_kernel=mf.Manifold.sphere(R_M+CAP_MARGIN_M,KSEG)
 cap_guard=cap_union.minkowski_sum(cap_kernel)
 if cap_guard.status()!=mf.Error.NoError:raise RuntimeError(f"shared cap guard Minkowski {cap_guard.status()}")
 report["cap_protection"]={"component_hulls":cap_records,"shared_union":h.metrics(cap_union),
   "guard":h.metrics(cap_guard),"guard_bbox_m":[x.tolist() for x in h.mesh_bbox(cap_guard)],
   "kernel_radius_m":R_M+CAP_MARGIN_M,"kernel_segments":KSEG,"same_guard_used_for_both_roles":True}
 report["checkpoints"]={}
 report["checkpoints"]["shared_cap_guard"]=stage(cap_guard,OUT/"shared-cap-guard-f64.npz")
 write(OUT/"progress.json",report)
 outer_safe=boolop(h,mf,outer,cap_guard,mf.OpType.Subtract,"outer support minus shared cap guard")
 inner_safe=boolop(h,mf,inner,cap_guard,mf.OpType.Subtract,"inner support minus shared cap guard")
 report["checkpoints"]["outer_support_protected"]=stage(outer_safe,OUT/"outer-support-protected-f64.npz")
 report["checkpoints"]["inner_support_protected"]=stage(inner_safe,OUT/"inner-support-protected-f64.npz")
 write(OUT/"progress.json",report)
 def role_support(support,role,parents):
  z=support.to_mesh64();sp=np.asarray(z.vert_properties,np.float64)[:,:3];sf=np.asarray(z.tri_verts,np.int64)
  if not len(sf):raise RuntimeError(f"{role}: protected support is empty")
  m,sid=h.make_mesh(sp,sf,face_ids=np.arange(len(sf),dtype=np.uint64))
  if m.status()!=mf.Error.NoError:raise RuntimeError(f"{role}: support reimport {m.status()}")
  support_path=OUT/f"{role}-added-source.npz"
  np.savez_compressed(support_path,points_world_m_f64=sp,faces_f64=sf,
      parent_run_original_ids=np.asarray(z.run_original_id,np.uint32),
      parent_run_index=np.asarray(z.run_index,np.uint64),parent_face_ids=np.asarray(z.face_id,np.uint64))
  ent={"role":role,"source_id":int(sid),"face_count":len(sf),"identity0":np.arange(len(sf),dtype=np.int64),
    "identity1":np.full(len(sf),-1,np.int64),"identity_keys":[f"{role}_protected_support_face_row","no_unique_parent_face"],
    "derived_operation":"regional_minkowski_support_minus_shared_cap_guard_and_minimum_skin_constraint",
    "derived_parent_role":parents,"derived_parent_source_ids":np.asarray(z.run_original_id,np.int64).astype(int).tolist(),
    "derived_parent_geometry_sha256":h.array_digest(sp,sf),"derived_kernel_sha256":h.manifold_mesh_digest(cap_kernel),
    "derived_radius_m":R_M+CAP_MARGIN_M,"path":str(support_path),"sha256":sha(support_path)}
  return m,ent
 outer_wrap,outer_ent=role_support(outer_safe,"O","source002 donor23 region23 skin+fat dilation minus shared cap guard")
 Onew=h.add_all([O["mesh"],outer_wrap],"O source plus protected regional support")
 if Onew.status()!=mf.Error.NoError:raise RuntimeError(f"O candidate union {Onew.status()}")
 report["checkpoints"]["O_candidate"]=stage(Onew,OUT/"O-candidate-f64.npz")
 write(OUT/"progress.json",report)
 skinK=mf.Manifold.sphere(MIN_SKIN_M,KSEG)
 erodedO=Onew.minkowski_difference(skinK)
 if erodedO.status()!=mf.Error.NoError:raise RuntimeError(f"minimum skin erosion {erodedO.status()}")
 report["checkpoints"]["O_minimum_skin_interior"]=stage(erodedO,OUT/"O-minskin-eroded-f64.npz")
 write(OUT/"progress.json",report)
 inner_clipped=boolop(h,mf,inner_safe,erodedO,mf.OpType.Intersect,"inner support clipped to minimum skin interior")
 if inner_clipped.status()!=mf.Error.NoError:raise RuntimeError(f"inner clip status {inner_clipped.status()}")
 report["checkpoints"]["inner_support_clipped"]=stage(inner_clipped,OUT/"inner-support-clipped-f64.npz")
 clipped_removed=boolop(h,mf,inner_safe,inner_clipped,mf.OpType.Subtract,"inner additions removed by minimum skin clip")
 report["minimum_skin_clip_removed_addition"]=h.metrics(clipped_removed)
 write(OUT/"progress.json",report)
 inner_wrap,inner_ent=role_support(inner_clipped,"I","source002 donor23 region23 fat dilation minus shared cap guard, clipped to 0.9mm interior of O candidate")
 Inew=h.add_all([I["mesh"],inner_wrap],"I source plus thickness-clipped protected support")
 if Inew.status()!=mf.Error.NoError:raise RuntimeError(f"I candidate union {Inew.status()}")
 report["checkpoints"]["I_candidate"]=stage(Inew,OUT/"I-candidate-f64.npz")
 write(OUT/"progress.json",report)
 # Emit complete exact source identities. Retain temporary caps as closed diagnostic faces.
 source_entries=[]
 for base in (O,I):
  source_entries.append({"role":base["role"],"source_id":base["sid"],"face_count":int(base["fid"].max(initial=-1))+1,
    "identity0":np.arange(int(base["fid"].max(initial=-1))+1,dtype=np.int64),
    "identity1":np.full(int(base["fid"].max(initial=-1))+1,-1,np.int64),
    "identity_keys":[f"source002_{base['role']}_closed_face_id","none"],"path":base["path"],"sha256":base["sha256"]})
 # Candidate manifold outputs carry only original role + explicit added role; this avoids shield lineage appearing as boundary.
 def emit_layer(role,base,manifold,added_wrap,added_ent,source32):
  sources=[x for x in source_entries if x["role"]==role]+[added_ent]
  prep=h.prepare_mesh_output(role,manifold,sources,{},np.asarray(source32,np.float32),OUT,[],report)
  Y,F32,em=emit(prep["P64"],prep["PF64"],np.asarray(source32,np.float32))
  orig=np.asarray(prep["orig"],np.int64);fid=np.asarray(prep["fid"],np.int64)
  capmask=(orig==base["sid"])&np.isin(fid,base["cap_ids"])
  path=OUT/f"{role}-protected-cap-f32.npz"
  np.savez_compressed(path,points_world_m_f64=prep["P64"],faces_f64=prep["F64"],
    points_world_m_f32=Y,faces_f32=F32,face_source_original_id=orig,face_source_face_id=fid,
    face_source_layer=np.asarray(prep["face_role"],np.int8),face_source_identity0=np.asarray(prep["identity0"],np.int64),
    face_source_identity1=np.asarray(prep["identity1"],np.int64),temporary_cap_descendant_face_mask=capmask,
    face_source_identity_schema_by_original_id_json=np.asarray([json.dumps(prep["source_schema"],sort_keys=True)]),
    float32_emission_json=np.asarray([json.dumps(em,separators=(",",":"))]))
  return {"path":str(path),"sha256":sha(path),"P64":prep["P64"],"F64":prep["F64"],"P32":Y,"F32":F32,
    "orig":orig,"fid":fid,"capmask":capmask,"emission":em,"topology":topology(Y,F32)}
 Xsource=load(CAND)
 X32=np.asarray(Xsource["points_reference_world_m_f64"],np.float32)
 # Preserve exact source and support Float32 coordinates when present.
 outer_srcP=np.asarray(outer_safe.to_mesh64().vert_properties,np.float64)[:,:3]
 inner_srcP=np.asarray(inner_clipped.to_mesh64().vert_properties,np.float64)[:,:3]
 srcO=np.vstack((O["P"],outer_srcP)).astype(np.float32);srcI=np.vstack((I["P"],inner_srcP)).astype(np.float32)
 oout=emit_layer("O",O,Onew,outer_wrap,outer_ent,srcO)
 iout=emit_layer("I",I,Inew,inner_wrap,inner_ent,srcI)
 report["emitted"]={"O":{k:oout[k] for k in ("path","sha256","topology","emission")},
   "I":{k:iout[k] for k in ("path","sha256","topology","emission")}}
 # Cap and exact source-face preservation outside each original dilation bbox.
 capcheck={"O":cap_identity_check(O,oout),"I":cap_identity_check(I,iout)}
 obb=h.mesh_bbox(outer);ibb=h.mesh_bbox(inner)
 back={"O":source_identity_check(O,oout,obb),"I":source_identity_check(I,iout,ibb)}
 report["cap_and_back"]={"caps":capcheck,"outside_original_support_bbox":back}
 exact_results={"self_O":exact_pairs(exact,oout["P32"],oout["F32"],"O"),
   "self_I":exact_pairs(exact,iout["P32"],iout["F32"],"I")}
 oopen=oout["F32"][~oout["capmask"]];iopen=iout["F32"][~iout["capmask"]]
 exact_results["open_O_I"]=exact_cross(exact,oout["P32"],oopen,iout["P32"],iopen,"open-O-I")
 report["exact_f32"]={"owner":str(EXACT_OWNER),"owner_sha256":sha(EXACT_OWNER),"predicate":"retained exact rational Float32 triangle predicate; no seam exemptions",**exact_results}
 MO,_=h.make_mesh(oout["P32"].astype(np.float64),oout["F32"],face_ids=np.arange(len(oout["F32"]),dtype=np.uint64))
 MI,_=h.make_mesh(iout["P32"].astype(np.float64),iout["F32"],face_ids=np.arange(len(iout["F32"]),dtype=np.uint64))
 if MO.status()!=mf.Error.NoError or MI.status()!=mf.Error.NoError:raise RuntimeError(f"F32 candidate import status O={MO.status()} I={MI.status()}")
 MD=D["mesh"]
 rIO=boolop(h,mf,MI,MO,mf.OpType.Subtract,"F32 I minus O")
 rDI=boolop(h,mf,MD,MI,mf.OpType.Subtract,"F32 D037 minus I")
 skin=boolop(h,mf,MO,MI,mf.OpType.Subtract,"F32 O minus I")
 fat=boolop(h,mf,MI,MD,mf.OpType.Subtract,"F32 I minus D037")
 local_skin=boolop(h,mf,outer_safe,MI,mf.OpType.Subtract,"region23 outer support minus final I")
 local_fat=boolop(h,mf,inner_clipped,MD,mf.OpType.Subtract,"region23 clipped inner support minus D037")
 report["f32_boolean"]={"O":h.metrics(MO),"I":h.metrics(MI),"D037":h.metrics(MD),
   "I_minus_O":h.metrics(rIO),"D037_minus_I":h.metrics(rDI),"O_minus_I":h.metrics(skin),"I_minus_D037":h.metrics(fat),
   "local_skin23_after_I":h.metrics(local_skin),"local_fat23_after_D":h.metrics(local_fat)}
 capacity=local_capacity(local_skin,local_fat,Xsource,load_parent_mass())
 report["regional_capacity_after_D037"]={"groups":capacity,"mass_mapping":False}
 report["gates"]={"exact_f32_self_O_zero":exact_results["self_O"]["count"]==0,
   "exact_f32_self_I_zero":exact_results["self_I"]["count"]==0,"exact_noncap_OI_cross_zero":exact_results["open_O_I"]["count"]==0,
   "F32_topology_clean":all(report["emitted"][r]["topology"]["boundary_edges"]==0 and report["emitted"][r]["topology"]["nonmanifold_edges"]==0 and report["emitted"][r]["topology"]["zero_area_faces"]==0 for r in ("O","I")),
   "source_caps_exact":all(capcheck[r]["pass"] for r in ("O","I")),
   "outside_support_source_triangles_exact":all(back[r]["outside_face_count"]==back[r]["exact_oriented_descendants"] and back[r]["missing"]==0 and back[r]["ambiguous"]==0 for r in ("O","I")),
   "I_inside_O_boolean":rIO.volume()<=1e-15,"D037_inside_I_boolean":rDI.volume()<=1e-15,
   "positive_skin_fat":skin.volume()>0 and fat.volume()>0,"region23_fixed_mass_capacity":all(x["pass"] for x in capacity.values())}
 report["status"]="COMPLETE_PROTECTED_CAP_MINIMUM_SKIN_COUNTERFACTUAL"
 report["physical_admission"]=False
 report["elapsed_s"]=time.monotonic()-t0
 report["limitations"]=["The cap guard uses convex hulls of the pinned source cap vertices before the existing sphere16 Minkowski kernel; it may protect volume beyond the cap triangles and is recorded as a conservative temporary-cap guard.",
  "The 0.9mm erosion uses the existing sphere16 kernel and is a construction counterfactual; its polyhedral inradius is below its parameter radius.",
  "Manifold CSG volumes are diagnostics; exact Float32 surface predicates and fixed regional mass-capacity checks are reported separately.",
  "No native mass mapping or physical admission is claimed."]
 write(OUT_RECEIPT,report)
 print(json.dumps({"status":report["status"],"receipt":str(OUT_RECEIPT),"receipt_sha256":sha(OUT_RECEIPT),
  "gates":report["gates"],"exact_f32":report["exact_f32"],"cap_and_back":report["cap_and_back"],
  "regional_capacity":capacity,"f32_boolean":report["f32_boolean"],"elapsed_s":report["elapsed_s"]},indent=2),flush=True)
except Exception as exc:
 report["status"]="FAILED_PROTECTED_CAP_MINIMUM_SKIN_COUNTERFACTUAL";report["error"]=repr(exc);report["traceback"]=traceback.format_exc();report["elapsed_s"]=time.monotonic()-t0
 write(OUT_RECEIPT,report);print(json.dumps({"status":report["status"],"error":repr(exc),"receipt":str(OUT_RECEIPT),"elapsed_s":report["elapsed_s"]},indent=2),flush=True);raise

