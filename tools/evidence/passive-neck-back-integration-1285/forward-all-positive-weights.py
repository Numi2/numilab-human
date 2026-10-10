#!/usr/bin/env python3
from __future__ import annotations
import ast, hashlib, json, math, signal, struct, sys, time, traceback
from fractions import Fraction
from pathlib import Path
import numpy as np

R=Path("/Users/n/numi-human-retained-delivery-20261009")
ROOT=R/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001"
OUT=ROOT/"attempt-010"
TISS=R/"source-seam-connectivity-1247/fhl-current-7b23-count-reconciliation-1258/attempt-002/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MAN=TISS.with_suffix(".manifest.json")
SCREEN=R/"passive-muscle-self-diagnosis-1227/all-muscle-captured-screen-002/report.json"
UNION_DIR=ROOT/"attempt-003"
OWNER=Path("/Users/n/numi-human-self-separation-partial-resume-1267/src")
SIGNED=Path("/Users/n/numi-human-signed-winding-1273/src")
ANATOMY=Path("/Users/n/numi-human-anatomy-completion-1276/src")
FORWARD=R/"signed-winding-self-union-1273/subscap-pilot-001/full-attribute-lift-007/run_offline_attribute_lift_pose_audit_007.py"
INV=Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
PAIR=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
STEPS=[0,47519,151999,152607,153215,153823,154431,155000]
SIDS=[29,30]
EXPECTED_TISS_SHA="6579d06d358bfcf51f964feb57425fe568c467fd361930bce46d376ec8aedaa4"
EXPECTED_MAN_SHA="80d775bef6b8a4108e159fb7882a84b2fb8dce5b492059cd8c7cf3644a137467"
EXPECTED_UNION_SCRIPTSHA="fd1c98a626ddb11e27a426b19c30f6647b7ea9ce54c40ec4e444f1b1e6b82d7d"
EXPECTED_UNION_REPORTSHA="802cee9e44d996b5ca94adc7458ab223e9b0188c98c346fee2595354fedba769"
EXPECTED_UNION_29_SHA="9cc5521c0c4b04c165beb3747c953fe7d541eba61e0299cb1527c451fccfe31c"
EXPECTED_UNION_30_SHA="aee6990edc4011dc70848cd23876901c69a1813cdfb4ba1468f20d41b80281fd"
EXPECTED_CA_SHA="151ba8925910ae164bdf9a41417ca2cc5c63b07c3b05eb09496d20c4771c5e9b"
EXPECTED_CG_SHA="f6e98744dad9e23cd3b505efc02e7faa618fccce81d2ee08b07f948ef182fb72"
EXPECTED_PARTITION_SHA="8f392640c599ed39ebbfc82bab9b0292139216d10b4bb36ae03f54ee9b2d8944"
EXPECTED_FORWARD_SHA="e78fe261b2f625de51378c34d238b6716ac38bf2eb667824d669cdd2bda6ad47"
EXPECTED_COMPOSER_SHA="732f9948813ad33566560494a279ac915e2758d997ef2d052d82220ed3a777d4"
EXPECTED_INVENTORY_SHA="a43484aa8d65b207cc498bb470097ec5f06ef2bb9872837bde9f197e19e815e3"
EXPECTED_CI_SHA="423180082e32d002375b612aad36f307ed439af07c04d655e6a8bedc87a338bd"
PROGRESS=OUT/"progress.jsonl"
START=time.monotonic()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def require(v,msg):
    if not v: raise RuntimeError(msg)

def progress(event,**kw):
    row={"event":event,"elapsed_seconds":time.monotonic()-START,**kw}
    with PROGRESS.open("a") as f: f.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps(row,sort_keys=True),flush=True)

def alarm(_sig,_frame):
    raise TimeoutError("bounded gluteus source/terminal audit exceeded 300 seconds")

def rational_pair(x):
    if isinstance(x,(list,tuple)) and len(x)==2:
        a,b=x
        return Fraction(int(a,0) if isinstance(a,str) else int(a),
                        int(b,0) if isinstance(b,str) else int(b))
    raise RuntimeError("union coordinate is not a numerator/denominator pair")

def round_f32_exact(value):
    estimate=np.float32(float(value))
    require(np.isfinite(estimate),"nonfinite rounded coordinate")
    candidates=[estimate,np.nextafter(estimate,np.float32(-np.inf),dtype=np.float32),
                np.nextafter(estimate,np.float32(np.inf),dtype=np.float32)]
    return min(candidates,key=lambda c:(abs(Fraction.from_float(float(c))-value),
                                        int(np.asarray(c,dtype=np.float32).view(np.uint32))&1))

def vol_component(points,faces,ids):
    terms=[]
    for k in ids:
        a,b,c=(np.asarray(points[int(j)],dtype=np.float64) for j in faces[int(k)])
        terms.append(float(np.dot(a,np.cross(b,c)))/6.0)
    return math.fsum(terms)

def area(points,faces):
    terms=[]
    for f in faces:
        a,b,c=(np.asarray(points[int(i)],dtype=np.float64) for i in f)
        terms.append(0.5*float(np.linalg.norm(np.cross(b-a,c-a))))
    return math.fsum(terms)

def vertex_normals(points,faces):
    p=np.asarray(points,dtype=np.float32); f=np.asarray(faces,dtype=np.int64)
    accum=np.zeros((len(p),3),dtype=np.float64)
    for a,b,c in f:
        tri=p[[a,b,c]].astype(np.float64)
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0])
        accum[[a,b,c]]+=n
    lengths=np.linalg.norm(accum,axis=1)
    require(np.isfinite(lengths).all() and np.all(lengths>0),"union has a vertex with undefined area-weighted normal")
    return (accum/lengths[:,None]).astype("<f4")

def load_tissue(p):
    raw=Path(p).read_bytes()
    magic,abi,nr,nb,nv,ni,fp,src=struct.unpack_from("<8s6I32s",raw)
    require(magic==b"NHTISS4\0" and abi==5,"unsupported NHTISS header")
    bo=64+32*nr; vo=bo+36*nb; io=vo+56*nv
    require(len(raw)==io+4*ni,"NHTISS byte length mismatch")
    rec=np.frombuffer(raw,dtype="<u4",count=nr*8,offset=64).reshape(-1,8)
    bind=np.frombuffer(raw,dtype=np.dtype([("core","<u4"),("value","<f4",(8,))]),count=nb,offset=bo)
    ind=np.frombuffer(raw,dtype="<u4",count=ni,offset=io)
    return {"path":Path(p),"raw":raw,"records":rec,"bindings":bind,"indices":ind,"bo":bo,"vo":vo}

def row_data(t,sid):
    hits=[r for r in t["records"] if int(r[6])==sid]
    require(len(hits)==1,f"stable {sid} row cardinality {len(hits)}")
    r=hits[0]; fb,bc,fv,vc,fi,ic=map(int,r[:6]); raw=t["raw"]; vo=t["vo"]
    pos=np.ndarray((vc,3),dtype="<f4",buffer=raw,offset=vo+fv*56,strides=(56,4)).copy()
    norms=np.ndarray((vc,3),dtype="<f4",buffer=raw,offset=vo+fv*56+12,strides=(56,4)).copy()
    local=np.ndarray((vc,4),dtype="<u4",buffer=raw,offset=vo+fv*56+24,strides=(56,4)).copy()
    weights=np.ndarray((vc,4),dtype="<f4",buffer=raw,offset=vo+fv*56+40,strides=(56,4)).copy()
    faces=t["indices"][fi:fi+ic].reshape(-1,3).astype(np.int64)-fv
    return {"sid":sid,"fb":fb,"bc":bc,"fv":fv,"vc":vc,"fi":fi,"ic":ic,
            "positions":pos,"normals":norms,"local":local,"weights":weights,"faces":faces,
            "vertex_bytes":raw[vo+fv*56:vo+(fv+vc)*56],
            "binding_bytes":raw[t["bo"]+fb*36:t["bo"]+(fb+bc)*36]}

def exact_rows(points,faces,ci):
    p=np.asarray(points,dtype="<f4"); f=np.asarray(faces,dtype=np.int64)
    require(p.ndim==2 and p.shape[1]==3 and f.ndim==2 and f.shape[1]==3,"bad exact mesh arrays")
    uq,inv=np.unique(p,axis=0,return_inverse=True); qf=inv[f]
    lattice=[ci.float32_point_lattice_key(tuple(float(v) for v in row)) for row in uq]
    deg=[]
    for i,t in enumerate(qf):
        n=ci._cross(ci._sub(lattice[int(t[1])],lattice[int(t[0])]),
                    ci._sub(lattice[int(t[2])],lattice[int(t[0])]))
        if not any(n): deg.append(int(i))
    if deg: return None,deg
    return ci._records(lattice,qf.tolist()),[]

def combine_records(a,b):
    off=max((int(r[3]) for r in a),default=-1)+1
    return list(a)+[(r[0],r[1],r[2],int(r[3])+off,r[4]) for r in b],off

def rotate32(q,p):
    f=np.float32; q=np.asarray(q,dtype=np.float32); p=np.asarray(p,dtype=np.float32); a=q[:3]
    cr=np.asarray([f(f(a[1]*p[2])-f(a[2]*p[1])),
                   f(f(a[2]*p[0])-f(a[0]*p[2])),
                   f(f(a[0]*p[1])-f(a[1]*p[0]))],dtype=np.float32)
    tw=np.asarray([f(f(2)*v) for v in cr],dtype=np.float32)
    co=np.asarray([f(f(f(q[3]*tw[0])+f(a[1]*tw[2]))-f(a[2]*tw[1])),
                   f(f(f(q[3]*tw[1])+f(a[2]*tw[0]))-f(a[0]*tw[2])),
                   f(f(f(q[3]*tw[2])+f(a[0]*tw[1]))-f(a[1]*tw[0]))],dtype=np.float32)
    return np.asarray([f(p[i]+co[i]) for i in range(3)],dtype=np.float32)

def forward(points,local,weights,row,tissue,poses):
    f=np.float32; out=np.zeros_like(points,dtype=np.float32)
    for vi in range(len(points)):
        accum=np.zeros(3,dtype=np.float32)
        for slot in range(4):
            w=f(weights[vi,slot])
            if w<=f(0): continue
            li=int(local[vi,slot]); require(li<row["bc"],"candidate local binding out of range")
            binding=tissue["bindings"][row["fb"]+li]; core=int(binding["core"])
            require(core in poses,"missing body pose "+str(core))
            val=binding["value"]; rr=rotate32(val[3:7],points[vi])
            loc=np.asarray([f(val[j]+f(f(rr[j])*val[7])) for j in range(3)],dtype=np.float32)
            bp,bq=poses[core]; wr=rotate32(bq,loc)
            world=np.asarray([f(bp[j]+wr[j]) for j in range(3)],dtype=np.float32)
            accum=np.asarray([f(accum[j]+f(w*world[j])) for j in range(3)],dtype=np.float32)
        out[vi]=accum
    return out

def read_pair_setup():
    declarations={}; executions={}; metadata={}
    for arm in ("baseline","intervention"):
        dpath=PAIR/arm/"run-declaration.json"
        epath=PAIR/arm/"execution.json"
        mpath=PAIR/arm/"native-run/run-metadata.json"
        d=json.loads(dpath.read_text()); e=json.loads(epath.read_text()); m=json.loads(mpath.read_text())
        require(d.get("capture_steps")==STEPS and d.get("accepted_steps")==155000 and d.get("seconds")==310.0,
                f"{arm} declaration pose grid mismatch")
        require(d.get("immutable_assets",{}).get(str(TISS))==EXPECTED_TISS_SHA,
                f"{arm} declaration does not bind current source TISS")
        require(e.get("returncode")==0 and e.get("declaration_sha256")==sha(dpath)
                and e.get("changed_inputs")=={} and e.get("native_argv_matches_prepared_cli_preview") is True
                and e.get("native_environment_matches_prepared_cli_preview") is True,
                f"{arm} native execution is not closed and hash-bound")
        require(m.get("exit_code")==0 and not m.get("source_files_changed_during_run"),
                f"{arm} run metadata is not successful/unchanged")
        declarations[arm]=d; executions[arm]=e; metadata[arm]=m
    return declarations,executions,metadata

def fraction_tuple(p):
    return tuple(Fraction.from_float(float(x)) for x in p)

def barycentric_exact(point,tri):
    a,b,c=tri
    ab=tuple(b[i]-a[i] for i in range(3)); ac=tuple(c[i]-a[i] for i in range(3))
    ap=tuple(point[i]-a[i] for i in range(3))
    dot=lambda x,y:sum((x[i]*y[i] for i in range(3)),Fraction(0))
    d00=dot(ab,ab); d01=dot(ab,ac); d11=dot(ac,ac); d20=dot(ap,ab); d21=dot(ap,ac)
    den=d00*d11-d01*d01
    require(den!=0,"source ancestry triangle is degenerate")
    v=(d11*d20-d01*d21)/den
    w=(d00*d21-d01*d20)/den
    u=Fraction(1)-v-w
    # Exact plane membership and closed-triangle membership.
    cross=(ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0])
    plane=sum((cross[i]*ap[i] for i in range(3)),Fraction(0))
    require(plane==0 and min(u,v,w)>=0 and max(u,v,w)<=1,
            "union vertex is not on its declared closed source-face ancestry")
    return (u,v,w)

def map_sig(m):
    return tuple((int(k),v.numerator,v.denominator) for k,v in sorted(m.items()) if v)

def signed_volumes(topology,points,faces):
    return [vol_component(points,faces,c) for c in topology["face_components"]]

def make_candidate(sid,tissue,row,doc,ci,geometry):
    pos=row["positions"]; faces=row["faces"]; local=row["local"]; weights=row["weights"]
    require(row["bc"]==2,"gluteus row binding cardinality changed")
    source_manifest=geometry["manifest_rows"][sid]
    expected_member={29:"FJ1418",30:"FJ1418M"}[sid]
    require(source_manifest["member_id"]==expected_member,f"stable {sid} source member mismatch")
    require(source_manifest["label"] and source_manifest["body_bindings"],"missing source binding identity")
    union_path=UNION_DIR/f"stable-{sid}-exact-union.json"
    docu=json.loads(union_path.read_text())
    require(docu["source_tissue_sha256"]==EXPECTED_TISS_SHA and docu["source_manifest_sha256"]==EXPECTED_MAN_SHA,
            f"stable {sid} union is not bound to current TISS")
    construction=docu["construction"]; mesh=construction["output_mesh"]
    out_exact=[tuple(rational_pair(v) for v in p) for p in mesh["vertices_m"]]
    out_faces=np.asarray(mesh["triangles"],dtype=np.int64)
    ancestry=docu["ancestry_by_output_face"]
    require(len(out_faces)==len(ancestry)==int(construction["output_face_count"] if "output_face_count" in construction else len(out_faces)),
            "union ancestry/face row mismatch")
    require(all(0<=int(x["source_face"])<len(faces) for x in ancestry),"union source-face ancestry out of range")
    src_unique, inverse=np.unique(pos,axis=0,return_inverse=True)
    qfaces=inverse[faces]
    src_exact=[fraction_tuple(x) for x in src_unique]
    source_coordinate_groups={}
    for vi,p in enumerate(pos):
        source_coordinate_groups.setdefault(fraction_tuple(p),[]).append(vi)
    def source_map(vi):
        m={}
        for slot in range(4):
            ww=Fraction.from_float(float(weights[vi,slot]))
            if ww:
                li=int(local[vi,slot]); require(li<row["bc"],"source active binding index out of range")
                m[li]=m.get(li,Fraction(0))+ww
        return m
    map_by_coord={}
    representative_by_coord={}
    for key,ids in source_coordinate_groups.items():
        maps={map_sig(source_map(i)):source_map(i) for i in ids}
        require(len(maps)==1,f"duplicate source xyz has conflicting route map on stable {sid}")
        map_by_coord[key]=next(iter(maps.values()))
        representative_by_coord[key]=min(ids)
    source_key_by_index={i:tuple(Fraction.from_float(float(x)) for x in p) for i,p in enumerate(src_unique)}
    out_pos=np.empty((len(out_exact),3),dtype="<f4")
    out_weights=np.zeros((len(out_exact),4),dtype="<f4")
    out_local=np.full((len(out_exact),4),np.uint32(0xffffffff),dtype="<u4")
    lift_rows=[]
    new_vertex_indices=[]
    face_parents=[int(x["source_face"]) for x in ancestry]
    incident=[[] for _ in out_exact]
    for fi,f in enumerate(out_faces):
        for vi in f: incident[int(vi)].append(fi)
    max_map_disagreement=Fraction(0)
    max_ancestry_map_sum_error=Fraction(0)
    for vi,p in enumerate(out_exact):
        source_key=tuple(p)
        if source_key in map_by_coord:
            rep=representative_by_coord[source_key]
            out_pos[vi]=pos[rep]
            out_local[vi]=local[rep]
            out_weights[vi]=weights[rep]
            lift_rows.append({"candidate_vertex":vi,"kind":"source_coordinate_preserved",
                              "source_duplicate_group_count":len(source_coordinate_groups[source_key]),
                              "source_representative":rep})
            continue
        out_pos[vi]=np.asarray([round_f32_exact(x) for x in p],dtype="<f4")
        variants={}
        parent_faces=[]
        for fi in incident[vi]:
            sf=face_parents[fi]; parent_faces.append(sf)
            tri_ids=qfaces[sf]
            lamb=barycentric_exact(p,[src_exact[int(k)] for k in tri_ids])
            m={}
            for coeff,src_i in zip(lamb,faces[sf]):
                sm=source_map(int(src_i))
                for li,wgt in sm.items(): m[li]=m.get(li,Fraction(0))+coeff*wgt
            map_sum_error=abs(sum(m.values(),Fraction(0))-1)
            max_ancestry_map_sum_error=max(max_ancestry_map_sum_error,map_sum_error)
            require(float(map_sum_error)<1e-5 and all(v>=0 for v in m.values()),
                    f"stable {sid} ancestry interpolation exceeds inherited composer route-sum tolerance")
            variants[map_sig(m)]=m
        require(bool(variants),f"new stable {sid} union vertex lacks incident parent ancestry")
        maps=[variants[k] for k in sorted(variants)]
        if len(maps)>1:
            l1=max((sum((abs(a.get(k,Fraction(0))-b.get(k,Fraction(0))) for k in range(row["bc"])),Fraction(0))
                    for i,a in enumerate(maps) for b in maps[i+1:]),default=Fraction(0))
            max_map_disagreement=max(max_map_disagreement,l1)
        mean={li:sum((m.get(li,Fraction(0)) for m in maps),Fraction(0))/len(maps)
              for li in range(row["bc"])}
        require(float(abs(sum(mean.values(),Fraction(0))-1))<1e-5 and all(x>=0 for x in mean.values()),
                "consensus map exceeds inherited composer route-sum tolerance")
        out_local[vi,:row["bc"]]=np.arange(row["bc"],dtype="<u4")
        out_weights[vi,:row["bc"]]=np.asarray([np.float32(float(mean[li])) for li in range(row["bc"])],dtype="<f4")
        lift_rows.append({"candidate_vertex":vi,"kind":"equal_mean_distinct_parent_maps",
                          "distinct_map_count":len(maps),"parent_faces":sorted(set(parent_faces)),
                          "max_pairwise_l1_source_map_difference":float(l1) if len(maps)>1 else 0.0,
                          "consensus_exact_local_slots":{str(k):str(v) for k,v in mean.items()},
                          "consensus_f32_local_slots":[float(x) for x in out_weights[vi,:row["bc"]]]})
        new_vertex_indices.append(vi)
    sums=out_weights.astype(np.float64).sum(axis=1)
    require(np.isfinite(out_pos).all() and np.isfinite(out_weights).all() and (out_weights>=0).all(),
            "candidate vertex or route maps are nonfinite/negative")
    require(float(np.max(np.abs(sums-1)))<1e-5,"candidate F32 route weights fail composer sum invariant")
    require(out_faces.min()>=0 and out_faces.max()<len(out_pos),"union candidate face index out of range")
    normals=vertex_normals(out_pos,out_faces)
    vertices6=np.concatenate([out_pos,normals],axis=1).astype("<f4")
    candidate=OUT/f"stable-{sid}-reference-union-row-patch.npz"
    np.savez(candidate,vertices6=vertices6,binding_indices=out_local,weights=out_weights,
             faces=out_faces.astype("<u4"))
    # Candidate source exact self/topology.
    candidate_records,deg=exact_rows(out_pos,out_faces,ci)
    source_records,source_deg=exact_rows(pos,faces,ci)
    require(candidate_records is not None and not deg,"candidate source union is Float32-degenerate")
    candidate_self=ci._audit_pair(candidate_records,candidate_records,same_surface=True)
    source_self=ci._audit_pair(source_records,source_records,same_surface=True)
    top=geometry["analyze_topology"](out_pos.astype(float).tolist(),out_faces.tolist())
    require(not deg and top["closed_oriented_manifold_candidate"],"candidate union topology not closed/oriented manifold")
    require(candidate_self["count"]==0,"candidate source union retains exact self-pairs")
    area_before=area(src_unique,qfaces); area_after=area(out_pos,out_faces)
    source_top=geometry["analyze_topology"](src_unique.astype(float).tolist(),qfaces.tolist())
    source_vols=signed_volumes(top, out_pos, out_faces)
    src_vols=signed_volumes(source_top,src_unique,qfaces)
    report={
      "stable_id":sid,"member_id":expected_member,"label":source_manifest["label"],
      "body_binding_rows":source_manifest["body_bindings"],
      "source_payload_sha256":EXPECTED_TISS_SHA,"source_manifest_sha256":EXPECTED_MAN_SHA,
      "source_row":{"vertex_rows":len(pos),"unique_f32_vertices":len(src_unique),"faces":len(faces),
                    "exact_self_pair_count":source_self["count"],"exact_self_pairs":source_self["triangle_pairs"],
                    "degenerate_face_rows":source_deg,"topology":source_top},
      "source_union":{"output_vertices":len(out_pos),"output_faces":len(out_faces),
                      "exact_self_pair_count":candidate_self["count"],"exact_self_pairs":candidate_self["triangle_pairs"],
                      "degenerate_face_rows":deg,"topology":top,
                      "components_signed_volume_m3":source_vols,"input_components_signed_volume_m3":src_vols,
                      "input_area_m2":area_before,"output_area_m2":area_after,
                      "area_delta_m2":area_after-area_before,"area_delta_relative":area_after/area_before-1,
                      "new_attribute_vertices":len(new_vertex_indices),
                      "ancestry_consensus_max_L1_weight_disagreement":float(max_map_disagreement),
                      "ancestry_exact_route_map_max_abs_sum_error":float(max_ancestry_map_sum_error),
                      "attribute_lift_policy":"For each new union vertex, compute exact barycentric maps through its source-face ancestry, deduplicate distinct exact maps, and take their equal arithmetic mean in the existing row-local binding-index order. Existing source-coordinate vertices preserve their original route arrays. No renormalization, body-binding, route or attachment edit.",
                      "candidate_npz_path":str(candidate),"candidate_npz_sha256":sha(candidate),
                      "npz_fields":["vertices6","binding_indices","weights","faces"],
                      "weights_max_sum_error_f32":float(np.max(np.abs(sums-1))),
                      "weights_renormalized":False,
                      "per_vertex_lift":lift_rows}
    }
    return {"row":row,"positions":out_pos,"weights":out_weights,"local":out_local,
            "faces":out_faces,"vertices6":vertices6,"records":candidate_records,
            "source_records":source_records,"source_self":source_self,"candidate_self":candidate_self,
            "report":report,"candidate_npz":candidate,"topology":top,
            "parents":face_parents,"source_unique":src_unique,"source_qfaces":qfaces}

def main():
    script_path=Path(__file__).resolve()
    require(OUT.is_dir() and not PROGRESS.exists() and not (OUT/"source-union-attribute-lift-terminal-audit-001.json").exists() and all(not (OUT/f"stable-{sid}-reference-union-row-patch.npz").exists() for sid in SIDS),"attempt-010 output paths must be fresh")
    require(not PROGRESS.exists(),"unexpected pre-existing progress file")
    sys.path.insert(0,str(SIGNED))
    sys.path.insert(0,str(OWNER))
    from numilab_human import cardiac_cavity_intersections as ci
    from numilab_human import common_atlas_skin_clearance as ca
    from numilab_human.cardiac_cavity_geometry import analyze_topology
    require(Path(ci.__file__).resolve()==OWNER/"numilab_human/cardiac_cavity_intersections.py","exact predicate import path mismatch")
    require(Path(ca.__file__).resolve()==OWNER/"numilab_human/common_atlas_skin_clearance.py","target audit import path mismatch")
    require(sha(Path(ci.__file__))==EXPECTED_CI_SHA,"exact predicate source hash mismatch")
    union_script=UNION_DIR/"run_gluteus_union_003.py"
    union_report=UNION_DIR/"union-execution-003.json"
    require(sha(union_script)==EXPECTED_UNION_SCRIPTSHA and sha(union_report)==EXPECTED_UNION_REPORTSHA,
            "attempt003 geometry union inputs changed")
    union_docs={sid:UNION_DIR/f"stable-{sid}-exact-union.json" for sid in SIDS}
    for sid,p in union_docs.items():
        require(p.is_file(),f"missing exact union row {sid}")
    require(sha(union_docs[29])==EXPECTED_UNION_29_SHA and sha(union_docs[30])==EXPECTED_UNION_30_SHA,
            "exact union result JSON pins changed")
    tissue=load_tissue(TISS)
    manifest=json.loads(MAN.read_text())
    require(sha(TISS)==EXPECTED_TISS_SHA and sha(MAN)==EXPECTED_MAN_SHA,"current TISS/manifest hash changed")
    require(manifest["payload"]["sha256"]==EXPECTED_TISS_SHA,"current manifest does not bind TISS")
    manifest_rows={int(x["stable_id"]):x for x in manifest["source"]["surfaces"]}
    screen=json.loads(SCREEN.read_text())
    screen_rows={int(x["stable_id"]):x for x in screen["surfaces"]}
    geometry={"manifest_rows":manifest_rows,"analyze_topology":analyze_topology}
    declarations,executions,metadata=read_pair_setup()
    # Import only exact function definitions from 007; do not execute its unrelated top-level audit.
    forward_src=ast.parse(FORWARD.read_text())
    function_namespace={"np":np,"Path":Path,"struct":struct}
    for node in forward_src.body:
        if isinstance(node,ast.FunctionDef) and node.name in {"need","load_tissue","row_data","f32","cross32","rotate32","forward"}:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(FORWARD),"exec"),function_namespace)
    load_tissue_007=function_namespace["load_tissue"]
    row_data_007=function_namespace["row_data"]
    fwd=function_namespace["forward"]
    tissue_fwd=load_tissue_007(TISS)
    rows={}
    for sid in SIDS:
        hits=[r for r in tissue["records"] if int(r[6])==sid]
        row= row_data_007(tissue_fwd,sid)
        require(row["vc"]==2053 and len(row["faces"])==2368,f"stable {sid} source shape changed")
        require(manifest_rows[sid]["member_id"]=={29:"FJ1418",30:"FJ1418M"}[sid],"stable ID source ancestry mismatch")
        require(screen_rows[sid]["triangle_pairs"]==[[278,279],[278,280],[280,281]],
                f"historical 150-surface screen mismatch for stable {sid}; retained screen is source history, not current output")
        rows[sid]=row
    inputs=[Path(__file__),TISS,MAN,SCREEN,INV,FORWARD,
            OWNER/"numilab_human/common_atlas_skin_clearance.py",
            OWNER/"numilab_human/cardiac_cavity_intersections.py",
            OWNER/"numilab_human/cardiac_cavity_geometry.py",
            SIGNED/"numilab_human/cardiac_cavity_partition.py",
            Path("/Users/n/numi-human-anatomy-completion-1276/src/numilab_human/passive_attachment_composition.py"),
            union_script,union_report,*union_docs.values()]
    for arm in ("baseline","intervention"):
        inputs += [PAIR/arm/"run-declaration.json",PAIR/arm/"execution.json",PAIR/arm/"native-run/run-metadata.json"]
        for step in STEPS:
            base=PAIR/arm/f"native-run/accepted-geometry/step-{step}"
            inputs += [base.with_suffix(".mrvpack"),base.with_suffix(".receipt.json")]
    pins_before={str(p):sha(p) for p in inputs}
    require(pins_before[str(TISS)]==EXPECTED_TISS_SHA and pins_before[str(MAN)]==EXPECTED_MAN_SHA,"pinned TISS changed after setup")
    for arm in ("baseline","intervention"):
        d=json.loads((PAIR/arm/"run-declaration.json").read_text())
        require(d["immutable_assets"][str(TISS)]==EXPECTED_TISS_SHA,f"{arm} declaration no longer binds TISS")
    progress("source_prepare_started",stable_ids=SIDS,source_tiss_sha256=EXPECTED_TISS_SHA)
    candidates={}
    for sid in SIDS:
        # Existing exact union output is hash-bound and does not rerun the geometry constructor.
        candidates[sid]=make_candidate(sid,tissue_fwd,rows[sid],json.loads(union_docs[sid].read_text()),ci,geometry)
        require(candidates[sid]["source_self"]["triangle_pairs"]==screen_rows[sid]["triangle_pairs"],
                f"current source exact pair list no longer matches the retained historical screen for stable {sid}")
        src_capture_count=screen_rows[sid]["triangle_pairs"]
        candidates[sid]["report"]["historical_150_screen"]={"path":str(SCREEN),"sha256":pins_before[str(SCREEN)],
            "source_pair_list":src_capture_count,"scope":"historical screen predates the current TISS; current row/step0 reproduction is separately checked below"}
        progress("source_union_candidate_ready",stable_id=sid,
                 source_exact_self_pairs=candidates[sid]["source_self"]["count"],
                 candidate_exact_self_pairs=candidates[sid]["candidate_self"]["count"],
                 candidate_vertices=len(candidates[sid]["positions"]),candidate_faces=len(candidates[sid]["faces"]),
                 candidate_npz=str(candidates[sid]["candidate_npz"]),
                 candidate_npz_sha256=sha(candidates[sid]["candidate_npz"]),
                 max_attribute_map_disagreement=candidates[sid]["report"]["source_union"]["ancestry_consensus_max_L1_weight_disagreement"])
    pose_rows=[]
    # First bounded accepted-pose gate: actual final capture for both arms.
    step=155000
    for arm in ("baseline","intervention"):
        pack=PAIR/arm/f"native-run/accepted-geometry/step-{step}.mrvpack"
        receipt_path=pack.with_name(f"step-{step}.receipt.json")
        receipt=json.loads(receipt_path.read_text())
        require(receipt.get("accepted_step")==step and receipt.get("physical_endpoint")=="accepted",
                f"{arm} terminal capture is not accepted at step {step}")
        require(receipt.get("pack_file_sha256")==sha(pack),f"{arm} terminal receipt does not bind pack bytes")
        poses={int(x["body_index"]):(np.asarray(x["position_m"],dtype=np.float32),
                                    np.asarray(x["quaternion_xyzw"],dtype=np.float32))
               for x in receipt["accepted_registered_body_poses"]}
        target_keys,clearance_keys,expected_counts,_pop=ca._load_target_inventory(INV)
        positions,surfaces,inv=ca._pack_surfaces(pack,{(51005,29),(51005,30)})
        require((51005,29) in surfaces and (51005,30) in surfaces,"terminal MRVPACK lacks a gluteus row")
        per_row={}
        base_records={}; cand_records={}
        candidate_world={}
        source_replay={}
        for sid in SIDS:
            row=rows[sid]; item=candidates[sid]
            cap_faces=np.asarray(surfaces[(51005,sid)]["faces"],dtype=np.int64)
            diff=cap_faces-row["faces"]
            require(diff.shape==row["faces"].shape and np.all(diff==diff.flat[0]),f"{arm}/{sid} terminal face mapping changed")
            off=int(diff.flat[0])
            require(off>=0 and off+row["vc"]<=len(positions),"captured row vertex window out of bounds")
            actual=np.asarray(positions[off:off+row["vc"]],dtype="<f4")
            require(np.all(cap_faces>=off) and np.all(cap_faces<off+row["vc"]),"terminal row faces escape mapped vertex window")
            replay=fwd(row["positions"],row["local"],row["weights"],row,tissue_fwd,poses)
            err=np.linalg.norm((replay.astype(np.float64)-actual.astype(np.float64)),axis=1)
            pred=fwd(item["positions"],item["local"],item["weights"],row,tissue_fwd,poses)
            brec,bdeg=exact_rows(actual,row["faces"],ci)
            prec,pdeg=exact_rows(pred,item["faces"],ci)
            require(brec is not None and prec is not None,"terminal baseline or candidate is exact Float32-degenerate")
            bself=ci._audit_pair(brec,brec,same_surface=True)
            pself=ci._audit_pair(prec,prec,same_surface=True)
            top=analyze_topology(pred.astype(float).tolist(),item["faces"].tolist())
            srcvol=signed_volumes(item["topology"],item["positions"],item["faces"])
            posedvol=signed_volumes(top,pred,item["faces"])
            same_sign=(len(srcvol)==len(posedvol) and all(a*b>0 for a,b in zip(srcvol,posedvol)))
            per_row[sid]={"captured_source_self_count":bself["count"],
                          "captured_source_self_pairs":bself["triangle_pairs"],
                          "candidate_forward_self_count":pself["count"],
                          "candidate_forward_self_pairs":pself["triangle_pairs"],
                          "candidate_forward_degenerate_faces":pdeg,
                          "candidate_topology":top,
                          "source_to_capture_forward_max_error_um":float(err.max(initial=0))*1e6,
                          "source_to_capture_replay_gate":"diagnostic_only_no_threshold",
                          "source_to_capture_replay_gate":"diagnostic_only_no_threshold",
                          "source_to_capture_forward_median_error_um":float(np.median(err))*1e6,
                          "source_to_capture_forward_exact_vertex_count":int(np.count_nonzero(np.all(replay==actual,axis=1))),
                          "terminal_source_to_pose_signed_volume_m3":srcvol,
                          "terminal_candidate_signed_volume_m3":posedvol,
                          "component_orientation_sign_preserved":same_sign}
            base_records[sid]=brec; cand_records[sid]=prec; candidate_world[sid]=pred; source_replay[sid]=replay
        base_mutual=ci._audit_pair(base_records[29],base_records[30],same_surface=False)
        cand_mutual=ci._audit_pair(cand_records[29],cand_records[30],same_surface=False)
        row={"arm":arm,"accepted_step":step,"receipt_sha256":sha(receipt_path),"pack_sha256":sha(pack),
             "forward_mapping_owner_path":str(FORWARD),"surface_forward_replay_max_error_um":max(per_row[s]["source_to_capture_forward_max_error_um"] for s in SIDS),
             "rows":{str(s):per_row[s] for s in SIDS},
             "captured_mutual_29_30_count":base_mutual["count"],"captured_mutual_29_30_pairs":base_mutual["triangle_pairs"],
             "candidate_mutual_29_30_count":cand_mutual["count"],"candidate_mutual_29_30_pairs":cand_mutual["triangle_pairs"]}
        pose_rows.append(row)
        progress("terminal_pose_self_audit_complete",arm=arm,accepted_step=step,
                 captured_self={str(s):per_row[s]["captured_source_self_count"] for s in SIDS},
                 candidate_self={str(s):per_row[s]["candidate_forward_self_count"] for s in SIDS},
                 mutual_captured=base_mutual["count"],mutual_candidate=cand_mutual["count"],
                 forward_max_error_um=row["surface_forward_replay_max_error_um"])
        require(sha(pack)==pins_before[str(pack)] and sha(receipt_path)==pins_before[str(receipt_path)],
                f"{arm} terminal inputs changed during audit")
    after={str(p):sha(p) for p in inputs}
    require(pins_before==after,"one or more frozen anatomy, owner, declaration, or accepted-pose inputs changed during source/terminal audit")
    summary={
      "schema":"numi.human.gluteus-maximus-source-union-attribute-lift-terminal-audit.v1",
      "status":"unadmitted_source_candidate_generated_and_terminal_offline_self_audited",
      "scope":"Stable 29/30 gluteus maximus rows only. The exact signed-winding union and inferred ancestry-mean route weights are offline candidates; this run audits source and the accepted native terminal pose in the two pair011 arms. No TISS composition, simulation, or whole-body admission.",
      "source":{"nhtiss_path":str(TISS),"nhtiss_sha256":EXPECTED_TISS_SHA,"manifest_path":str(MAN),
                "manifest_sha256":EXPECTED_MAN_SHA,"current_row_source_sha_match":True},
      "pose_fixture":{"pair_directory":str(PAIR),"arms":["baseline","intervention"],"accepted_step":155000,
                      "seconds":310.0,"pose_receipts_are_accepted":True,"fixture_capture_steps":STEPS},
      "historical_screen":{"path":str(SCREEN),"sha256":pins_before[str(SCREEN)],
                           "predates_current_6579_payload":True,"selected_source_pair_rows_reproduced":True},
      "owner_pins":{str(p):pins_before[str(p)] for p in inputs if str(p) in {
          str(FORWARD),str(OWNER/"numilab_human/common_atlas_skin_clearance.py"),
          str(OWNER/"numilab_human/cardiac_cavity_intersections.py"),
          str(SIGNED/"numilab_human/cardiac_cavity_partition.py"),
          str(ANATOMY/"numilab_human/passive_attachment_composition.py")}},
      "rows":[candidates[s]["report"] for s in SIDS],
      "terminal_accepted_pose_audits":pose_rows,
      "input_sha256_before":pins_before,"input_sha256_after":after,"inputs_unchanged":True,
      "limitations":["The offline 007 forward replay is not a native candidate capture; original source rows differ from terminal captured Float32 positions by the reported sub-micrometre residual.",
                    "Only exact self, topology/orientation, and mutual stable29/30 checks were run at terminal in this increment; external target scans and the remaining 15 accepted poses are not yet complete.",
                    "An ancestry-mean route map is an explicit source-derived inference at new union vertices, not measured anatomy. Existing route/body binding records are unchanged.",
                    "The old 150-row screen predates the current full TISS payload; its matching pair list is corroboration for source history only."]}
    out=OUT/"source-union-attribute-lift-terminal-audit-001.json"
    out.write_text(json.dumps(summary,sort_keys=True,separators=(",",":"))+"\n")
    progress("terminal_source_audit_report_written",report=str(out),report_sha256=sha(out),
             candidate_nps={str(s):{"path":str(candidates[s]["candidate_npz"]),"sha256":sha(candidates[s]["candidate_npz"])} for s in SIDS})
    print(json.dumps({"status":summary["status"],"report":str(out),"report_sha256":sha(out),
       "candidate_npz":{str(s):{"path":str(candidates[s]["candidate_npz"]),"sha256":sha(candidates[s]["candidate_npz"])} for s in SIDS},
       "terminal":[{"arm":x["arm"],"step":x["accepted_step"],"source_self":{k:v["captured_source_self_count"] for k,v in x["rows"].items()},
                    "candidate_self":{k:v["candidate_forward_self_count"] for k,v in x["rows"].items()},
                    "mutual_candidate":x["candidate_mutual_29_30_count"]} for x in pose_rows],
       "inputs_unchanged":True,"wall_seconds":time.monotonic()-START},sort_keys=True))
    return 0

if __name__=="__main__":
    signal.signal(signal.SIGALRM,alarm); signal.alarm(300)
    try:
        raise SystemExit(main())
    except BaseException as e:
        if OUT.exists():
            fail={"schema":"numi.human.gluteus-maximus-source-union-attempt-failure.v1",
                  "status":"retained_failure","error_type":type(e).__name__,"error":str(e),
                  "traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-START,
                  "input_hashes_current":{}}
            for p in [TISS,MAN,SCREEN,FORWARD,UNION_DIR/"run_gluteus_union_003.py",
                      UNION_DIR/"union-execution-003.json"]:
                if p.is_file(): fail["input_hashes_current"][str(p)]=sha(p)
            (OUT/"failure.json").write_text(json.dumps(fail,indent=2,sort_keys=True)+"\n")
        raise
    finally:
        signal.alarm(0)

