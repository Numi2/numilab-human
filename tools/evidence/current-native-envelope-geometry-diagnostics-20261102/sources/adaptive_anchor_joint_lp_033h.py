#!/usr/bin/env python3
"""Extend frozen 032d source-only quadrature LP with exact 030c cut components."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, re, sys, time
from pathlib import Path
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
TASK=H/"current-source-carrier-joint-initialization-1054/cut-closure-1054"
OWNER32=TASK/"actual-032-all-witness-adaptive-joint-lp-001/adaptive_anchor_joint_lp_032d.py"
PREP32=TASK/"actual-032-all-witness-adaptive-joint-lp-001/run-008/preparation"
PROP32=TASK/"actual-032-all-witness-adaptive-joint-lp-001/run-008/proposal/adaptive-joint-weight-proposal-032d.npz"
CENSUS=H/"current-fractional-semantic-native-1047/actual-030c-component-census-001/run-001/component-census/component-census.json"
CUT_RUN=H/"current-fractional-semantic-native-1047/actual-030c-032d-diagnostic/run-001/run-manifest.json"
EXPORT_DEFAULT=TASK/"actual-030c-component-batch-export-005/outputs/export-001/run-001"
EXPORT_RUNNER=TASK/"actual-030c-component-batch-export-005/run_component_export_030c_005.py"
OUTROOT=TASK/"actual-033h-current030c-component-joint-lp-001"
EXPECTED_032D="1ca324fe7ff049ab4e5f5f4e01bff27dedc2ad3570bd36e3bcaf800520870974"
EXPECTED_CENSUS="d8ac2770b3fa5bde237aebe20a97a1b28bbae9a5ce1f58bc3406341305f13c62"
EXPECTED_CUT_RUN="dc598970234cac687249fa1040285c836380c86fc515ee5bb7919224521d3882"
EXPECTED_CENSUS_RAW_P1="16fd59b950211957ec3139363e0aa71ad870a583c3a71a5ab84c82d378d21912"
EXPECTED_CENSUS_EXPANDED_SUPPORT="bdd45ffb4b8e340be7bbcd97355f3668a8b0c019c17deb51fe78b184cc64dc08"
EXPECTED_PARENT_SOURCE="8e2424901318f25caca1540aeca412ca779fc014f68f49fe0cb6c17fb5aa0c1f"
EXPECTED_CHILD_MESH="9383228dfb17954490994b102055ddc7e8dc074c8584353ac9bf4928e353ccbd"
EXPECTED_FRAME="0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a"
EXPECTED_ADAPTIVE_BATCH="fa508d40fd2383d72185adb69aae032426a8555c8ea497593b0a63042faef203"
EXPECTED_TARGETS="14252ebfe9c5c42adb5f1641dbd3446a5d9db739620251eafa58be91e2945da6"
EXPECTED_EXPORT_RUNNER="ac6efe2a84136758ece21f92822e1a47825e928b12cbfc90800b3ab5fd6e8d23"
EXPECTED_EXPORT_CPP="fc0e880a5eba7955be7298d44cb432d2fcc3f350d25c72c5888ceacab6c59d4a"
EXPECTED_EXPORT_OWNER_HEADER="fc27ee95ea9e4ca3cdf956c90011ddea5e38b73c85b2046b4bf453187ea8cde2"
EXPECTED_EXPORT_CENSUS_HEADER="e65668ce19c9d325ac9fecd1703ddf209164cf58b5d0ceda18d71b4723aa990a"
EXPECTED_EXPORT_DEPS="c136a6f9fbda94de169c935778b0a22acf14de5df827ba8bbdd977705ef3fb62"
EXPECTED_EXPORT_BINARY="0b7262f30d5e39afdb274a99b7506b9c3ea046dc819a38e73107e515c4e3bf7a"
EXPECTED_PREP32="603857489fd5eef26ab8ec7bf3e3a499623b47121245d9d7f7e5b2f805d14078"
EXPECTED_PROP32="acde169351151266192500724ebf332fb9b6c1beba6c1c0245263603ed85fffa"
EXPECTED_SITES32="67d5b9cee6fb251a6e6be29928c4ce69f96eeaebf146a64feaed6e22c037aa44"
R=0.019353
NNEW=9
BASE_BITS=26


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def pin(p,d):
    p=Path(p)
    if not p.is_file() or sha(p)!=d: raise RuntimeError(f"pin mismatch: {p}")

def load32():
    pin(OWNER32,EXPECTED_032D)
    sp=importlib.util.spec_from_file_location("h706_032d_for_033h",OWNER32)
    if not sp or not sp.loader: raise RuntimeError("cannot load 032d")
    m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
    return m

def key(p,b):
    return (int(p),np.asarray(b,dtype=np.float64).tobytes())

def load_census(m,data):
    pin(CENSUS,EXPECTED_CENSUS)
    d=json.loads(CENSUS.read_text())
    owner=data["m"]
    # 031h's matrix digest and the 030c carrier-expanded digest are different
    # domains. The latter is the C++ hash over the pinned typed-support contents,
    # carrier-array digest and 032d packet digest; never compare them as aliases.
    if d.get("component_count")!=9 or d.get("typed_support_sha256")!=EXPECTED_CENSUS_EXPANDED_SUPPORT or d.get("expanded_typed_support_sha256")!=EXPECTED_CENSUS_EXPANDED_SUPPORT:
        raise RuntimeError("030c census expanded-carrier support pin mismatch")
    if d.get("raw_p1_support_sha256")!=EXPECTED_CENSUS_RAW_P1 or d.get("parent_mass_sha256")!=data["named"]["source330_mass"]["sha256"] or d.get("parent_source_sha256")!=EXPECTED_PARENT_SOURCE:
        raise RuntimeError("030c census source mass/source contribution/P1 pin mismatch")
    if d.get("child_mesh_sha256")!=EXPECTED_CHILD_MESH or d.get("region_sha256")!=data["named"]["child_skin_labels"]["sha256"] or d.get("coordinate_frame_sha256")!=EXPECTED_FRAME:
        raise RuntimeError("030c census child mesh/region/frame pin mismatch")
    if d.get("base_run_manifest_024_sha256")!=data["m"].EXPECTED_CURRENT027_MANIFEST or d.get("cut_run_manifest_025_sha256")!=EXPECTED_CUT_RUN or d.get("target_rows_sha256")!=EXPECTED_TARGETS:
        raise RuntimeError("030c census source-run/target pins mismatch")
    pin(CUT_RUN,EXPECTED_CUT_RUN)
    cut_manifest=json.loads(CUT_RUN.read_text())
    cut_named=cut_manifest.get("named_input_pins",{})
    shared=("candidate_fat_volume","candidate_skin_volume","child_fat_elements","child_fat_labels","child_fat_node",
        "child_skin_elements","child_skin_labels","child_skin_node","current_state","p1_fat_support","p1_skin_support",
        "projection_evidence","source330_labels","source330_mass","source330_nodes","source330_tets","source_centroid_slots")
    for name in shared:
        a=data["named"].get(name,{}).get("sha256")
        b=cut_named.get(name,{}).get("sha256")
        if not a or a!=b: raise RuntimeError(f"030c census and 032d matrix disagree on shared input {name}")
    if cut_named.get("adaptive_packet_batch",{}).get("sha256")!=EXPECTED_ADAPTIVE_BATCH:
        raise RuntimeError("030c census is not bound to the pinned 032d packet batch")
    comp_by=np.full(data["nchild"],-1,np.int16)
    components=sorted(d["components"],key=lambda x:int(x["component_id"]))
    if [int(x["component_id"]) for x in components]!=list(range(NNEW)): raise RuntimeError("030c IDs not 0..8")
    demands=[];targets=[];refs=[]
    for c in components:
        p=Path(c["target_rows_path"]);pin(p,c["target_rows_sha256"])
        rr=owner.read_targets(data["mod"],p,int(c["target_child_count"]))
        if np.any(rr<0)|np.any(rr>=data["nchild"])|np.any(comp_by[rr]>=0): raise RuntimeError("030c targets overlap/range error")
        comp_by[rr]=int(c["component_id"]);demands.append(float(c["minimum_demand_kg"]));targets.append(rr)
        refs.append({"component_id":int(c["component_id"]),"path":str(p),"sha256":c["target_rows_sha256"],
                     "rows":len(rr),"minimum_demand_kg":float(c["minimum_demand_kg"]),
                     "existing_anchor_capacity_kg":float(c["existing_anchor_capacity_kg"])})
    globalp=Path(d["target_rows_path"]);pin(globalp,EXPECTED_TARGETS)
    union=owner.read_targets(data["mod"],globalp,int(d["target_child_count"]))
    if int(np.count_nonzero(comp_by>=0))!=len(union) or not np.array_equal(np.flatnonzero(comp_by>=0),np.sort(union)):
        raise RuntimeError("030c nine targets do not exactly partition authenticated full-cut target rows")
    return d,comp_by,np.asarray(demands,np.float64),targets,refs

def read_export(m,data,comp_by,targets,export_dir):
    """Read exact owner-export routes and canonical candidate sites; output a bounded map."""
    source_mass=data["mass"];source_mat=data["material"];source_don=data["donor"];source_reg=data["region"]
    nparent=len(source_mass)
    pin(EXPORT_RUNNER,EXPECTED_EXPORT_RUNNER)
    runmf=export_dir/"run-manifest.json"
    if not runmf.is_file(): raise RuntimeError("030c batch export run-manifest missing")
    run=json.loads(runmf.read_text())
    if run.get("schema")!="numi.h706.current-030c-032d-component-owner-export.v1" or run.get("status")!="PASS_PROPOSAL_ONLY" or run.get("allocator_run") is not False or run.get("native_steps")!=0 or run.get("physical_admission") is not False:
        raise RuntimeError("030c batch export is not a source-only owner pass")
    if run.get("census",{}).get("sha256")!=EXPECTED_CENSUS or run.get("global_target",{}).get("sha256")!=EXPECTED_TARGETS or run.get("030c_run_manifest",{}).get("sha256")!="dc598970234cac687249fa1040285c836380c86fc515ee5bb7919224521d3882":
        raise RuntimeError("030c export manifest does not bind expected census/target/source run")
    build=export_dir.parent/"builds/030c-032d"
    deps=build/"dependencies.d";binary=build/"bin/current-fractional-component-batch-export-030c-032d-o3"
    pin(deps,EXPECTED_EXPORT_DEPS);pin(binary,EXPECTED_EXPORT_BINARY)
    deptext=re.sub(r"\\\s*\n", " ", deps.read_text())
    dep_paths=[Path(x.rstrip(":")) for x in deptext.split() if x.startswith("/")]
    if not any(x.name=="NumiHumanCurrentFractionalSupportNPZ_029.hpp" for x in dep_paths) or any(x.name=="NumiHumanCurrentFractionalSupportNPZ_011.hpp" for x in dep_paths):
        raise RuntimeError("compiler closure lacks 029 reader or includes stale 011 loader")
    expected_deps={"NumiHumanCurrentFractionalSupportNPZ_029.hpp":"84f23778ccde7cfddd85564a62056906b66dba09f9e57924a9eeae3a7bf14ca0",
        "ConservativeSpatialFractionalSemanticSupport.hpp":"fe75054637a78716ee015e74e82fb4e5c7d190690b6b04ffb2383e4426dfaa91",
        "ConservativeSpatialSourceCarrierQuadrature.hpp":"9d0afe443ec1d753eb5daaa82475b1dbd2afd12979812bc85af57ff3a22fa47a",
        "NumiHumanPhysicalSkinChildMassMapping.hpp":"1fce135c3fd0c5564f8358d464c54283f3164e70d66bbf34c86e8991774d304d"}
    expected_deps["CurrentSourceCarrierHallCutExport_1054.hpp"]=EXPECTED_EXPORT_OWNER_HEADER
    expected_deps["HallCutComponentCensus_006.hpp"]=EXPECTED_EXPORT_CENSUS_HEADER
    for name,digest in expected_deps.items():
        paths=[x for x in dep_paths if x.name==name]
        if not paths or all(not q.is_file() or sha(q)!=digest for q in paths): raise RuntimeError(f"compile dependency hash mismatch: {name}")
    source=run.get("source",{});sourcepath=Path(source.get("path",""))
    expected_source=export_dir.parent/"NumiHumanCurrentFractionalHallCutComponentBatchExportCLI_030c_001.cpp"
    if sourcepath.resolve()!=expected_source.resolve() or not sourcepath.is_file() or sha(sourcepath)!=EXPECTED_EXPORT_CPP or sha(sourcepath)!=source.get("sha256"): raise RuntimeError("compiled source path/hash mismatch")
    batchref=run.get("batch_report",{});batchpath=Path(batchref.get("path",""))
    if not batchpath.is_file() or sha(batchpath)!=batchref.get("sha256"): raise RuntimeError("batch report hash mismatch")
    batch=json.loads(batchpath.read_text())
    if batch.get("status")!="PASS_PROPOSAL_ONLY" or batch.get("component_count")!=NNEW or batch.get("allocator_run") is not False:
        raise RuntimeError("C++ batch owner did not pass nine component exports")
    comprefs=sorted(run.get("components",[]),key=lambda x:int(x["id"]))
    if [int(x["id"]) for x in comprefs]!=list(range(NNEW)) or len(batch.get("components",[]))!=NNEW:
        raise RuntimeError("run/batch component roster mismatch")
    witness={}; canonical={}
    child_union=set()
    eligible_rows=0; csvrows=0
    component_files=[]
    for cid,ref in enumerate(comprefs):
        p=Path(ref["coefficients_csv"]["path"])
        expected=export_dir/f"components/component-{cid:05d}/hall-cut-coefficients.csv"
        if p.resolve()!=expected.resolve() or not p.is_file() or sha(p)!=ref["coefficients_csv"]["sha256"] or p.stat().st_size!=int(ref["coefficients_csv"]["bytes"]):
            raise RuntimeError(f"component export path/hash/size mismatch {cid}")
        component_files.append(p)
        with p.open(newline="",encoding="utf-8") as f:
            rd=csv.DictReader(f)
            expect={"record_kind","source_parent_row","parent_input_row","material","donor","region","parent_mass_kg",
                    "bary0","bary1","bary2","bary3","source_x_m","source_y_m","source_z_m",
                    "cut_eligible","witness_child_row","witness_distance_m"}
            if not rd.fieldnames or not expect.issubset(rd.fieldnames): raise RuntimeError(f"component{cid} header incomplete")
            for row in rd:
                csvrows+=1
                pglobal=int(row["source_parent_row"]);plocal=int(row["parent_input_row"])
                if pglobal!=plocal or pglobal<0 or pglobal>=nparent: raise RuntimeError("export parent row alias/range mismatch")
                mat=int(row["material"]);don=int(row["donor"]);reg=int(row["region"])
                if (mat,don,reg)!=(int(source_mat[pglobal]),int(source_don[pglobal]),int(source_reg[pglobal])):
                    raise RuntimeError("export source tuple mismatch")
                if abs(float(row["parent_mass_kg"])-float(source_mass[pglobal]))>2e-15: raise RuntimeError("export parent mass mismatch")
                bary=np.array([float(row[f"bary{k}"]) for k in range(4)],np.float64)
                if np.any(~np.isfinite(bary)) or np.any(bary<0) or abs(float(bary.sum())-1)>2e-12: raise RuntimeError("export bary invalid")
                xyz=np.einsum("i,ij->j",bary,data["tetxyz"][pglobal])
                declared=np.array([float(row["source_x_m"]),float(row["source_y_m"]),float(row["source_z_m"])])
                if np.max(np.abs(xyz-declared))>2e-12: raise RuntimeError("export source point does not replay")
                eligible=int(row["cut_eligible"])
                if eligible not in (0,1): raise RuntimeError("export cut flag invalid")
                kind=row["record_kind"]
                if kind not in ("existing_packet_anchor","implicit_centroid","source_tet_candidate"):
                    raise RuntimeError("unexpected owner export record kind")
                if kind=="source_tet_candidate" and eligible:
                    k=key(pglobal,bary)
                    old=canonical.get(k)
                    if old is None: canonical[k]={"parent":pglobal,"bary":bary,"xyz":xyz.copy(),"kind":row["site_proof"]}
                    elif np.max(np.abs(old["xyz"]-xyz))>2e-12: raise RuntimeError("duplicate canonical site mismatch")
                if not eligible:
                    if row["witness_child_row"] or row["witness_distance_m"]: raise RuntimeError("ineligible export row has witness")
                    continue
                child=int(row["witness_child_row"]);dist=float(row["witness_distance_m"])
                if child<0 or child>=data["nchild"] or not math.isfinite(dist) or dist<0 or dist>R+1e-12:
                    raise RuntimeError("export witness range/distance invalid")
                if comp_by[child]!=cid: raise RuntimeError("owner witness outside its exact component target")
                witness.setdefault((pglobal,cid),set()).add(child);child_union.add(child);eligible_rows+=1
    if not eligible_rows: raise RuntimeError("owner exports contain no eligible witnesses")
    records=[]
    for (p,cid),children in witness.items():
        for child in children: records.append((p,cid,child,0.0,0))
    records=np.asarray(records,dtype=np.float64)
    rows,childxyz,childmat,rolelocal=data["m"].selected_child_geometry(
        np.asarray(sorted(child_union),np.int64),
        Path(data["named"]["child_skin_node"]["path"]),Path(data["named"]["child_skin_elements"]["path"]),
        Path(data["named"]["child_skin_labels"]["path"]),data["nchild"])
    # Every C++-authorized parent-child pair is independently checked against exact material,
    # P1 one-hop admission, target membership; sourcepoint-to-child distance is replayed in pass 2.
    p1=m.load_p1(data)
    m.validate_witness_edges(data,records,rows,childxyz,childmat,rolelocal,p1)
    rowpos={int(r):i for i,r in enumerate(rows)}
    # Re-read to replay each serialized source point and distance against actual child centroids.
    verified=0
    for cid,p in enumerate(component_files):
        with p.open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["cut_eligible"])!=1: continue
                parent=int(row["source_parent_row"]);child=int(row["witness_child_row"])
                b=np.array([float(row[f"bary{k}"]) for k in range(4)],np.float64)
                point=np.einsum("i,ij->j",b,data["tetxyz"][parent])
                actual=float(np.linalg.norm(point-childxyz[rowpos[child]]))
                claimed=float(row["witness_distance_m"])
                if abs(actual-claimed)>2e-12 or actual>R+1e-12:
                    raise RuntimeError("owner source-to-child distance fails independent coordinate replay")
                verified+=1
    if verified!=eligible_rows: raise RuntimeError("owner eligible-row replay count mismatch")
    return witness,canonical,rows,childxyz,{
        "export_runner":{"path":str(EXPORT_RUNNER),"sha256":sha(EXPORT_RUNNER)},
        "export_run_manifest":{"path":str(runmf),"sha256":sha(runmf)},
        "batch_owner_report":{"path":str(batchpath),"sha256":sha(batchpath)},
        "compile_dependency_file":{"path":str(deps),"sha256":sha(deps)},"binary":{"path":str(binary),"sha256":sha(binary)},
        "compiled_source":{"path":str(sourcepath),"sha256":sha(sourcepath)},
        "owner_header":{"path":str(export_dir.parent/"CurrentSourceCarrierHallCutExport_1054.hpp"),"sha256":sha(export_dir.parent/"CurrentSourceCarrierHallCutExport_1054.hpp")},
        "census_header":{"path":str(export_dir.parent/"HallCutComponentCensus_006.hpp"),"sha256":sha(export_dir.parent/"HallCutComponentCensus_006.hpp")},
        "component_csv_paths":[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in component_files],
        "csv_rows":csvrows,"eligible_row_count":eligible_rows,"independently_replayed_eligible_rows":verified,
        "unique_parent_component_groups":len(witness),"unique_parent_child_pairs":len(records),
        "canonical_candidate_site_count":len(canonical),"unique_child_witnesses":len(rows),
        "export_runner":{"path":str(EXPORT_RUNNER),"sha256":sha(EXPORT_RUNNER)}}

def route_new_components(ap,bary,pts,witness,rows,childxyz):
    n=len(ap);mask=np.zeros(n,np.uint64);cw=np.full((n,NNEW),-1,np.int32);cd=np.full((n,NNEW),np.inf,np.float64)
    if not n:return mask,cw,cd
    rowpos={int(r):i for i,r in enumerate(rows)}
    order=np.argsort(ap,kind="stable");sp=ap[order];bounds=np.r_[0,np.flatnonzero(sp[1:]!=sp[:-1])+1,len(sp)]
    for a,z in zip(bounds[:-1],bounds[1:]):
        parent=int(sp[a]);ix=order[a:z]
        points=pts[ix]
        for cid in range(NNEW):
            kids=sorted(witness.get((parent,cid),()))
            if not kids: continue
            xyz=childxyz[[rowpos[int(x)] for x in kids]]
            delta=points[:,None,:]-xyz[None,:,:]
            dm=np.sqrt(np.einsum("nmk,nmk->nm",delta,delta))
            arg=np.argmin(dm,axis=1);dmin=dm[np.arange(len(ix)),arg];good=dmin<=R+1e-12
            if np.any(good):
                out=ix[good];mask[out]|=np.uint64(1) << np.uint64(BASE_BITS+cid)
                cw[out,cid]=np.asarray(kids,np.int32)[arg[good]];cd[out,cid]=dmin[good]
    return mask,cw,cd

def route_old_candidates(m,data,comp11,records,ap,pts,rows,childxyz):
    n=len(ap);mask=np.zeros(n,np.uint32);cutw=np.full((n,15),-1,np.int32);cutd=np.full((n,15),np.inf,np.float64)
    curw=np.full((n,11),-1,np.int32);curd=np.full((n,11),np.inf,np.float64);gw=np.full(n,-1,np.int32);gd=np.full(n,np.inf,np.float64)
    rowpos={int(r):i for i,r in enumerate(rows)}
    children={};cutchildren={}
    for p,j,c,_,_ in records:
        p=int(p);j=int(j);c=int(c);children.setdefault(p,set()).add(c)
        if j>=0:cutchildren.setdefault((p,j),set()).add(c)
    order=np.argsort(ap,kind="stable");sp=ap[order];bounds=np.r_[0,np.flatnonzero(sp[1:]!=sp[:-1])+1,len(sp)]
    for a,z in zip(bounds[:-1],bounds[1:]):
        parent=int(sp[a]);ix=order[a:z];kids=sorted(children.get(parent,()))
        if not kids: continue
        kloc=np.asarray([rowpos[int(c)] for c in kids],np.int64)
        delta=pts[ix,None,:]-childxyz[kloc][None,:,:];dm=np.sqrt(np.einsum("nmk,nmk->nm",delta,delta))
        arg=np.argmin(dm,axis=1);dmin=dm[np.arange(len(ix)),arg];ok=dmin<=R+1e-12
        if np.any(ok): gw[ix[ok]]=np.asarray(kids,np.int32)[arg[ok]];gd[ix[ok]]=dmin[ok]
        kidarr=np.asarray(kids,np.int64);ci=comp11[kidarr]
        for j in range(15):
            ck=sorted(cutchildren.get((parent,j),()))
            if not ck: continue
            pos=np.searchsorted(kidarr,np.asarray(ck,np.int64));sub=dm[:,pos];arg=np.argmin(sub,axis=1);dmin=sub[np.arange(len(ix)),arg];ok=dmin<=R+1e-12
            if np.any(ok):mask[ix[ok]]|=np.uint32(1<<j);cutw[ix[ok],j]=np.asarray(ck,np.int32)[arg[ok]];cutd[ix[ok],j]=dmin[ok]
        for c in range(11):
            pos=np.flatnonzero(ci==c)
            if not len(pos):continue
            sub=dm[:,pos];arg=np.argmin(sub,axis=1);dmin=sub[np.arange(len(ix)),arg];ok=dmin<=R+1e-12
            if np.any(ok):mask[ix[ok]]|=np.uint32(1<<(15+c));curw[ix[ok],c]=kidarr[pos[arg[ok]]].astype(np.int32);curd[ix[ok],c]=dmin[ok]
    return mask,cutw,cutd,curw,curd,gw,gd

def post_blend(lp,prior,ap,bary,masks,pcount,epsilon=1e-6):
    order=np.argsort(ap,kind="stable");sp=ap[order];bounds=np.r_[0,np.flatnonzero(sp[1:]!=sp[:-1])+1,len(sp)]
    fixed=np.zeros(pcount,bool)
    for a,z in zip(bounds[:-1],bounds[1:]):fixed[int(sp[a])]=len(np.unique(masks[order[a:z]]))==1
    out=(1-epsilon)*lp+epsilon*prior
    for a,z in zip(bounds[:-1],bounds[1:]):
        p=int(sp[a])
        if fixed[p]:out[order[a:z]]=prior[order[a:z]]
    sums=np.bincount(ap,weights=out,minlength=pcount)
    moms=np.column_stack([np.bincount(ap,weights=out*bary[:,j],minlength=pcount) for j in range(4)])
    return out,fixed,{"epsilon":epsilon,"fixed_route_parents_restored":int(fixed.sum()),"weight_sum_error_max":float(np.max(np.abs(sums-1))),
        "barycentric_moment_error_max":float(np.max(np.abs(moms-.25))),"passed":bool(np.isfinite(out).all() and np.all(out>=0) and np.max(np.abs(sums-1))<=2e-10 and np.max(np.abs(moms-.25))<=2e-10)}

def prepare(out,export_dir):
    if out.exists():raise RuntimeError("refuse overwrite "+str(out))
    m=load32();data=m.load_inputs(m.load_base())
    pin(PREP32/"preparation-report.json",EXPECTED_PREP32)
    pin(PREP32/"adaptive-sites-032.npz",EXPECTED_SITES32)
    pin(PROP32,EXPECTED_PROP32)
    prep32=json.loads((PREP32/"preparation-report.json").read_text())
    # Current-11 row map comes from the immutable reconciled 032d preparation.
    comp11,dem11,refs11=m.current_component_map(data,prep32["current11_reconciliation"])
    d,comp_new,dem_new,targets,refs=load_census(m,data)
    witness,canonical,rows,childxyz,exportproof=read_export(m,data,comp_new,targets,export_dir)
    records=m.matrix_witness_pairs(data,comp11)
    # Include all matrix and new-export witness child rows for geometric routing.
    oldchildren=np.unique(records[:,2].astype(np.int64))
    allrows=np.unique(np.r_[rows,oldchildren])
    # Re-read exact centroids for the union; selected_child_geometry is deterministic and row keyed.
    rows,childxyz,childmat,rolelocal=data["m"].selected_child_geometry(
        allrows,Path(data["named"]["child_skin_node"]["path"]),Path(data["named"]["child_skin_elements"]["path"]),
        Path(data["named"]["child_skin_labels"]["path"]),data["nchild"])
    m.validate_witness_edges(data,records,rows,childxyz,childmat,rolelocal,m.load_p1(data))
    with np.load(PREP32/"adaptive-sites-032.npz",allow_pickle=False) as z:
        ca=z["source_parent_row"].astype(np.int64);cb=z["barycentric"].astype(np.float64);cxyz=z["source_point_world_m"].astype(np.float64)
        cm=z["route_mask_26"].astype(np.uint32);ccw=z["cut_witness_child_rows"].astype(np.int32);ccd=z["cut_witness_distance_m"].astype(np.float64)
        ccurw=z["current11_witness_child_rows"].astype(np.int32);ccurd=z["current11_witness_distance_m"].astype(np.float64)
        cgw=z["global_witness_child_row"].astype(np.int32);cgd=z["global_witness_distance_m"].astype(np.float64)
        cglobal=z["global_route"].astype(bool)
    if cglobal.shape!=(len(ca),) or not np.all(cglobal) or np.any(cgw<0) or np.any(~np.isfinite(cgd)) or np.any(cgd>R+1e-12):
        raise RuntimeError("032d adaptive primary global witnesses are incomplete or invalid")
    basefile=PREP32/"base-route-expansion-032.npz"
    m.pin(basefile,prep32["base_route_expansion_npz"]["sha256"])
    with np.load(basefile,allow_pickle=False) as z:
        bm=z["route_mask_26"].astype(np.uint32);bcw=z["current11_witness_child_rows"].astype(np.int32);bcd=z["current11_witness_distance_m"].astype(np.float64)
    # 032d's run-008 solution, not the pre-LP matrix prior, is the L1 baseline.
    with np.load(PROP32,allow_pickle=False) as z:
        priorbase=z["matrix_anchor_row_weights"].astype(np.float64)
        pcp=z["source_parent_row"].astype(np.int64);pcb=z["barycentric"].astype(np.float64);pw=z["weight"].astype(np.float64)
    if len(priorbase)!=len(data["parents"]) or np.any(priorbase<0) or not np.isfinite(priorbase).all():raise RuntimeError("032d base prior invalid")
    ckey={key(p,b):i for i,(p,b) in enumerate(zip(ca,cb))}
    if len(ckey)!=len(ca):raise RuntimeError("032d candidate pool exact key duplicated")
    priorcand=np.zeros(len(ca),np.float64)
    for p,b,w in zip(pcp,pcb,pw):
        ix=ckey.get(key(p,b))
        if ix is None:raise RuntimeError("032d positive proposal not in its pinned candidate pool")
        priorcand[ix]+=w
    if np.max(np.abs(np.bincount(ca,weights=priorcand,minlength=len(data["mass"]))+np.bincount(data["parents"],weights=priorbase,minlength=len(data["mass"]))-1))>2e-10:
        raise RuntimeError("032d proposal fails per-parent mass closure before extension")
    # New component witness routes for inherited matrix/adaptive anchors.
    basexyz=np.einsum("ni,nij->nj",data["baseb"],data["tetxyz"][data["parents"]])
    bn,bnw,bnd=route_new_components(data["parents"],data["baseb"],basexyz,witness,rows,childxyz)
    cn,cnw,cnd=route_new_components(ca,cb,cxyz,witness,rows,childxyz)
    base35=bm.astype(np.uint64)|(bn)
    cand35=cm.astype(np.uint64)|(cn)
    # Add unique canonical source-tet candidates, excluding every exact existing base/adaptive anchor.
    existing={key(p,b) for p,b in zip(data["parents"],data["baseb"])}
    existing.update(key(p,b) for p,b in zip(ca,cb))
    newsites=[v for k,v in canonical.items() if k not in existing]
    if newsites:
        npnew=np.asarray([x["parent"] for x in newsites],np.int64);nbnew=np.vstack([x["bary"] for x in newsites]);nxnew=np.vstack([x["xyz"] for x in newsites])
        nmask,ncw,ncd=route_new_components(npnew,nbnew,nxnew,witness,rows,childxyz)
        om,ocw,ocd,orw,ord_,ogw,ogd=route_old_candidates(m,data,comp11,records,npnew,nxnew,rows,childxyz)
        # A global primary route may come from either old matrix edges or new exact target witnesses.
        nglobal=nmask!=0
        ogvalid=ogw>=0
        globalok=nglobal|ogvalid
        nkeep=np.flatnonzero(globalok)
        npnew=npnew[nkeep];nbnew=nbnew[nkeep];nxnew=nxnew[nkeep];nmask=nmask[nkeep];ncw=ncw[nkeep];ncd=ncd[nkeep]
        om=om[nkeep];ocw=ocw[nkeep];ocd=ocd[nkeep];orw=orw[nkeep];ord_=ord_[nkeep]
        choose_new=~ogvalid[nkeep]
        ngw=ogw[nkeep].copy();ngd=ogd[nkeep].copy()
        # For missing old global routes, select nearest among all new component witnesses.
        if np.any(choose_new):
            # route_new's first eligible component witness is guaranteed to be within R.
            subcw=ncw[choose_new];subcd=ncd[choose_new];arg=np.argmin(subcd,axis=1);rr=np.arange(int(choose_new.sum()))
            ngw[choose_new]=subcw[rr,arg];ngd[choose_new]=subcd[rr,arg]
        nmask35=om.astype(np.uint64)|nmask
    else:
        npnew=np.empty(0,np.int64);nbnew=np.empty((0,4),np.float64);nxnew=np.empty((0,3),np.float64)
        nmask35=np.empty(0,np.uint64);ncw=np.empty((0,NNEW),np.int32);ncd=np.empty((0,NNEW),np.float64)
        ocw=np.empty((0,15),np.int32);ocd=np.empty((0,15),np.float64);orw=np.empty((0,11),np.int32);ord_=np.empty((0,11),np.float64)
        ngw=np.empty(0,np.int32);ngd=np.empty(0,np.float64)
    allp=np.r_[data["parents"],ca,npnew]
    allb=np.vstack([data["baseb"],cb,nbnew])
    allxyz=np.vstack([basexyz,cxyz,nxnew])
    allmask=np.r_[base35,cand35,nmask35].astype(np.uint64)
    allglobal=np.r_[data["global_route"],cglobal,np.ones(len(npnew),bool)]
    prior=np.r_[priorbase,priorcand,np.zeros(len(npnew),np.float64)]
    # New routes are exactly bits 26..34; 0..25 remain byte-equivalent to 032d.
    if np.any((allmask>>np.uint64(35))!=0):raise RuntimeError("route mask exceeds declared 35 constraints")
    reqs=np.r_[data["demands15"],dem11,dem_new]
    names=list(data["m"].CUT_NAMES)+list(m.CURRENT_NAMES)+[f"current030c_component_{i:02d}" for i in range(NNEW)]
    if len(reqs)!=35:raise RuntimeError("expected 35 distinct cut constraints")
    out.mkdir(parents=True)
    arrays=out/"source-anchors-033h.npz"
    np.savez_compressed(arrays,parent_row=allp.astype(np.int32),barycentric=allb,source_point_world_m=allxyz,
        route_mask_35=allmask,global_route=allglobal.astype(np.uint8),prior_weight=prior,
        matrix_row_count=np.asarray([len(data["parents"])],np.int32),candidate032_row_count=np.asarray([len(ca)],np.int32),
        new_canonical_row_count=np.asarray([len(npnew)],np.int32))
    # Save canonical separate arrays and exact 030c witness routing.
    np.savez_compressed(out/"new-canonical-sites-033h.npz",source_parent_row=npnew.astype(np.int32),barycentric=nbnew,
        source_point_world_m=nxnew,new_component_witness_child_rows=ncw,new_component_witness_distance_m=ncd,
        historical_cut_witness_child_rows=ocw,historical_cut_witness_distance_m=ocd,
        current11_witness_child_rows=orw,current11_witness_distance_m=ord_,
        global_witness_child_row=ngw,global_witness_distance_m=ngd,route_mask_35=nmask35)
    # Save the exact route evidence needed to serialize all positive adaptive weights after solve.
    base_route_out=out/"base-routes-033h.npz"
    np.savez_compressed(base_route_out,route_mask_35=base35,new_component_witness_child_rows=bnw,new_component_witness_distance_m=bnd,
        old_current11_witness_child_rows=bcw,old_current11_witness_distance_m=bcd)
    candidate_route_out=out/"candidate032-routes-033h.npz"
    np.savez_compressed(candidate_route_out,source_parent_row=ca.astype(np.int32),barycentric=cb,source_point_world_m=cxyz,
        route_mask_35=cand35,cut_witness_child_rows=ccw,cut_witness_distance_m=ccd,
        current11_witness_child_rows=ccurw,current11_witness_distance_m=ccurd,
        new_component_witness_child_rows=cnw,new_component_witness_distance_m=cnd,
        global_witness_child_row=cgw,global_witness_distance_m=cgd)
    canonical_path=out/"new-canonical-sites-033h.npz"
    report={"schema":"numi.h706.current030c-nine-component-032d-joint-lp-preparation.v1","status":"PASS_SOURCE_ONLY_PREPARATION",
        "physical_admission":False,"native_steps":0,"allocator_run":False,"radius_m":R,
        "input_pins":{"032d_owner":{"path":str(OWNER32),"sha256":sha(OWNER32)},
            "032d_preparation":{"path":str(PREP32/"preparation-report.json"),"sha256":sha(PREP32/"preparation-report.json")},
            "032d_proposal":{"path":str(PROP32),"sha256":sha(PROP32)},
            "030c_census":{"path":str(CENSUS),"sha256":sha(CENSUS)},
            "030c_cut_run_manifest":{"path":str(CUT_RUN),"sha256":sha(CUT_RUN)},
            "030c_export_csvs":exportproof["component_csv_paths"],
            "032d_matrix_typed_support_sha256":data["m"].EXPECTED_SUPPORT,
            "030c_raw_p1_support_sha256":d["raw_p1_support_sha256"],
            "030c_carrier_expanded_typed_support_sha256":d["expanded_typed_support_sha256"],
            "030c_support_hash_domain":"numi.human.source-tetra-centroid-carrier-expanded-typed-support.v1 = domain tag + supportContentsSHA + carrierArraySHA + sourceCarrierPacketSHA; intentionally distinct from the 031h matrix typed-support digest",
            "source_parent_mass_sha256":data["manifest"]["input_named_pins"]["source330_mass"]["sha256"],
            "source_parent_labels_sha256":data["manifest"]["input_named_pins"]["source330_labels"]["sha256"]},
        "new_components":refs,"historical_and_existing_components":{"count":26,"names":list(data["m"].CUT_NAMES)+list(m.CURRENT_NAMES),"demands_kg":np.r_[data["demands15"],dem11].tolist()},
        "objective":{"constraint_count":35,"names":list(data["m"].CUT_NAMES)+list(m.CURRENT_NAMES)+[f"current030c_component_{i:02d}" for i in range(NNEW)],"demands_kg":reqs.tolist()},
        "owner_export_replay":exportproof,"source_anchor_counts":{"base_matrix":len(data["parents"]),"032d_adaptive_pool":len(ca),"positive_032d_prior_adaptive":int(np.count_nonzero(priorcand)),
            "new_canonical_sites":len(npnew),"total_lp_anchor_rows":len(allp),"routeable_new_components_by_component":[int(np.count_nonzero((allmask&(np.uint64(1)<<np.uint64(BASE_BITS+i)))!=0)) for i in range(NNEW)]},
        "candidate_arrays":{"path":str(arrays),"sha256":sha(arrays),"bytes":arrays.stat().st_size},
        "base_routes":{"path":str(base_route_out),"sha256":sha(base_route_out),"bytes":base_route_out.stat().st_size},
        "candidate032_routes":{"path":str(candidate_route_out),"sha256":sha(candidate_route_out),"bytes":candidate_route_out.stat().st_size},
        "canonical_routes":{"path":str(canonical_path),"sha256":sha(canonical_path),"bytes":canonical_path.stat().st_size},
        "limitations":["The LP will use 35 independent cut-capacity inequalities, not a shared child-flow proof.",
            "New routes only use exact owner-exported source-parent/target-child witnesses; candidate anchor distances are recomputed to those witnesses.",
            "No geometry, source mass, density, donor/region semantics, radius, or native allocator state is changed."]}
    rp=out/"preparation-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"report":str(rp),"sha256":sha(rp),"anchors":len(allp),"newcanonical":len(npnew),
        "owner_rows":exportproof["csv_rows"],"newcomponent_routes":report["source_anchor_counts"]["routeable_new_components_by_component"]},sort_keys=True))
    return 0

def solve(prepdir,out,time_limit,expected_sha):
    if out.exists():raise RuntimeError("refuse overwrite "+str(out))
    m=load32();data=m.load_inputs(m.load_base());pr=prepdir/"preparation-report.json";pin(pr,expected_sha)
    rep=json.loads(pr.read_text())
    if rep.get("status")!="PASS_SOURCE_ONLY_PREPARATION":raise RuntimeError("prep status not PASS")
    arrays=prepdir/"source-anchors-033h.npz";pin(arrays,rep["candidate_arrays"]["sha256"])
    with np.load(arrays,allow_pickle=False) as z:
        ap=z["parent_row"].astype(np.int64);bary=z["barycentric"].astype(np.float64);mask=z["route_mask_35"].astype(np.uint64)
        prior=z["prior_weight"].astype(np.float64);globalroute=z["global_route"].astype(bool)
        nbase=int(z["matrix_row_count"][0]);ncand=int(z["candidate032_row_count"][0]);nnew=int(z["new_canonical_row_count"][0])
    if ap.shape!=(len(prior),) or bary.shape!=(len(ap),4) or mask.shape!=(len(ap),):raise RuntimeError("LP array shape mismatch")
    if np.any(ap<0)|np.any(ap>=len(data["mass"])) or np.any(bary<0) or np.max(np.abs(bary.sum(axis=1)-1))>2e-12:raise RuntimeError("LP source anchor invalid")
    demands=np.asarray(rep["objective"]["demands_kg"],np.float64) if "objective" in rep else None
    if demands is None:
        # Reconstruct exact frozen 15+11+9 demand order from validated reports.
        demands=np.r_[data["demands15"],np.asarray([x["minimum_demand_kg"] for x in rep["current11_components"]],np.float64),
                     np.asarray([x["minimum_demand_kg"] for x in rep["new_components"]],np.float64)]
    if len(demands)!=35:raise RuntimeError("35 cut demand vector missing")
    names=list(data["m"].CUT_NAMES)+list(m.CURRENT_NAMES)+[f"current030c_component_{i:02d}" for i in range(NNEW)]
    print(f"LP_START anchors={len(ap)} parents={len(data['mass'])} constraints={len(demands)}",flush=True)
    t=time.monotonic();res,w,lr=m.lp_solve_lexicographic(data["m"],data["mass"],ap,bary,mask,globalroute,demands,prior,time_limit);elapsed=time.monotonic()-t
    out.mkdir(parents=True)
    blend=None;output=None
    if len(w):
        wf,fixed,blend=post_blend(np.asarray(w,np.float64),prior,ap,bary,mask,len(data["mass"]),1e-6)
        caps=np.array([np.sum(data["mass"][ap[(mask&(np.uint64(1)<<np.uint64(j)))!=0]]*wf[(mask&(np.uint64(1)<<np.uint64(j)))!=0]) for j in range(35)])
        req=demands+data["m"].MARGIN;short=np.maximum(0,req-caps)
        blend.update({"capacity_after_restore_and_blend_kg":caps.tolist(),"shortfall_after_restore_and_blend_kg":short.tolist(),
            "all_35_constraints_met_after_blend":bool(np.all(short<=1e-10))})
        base_route=prepdir/"base-routes-033h.npz";pin(base_route,rep["base_routes"]["sha256"])
        cand_route=prepdir/"candidate032-routes-033h.npz";pin(cand_route,rep["candidate032_routes"]["sha256"])
        canon_path=prepdir/"new-canonical-sites-033h.npz";pin(canon_path,rep["canonical_routes"]["sha256"])
        with np.load(base_route,allow_pickle=False) as bq:
            bnewmask=bq["route_mask_35"];bnw=bq["new_component_witness_child_rows"];bnd=bq["new_component_witness_distance_m"]
            bcw=bq["old_current11_witness_child_rows"];bcd=bq["old_current11_witness_distance_m"]
        with np.load(cand_route,allow_pickle=False) as cq:
            cparent=cq["source_parent_row"];cbary=cq["barycentric"];cpoint=cq["source_point_world_m"]
            cmask=cq["route_mask_35"];ccw=cq["cut_witness_child_rows"];ccd=cq["cut_witness_distance_m"]
            ccurw=cq["current11_witness_child_rows"];ccurd=cq["current11_witness_distance_m"]
            cnw=cq["new_component_witness_child_rows"];cnd=cq["new_component_witness_distance_m"]
            cgw=cq["global_witness_child_row"];cgd=cq["global_witness_distance_m"]
        with np.load(canon_path,allow_pickle=False) as q:
            pnew=q["source_parent_row"];bnew=q["barycentric"];xnew=q["source_point_world_m"]
            nm=q["route_mask_35"];ncw=q["new_component_witness_child_rows"];ncd=q["new_component_witness_distance_m"]
            hw=q["historical_cut_witness_child_rows"];hd=q["historical_cut_witness_distance_m"]
            rw=q["current11_witness_child_rows"];rd=q["current11_witness_distance_m"]
            ngw=q["global_witness_child_row"];ngd=q["global_witness_distance_m"]
        selcand=np.flatnonzero(wf[nbase:nbase+ncand]>0);selnew=np.flatnonzero(wf[nbase+ncand:]>0)
        op=out/"adaptive-joint-weight-proposal-033h.npz"
        np.savez_compressed(op,source_parent_row=np.r_[cparent[selcand],pnew[selnew]],
            barycentric=np.vstack([cbary[selcand],bnew[selnew]]),weight=np.r_[wf[nbase:nbase+ncand][selcand],wf[nbase+ncand:][selnew]],
            source_point_world_m=np.vstack([cpoint[selcand],xnew[selnew]]),
            route_mask_35=np.r_[cmask[selcand],nm[selnew]],
            cut_witness_child_rows=np.vstack([ccw[selcand],hw[selnew]]),cut_witness_distance_m=np.concatenate([ccd[selcand],hd[selnew]]),
            current11_witness_child_rows=np.vstack([ccurw[selcand],rw[selnew]]),current11_witness_distance_m=np.vstack([ccurd[selcand],rd[selnew]]),
            new_component_witness_child_rows=np.vstack([cnw[selcand],ncw[selnew]]),new_component_witness_distance_m=np.vstack([cnd[selcand],ncd[selnew]]),
            global_witness_child_row=np.r_[cgw[selcand],ngw[selnew]],global_witness_distance_m=np.r_[cgd[selcand],ngd[selnew]],
            matrix_anchor_row_weights=wf[:nbase],matrix_anchor_route_mask_35=mask[:nbase],
            matrix_anchor_cut_witness_child_rows=data["wit"].astype(np.int32),matrix_anchor_cut_witness_distance_m=data["dist"],
            matrix_anchor_current11_witness_child_rows=bcw,matrix_anchor_current11_witness_distance_m=bcd,
            matrix_anchor_new_component_witness_child_rows=bnw,matrix_anchor_new_component_witness_distance_m=bnd,
            matrix_anchor_global_route=data["ints"]["global_route"].astype(np.uint8),
            matrix_anchor_global_witness_child_row=data["ints"]["global_witness_child_row"].astype(np.int32),
            matrix_anchor_global_witness_distance_m=data["fl"]["global_witness_distance_m"],matrix_anchor_prior_weight=prior[:nbase])
        output={"path":str(op),"sha256":sha(op),"bytes":op.stat().st_size,"positive_032d_adaptive_rows":len(selcand),"positive_new_canonical_rows":len(selnew)}
    if blend:
        caps=np.asarray(blend["capacity_after_restore_and_blend_kg"]);short=np.asarray(blend["shortfall_after_restore_and_blend_kg"])
    else: caps=short=np.full(35,np.nan)
    results=[{"index":j,"name":names[j],"demand_kg":float(demands[j]),"capacity_kg":float(caps[j]),"shortfall_kg":float(short[j]),
              "routed_anchor_count":int(np.count_nonzero(mask&(np.uint64(1)<<np.uint64(j))))} for j in range(35)]
    passed=bool(res.success and lr.get("passed_parent_moments") and lr.get("passed_slack_replay") and blend and blend["passed"] and output)
    report={"schema":"numi.h706.current030c-nine-component-032d-joint-lp.v1","status":"PASS_SOURCE_ONLY_35_CUT_SLACK_PROPOSAL" if passed else "INCONCLUSIVE_SOURCE_ONLY_35_CUT_LP",
        "physical_admission":False,"native_steps":0,"allocator_run":False,"objective":{"constraint_count":35,"names":names,"demands_kg":demands.tolist(),
            "margin_kg":float(data["m"].MARGIN),"minimized":"sum normalized deficits, then mass-weighted L1 change from the 032d run008 solution"},
        "input_pins":{"preparation_report":{"path":str(pr),"sha256":sha(pr)},"anchor_arrays":{"path":str(arrays),"sha256":sha(arrays)},
            "032d_owner":{"path":str(OWNER32),"sha256":sha(OWNER32)}},
        "solver":{"method":"two-stage sparse SciPy HiGHS LP; exact per-parent sum and first moments; uint64 route mask","status":str(res.message),"status_code":int(res.status),
            "optimal":bool(res.success),"elapsed_seconds":elapsed,"lexicographic":lr.get("lexicographic"),"anchor_variables":len(ap),"parent_count":len(data["mass"])},
        "lp_replay_before_blend":lr,"prior_support_preservation":blend,"cut_results":results,"output":output,
        "limitations":["This is a source-only independent-cut capacity relaxation, not the shared-flow allocator. A PASS status only means the 35 separate capacities can meet their lower bounds under per-parent mass and first-moment constraints.",
            "The output retains exact source-tet coordinates and parent tuple/radius witnesses; no geometry, mass, density, material, donor, region, radius, or native physics changed.",
            "A fresh 1047 shared-capacity flow remains the deciding test; no physical admission or native step is claimed."]}
    rp=out/"joint-lp-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    ex={"status":report["status"],"physical_admission":False,"native_steps":0,"allocator_run":False,"report":{"path":str(rp),"sha256":sha(rp)},"output":output,"elapsed_seconds":elapsed}
    (out/"execution.json").write_text(json.dumps(ex,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"report":str(rp),"sha256":sha(rp),"output":output,"elapsed_seconds":elapsed,
        "all35constraints_met":None if not blend else blend["all_35_constraints_met_after_blend"]},sort_keys=True))
    return 0 if passed else 3

def self_test():
    # 35th constraint must survive the uint64 route representation; exact parent
    # grouping prevents a witness for one source tetra from leaking to another.
    rows=np.array([101,202],np.int64)
    childxyz=np.array([[0.0,0,0],[0.05,0,0]],np.float64)
    witness={(0,8):{101},(1,0):{202}}
    ap=np.array([0,0,1],np.int64)
    bary=np.array([[.25]*4,[1,0,0,0],[.25]*4],np.float64)
    pts=np.array([[.01,0,0],[.015,0,0],[.05,0,0]],np.float64)
    mask,cw,cd=route_new_components(ap,bary,pts,witness,rows,childxyz)
    assert mask.dtype==np.uint64 and int(mask[0])==(1<<34) and int(mask[1])==(1<<34) and int(mask[2])==(1<<26)
    assert cw[0,8]==101 and cw[2,0]==202 and np.all(cd[[0,1,2],[8,8,0]]<=R)
    # Convex prior restoration must preserve each parent's mass and all four moments.
    ap2=np.array([0]*5+[1]*4,np.int64)
    b2=np.vstack([np.eye(4),np.full((1,4),.25),np.eye(4)])
    prior=np.array([.2]*5+[.25]*4,np.float64)
    lp=np.array([.1]*4+[.6]+[.25]*4,np.float64)
    masks=np.array([1,2,4,8,16]+[1]*4,dtype=np.uint64)
    out,fixed,stats=post_blend(lp,prior,ap2,b2,masks,2,1e-6)
    sums=np.bincount(ap2,weights=out,minlength=2)
    moms=np.column_stack([np.bincount(ap2,weights=out*b2[:,j],minlength=2) for j in range(4)])
    assert fixed.tolist()==[False,True] and stats["passed"]
    assert np.max(np.abs(sums-1))<2e-12 and np.max(np.abs(moms-.25))<2e-12
    assert np.array_equal(out[5:],prior[5:])
    # Exercise the real 032d lexicographic LP using its 031h base owner and bit 34.
    owner=load32();base=owner.load_base()
    demands=np.zeros(35,np.float64);demands[0]=.5;demands[34]=.5
    res,w,lp=owner.lp_solve_lexicographic(base,np.array([1.0]),np.array([0],np.int64),
        np.array([[.25]*4]),np.array([1 | (1<<34)],np.uint64),np.array([True]),demands,np.array([1.0]),10.0)
    assert res.success and len(w)==1 and abs(w[0]-1.0)<1e-9
    assert lp["passed_parent_moments"] and lp["passed_slack_replay"]
    assert lp["capacity_kg"][0]>=.5 and lp["capacity_kg"][34]>=.5
    print(json.dumps({"status":"PASS_033H_SYNTHETIC_CONTROLS","uint64_bit34":True,"parent_group_isolation":True,
        "prior_restore_and_blend_moment_gate":stats,"35_cut_lp_bit34":True},sort_keys=True))

def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("prepare","solve"));p.add_argument("--self-test",action="store_true");p.add_argument("--out-dir",type=Path)
    p.add_argument("--export-dir",type=Path,default=EXPORT_DEFAULT);p.add_argument("--prepared-dir",type=Path);p.add_argument("--expected-preparation-sha256")
    p.add_argument("--time-limit-seconds",type=float,default=300.)
    a=p.parse_args()
    if a.self_test:self_test();return 0
    if a.mode is None:raise RuntimeError("--mode is required unless running --self-test")
    if a.out_dir is None:raise RuntimeError("--out-dir is required")
    if a.mode=="prepare":return prepare(a.out_dir.resolve(),a.export_dir.resolve())
    if a.prepared_dir is None or not a.expected_preparation_sha256:raise RuntimeError("solve requires prepared-dir and expected-preparation-sha256")
    return solve(a.prepared_dir.resolve(),a.out_dir.resolve(),a.time_limit_seconds,a.expected_preparation_sha256)
if __name__=="__main__":raise SystemExit(main())

