#!/usr/bin/env python3
"""1046d: positive source-tetra anchor refinement for bounded current receiver coverage."""
import argparse,csv,hashlib,json,sys,importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linprog
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
_spec=importlib.util.spec_from_file_location("bridge1046",H/"current-source-carrier-bridge-1046.py")
B=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(B)
PROJDIR=H/"current-native-1024-root/current-433-1044/mass-group-audit-006"
PROJ=PROJDIR/"qualified-projection-support.csv"
PROJREPORT=PROJDIR/"exact-tuple-projection-report.json"
PROJOWNER=PROJDIR/"audit_exact_tuple_projection_006.py"
PARENT_PACKET={
 "Skin":H/"current-source-carrier-bridge-1046c-r5/skin-current-source-quadrature-1046c.npz",
 "Fat":H/"current-source-carrier-bridge-1046c-r5/fat-current-source-quadrature-1046c.npz",
}
PARENT_RECEIPT={
 "Skin":H/"current-source-carrier-bridge-1046c-r5/skin-current-source-quadrature-1046c-receipt.json",
 "Fat":H/"current-source-carrier-bridge-1046c-r5/fat-current-source-quadrature-1046c-receipt.json",
}
PARENT_WITNESS={
 "Skin":H/"current-native-1024-root/current-433-1044/mass-group-audit-011/carrier-witnesses-extended-011-skin.csv",
 "Fat":H/"current-native-1024-root/current-433-1044/mass-group-audit-011/carrier-witnesses-extended-011-fat.csv",
}
EXPECTED_PARENT_PACKET_SHA={"Skin":"12af1b30eb5bf015a67f78b219c0d6aa0a28e3042f051520e960f18cb007e91e","Fat":"33aaf568d056248788d17d8e5b4f36f0c31a4dc3b362ff52c3339170d7f9546f"}
EXPECTED_PARENT_WITNESS_SHA={"Skin":"9e3f4882d618786a37b416b03ae589e8b701b524af855984c8977e2bffe9ad67","Fat":"5b13685e14316a987168471f60d9438dd7b456273b2a330b40e19d848864cd80"}
REPROJECTOR=H/"batched_reference_tet_projection_149_root.py"
SCHEMA="numi.h706.current-source-carrier-rebind.1046h.v1"
NEW_CONSTRUCTION="source_tetra_anchor_refinement_1046d"
EDGE_CONSTRUCTION="current_source_carrier_edge_support"
ZERO_CHILDREN_BY_PARENT={
  9438:[637056],9439:[666620],
  9628:[722355,722356,722358,722363,722365,722366,722367,722369,722372],
  19252:[849740],19253:[781973],19789:[1330159],19631:[1432859],
}
CAPACITY_PARENTS_BY_ROLE={"Skin":set(),"Fat":{18441,18516}}
REFINEMENT_PARENTS={
  15:set(k for k in ZERO_CHILDREN_BY_PARENT if k<14316),
  16:(set(k for k in ZERO_CHILDREN_BY_PARENT if k>=14316)|CAPACITY_PARENTS_BY_ROLE["Fat"]),
}
WITNESS_FIELDS=["record_kind","role","carrier_construction","packet_carrier_row","local_parent_row","global_source330_tet_row","child_local_row","child_global_row","material","donor","region","provenance_mode","p1_group_id","projection_csv_row","bary0","bary1","bary2","bary3","weight","source_x_m","source_y_m","source_z_m","child_centroid_x_m","child_centroid_y_m","child_centroid_z_m","distance_m"]

def load_projection(role, labels, node_rows, current, mats, regions, donors, cglobal, cc, p1ro, p1rid, p1tuples, used_groups):
    report=json.loads(PROJREPORT.read_text())
    if report.get("schema")!="numi.h706.436-missing-group-exact-current-centroid-source-tuple-projection.v1" or report.get("status")!="PASS_EXACT_CONSERVATIVE_ENVELOPE_PROJECTION_CENSUS":
        raise ValueError("projection report schema/status")
    if report.get("geometry_or_source_mutated") is not False or report.get("physical_admission") is not False or report.get("native_steps")!=0:
        raise ValueError("projection report claims mutation/admission")
    qmeta=report.get("qualified_projection_csv",{})
    if qmeta.get("path")!=str(PROJ) or qmeta.get("sha256")!=B.sha(PROJ) or qmeta.get("rows")!=55336:
        raise ValueError("projection CSV report pin/count mismatch")
    if report.get("counts",{}).get("qualified_group_receiver_pairs")!=55336 or report.get("counts",{}).get("no_P1_source_groups")!=51:
        raise ValueError("projection census counts differ")
    if B.sha(PROJOWNER)!=report.get("producer",{}).get("sha256"):
        raise ValueError("projection owner source hash mismatch")
    # Authenticate every report input which defines source, current row domains,
    # semantic labels, and exact 217 registration geometry.
    expected={
      "candidate":B.CAND,"state":B.STATE,"parent_mass":B.TAB,
      "parent_nodes":B.TAB.with_name(B.TAB.name+".nodes.csv"),
      "parent_tets":B.TAB.with_name(B.TAB.name+".tets.csv"),
      "child_node":Path(report["input_pins"]["child_node"]["path"]),
      "child_ele":Path(report["input_pins"]["child_ele"]["path"]),
      "child_labels_433":Path(report["input_pins"]["child_labels_433"]["path"]),
      "source_proof_808":Path(report["input_pins"]["source_proof_808"]["path"]),
      "v2_group_audit":Path(report["input_pins"]["v2_group_audit"]["path"]),
    }
    for key,path in expected.items():
        if not path.is_file() or B.sha(path)!=report["input_pins"][key]["sha256"]:
            raise ValueError("projection input pin mismatch "+key)
    # Build the actual admitted P1 row map; projections may only serve groups
    # with no exact P1 child group slot.
    tuple_to_gid={tuple(map(int,x)):i for i,x in enumerate(p1tuples)}
    rows_by_group={}
    all_groups=set();all_pairs=0;seen_pairs=set()
    with open(PROJ,newline="") as f:
        for row_index,row in enumerate(csv.DictReader(f)):
            mat=int(row["source_material"]);donor=int(row["source_donor"]);reg=int(row["source_region"]);tup=(mat,donor,reg)
            all_groups.add(tup)
            if (tup,int(row["receiver_child_tet_row"])) in seen_pairs:raise ValueError("duplicate exact projection tuple-child pair")
            seen_pairs.add((tup,int(row["receiver_child_tet_row"])));all_pairs+=1
            if mat not in (15,16) or row["projection_status"]!="PASS_EXACT_CURRENT_CENTROID_SOURCE_TUPLE_PROJECTION":
                raise ValueError("invalid exact tuple projection row")
            if int(row["receiver_current_material"])!=mat or int(row["receiver_current_region"])!=reg:
                raise ValueError("projection receiver fails exact material/region")
            if mat!=B.MAT[role]:
                continue
            global_child=int(row["receiver_child_tet_row"])
            local=int(np.searchsorted(cglobal,global_child))
            if local>=len(cglobal) or int(cglobal[local])!=global_child:
                raise ValueError("projection receiver outside exact role child row domain")
            if int(mats[global_child])!=mat or int(regions[global_child])!=reg:
                raise ValueError("projection row current material/region differs from candidate")
            if int(donors[global_child])!=int(row["receiver_current_donor"]):
                raise ValueError("projection current donor row does not match candidate")
            xyz=np.asarray([float(row[f"receiver_centroid_{a}_m"]) for a in "xyz"])
            if np.max(np.abs(xyz-cc[local]))>2e-12:
                raise ValueError("projection receiver centroid differs from exact candidate F32 child")
            # Source projection record must be a valid point in the exact named
            # source330 tetrahedron and its reported distance must replay.
            srcrow=int(row["source_projection_parent_tet_row"])
            if srcrow<0 or srcrow>=len(labels):raise ValueError("projection source tet row range")
            srctup=tuple(int(labels[srcrow][k]) for k in ("material_index","geometric_donor_body","anatomical_region_id"))
            if srctup!=tup:raise ValueError("projection source tet tuple mismatch")
            bary=np.asarray([float(row[f"source_projection_bary{k}"]) for k in range(4)])
            if np.any(bary<0) or abs(float(bary.sum())-1)>2e-12:raise ValueError("projection source barycentric")
            point=bary@current[node_rows[srcrow]]
            claimed=np.asarray([float(row[f"projected_source_{a}_m"]) for a in "xyz"])
            dist=float(np.linalg.norm(point-cc[local]))
            if np.max(np.abs(point-claimed))>2e-12 or abs(dist-float(row["exact_projection_distance_m"]))>2e-12 or dist>B.R+1e-12:
                raise ValueError("projection row source point/distance does not replay")
            # One-hop P1 region admission remains mandatory for each receiver.
            if reg not in p1rid[p1ro[local]:p1ro[local+1]]:
                raise ValueError("projection receiver is outside authenticated one-hop region list")
            gid=tuple_to_gid.get(tup)
            if gid is not None and gid in used_groups:
                raise ValueError("projection row offered for a tuple that already has exact P1 slots")
            rows_by_group.setdefault(tup,[]).append((local,row_index,dist))
    groups=set(rows_by_group)
    if len(all_groups)!=51:
        raise ValueError(f"projection source group coverage expected 51, found {len(all_groups)}")
    if all_pairs!=55336:raise ValueError(f"projection CSV global row count mismatch {all_pairs}")
    pair_count=sum(len(v) for v in rows_by_group.values())
    return rows_by_group, B.sha(PROJ), B.sha(PROJREPORT)

def regenerate_degree1(parent_local, parent_global, allowed, child_centroids, source_node_xyz, source_tet_nodes):
    # Project each exact-tuple child centroid onto the source tetra, then move
    # toward tetra vertices/edges/interior only as far as the same receiver
    # remains within the unchanged transport cap.
    spec=importlib.util.spec_from_file_location("projection149_for_1046b",REPROJECTOR)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    xyz=source_node_xyz[source_tet_nodes[parent_global]]
    mat=np.transpose(xyz[1:]-xyz[0]);inv=np.linalg.inv(mat);query=child_centroids[allowed]
    pr=mod.project_pairs(query,np.zeros(len(query),dtype=np.int64),source_node_xyz,
                         np.asarray([source_tet_nodes[parent_global]],dtype=np.int64),inv.reshape(1,3,3))
    keep=np.flatnonzero(pr["distance"]<=B.R+1e-12)
    if len(keep)<1:return None,{"candidate_rows":len(allowed),"inside_cap_projection_sites":len(keep),"reason":"no_inside_cap_projection_site"}
    base_bary=pr["bary"][keep];base_points=pr["projected"][keep];base_targets=allowed[keep]
    bary_parts=[base_bary];target_parts=[base_targets];distance_parts=[pr["distance"][keep]]
    goals=list(np.eye(4))
    for j in range(4):
        for k in range(j+1,4):
            g=np.zeros(4);g[j]=g[k]=.5;goals.append(g)
    goals.append(np.full(4,.25))
    for goal in goals:
        goal_point=goal@xyz;direction=goal_point[None,:]-base_points;relative=base_points-query[keep]
        aa=np.einsum("ij,ij->i",direction,direction);bb=2.*np.einsum("ij,ij->i",direction,relative)
        cc=np.einsum("ij,ij->i",relative,relative)-B.R*B.R;disc=np.maximum(0.,bb*bb-4.*aa*cc)
        tmax=np.divide(-bb+np.sqrt(disc),2.*aa,out=np.zeros_like(aa),where=aa>0.)
        tt=.95*np.minimum(1.,np.maximum(0.,tmax))
        bs=base_bary*(1.-tt[:,None])+goal[None,:]*tt[:,None]
        pp=bs@xyz;dd=np.linalg.norm(pp-query[keep],axis=1);ok=(dd<=B.R+1e-12)&(tt>1e-14)
        if np.any(ok):
            bary_parts.append(bs[ok]);target_parts.append(base_targets[ok]);distance_parts.append(dd[ok])
    bary=np.concatenate(bary_parts);targets=np.concatenate(target_parts);dist=np.concatenate(distance_parts)
    cost=dist*dist+np.arange(len(dist),dtype=np.float64)*1e-16
    lp=linprog(cost,A_eq=bary.T,b_eq=np.full(4,.25),bounds=(0,None),method="highs",
               options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
    if not lp.success:return None,{"candidate_rows":len(allowed),"inside_cap_projection_sites":len(keep),"generated_valid_sites":len(dist),"reason":"degree1_positive_lp_"+str(lp.message)}
    take=np.flatnonzero(lp.x>1e-13);weights=lp.x[take];bs=bary[take];tar=targets[take];ds=dist[take]
    if not len(take) or np.any(weights<=0) or abs(float(weights.sum())-1)>1e-12 or np.max(np.abs(weights@bs-.25))>2e-10 or np.any(ds>B.R+1e-12):
        return None,{"candidate_rows":len(allowed),"inside_cap_projection_sites":len(keep),"generated_valid_sites":len(dist),"reason":"independent_degree1_solution_replay_failed"}
    return (weights,bs,tar,ds),{"candidate_rows":len(allowed),"inside_cap_projection_sites":len(keep),"generated_valid_sites":len(dist),"active_sites":len(take),"weight_sum_error":abs(float(weights.sum())-1),"barycentric_first_moment_error":float(np.max(np.abs(weights@bs-.25))),"maximum_distance_m":float(ds.max(initial=0))}

def solve_refined_weights(base_weights,base_barycentric,target_barycentrics,target_min_fractions,old_weights=None,old_barycentric=None,centroid_fraction=0.0):
    """Positive parent-mass/first-moment LP retaining half of every old anchor."""
    bw=np.asarray(base_weights,dtype=np.float64);bb=np.asarray(base_barycentric,dtype=np.float64).reshape(-1,4)
    tb=np.asarray(target_barycentrics,dtype=np.float64).reshape(-1,4);lower=np.asarray(target_min_fractions,dtype=np.float64)
    ow=np.empty(0,dtype=np.float64) if old_weights is None else np.asarray(old_weights,dtype=np.float64)
    ob=np.empty((0,4),dtype=np.float64) if old_barycentric is None else np.asarray(old_barycentric,dtype=np.float64).reshape(-1,4)
    if len(bw)!=len(bb) or len(ow)!=len(ob) or len(tb)!=len(lower) or not len(bw):raise ValueError("refined quadrature shape")
    if np.any(bw<=0) or np.any(bb<0) or np.any(ow<=0) or np.any(ob<0) or np.any(tb<0) or np.any(lower<=0):raise ValueError("refined quadrature positivity")
    cf=float(centroid_fraction)
    if cf<0 or cf>=1:raise ValueError("centroid anchor fraction")
    include_centroid=cf>0
    parts=[ob,bb]
    if include_centroid:parts.append(np.full((1,4),0.25,dtype=np.float64))
    parts.append(tb)
    all_b=np.vstack(parts);no=len(ow);nbase=len(bw);centroid_idx=no+nbase if include_centroid else None
    target_start=no+nbase+(1 if include_centroid else 0);n=len(all_b)
    aeq=np.vstack([np.ones(n),all_b[:,0],all_b[:,1],all_b[:,2]])
    beq=np.asarray([1.,.25,.25,.25],dtype=np.float64)
    objective=np.zeros(n,dtype=np.float64)
    if no:objective[:no]=1.0
    elif len(tb):objective[target_start:]=-1.0
    bounds=[(0.5*float(w),None) for w in ow]+[(1e-9,None)]*nbase
    if include_centroid:bounds.append((cf,cf))
    bounds.extend((float(x),None) for x in lower)
    lp=linprog(objective,A_eq=aeq,b_eq=beq,bounds=bounds,method="highs",
        options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
    if not lp.success:return None,{"status":"NO_POSITIVE_MOMENT_SPLIT","solver_status":int(lp.status),"solver_message":str(lp.message),"old_anchor_count":no}
    weights=np.asarray(lp.x,dtype=np.float64);moment=weights@all_b
    if np.any(weights<=0) or abs(float(weights.sum())-1)>1e-12 or np.max(np.abs(moment-.25))>2e-10:
        return None,{"status":"REFINED_MOMENT_REPLAY_FAILED","weight_sum":float(weights.sum()),"moment_error":float(np.max(np.abs(moment-.25)))}
    if no and np.any(weights[:no]+1e-14<0.5*ow):return None,{"status":"OLD_ANCHOR_HALF_SHARE_FAILED"}
    if len(lower) and np.any(weights[target_start:]+1e-15<lower):return None,{"status":"TARGET_MASS_FLOOR_FAILED"}
    return (weights,all_b),{"status":"PASS_POSITIVE_PARENT_MOMENT_SPLIT","solver_status":int(lp.status),"solver_message":str(lp.message),
        "old_anchor_count":no,"old_anchor_minimum_share_per_row":0.5 if no else None,"old_anchor_weight_sum_after":float(weights[:no].sum()),
        "degree1_anchor_count":nbase,"centroid_anchor_included":include_centroid,
        "centroid_anchor_weight":float(weights[centroid_idx]) if include_centroid else 0.0,
        "target_anchor_count":len(tb),"target_weights":weights[target_start:].tolist(),
        "target_min_fractions":lower.tolist(),"weight_sum_error":abs(float(weights.sum())-1),"first_moment_max_error":float(np.max(np.abs(moment-.25)))}

def refine_parent_quadrature(parent_local,parent_global,regional_children,target_children,child_centroids,
        child_volumes,parent_mass,source_node_xyz,source_tet_nodes,role_min_density,
        old_weights=None,old_barycentric=None,old_targets=None):
    """Add positive in-tet anchors while retaining all old support at >= half weight."""
    base,baseproof=regenerate_degree1(parent_local,parent_global,regional_children,child_centroids,source_node_xyz,source_tet_nodes)
    if base is None:return None,{"base_degree1":baseproof}
    bw,bb,bt,bd=base;verts=source_node_xyz[source_tet_nodes[parent_global]]
    target_bary=[]
    for child in target_children:
        rhs=child_centroids[int(child)]-verts[0]
        u=np.linalg.solve(np.transpose(verts[1:]-verts[0]),rhs);bary=np.r_[1.0-float(u.sum()),u]
        if np.min(bary)<-1e-10 or np.max(bary)>1+1e-10:return None,{"status":"TARGET_NOT_IN_SOURCE_TET","child_local_row":int(child),"barycentric":bary.tolist()}
        bary=np.maximum(bary,0.);bary/=float(bary.sum());dist=float(np.linalg.norm(bary@verts-child_centroids[int(child)]))
        if dist>B.R+1e-12:return None,{"status":"TARGET_EXCEEDS_TRANSPORT_RADIUS","child_local_row":int(child),"distance_m":dist}
        target_bary.append(bary)
    tb=np.asarray(target_bary,dtype=np.float64).reshape(-1,4)
    minmass=np.asarray(child_volumes[np.asarray(target_children,dtype=np.int64)]*float(role_min_density)*1.01,dtype=np.float64)
    minfrac=minmass/float(parent_mass) if len(minmass) else np.empty(0,dtype=np.float64)
    ow=np.empty(0,dtype=np.float64) if old_weights is None else np.asarray(old_weights,dtype=np.float64)
    ob=np.empty((0,4),dtype=np.float64) if old_barycentric is None else np.asarray(old_barycentric,dtype=np.float64).reshape(-1,4)
    ot=np.empty(0,dtype=np.int32) if old_targets is None else np.asarray(old_targets,dtype=np.int32)
    if len(ow)!=len(ob) or len(ow)!=len(ot):return None,{"status":"OLD_QUADRATURE_SHAPE_MISMATCH"}
    if len(ow):
        if np.any(ow<=0) or np.any(ob<0) or np.any(np.abs(ob.sum(axis=1)-1)>1e-12) or abs(float(ow.sum())-1)>1e-12 or np.max(np.abs(ow@ob-.25))>2e-10:return None,{"status":"OLD_QUADRATURE_MOMENT_INVALID"}
        olddist=np.linalg.norm(ob@verts-child_centroids[ot],axis=1)
        if np.any(olddist>B.R+1e-12):return None,{"status":"OLD_QUADRATURE_DISTANCE_INVALID","maximum_distance_m":float(olddist.max())}
    else:olddist=np.empty(0,dtype=np.float64)
    center_fraction=0.0;center_child=None;center_distance=None
    if not len(ow):
        center=np.full(4,0.25,dtype=np.float64)@verts
        center_tree=cKDTree(child_centroids[np.asarray(regional_children,dtype=np.int32)])
        center_distance,center_j=center_tree.query(center,k=1)
        center_child=int(regional_children[int(center_j)]);center_distance=float(center_distance)
        if center_distance>B.R+1e-12:
            return None,{"status":"ORIGINAL_CENTROID_ANCHOR_HAS_NO_VALID_RECEIVER","nearest_distance_m":center_distance,"cap_m":B.R}
        center_fraction=0.10
    solved,splitproof=solve_refined_weights(bw,bb,tb,minfrac,ow,ob,center_fraction)
    if solved is None:return None,{"base_degree1":baseproof,"moment_split":splitproof,"centroid_anchor_weight":center_fraction,"centroid_anchor_child":center_child,"centroid_distance_m":center_distance}
    weights,bary=solved
    targets=np.r_[ot,np.asarray(bt,dtype=np.int32),
                  np.asarray([center_child],dtype=np.int32) if center_fraction>0 else np.empty(0,dtype=np.int32),
                  np.asarray(target_children,dtype=np.int32)]
    distances=np.linalg.norm(bary@verts-child_centroids[targets],axis=1)
    no=len(ow);target_start=no+len(bw)+(1 if center_fraction>0 else 0)
    assigned=weights[target_start:]*float(parent_mass)
    if len(targets)!=len(weights) or np.any(weights<=0) or np.any(bary<0) or np.max(np.abs(bary.sum(axis=1)-1))>1e-12 or abs(float(weights.sum())-1)>1e-12 or np.max(np.abs(weights@bary-.25))>2e-10 or np.any(distances>B.R+1e-12):
        return None,{"status":"REFINED_GEOMETRY_OR_MOMENT_REPLAY_FAILED","max_distance_m":float(distances.max(initial=0))}
    if len(assigned)!=len(minmass) or np.any(assigned+1e-15<minmass):return None,{"status":"TARGET_MINIMUM_MASS_FLOOR_FAILED","assigned":assigned.tolist(),"minimum":minmass.tolist()}
    proof={"status":"PASS_SOURCE_TET_MOMENT_PRESERVING_REFINEMENT","base_degree1":baseproof,"moment_split":splitproof,
        "parent_mass_kg":float(parent_mass),"target_child_local_rows":list(map(int,target_children)),"target_min_mass_kg":minmass.tolist(),
        "target_assigned_mass_kg":assigned.tolist(),"preserved_old_anchor_count":no,"old_anchor_lower_bound_fraction":0.5 if no else None,
        "old_anchor_weight_sum_after":float(weights[:no].sum()),"degree1_anchor_count":len(bw),"target_anchor_count":len(tb),
        "retained_original_centroid_anchor":center_fraction>0,"original_centroid_anchor_weight":center_fraction,
        "original_centroid_receiver_child":center_child,"original_centroid_nearest_distance_m":center_distance,
        "carrier_count":int(len(weights)),"weight_sum_error":abs(float(weights.sum())-1),"first_moment_max_error":float(np.max(np.abs(weights@bary-.25))),
        "max_transport_distance_m":float(distances.max(initial=0))}
    return (weights,bary,targets,distances),proof

def build(role,outdir):
    mat=B.MAT[role];p1path=B.P1BASE/f"p1-{role.lower()}"/f"{role.lower()}-current-p1-support.npz"
    labels=B.csvread(B.TAB.with_name(B.TAB.name+".labels.csv"));tetr=B.csvread(B.TAB.with_name(B.TAB.name+".tets.csv"));nodes=B.csvread(B.TAB.with_name(B.TAB.name+".nodes.csv"));mass_rows=B.csvread(B.TAB)
    source_mass_by_row={int(x["source_tet_row"]):float(x["mass_kg"]) for x in mass_rows}
    allmat=np.fromiter((int(x["material_index"]) for x in labels),dtype=np.int64);grows=np.flatnonzero(allmat==mat).astype(np.uint32)
    if len(grows)!=14316:raise ValueError("source role parent count")
    node_rows=np.asarray([[int(x[f"parent_node{k}"]) for k in range(4)] for x in tetr],np.int64)
    rest=np.asarray([[float(x[f"rest_{a}_m"]) for a in "xyz"] for x in nodes],np.float64)
    current=np.asarray([[float(x[f"current_{a}_m"]) for a in "xyz"] for x in nodes],np.float64)
    verts=rest[node_rows[grows]];cent=verts.mean(axis=1)
    regs=np.fromiter((int(labels[int(g)]["anatomical_region_id"]) for g in grows),dtype=np.int64)
    tuples=[tuple(int(labels[int(g)][k]) for k in ("material_index","geometric_donor_body","anatomical_region_id")) for g in grows]
    parent_receipt=json.loads(PARENT_RECEIPT[role].read_text())
    if parent_receipt.get("status")!="PASS_SOURCE_ONLY_CURRENT_433_EXACT_TUPLE_CARRIER_REBIND" or parent_receipt.get("role")!=role or parent_receipt.get("outputs",{}).get("npz",{}).get("sha256")!=EXPECTED_PARENT_PACKET_SHA[role] or B.sha(PARENT_PACKET[role])!=EXPECTED_PARENT_PACKET_SHA[role]:raise ValueError("1046c parent packet/receipt pin")
    if B.sha(PARENT_WITNESS[role])!=EXPECTED_PARENT_WITNESS_SHA[role]:raise ValueError("011 extended witness pin")
    with np.load(PARENT_PACKET[role],allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    old_side=list(csv.DictReader(open(PARENT_WITNESS[role],newline="")))
    old_primary_by_packet={int(x["packet_carrier_row"]):x for x in old_side if x["record_kind"]=="carrier"}
    if len(old_primary_by_packet)!=len(old["carrier_weights"]) or set(old_primary_by_packet)!=set(range(len(old["carrier_weights"]))):raise ValueError("1046c/011 primary witness domain")
    if not np.array_equal(old["global_source330_tet_rows"],grows) or not np.array_equal(old["source_parent_tet_rows"],np.arange(len(grows),dtype=np.uint32)):raise ValueError("old source330 map")
    with np.load(B.CAND,allow_pickle=False) as z:
        pts=z["points_reference_world_m_f64"].copy();tets=z["volume_tetrahedra"].astype(np.int64,copy=True);mats=z["material_index"].astype(np.int64,copy=True);regions=z["anatomical_region_id"].astype(np.int64,copy=True);donors=z["geometric_donor_body"].astype(np.int64,copy=True);primary=z["current_typed_p1_anchor_primary_group_tuple"].astype(np.int64,copy=True)
    if not np.array_equal(pts,pts.astype(np.float32).astype(np.float64)):raise ValueError("candidate is not widened F32")
    cglobal=np.flatnonzero(mats==mat);ct=tets[cglobal];cc=pts[ct].mean(axis=1)
    child_xyz=pts[ct];child_det=np.einsum("ij,ij->i",child_xyz[:,1]-child_xyz[:,0],np.cross(child_xyz[:,2]-child_xyz[:,0],child_xyz[:,3]-child_xyz[:,0]))
    child_volumes=np.abs(child_det)/6.0
    if not np.isfinite(child_volumes).all() or np.any(child_volumes<=0):raise ValueError("current role child volume positivity")
    with np.load(p1path,allow_pickle=False) as z:
        p={k:z[k].copy() for k in ("cell_group_offsets","cell_group_ids","cell_admissible_region_offsets","cell_admissible_region_ids","group_tuples","admissibility_max_source_region_hops","current_state_path","current_state_sha256","source330_mass_sha256")}
    if len(cglobal)!=len(p["cell_group_offsets"])-1 or int(p["admissibility_max_source_region_hops"])!=1:raise ValueError("P1 child domain or hop")
    if str(np.asarray(p["current_state_path"]).reshape(-1)[0])!=str(B.STATE) or str(np.asarray(p["current_state_sha256"]).reshape(-1)[0])!=B.sha(B.STATE):raise ValueError("P1 state pin")
    if str(np.asarray(p["source330_mass_sha256"]).reshape(-1)[0])!=B.sha(B.TAB):raise ValueError("P1 source330 mass pin")
    po=p["cell_group_offsets"];pg=p["cell_group_ids"];tuple_to_gid={tuple(map(int,x)):i for i,x in enumerate(p["group_tuples"])}
    if len(tuple_to_gid)!=len(p["group_tuples"]):raise ValueError("duplicate P1 group tuple")
    # Validate candidate-to-P1 row alignment through the typed tuple slot, not
    # the lossy single selected-group scalar.
    for i,key in enumerate(primary[cglobal]):
        gid=tuple_to_gid.get(tuple(map(int,key)))
        if gid is None or not np.any(pg[po[i]:po[i+1]]==gid):raise ValueError(f"candidate/P1 typed row mismatch {i}")
    slot_child=np.repeat(np.arange(len(cc),dtype=np.int32),np.diff(po))
    group_rows={}
    for g in np.unique(pg):
        group_rows[int(g)]=np.unique(slot_child[pg==g])
    used_groups=set(group_rows)
    ro=p["cell_admissible_region_offsets"];rid=p["cell_admissible_region_ids"].astype(np.int64)
    if np.any(np.diff(ro)<=0):raise ValueError("empty one-hop P1 region row")
    reg_child=np.repeat(np.arange(len(cc),dtype=np.int32),np.diff(ro));order=np.argsort(rid,kind="stable");sr=rid[order];sc=reg_child[order]
    ur,st,n=np.unique(sr,return_index=True,return_counts=True);region_rows={int(q):np.unique(sc[i:i+j]) for q,i,j in zip(ur,st,n)}
    projection,projection_sha,projection_report_sha=load_projection(role,labels,node_rows,current,mats,regions,donors,cglobal,cc,ro,rid,p["group_tuples"],used_groups)
    eligible={};mode={};p1gid={};projrows={}
    all_parent_groups=set(tuples)
    for tup in all_parent_groups:
        gid=tuple_to_gid.get(tup)
        if gid is not None and gid in used_groups:
            rr=region_rows.get(tup[2],np.empty(0,dtype=np.int32));children=np.intersect1d(group_rows[gid],rr,assume_unique=True)
            if len(children):
                eligible[tup]=children;mode[tup]="p1_exact_tuple_slot";p1gid[tup]=gid;projrows[tup]={}
                continue
        records=projection.get(tup)
        if records:
            bychild={}
            for child,csvrow,projecteddist in records:
                if child in bychild:raise ValueError("duplicate projection tuple-child pair")
                bychild[child]=csvrow
            eligible[tup]=np.asarray(sorted(bychild),dtype=np.int32);mode[tup]="authenticated_1044_exact_tuple_projection";p1gid[tup]=-1;projrows[tup]=bychild
        else:
            # Explicit third mode: the 1047 CurrentCarrierPointProjection proof
            # authenticates a positive source330 carrier point against an
            # exact parent tuple and the child's one-hop source-region CSR.
            eligible[tup]=np.empty(0,dtype=np.int32);mode[tup]="CurrentCarrierPointProjection";p1gid[tup]=-1;projrows[tup]={}
    offsets=old["parent_offsets"].astype(np.int64);w0=old["carrier_weights"].astype(np.float64);b0=old["carrier_barycentric"].astype(np.float64)
    if offsets.shape!=(len(grows)+1,) or offsets[0]!=0 or offsets[-1]!=len(w0) or b0.shape!=(len(w0),4):raise ValueError("old carrier CSR")
    newoff=[0];nw=[];nb=[];nt=[];nd=[];side=[];fallback=[];reprojected=[];maxd=0.;nonempty=0;mode_counts={};old_to_new_packet={}
    for i,tup in enumerate(tuples):
        a,e=map(int,offsets[i:i+2])
        parent_global=int(grows[i]);force_refine=parent_global in REFINEMENT_PARENTS[mat]
        if a==e and not force_refine:
            newoff.append(len(nw));continue
        exact_allowed=eligible[tup]
        exact_mode=mode[tup]
        # The source carrier geometry is parent.referenceWorldPoints, i.e. the
        # source330 rest_x/y/z columns. The 1044 projection CSV above was
        # independently validated in its current_x/y/z basis.
        qpts=b0[a:e]@verts[i]
        if np.any(~np.isfinite(qpts)) or np.any(b0[a:e]<0) or np.any(np.abs(b0[a:e].sum(axis=1)-1)>1e-12):
            raise ValueError("source-reference carrier point/barycentric invalid")
        selected_mode=None;selected_gid=-1;selected_proj={};construction="inherited_090_quadrature"
        if force_refine:
            regional_children=region_rows.get(tup[2],np.empty(0,dtype=np.int32))
            if not len(regional_children):raise ValueError(f"no authenticated one-hop children for refinement parent={parent_global}")
            target_global=ZERO_CHILDREN_BY_PARENT.get(parent_global,[])
            target_local=[]
            for global_child in target_global:
                local_child=int(np.searchsorted(cglobal,int(global_child)))
                if local_child>=len(cglobal) or int(cglobal[local_child])!=int(global_child):raise ValueError("refinement target child outside role row map")
                if local_child not in regional_children:raise ValueError("refinement target child not in authenticated one-hop CSR")
                target_local.append(local_child)
            parent_mass=float(source_mass_by_row[parent_global])
            refined,proof=refine_parent_quadrature(i,parent_global,regional_children,target_local,cc,child_volumes,parent_mass,rest,node_rows,900.0 if mat==15 else 850.0,w0[a:e],b0[a:e],old["carrier_target_child_rows"][a:e])
            if refined is None:raise ValueError(f"source-tet quadrature refinement failed parent={parent_global} proof={proof}")
            selected=refined;selected_mode="CurrentCarrierPointProjection";selected_gid=-1;selected_proj={};construction=NEW_CONSTRUCTION
            refine_children=np.asarray(regional_children,dtype=np.int32);refine_tree=cKDTree(cc[refine_children])
        else:
            selected=None;proof=None;refine_children=np.empty(0,dtype=np.int32);refine_tree=None
            if e>a:
                inherited_targets=old["carrier_target_child_rows"][a:e].astype(np.int32,copy=True)
                inherited_dist=np.linalg.norm(b0[a:e]@verts[i]-cc[inherited_targets],axis=1)
                selected=(w0[a:e].copy(),b0[a:e].copy(),inherited_targets,inherited_dist)
                selected_mode="CurrentCarrierPointProjection";selected_gid=-1;selected_proj={}
        # First preserve the exact source-tuple receiver set (P1 slot or 1044
        # authenticated tuple projection) when it supports every original point.
        if not force_refine and len(exact_allowed):
            exact_tree=cKDTree(cc[exact_allowed]);ds,js=exact_tree.query(qpts,k=1)
            ds=np.asarray(ds,float).reshape(-1);js=np.asarray(js,np.int64).reshape(-1)
            if np.all(np.isfinite(ds)) and float(ds.max(initial=0))<=B.R+1e-12:
                selected=exact_allowed[js];selected_mode=exact_mode;selected_gid=p1gid[tup];selected_proj=projrows[tup]
        # Preserve the existing centroid fallback only when the exact tuple
        # receiver set proves the unchanged source centroid route.
        if not force_refine and selected is None and len(exact_allowed):
            exact_tree=cKDTree(cc[exact_allowed]);d,j=exact_tree.query(cent[i])
            if float(d)<=B.R+1e-12:
                child=int(exact_allowed[int(j)]);csvrow=int(projrows[tup].get(child,-1))
                fallback.append({"role":role,"local_parent_row":i,"global_source330_tet_row":int(grows[i]),"tuple":list(tup),"target_child_local_row":child,"target_child_global_row":int(cglobal[child]),"provenance_mode":exact_mode,"p1_group_id":int(p1gid[tup]),"projection_csv_row":csvrow,"distance_m":float(d)})
                side.append({"record_kind":"centroid_route_fallback","role":role,"carrier_construction":"unchanged_source_centroid","packet_carrier_row":-1,"local_parent_row":i,"global_source330_tet_row":int(grows[i]),"child_local_row":child,"child_global_row":int(cglobal[child]),"material":tup[0],"donor":tup[1],"region":tup[2],"provenance_mode":exact_mode,"p1_group_id":int(p1gid[tup]),"projection_csv_row":csvrow,"bary":np.full(4,.25).tolist(),"weight":1.0,"source":cent[i].tolist(),"child":cc[child].tolist(),"distance":float(d)})
                newoff.append(len(nw));continue
        if not force_refine:proof=None
        # If exact recipients are insufficient, first try the existing exact
        # tuple set with a regenerated positive degree-one quadrature.
        if not force_refine and selected is None and len(exact_allowed):
            replacement,proof=regenerate_degree1(i,int(grows[i]),exact_allowed,cc,rest,node_rows)
            if replacement is not None:
                rw,rb,rt,rd=replacement;selected=(rw,rb,rt,rd);selected_mode=exact_mode;selected_gid=p1gid[tup];selected_proj=projrows[tup];construction="current_exact_tuple_projection_degree1_lp"
        # Exact tuple slots can be geometrically infeasible. The separately
        # authorized CurrentCarrierPointProjection mode uses the exact source
        # parent tuple plus authenticated one-hop region CSR; it is never
        # mislabeled as a P1 slot or an 1044 exact-tuple projection.
        if not force_refine and selected is None:
            regional=region_rows.get(tup[2],np.empty(0,dtype=np.int32))
            if len(regional):
                rtree=cKDTree(cc[regional]);rds,rjs=rtree.query(qpts,k=1)
                rds=np.asarray(rds,float).reshape(-1);rjs=np.asarray(rjs,np.int64).reshape(-1)
                if np.all(np.isfinite(rds)) and float(rds.max(initial=0))<=B.R+1e-12:
                    selected=regional[rjs];selected_mode="CurrentCarrierPointProjection";selected_gid=-1;selected_proj={}
                else:
                    # Conservative candidate reduction: a valid point in the
                    # parent tetra cannot be within R of a child centroid whose
                    # distance from the tet centroid exceeds R+max_vertex_radius.
                    rad=float(np.linalg.norm(verts[i]-cent[i],axis=1).max(initial=0.0))
                    candidate=regional[np.asarray(rtree.query_ball_point(cent[i],B.R+rad+1e-12),dtype=np.int32)]
                    if len(candidate):
                        replacement,proof_geom=regenerate_degree1(i,int(grows[i]),candidate,cc,rest,node_rows)
                        if replacement is not None:
                            rw,rb,rt,rd=replacement;selected=(rw,rb,rt,rd);selected_mode="CurrentCarrierPointProjection";selected_gid=-1;selected_proj={};construction="current_carrier_point_geometry_projection"
                        else:
                            proof={"exact_tuple_attempt":proof,"carrier_point_geometry_attempt":proof_geom,"bounded_region_candidates":int(len(candidate))}
            if selected is None:
                raise ValueError(f"no positive degree-one carrier witness within exact tuple or explicit carrier-point projection mode parent={int(grows[i])} tuple={tup} exact_mode={exact_mode} exact_children={len(exact_allowed)} proof={proof}")
        if selected_mode=="CurrentCarrierPointProjection" and isinstance(selected,tuple) and len(selected)==2:
            raise ValueError("internal carrier-point selection state")
        # Exact inherited quadrature case: selected is its child-row vector.
        if isinstance(selected,np.ndarray):
            targets=np.asarray(selected,dtype=np.int32);weights=w0[a:e];baries=b0[a:e];dist=np.linalg.norm(baries@verts[i]-cc[targets],axis=1)
        else:
            weights,baries,targets,dist=selected
            targets=np.asarray(targets,dtype=np.int32);weights=np.asarray(weights,float);baries=np.asarray(baries,float);dist=np.asarray(dist,float)
        if len(targets)!=len(weights) or baries.shape!=(len(targets),4) or dist.shape!=(len(targets),):raise ValueError("selected carrier quadrature shape")
        if np.any(weights<=0) or np.any(baries<0) or np.any(np.abs(baries.sum(axis=1)-1)>1e-12) or abs(float(weights.sum())-1)>1e-12 or np.max(np.abs(weights@baries-.25))>2e-10 or np.any(dist>B.R+1e-12):raise ValueError(f"selected carrier quadrature violates positivity/moment/distance parent={parent_global} minw={float(weights.min(initial=0))} minb={float(baries.min(initial=0))} sumerr={abs(float(weights.sum())-1)} baryerr={float(np.max(np.abs(weights@baries-.25)))} maxdist={float(dist.max(initial=0))}")
        # Recompute recipient distances exclusively in the authoritative rest
        # reference basis, regardless of the 1044 current-basis projection lane.
        actual=np.linalg.norm(baries@verts[i]-cc[targets],axis=1)
        if np.max(np.abs(actual-dist),initial=0)>1e-12 or np.any(actual>B.R+1e-12):raise ValueError("source-reference carrier distance replay")
        new_packet_base=len(nw)
        preserved_old_count=int(proof.get("preserved_old_anchor_count",0)) if force_refine else e-a
        if preserved_old_count!=e-a:raise ValueError("refinement did not retain complete old anchor prefix")
        for old_index in range(a,e):old_to_new_packet[old_index]=new_packet_base+(old_index-a)
        for k,(target,distance) in enumerate(zip(targets,actual)):
            child=int(target);old_record=old_primary_by_packet.get(a+k) if k<preserved_old_count else None
            row_mode=old_record["provenance_mode"] if old_record else selected_mode
            row_gid=int(old_record["p1_group_id"]) if old_record else selected_gid
            row_proj=int(old_record["projection_csv_row"]) if old_record else int(selected_proj.get(child,-1))
            if row_mode=="p1_exact_tuple_slot":
                if row_gid<0 or not np.any(pg[po[child]:po[child+1]]==row_gid) or row_proj!=-1:raise ValueError("reprojected P1 slot identity")
            elif row_mode=="authenticated_1044_exact_tuple_projection":
                if row_proj<0 or child not in projrows[tup] or projrows[tup][child]!=row_proj:raise ValueError("reprojected projection row missing")
            elif row_mode=="CurrentCarrierPointProjection":
                if int(mats[int(cglobal[child])])!=tup[0] or int(regions[int(cglobal[child])]) not in rid[ro[child]:ro[child+1]]:raise ValueError("carrier-point receiver outside authenticated one-hop material/region")
            else:raise ValueError("unsupported selected carrier provenance mode")
            if force_refine:
                side.append({"record_kind":"carrier","role":role,"carrier_construction":construction,"packet_carrier_row":new_packet_base+k,"local_parent_row":i,"global_source330_tet_row":int(grows[i]),"child_local_row":child,"child_global_row":int(cglobal[child]),"material":tup[0],"donor":tup[1],"region":tup[2],"provenance_mode":row_mode,"p1_group_id":row_gid,"projection_csv_row":row_proj,"bary":baries[k].tolist(),"weight":float(weights[k]),"source":(baries[k]@verts[i]).tolist(),"child":cc[child].tolist(),"distance":float(distance)})
            if force_refine:
                packet_row=len(nw)+k;source_point=baries[k]@verts[i]
                near=refine_tree.query_ball_point(source_point,B.R+1e-12)
                for extra_child in refine_children[np.asarray(near,dtype=np.int64)]:
                    extra_child=int(extra_child)
                    if extra_child==child:continue
                    extra_distance=float(np.linalg.norm(source_point-cc[extra_child]))
                    if extra_distance>B.R+1e-12:raise ValueError("extra carrier child slot exceeds unchanged radius")
                    side.append({"record_kind":"carrier_child_slot","role":role,"carrier_construction":EDGE_CONSTRUCTION,"packet_carrier_row":packet_row,"local_parent_row":i,"global_source330_tet_row":int(grows[i]),"child_local_row":extra_child,"child_global_row":int(cglobal[extra_child]),"material":tup[0],"donor":tup[1],"region":tup[2],"provenance_mode":"CurrentCarrierPointProjection","p1_group_id":-1,"projection_csv_row":-1,"bary":baries[k].tolist(),"weight":float(weights[k]),"source":source_point.tolist(),"child":cc[extra_child].tolist(),"distance":extra_distance})
        nw.extend(map(float,weights));nb.extend(baries.tolist());nt.extend(map(int,targets));nd.extend(map(float,actual));maxd=max(maxd,float(actual.max(initial=0)));newoff.append(len(nw));nonempty+=1;mode_counts[selected_mode]=mode_counts.get(selected_mode,0)+len(weights)
        if construction!="inherited_090_quadrature":reprojected.append({"local_parent_row":i,"global_source330_tet_row":int(grows[i]),"tuple":list(tup),"construction":construction,"proof":proof})
    for oldrow in old_side:
        lp=int(oldrow["local_parent_row"])
        if int(grows[lp]) in REFINEMENT_PARENTS[mat]:continue
        kind=oldrow["record_kind"];row=dict(oldrow)
        if kind in ("carrier","carrier_child_slot"):
            oi=int(oldrow["packet_carrier_row"])
            if oi not in old_to_new_packet:raise ValueError("old 011 support row has no new packet binding")
            row["packet_carrier_row"]=old_to_new_packet[oi]
        elif kind!="centroid_route_fallback":raise ValueError("unknown inherited 011 witness kind")
        row["role"]=role
        row["bary"]=[float(oldrow[f"bary{k}"]) for k in range(4)]
        row["weight"]=float(oldrow["weight"])
        row["source"]=[float(oldrow[f"source_{axis}_m"]) for axis in "xyz"]
        row["child"]=[float(oldrow[f"child_centroid_{axis}_m"]) for axis in "xyz"]
        row["distance"]=float(oldrow["distance_m"])
        side.append(row)
    side.sort(key=lambda x:(0 if x["record_kind"]=="carrier" else 1 if x["record_kind"]=="carrier_child_slot" else 2,int(x.get("packet_carrier_row",-1)),int(x.get("local_parent_row",-1)),int(x.get("child_global_row",-1))))
    arr={"source_parent_tet_rows":np.arange(len(grows),dtype="<u4"),"global_source330_tet_rows":grows.astype("<u4"),"parent_offsets":np.asarray(newoff,dtype="<u4"),"carrier_weights":np.asarray(nw,dtype="<f8"),"carrier_barycentric":np.asarray(nb,dtype="<f8").reshape(-1,4),"carrier_target_child_rows":np.asarray(nt,dtype="<u4"),"carrier_target_distance_m":np.asarray(nd,dtype="<f8")}
    outdir.mkdir(parents=True,exist_ok=True);stem=f"{role.lower()}-current-source-quadrature-1046d";npz=outdir/(stem+".npz");binp=outdir/(stem+".bin");np.savez_compressed(npz,**arr);binp.write_bytes(B.packet(arr));loaded,bsha=B.verify_packet(binp,npz,grows)
    # Independent all-witness readback from the serialized packet plus aligned
    # sidecar; each row is bound to exact source tuple and authenticated mode.
    sidepath=outdir/(role.lower()+"-witnesses-1046d.csv")
    with open(sidepath,"w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=WITNESS_FIELDS);wr.writeheader()
        for x in side:
            row={k:"" for k in WITNESS_FIELDS};row.update({k:x[k] for k in ("record_kind","role","carrier_construction","packet_carrier_row","local_parent_row","global_source330_tet_row","child_local_row","child_global_row","material","donor","region","provenance_mode","p1_group_id","projection_csv_row","weight") if k in x})
            row["distance_m"]=x.get("distance",x.get("distance_m"))
            row.update({f"bary{i}":x["bary"][i] for i in range(4)})
            for i,a in enumerate("xyz"):row[f"source_{a}_m"]=x["source"][i];row[f"child_centroid_{a}_m"]=x["child"][i]
            wr.writerow(row)
    # Replay sidecar membership and exact geometry from serialized NPZ.
    sidehash=B.sha(sidepath);records=list(csv.DictReader(open(sidepath,newline="")))
    carrier_records=[x for x in records if x["record_kind"]=="carrier"]
    if len(carrier_records)!=len(nw):raise ValueError("sidecar carrier row count")
    for index,x in enumerate(carrier_records):
        if int(x["packet_carrier_row"])!=index:raise ValueError("sidecar carrier order")
        lp=int(x["local_parent_row"]);child=int(x["child_local_row"]);tup=tuples[lp]
        if (int(x["global_source330_tet_row"])!=int(grows[lp]) or tuple(int(x[k]) for k in ("material","donor","region"))!=tup or int(x["child_global_row"])!=int(cglobal[child])):raise ValueError("sidecar tuple/row identity")
        a,e=map(int,loaded["parent_offsets"][lp:lp+2]); ix=index-a
        if not (a<=index<e):raise ValueError("sidecar packet CSR index")
        bary=loaded["carrier_barycentric"][index];source=bary@verts[lp];dist=float(np.linalg.norm(source-cc[child]))
        if abs(dist-float(x["distance_m"]))>1e-12 or abs(dist-float(loaded["carrier_target_distance_m"][index]))>1e-12 or dist>B.R+1e-12:raise ValueError("sidecar carrier point geometry")
        if tuple(float(x[f"bary{k}"]) for k in range(4))!=tuple(map(float,bary)):raise ValueError("sidecar barycentric differs from packet")
        if abs(float(x["weight"])-float(loaded["carrier_weights"][index]))>1e-15:raise ValueError("sidecar weight differs from packet")
        srcrow=np.asarray([float(x[f"source_{axis}_m"]) for axis in "xyz"])
        childrow=np.asarray([float(x[f"child_centroid_{axis}_m"]) for axis in "xyz"])
        if np.max(np.abs(srcrow-source))>2e-12 or np.max(np.abs(childrow-cc[child]))>2e-12:raise ValueError("sidecar source/child coordinate mismatch")
        md=x["provenance_mode"]
        if md=="p1_exact_tuple_slot":
            gid=int(x["p1_group_id"])
            if gid<0 or not np.any(pg[po[child]:po[child+1]]==gid):raise ValueError("sidecar exact P1 slot missing")
        elif md=="authenticated_1044_exact_tuple_projection":
            pr=int(x["projection_csv_row"])
            if pr<0 or child not in projrows[tup] or projrows[tup][child]!=pr:raise ValueError("sidecar exact projection row missing")
        elif md=="CurrentCarrierPointProjection":
            if int(mats[int(cglobal[child])])!=tup[0]:raise ValueError("carrier-point receiver material")
            if int(regions[int(cglobal[child])]) not in rid[ro[child]:ro[child+1]]:raise ValueError("carrier-point receiver outside authenticated one-hop CSR")
            if int(x["p1_group_id"])!=-1 or int(x["projection_csv_row"])!=-1:raise ValueError("carrier-point mode mislabeled as exact tuple row")
            if x["carrier_construction"]==NEW_CONSTRUCTION and int(grows[lp]) not in REFINEMENT_PARENTS[mat]:raise ValueError("new quadrature tag on undeclared parent")
        else:raise ValueError("sidecar provenance mode")
    # Centroid-route fallbacks are not serialized as quadrature rows, but their
    # exact tuple eligibility and reference-basis distance are independently replayed.
    extra_records=[x for x in records if x["record_kind"]=="carrier_child_slot"]
    extra_seen=set()
    for x in extra_records:
        index=int(x["packet_carrier_row"]);lp=int(x["local_parent_row"]);child=int(x["child_local_row"]);tup=tuples[lp]
        pair=(index,int(x["child_global_row"]))
        if pair in extra_seen:raise ValueError("duplicate carrier child-slot pair")
        extra_seen.add(pair)
        if not (0<=index<len(loaded["carrier_weights"])) or int(loaded["global_source330_tet_rows"][lp])!=int(x["global_source330_tet_row"]):raise ValueError("extra slot carrier/parent binding")
        a,e=map(int,loaded["parent_offsets"][lp:lp+2])
        if not (a<=index<e):raise ValueError("extra slot carrier outside parent CSR")
        if x["carrier_construction"]!=EDGE_CONSTRUCTION or x["provenance_mode"]!="CurrentCarrierPointProjection" or int(x["p1_group_id"])!=-1 or int(x["projection_csv_row"])!=-1:raise ValueError("extra slot provenance label")
        if tuple(int(x[k]) for k in ("material","donor","region"))!=tup or int(x["child_global_row"])!=int(cglobal[child]):raise ValueError("extra slot tuple/row mapping")
        bary=loaded["carrier_barycentric"][index];weight=float(loaded["carrier_weights"][index]);source=bary@verts[lp];dist=float(np.linalg.norm(source-cc[child]))
        if tuple(float(x[f"bary{k}"]) for k in range(4))!=tuple(map(float,bary)) or abs(float(x["weight"])-weight)>1e-15:raise ValueError("extra slot packet weight/bary mismatch")
        srcrow=np.asarray([float(x[f"source_{axis}_m"]) for axis in "xyz"]);childrow=np.asarray([float(x[f"child_centroid_{axis}_m"]) for axis in "xyz"])
        if np.max(np.abs(srcrow-source))>2e-12 or np.max(np.abs(childrow-cc[child]))>2e-12 or abs(dist-float(x["distance_m"]))>1e-12 or dist>B.R+1e-12:raise ValueError("extra slot source/child/distance replay")
        if int(mats[int(cglobal[child])])!=tup[0] or int(regions[int(cglobal[child])]) not in rid[ro[child]:ro[child+1]]:raise ValueError("extra slot material/one-hop gate")
    fallback_records=[x for x in records if x["record_kind"]=="centroid_route_fallback"]
    for x in fallback_records:
        lp=int(x["local_parent_row"]);child=int(x["child_local_row"]);tup=tuples[lp]
        if tuple(int(x[k]) for k in ("material","donor","region"))!=tup or int(x["global_source330_tet_row"])!=int(grows[lp]):raise ValueError("fallback source tuple identity")
        if child not in eligible[tup]:raise ValueError("fallback recipient outside exact tuple receiver set")
        if tuple(float(x[f"bary{k}"]) for k in range(4))!=(0.25,0.25,0.25,0.25):raise ValueError("fallback centroid barycentric")
        srcrow=np.asarray([float(x[f"source_{axis}_m"]) for axis in "xyz"]);childrow=np.asarray([float(x[f"child_centroid_{axis}_m"]) for axis in "xyz"])
        if np.max(np.abs(srcrow-cent[lp]))>2e-12 or np.max(np.abs(childrow-cc[child]))>2e-12:raise ValueError("fallback source/child coordinate mismatch")
        dist=float(np.linalg.norm(cent[lp]-cc[child]))
        if abs(dist-float(x["distance_m"]))>1e-12 or dist>B.R+1e-12:raise ValueError("fallback reference-basis distance")
    inputs={"old_packet_sha256":B.sha(PARENT_PACKET[role]),"old_packet_receipt_sha256":B.sha(PARENT_RECEIPT[role]),"old_extended_witness_sha256":B.sha(PARENT_WITNESS[role]),"candidate_npz_sha256":B.sha(B.CAND),"candidate_receipt_sha256":B.sha(B.CREC),"current_state_sha256":B.sha(B.STATE),"p1_support_sha256":B.sha(p1path),"source330_mass_sha256":B.sha(B.TAB),"source330_labels_sha256":B.sha(B.TAB.with_name(B.TAB.name+".labels.csv")),"source330_nodes_sha256":B.sha(B.TAB.with_name(B.TAB.name+".nodes.csv")),"source330_tets_sha256":B.sha(B.TAB.with_name(B.TAB.name+".tets.csv")),"projection_support_csv_sha256":projection_sha,"projection_report_sha256":projection_report_sha,"projection_owner_sha256":B.sha(PROJOWNER),"old_packet_parent_sha256":B.sha(B.OLD[role])}
    receipt={"schema":SCHEMA,"status":"PASS_SOURCE_ONLY_CURRENT_433_EXACT_TUPLE_CARRIER_REBIND","role":role,"material_index":mat,"native_steps":0,"physical_admission":False,"maximum_transport_distance_m":B.R,"source_parent_count":len(grows),"carrier_count":len(nw),"carrierized_parent_count":nonempty,"centroid_fallbacks":fallback,"reprojected_parent_rows":reprojected,"reprojected_parent_count":len(reprojected),"receiver_provenance_carrier_counts":mode_counts,"p1_slot_group_count":len(used_groups),"authenticated_projection_group_count":len(projection),"projection_pair_count":sum(map(len,projection.values())),"max_distance_m":maxd,"preserved_old_extended_support_rows":sum(x["record_kind"]=="carrier_child_slot" for x in side if int(x["global_source330_tet_row"]) not in REFINEMENT_PARENTS[mat]),"refined_source_parent_rows":sorted(REFINEMENT_PARENTS[mat]),"geometry_basis":{"carrier_source":"source330 parent.referenceWorldPoints from rest_x/y/z_m","1044_projection_replay":"pinned 217 read_parent_current from current_x/y/z_m"},"sidecar":{"schema":"numi.h706.current-source-carrier-witness-tuples.1046h.csv.v1","path":str(sidepath),"sha256":sidehash,"rows":len(records),"carrier_rows":len(carrier_records),"fallback_rows":len(fallback_records),"extra_child_slot_rows":len(extra_records)},"inputs_sha256":inputs,"outputs":{"npz":{"path":str(npz),"sha256":B.sha(npz)},"binary":{"path":str(binp),"sha256":bsha}},"witness_rule":"exact P1 admitted tuple slot or authenticated 1044 exact-tuple projection row when feasible; otherwise explicit CurrentCarrierPointProjection binds the exact source330 parent tuple and positive barycentric referenceWorldPoints carrier to a same-role-material child admitted by authenticated one-hop region CSR; every distance is recomputed in rest_x/y/z reference coordinates against current child centroid and remains <= unchanged cap","carrier_geometry_basis":"source330 parent.referenceWorldPoints from rest_x/y/z_m","1044_projection_validation_basis":"pinned 217 read_parent_current from current_x/y/z_m; projection CSV replay remains on that basis","native_admission":"not claimed"}
    recpath=outdir/(stem+"-receipt.json");recpath.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    verification={"schema":"numi.h706.current-source-carrier-rebind.1046h.readback.v1","status":"PASS_SOURCE_ONLY_EXACT_TUPLE_WITNESS_READBACK","role":role,"receipt_sha256":B.sha(recpath),"packet_npz_sha256":B.sha(npz),"packet_binary_sha256":bsha,"witness_sidecar_sha256":sidehash,"carrier_rows_replayed":len(carrier_records),"fallback_rows_replayed":len(records)-len(carrier_records),"provenance_modes":mode_counts,"projection_csv_sha256":projection_sha,"max_distance_m":maxd,"native_steps":0,"physical_admission":False}
    vpath=outdir/(stem+"-independent-verification.json");vpath.write_text(json.dumps(verification,sort_keys=True,indent=2)+"\n")
    return verification

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true");ap.add_argument("--output",type=Path,default=H/"current-source-carrier-bridge-1046h");a=ap.parse_args()
    if a.self_test:
        # Check disjoint P1 vs projection admission labels and fail closed.
        modes={"p1":"p1_exact_tuple_slot","projection":"authenticated_1044_exact_tuple_projection","geometry":"CurrentCarrierPointProjection"}
        assert len(set(modes.values()))==3 and modes["geometry"] not in (modes["p1"],modes["projection"])
        assert B.R==0.019353
        base_b=np.eye(4,dtype=np.float64);base_w=np.full(4,0.25)
        targets=np.asarray([[0.4,0.2,0.2,0.2],[0.1,0.2,0.3,0.4]],dtype=np.float64)
        result,proof=solve_refined_weights(base_w,base_b,targets,np.asarray([0.05,0.04]),None,None)
        assert result is not None and np.min(result[0])>0 and abs(float(result[0].sum())-1)<1e-12 and np.max(np.abs(result[0]@result[1]-.25))<2e-10
        bad_b=result[1].copy();bad_b[-1,0]+=1e-3
        assert np.max(np.abs(result[0]@bad_b-.25))>1e-5
        mixed,mproof=solve_refined_weights(base_w,base_b,targets,np.asarray([0.05,0.04]),base_w,base_b)
        assert mixed is not None and np.all(mixed[0][:4]+1e-14>=.5*base_w)
        assert np.max(np.abs(mixed[0]@mixed[1]-.25))<2e-10 and np.all(mixed[0][8:]+1e-15>=np.asarray([.05,.04]))
        print(json.dumps({"status":"PASS","three_receiver_provenance_modes_distinct":True,"refined_anchor_mass_and_first_moment":True,"centroid_anchor_not_required":True,"old_support_half_weight_and_target_floor":True,"tampered_target_moment_detected":True,"reference_basis":"rest_x/y/z_m","1044_projection_basis":"current_x/y/z_m","unchanged_transport_cap_m":B.R}));return
    a.output.mkdir(parents=True,exist_ok=True);v={r:build(r,a.output) for r in ("Skin","Fat")}
    report={"schema":"numi.h706.current-source-carrier-rebind.1046h.batch.v1","status":"PASS_SOURCE_ONLY_EXACT_TUPLE_WITNESS_READBACK","roles":v,"native_steps":0,"physical_admission":False}
    path=a.output/"batch-independent-verification.json";path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"report":str(path),"roles":v}))
if __name__=="__main__":main()

