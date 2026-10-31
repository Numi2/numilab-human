"""Independent verification of deterministic and solver-free recorded carrier allocations."""
from pathlib import Path
import argparse, csv, hashlib, json, math, time
import numpy as np

from source_carrier_packet_188_root import load_packet
from source_semantic_support_191_root import load_support, check_edges
from source_carrier_receipt_197_root import (
    validate_current_input_pins, validate_receipt_mode,
    source_carrier_packet_digest, source_carrier_array_digest,
)

INVALID = 0xffffffff
FLOW_NAMES = {
    "transport-child-mass.csv", "transport-flow.csv", "source-carrier-flow.csv",
    "transport-child-source-donor.csv", "transport-source-parent.csv",
    "transport-source-parent-donor.csv",
}

def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def load_numeric(directory, name, columns):
    a = np.loadtxt(directory / name, delimiter=",", skiprows=1, ndmin=2)
    if a.ndim != 2 or a.shape[1] != columns or not np.isfinite(a).all():
        raise ValueError("CSV shape/finite " + name)
    return a

def close(actual, expected, atol, rtol, what):
    a = np.asarray(actual)
    b = np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(what + " shape or finite")
    err = np.abs(a-b)
    if np.any(err > np.maximum(atol, rtol*np.abs(b))):
        raise ValueError(what + " maximumerror " + str(float(err.max(initial=0))))
    return float(err.max(initial=0))

def read_child(node_path, ele_path):
    raw_nodes = np.loadtxt(node_path, skiprows=1, usecols=(0,1,2,3), ndmin=2)
    node_ids = raw_nodes[:,0].astype(np.int64)
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("child node IDs are not unique")
    points = raw_nodes[:,1:].astype(np.float32).astype(np.float64)
    raw_ele = np.loadtxt(ele_path, skiprows=1, ndmin=2)
    if raw_ele.ndim != 2 or raw_ele.shape[1] != 6 or not np.isfinite(raw_ele).all():
        raise ValueError("child element table must contain 4 corners and one material")
    if len(np.unique(raw_ele[:,0])) != len(raw_ele):
        raise ValueError("child element IDs are not unique")
    if not np.array_equal(raw_ele[:,5], np.floor(raw_ele[:,5])):
        raise ValueError("child material values are not integer labels")
    id_to_row = {int(node_id): row for row,node_id in enumerate(node_ids)}
    try:
        tets = np.asarray([[id_to_row[int(x)] for x in row[1:5]] for row in raw_ele], dtype=np.uint32)
    except KeyError as e:
        raise ValueError("child element references missing node ID " + str(e))
    materials = raw_ele[:,5].astype(np.uint32)
    return node_ids, points, tets, materials

def child_mesh_digest(node_path, ele_path, points, tets):
    from source_carrier_receipt_197_root import CanonicalDigest
    h=CanonicalDigest()
    h.string("existing-tetgen-reference-child.v1")
    h.string(sha(node_path))
    h.string(sha(ele_path))
    h.u64(len(points))
    for point in points:
        for x in point: h.real(x)
    h.u64(len(tets))
    for tet in tets:
        for x in tet: h.u32(x)
    return h.finish()

def bary_point(tet, bary):
    point=np.zeros(3,dtype=np.float64)
    for node in range(4):
        for axis in range(3):
            point[axis]+=float(bary[node])*float(tet[node][axis])
    return point

def point_distance(a,b):
    dx=float(a[0])-float(b[0]);dy=float(a[1])-float(b[1]);dz=float(a[2])-float(b[2])
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def ordered_centroids(points,tets):
    out=np.zeros((len(tets),3),dtype=np.float64)
    for corner in range(4):
        out+=points[tets[:,corner]]*0.25
    return out

def read_held_rows(path, parent_count, child_count, parent_mass, parent_volume,
                   parent_centroids, parent_tets, child_centroids, q, scale, max_distance):
    required = {"child_row","parent_row","expanded_parent_row","carrier_row",
                "mass_kg","source_measure_m3","distance_m"}
    old_offsets=q["parent_offsets"].astype(np.int64)
    old_carrier_to_expanded={}
    old_default_expanded={}
    next_row=0
    for p in range(parent_count):
        a,b=map(int,old_offsets[p:p+2])
        if a==b:
            old_default_expanded[p]=next_row
            next_row+=1
        else:
            for k in range(a,b):
                old_carrier_to_expanded[k]=next_row
                next_row+=1
    grouped={}
    rows_seen=0
    with open(path,newline="") as f:
        reader=csv.DictReader(f)
        if set(reader.fieldnames or []) != required:
            raise ValueError("held carrier-flow CSV header differs from source-007 contract")
        for row in reader:
            try:
                c=int(row["child_row"]); p=int(row["parent_row"])
                old_exp=int(row["expanded_parent_row"]); old=int(row["carrier_row"])
                mass=float(row["mass_kg"]); measure=float(row["source_measure_m3"]); dist=float(row["distance_m"])
            except Exception as e:
                raise ValueError("invalid held carrier-flow row: "+str(e))
            if not (0<=c<child_count and 0<=p<parent_count) or not (math.isfinite(mass) and mass>0 and math.isfinite(measure) and measure>0 and math.isfinite(dist) and dist>=0):
                raise ValueError("held carrier-flow row is outside bounds or nonpositive")
            if old==INVALID:
                if old_offsets[p] != old_offsets[p+1]:
                    raise ValueError("held default-centroid route conflicts with explicit old quadrature")
                expected=old_default_expanded[p]
                point=parent_centroids[p]
            else:
                if old not in old_carrier_to_expanded or not (old_offsets[p] <= old < old_offsets[p+1]):
                    raise ValueError("held old carrier row is outside its parent quadrature slice")
                expected=old_carrier_to_expanded[old]
                point=bary_point(parent_tets[p],q["carrier_barycentric"][old])
            if old_exp != expected:
                raise ValueError("held expanded-parent row differs from the pinned original quadrature")
            measured=point_distance(point,child_centroids[c])
            source_measure=mass/(parent_mass[p]/parent_volume[p])
            if abs(measured-dist)>max(1e-12,64*np.finfo(float).eps*max(1.,abs(measured),abs(dist))):
                raise ValueError("held flow distance differs from recomputed source-carrier distance")
            if abs(source_measure-measure)>max(1e-12,64*np.finfo(float).eps*max(1.,abs(source_measure),abs(measure))):
                raise ValueError("held flow source measure differs from mass/density")
            if measured>max_distance+1e-12:
                raise ValueError("held source carrier exceeds role transport radius")
            rows_seen+=1
            if scale[p]>0:
                grouped.setdefault((p,old),[]).append((mass,c))
    return grouped, rows_seen

def build_lp_carriers(lp_path, held_path, held_manifest_path, lp_manifest_path,
                      feasibility_path, receipt, q, q_sha, global_rows, parent_mass,
                      parent_volume, parent_tets, parent_centroids, child_centroids,
                      max_distance):
    expected_paths = {
        "held_partial_manifest_sha256": held_manifest_path,
        "held_carrier_flow_csv_sha256": held_path,
        "lp_proposal_sha256": lp_path,
        "lp_input_manifest_sha256": lp_manifest_path,
        "feasibility_source_sha256": feasibility_path,
    }
    for key,path in expected_paths.items():
        if sha(path) != receipt[key]:
            raise ValueError("LP source input hash mismatch: "+key)
    z=np.load(lp_path,allow_pickle=False)
    required={"variable_flow_mass_kg","variable_positive_indices","variable_child_rows",
              "variable_parent_rows","variable_barycentric","variable_distance_m","held_parent_scale"}
    if not required.issubset(z.files):
        raise ValueError("LP proposal misses required source-007 arrays")
    lp={name:z[name] for name in required}
    np_=len(parent_mass);nc=len(child_centroids)
    scale=np.asarray(lp["held_parent_scale"],dtype=np.float64)
    if scale.shape!=(np_,) or not np.isfinite(scale).all() or np.any(scale<0) or np.any(scale>1):
        raise ValueError("held-parent scale domain or bounds invalid")
    held_groups,held_count=read_held_rows(held_path,np_,nc,parent_mass,parent_volume,
        parent_centroids,parent_tets,child_centroids,q,scale,max_distance)
    if held_count!=receipt["held_positive_flow_rows"]:
        raise ValueError("held flow row count differs from LP receipt")
    if len(held_groups)==0:
        raise ValueError("LP receipt has no positive held carrier sites")
    flow=np.asarray(lp["variable_flow_mass_kg"],dtype=np.float64)
    indices=np.asarray(lp["variable_positive_indices"],dtype=np.int64)
    children=np.asarray(lp["variable_child_rows"],dtype=np.int64)
    parents=np.asarray(lp["variable_parent_rows"],dtype=np.int64)
    bary=np.asarray(lp["variable_barycentric"],dtype=np.float64)
    claimed=np.asarray(lp["variable_distance_m"],dtype=np.float64)
    n=len(flow)
    if flow.shape!=(n,) or indices.shape!=(n,) or children.shape!=(n,) or parents.shape!=(n,) or bary.shape!=(n,4) or claimed.shape!=(n,):
        raise ValueError("LP positive-variable array shapes differ")
    if n!=receipt["lp_positive_variable_count"] or n==0 or len(np.unique(indices))!=n:
        raise ValueError("LP positive-variable count or IDs differ from receipt")
    if np.any(flow<=0) or not np.isfinite(flow).all() or np.any(indices<0):
        raise ValueError("LP positive variables must have positive finite flow and unique nonnegative IDs")
    if np.any(children<0) or np.any(children>=nc) or np.any(parents<0) or np.any(parents>=np_):
        raise ValueError("LP variable child/parent rows outside current geometry")
    if not np.isfinite(bary).all() or np.any(bary < -1e-13) or np.any(bary > 1+1e-13) or np.any(np.abs(bary.sum(axis=1)-1)>2e-12):
        raise ValueError("LP carrier barycentric rows invalid")
    if not np.isfinite(claimed).all() or np.any(claimed<0):
        raise ValueError("LP distance claims invalid")
    for i in range(n):
        point=bary_point(parent_tets[parents[i]],bary[i])
        d=point_distance(point,child_centroids[children[i]])
        if abs(d-claimed[i])>max(1e-12,64*np.finfo(float).eps*max(1.,abs(d),abs(claimed[i]))) or d>max_distance+1e-12:
            raise ValueError("LP carrier distance fails independent geometry/radius check")

    records=[[] for _ in range(np_)]
    for (p,old),rows in sorted(held_groups.items()):
        mass=math.fsum(m*scale[p] for m,_ in rows)
        if mass<=0 or not math.isfinite(mass):
            continue
        if old==INVALID:
            carrier_bary=np.asarray([.25,.25,.25,.25],dtype=np.float64)
            witness=rows[0][1]
        else:
            carrier_bary=q["carrier_barycentric"][old].astype(np.float64)
            witness=int(q["carrier_target_child_rows"][old])
        point=bary_point(parent_tets[p],carrier_bary)
        dist=point_distance(point,child_centroids[witness])
        records[p].append({"is_lp":False,"sort":old,"weight":mass/parent_mass[p],
                           "bary":carrier_bary,"witness":witness,"distance":dist})
    order=np.lexsort((indices,parents))
    for i in order:
        p=int(parents[i]);witness=int(children[i]);carrier_bary=bary[i]
        point=bary_point(parent_tets[p],carrier_bary)
        dist=point_distance(point,child_centroids[witness])
        records[p].append({"is_lp":True,"sort":int(indices[i]),"weight":flow[i]/parent_mass[p],
                           "bary":carrier_bary.copy(),"witness":witness,"distance":dist})
    offsets=[0];weights=[];barys=[];witnesses=[];witness_distances=[]
    virtual_source=[];virtual_carrier=[];sites=[];expanded_weights=[]
    carrierized=0
    for p,items in enumerate(records):
        items.sort(key=lambda x:(x["is_lp"],x["sort"]))
        if items:
            carrierized+=1
            for item in items:
                weights.append(float(item["weight"]));barys.append(item["bary"])
                witnesses.append(int(item["witness"]));witness_distances.append(float(item["distance"]))
                virtual_source.append(p);virtual_carrier.append(len(weights)-1)
                sites.append(bary_point(parent_tets[p],item["bary"]))
                expanded_weights.append(float(item["weight"]))
        else:
            virtual_source.append(p);virtual_carrier.append(INVALID)
            sites.append(parent_centroids[p]);expanded_weights.append(1.)
        offsets.append(len(weights))
    weights=np.asarray(weights,dtype=np.float64)
    barys=np.asarray(barys,dtype=np.float64).reshape((-1,4))
    witnesses=np.asarray(witnesses,dtype=np.uint32)
    witness_distances=np.asarray(witness_distances,dtype=np.float64)
    for p in range(np_):
        a,b=offsets[p:p+2]
        if a==b:
            continue
        if abs(math.fsum(weights[a:b])-1)>1e-12 or np.max(np.abs(weights[a:b]@barys[a:b]-.25))>1e-12:
            raise ValueError("finalized LP carrier weights do not preserve source-cell mass and first moments")
    packet_sha=source_carrier_packet_digest(q_sha,receipt["held_partial_manifest_sha256"],
        receipt["held_carrier_flow_csv_sha256"],receipt["lp_proposal_sha256"],
        receipt["lp_input_manifest_sha256"],offsets,global_rows,weights,barys,witnesses,witness_distances)
    return (np.asarray(offsets,dtype=np.uint32),weights,barys,witnesses,witness_distances,
            np.asarray(virtual_source,dtype=np.uint32),np.asarray(virtual_carrier,dtype=np.uint32),
            np.asarray(sites,dtype=np.float64),np.asarray(expanded_weights,dtype=np.float64),
            packet_sha,carrierized)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--role",required=True,choices=["Skin","Fat"])
    ap.add_argument("--result-directory",required=True,type=Path)
    ap.add_argument("--result-receipt",required=True)
    ap.add_argument("--quadrature-directory",required=True,type=Path)
    ap.add_argument("--support-bin",required=True,type=Path)
    ap.add_argument("--child-node",type=Path)
    ap.add_argument("--child-ele",type=Path)
    ap.add_argument("--child-labels",type=Path)
    ap.add_argument("--held-carrier-flow-csv",type=Path)
    ap.add_argument("--held-partial-manifest",type=Path)
    ap.add_argument("--lp-proposal",type=Path)
    ap.add_argument("--lp-input-manifest",type=Path)
    ap.add_argument("--feasibility-source",type=Path)
    args=ap.parse_args()
    role=args.role;material=15 if role=="Skin" else 16;role_lower=role.lower()
    density_lower=900 if role=="Skin" else 850;density_upper=1100 if role=="Skin" else 950
    D=args.result_directory.resolve();H=Path(__file__).resolve().parent;P=H.parent
    Q=args.quadrature_directory.resolve()
    child_base=H/"combined-child-handoff-142"
    node_path=(args.child_node or child_base/f"{role_lower}.node").resolve()
    ele_path=(args.child_ele or child_base/f"{role_lower}.ele").resolve()
    labels_path=(args.child_labels or node_path.with_suffix(".labels.csv")).resolve()
    if (args.child_node is None) != (args.child_ele is None):
        raise ValueError("--child-node and --child-ele must be supplied together")
    S=P/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
    source_path=S.with_name(S.name+".source.csv")
    source_tet_path=S.with_name(S.name+".tets.csv")
    source_node_path=S.with_name(S.name+".nodes.csv")
    parent_labels_path=S.with_name(S.name+".labels.csv")
    start=time.monotonic()
    receipt_path=D/args.result_receipt
    r=json.loads(receipt_path.read_text())
    mode=validate_receipt_mode(r,role)
    if set(r["flow_csv"])!=FLOW_NAMES:
        raise ValueError("receipt flow set differs from existing six-CSV contract")
    for name,pin in r["flow_csv"].items():
        p=D/name
        if sha(p)!=pin["sha256"] or p.stat().st_size!=pin["size_bytes"]:
            raise ValueError("flow CSV pin mismatch: "+name)

    source=read_rows(source_tet_path);labels=read_rows(parent_labels_path);nodes=read_rows(source_node_path)
    source_contrib=read_rows(source_path)
    globalrows=np.asarray([i for i,x in enumerate(source) if int(x["material_index"])==material],dtype=np.uint32)
    pt=np.asarray([[int(x["parent_node"+str(k)]) for k in range(4)] for x in source],dtype=np.int64)
    px=np.asarray([[float(x["rest_"+axis+"_m"]) for axis in "xyz"] for x in nodes],dtype=np.float64)
    parent_mass=np.asarray([float(source[g]["parent_mass_kg"]) for g in globalrows])
    parent_volume=np.asarray([float(source[g]["parent_reference_volume_m3"]) for g in globalrows])
    parent_regions=np.asarray([int(labels[g]["anatomical_region_id"]) for g in globalrows],dtype=np.uint32)
    parent_donors=np.asarray([int(labels[g]["geometric_donor_body"]) for g in globalrows],dtype=np.uint32)
    parent_faces=np.asarray([int(labels[g]["stable_source_face_id"]) for g in globalrows],dtype=np.uint32)
    parent_tet_nodes=pt[globalrows].astype(np.uint32)
    parent_tets=px[parent_tet_nodes]
    parent_centroids=ordered_centroids(px,parent_tet_nodes)

    node_ids,child_points,child_tets,child_material=read_child(node_path,ele_path)
    child_count=len(child_tets)
    if child_count==0 or not np.all(child_material==material):
        raise ValueError("current child element table is empty or has wrong material")
    expected_parent_key="parent_skin_tets" if role=="Skin" else "parent_fat_tets"
    expected_child_key="child_skin_tets" if role=="Skin" else "child_fat_tets"
    if r.get(expected_parent_key)!=len(globalrows) or r.get(expected_child_key)!=child_count:
        raise ValueError("receipt source/child row counts differ from the selected inputs")
    if r.get("maximum_transport_distance_allowed_m") is not None and not math.isclose(
            r["maximum_transport_distance_allowed_m"],.019353,rel_tol=0,abs_tol=1e-15):
        raise ValueError("receipt changes the role transport radius")
    if r.get("density_bounds_kg_m3") is not None and list(r["density_bounds_kg_m3"]) != [density_lower,density_upper]:
        raise ValueError("receipt changes the role density bounds")
    child_label_rows=read_rows(labels_path)
    expected_label_header={"child_tet_row","material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"}
    if len(child_label_rows)!=child_count or set(child_label_rows[0])!=expected_label_header:
        raise ValueError("child labels are incomplete or have an unexpected typed header")
    for i,row in enumerate(child_label_rows):
        if int(row["child_tet_row"])!=i or int(row["material_index"])!=material:
            raise ValueError("child label row identity/material mismatch")
    child_centroids=ordered_centroids(child_points,child_tets)
    dv=child_points[child_tets][:,1:]-child_points[child_tets][:,:1]
    child_volume=np.abs(np.einsum("ij,ij->i",dv[:,0],np.cross(dv[:,1],dv[:,2]))/6.)
    child_mesh_sha=child_mesh_digest(node_path,ele_path,child_points,child_tets)
    region_label_sha=sha(labels_path)
    expected_pins={
        "input_child_node_sha256":sha(node_path),
        "input_child_element_sha256":sha(ele_path),
        "input_parent_mass_csv_sha256":sha(S),
        "input_parent_label_csv_sha256":sha(parent_labels_path),
    }
    if mode=="solver_free_lp_feasibility":
        if not math.isclose(r.get("maximum_transport_distance_allowed_m",float("nan")),
                            .019353,rel_tol=0,abs_tol=1e-15):
            raise ValueError("LP-feasibility receipt changes the fixed role transport radius")
        if list(r.get("density_bounds_kg_m3",[])) != [density_lower,density_upper]:
            raise ValueError("LP-feasibility receipt changes the role density bounds")
        expected_pins.update({
            "support_binary_sha256":sha(args.support_bin),
            "input_region_label_sha256":region_label_sha,
            "input_parent_source_sha256":sha(source_path),
        })
        validate_current_input_pins(r,expected_pins)
        if r["input_region_label_sha256"] != region_label_sha:
            raise ValueError("region-label pin differs from current child typed labels")
        if r["input_parent_mass_sha256"] != sha(S):
            raise ValueError("parent mass packet alias differs from compiled source-330 mass CSV")
        if r["child_mesh_sha256"] != child_mesh_sha:
            raise ValueError("child mesh digest differs from current selected TetGen files")
    else:
        for key,digest in expected_pins.items():
            if r.get(key)!=digest:
                raise ValueError("legacy input pin mismatch: "+key)

    parent=load_numeric(D,"transport-source-parent.csv",11)
    child=load_numeric(D,"transport-child-mass.csv",4)
    if len(parent)!=len(globalrows) or len(child)!=child_count:
        raise ValueError("transport parent/child table row count differs from selected inputs")
    if not np.array_equal(parent[:,0],np.arange(len(parent))) or not np.array_equal(parent[:,1],globalrows) or not np.all(parent[:,2]==material):
        raise ValueError("transport parent source identity rows differ")
    if not np.array_equal(parent[:,3],parent_donors) or not np.array_equal(parent[:,4],parent_regions) or not np.array_equal(parent[:,5],parent_faces):
        raise ValueError("transport parent typed labels differ")
    close(parent[:,6],parent_mass,0,1e-12,"parent mass")
    close(parent[:,7],parent_volume,0,1e-12,"parent reference volume")
    close(parent[:,8:],parent_centroids,1e-12,0,"source centroid")
    if not np.array_equal(child[:,0],np.arange(child_count)) or not np.all(child[:,1:]>0):
        raise ValueError("child mass table rows or physical values invalid")
    density=child[:,2]/child[:,1]
    if np.any(density<density_lower-1e-7) or np.any(density>density_upper+1e-7):
        raise ValueError("child density outside role bounds")
    close(child[:,3],density,1e-8,1e-12,"child density")
    close(child[:,1],child_volume,0,1e-10,"recorded/current F32 child volume")

    edge=load_numeric(D,"transport-flow.csv",5)
    carrier=load_numeric(D,"source-carrier-flow.csv",7)
    if r.get("transport_edges") is not None and r.get("transport_edges")!=len(edge):
        raise ValueError("receipt aggregate edge count differs from recorded flow CSV")
    if not np.array_equal(edge[:,:2],np.rint(edge[:,:2])) or not np.array_equal(carrier[:,:4],np.rint(carrier[:,:4])):
        raise ValueError("flow row identifiers are not integers")
    ec=edge[:,0].astype(np.int64);ep=edge[:,1].astype(np.int64)
    fc=carrier[:,0].astype(np.int64);fp=carrier[:,1].astype(np.int64)
    if not (np.all(edge[:,2:4]>0) and np.all(edge[:,4]>=0) and np.all(carrier[:,4:6]>0) and np.all(carrier[:,6]>=0)):
        raise ValueError("nonpositive or invalid flow values")
    if np.any(fc<0)|np.any(fc>=child_count)|np.any(fp<0)|np.any(fp>=len(parent)):
        raise ValueError("carrier flow child/parent row outside source domains")
    if np.any(ec<0)|np.any(ec>=child_count)|np.any(ep<0)|np.any(ep>=len(parent)):
        raise ValueError("aggregate flow child/parent row outside source domains")
    child_error=close(np.bincount(ec,weights=edge[:,2],minlength=child_count),child[:,2],0,1e-10,"child inflow")
    parent_error=close(np.bincount(ep,weights=edge[:,2],minlength=len(parent)),parent_mass,1e-13,1e-10,"parent mass closure")
    measure_error=close(np.bincount(ep,weights=edge[:,3],minlength=len(parent)),parent_volume,1e-15,1e-10,"parent source measure closure")
    close(edge[:,3],edge[:,2]*(parent_volume/parent_mass)[ep],0,1e-10,"edge source measure")

    q,q_sha=load_packet(Q/"source-quadrature.bin",Q/"source-quadrature.npz",globalrows)
    if mode=="deterministic_optimizer_replay":
        if r["source_carrier_packet_sha256"]!=q_sha:
            raise ValueError("deterministic source-carrier packet differs from authenticated quadrature")
        offsets=q["parent_offsets"].astype(np.int64)
        counts=np.maximum(np.diff(offsets),1)
        expanded_source=np.repeat(np.arange(len(parent)),counts)
        expanded_carrier=np.full(len(expanded_source),INVALID,dtype=np.uint32)
        sites=np.empty((len(expanded_source),3),dtype=np.float64)
        expanded_weight=np.ones(len(expanded_source),dtype=np.float64)
        n=0
        for p,count in enumerate(counts):
            a,b=map(int,offsets[p:p+2])
            if a==b:
                sites[n]=parent_centroids[p]
            else:
                sites[n:n+count]=q["carrier_barycentric"][a:b]@parent_tets[p]
                expanded_carrier[n:n+count]=np.arange(a,b,dtype=np.uint32)
                expanded_weight[n:n+count]=q["carrier_weights"][a:b]
            n+=count
        carrierized=int(np.count_nonzero(np.diff(offsets)))
    else:
        supplied=(args.held_carrier_flow_csv,args.held_partial_manifest,args.lp_proposal,
                  args.lp_input_manifest,args.feasibility_source)
        if any(x is None for x in supplied):
            raise ValueError("LP-feasibility verification requires held CSV, held manifest, LP proposal, LP input manifest, and feasibility source")
        (offsets,weights,barys,witnesses,witness_distances,expanded_source,expanded_carrier,
         sites,expanded_weight,packet_sha,carrierized)=build_lp_carriers(
            args.lp_proposal,args.held_carrier_flow_csv,args.held_partial_manifest,
            args.lp_input_manifest,args.feasibility_source,r,q,q_sha,globalrows,
            parent_mass,parent_volume,parent_tets,parent_centroids,child_centroids,.019353)
        if r["source_carrier_packet_sha256"]!=packet_sha:
            raise ValueError("composite LP source-carrier packet digest mismatch")
        array_sha=source_carrier_array_digest(packet_sha,sha(source_tet_path),sha(source_node_path),
            offsets,globalrows,parent_tet_nodes,parent_tets,weights,barys,witnesses,witness_distances,
            len(parent),r["input_parent_mass_sha256"],r["input_parent_source_sha256"],
            child_mesh_sha,region_label_sha,r["input_coordinate_frame_sha256"])
        if r["source_carrier_array_sha256"]!=array_sha:
            raise ValueError("composite LP source-carrier array digest mismatch")
        if r["source_carrierized_parents"]!=carrierized or r["source_carriers"]!=len(weights):
            raise ValueError("composite LP carrier counts differ from receipt")

    ef=carrier[:,2].astype(np.int64);carrier_ids=carrier[:,3].astype(np.uint64)
    if np.any(ef<0)|np.any(ef>=len(expanded_source)):
        raise ValueError("carrier flow references invalid expanded source parent")
    if not np.array_equal(expanded_source[ef],fp) or not np.array_equal(expanded_carrier[ef],carrier_ids.astype(np.uint32)):
        raise ValueError("carrier flow expanded parent/carrier mapping differs")
    actual_distance=np.linalg.norm(sites[ef]-child_centroids[fc],axis=1)
    close(carrier[:,6],actual_distance,1e-12,0,"carrier actual distance")
    if np.any(actual_distance>.019353+1e-12):
        raise ValueError("carrier flow exceeds unchanged role transport radius")
    expected_carrier_mass=parent_mass[fp]*expanded_weight[ef]
    expected_carrier_measure=parent_volume[fp]*expanded_weight[ef]
    carrier_mass_error=close(np.bincount(ef,weights=carrier[:,4],minlength=len(expanded_source)),
                             expected_carrier_mass,0,1e-10,"individual source carrier mass closure")
    carrier_measure_error=close(np.bincount(ef,weights=carrier[:,5],minlength=len(expanded_source)),
                                expected_carrier_measure,0,1e-10,"individual source carrier measure closure")
    close(carrier[:,5],carrier[:,4]*(parent_volume/parent_mass)[fp],0,1e-10,"carrier source measure")
    key=(fc.astype(np.uint64)<<32)|fp.astype(np.uint64);order=np.argsort(key,kind="stable");unique,starts=np.unique(key[order],return_index=True)
    ekey=(ec.astype(np.uint64)<<32)|ep.astype(np.uint64);eo=np.argsort(ekey,kind="stable")
    if not np.array_equal(unique,ekey[eo]) or len(unique)!=len(edge):
        raise ValueError("aggregate and carrier flow edge sets differ")
    mass=np.add.reduceat(carrier[order,4],starts);measure=np.add.reduceat(carrier[order,5],starts)
    close(mass,edge[eo,2],0,1e-10,"aggregate carrier mass")
    close(measure,edge[eo,3],0,1e-10,"aggregate carrier measure")
    close(np.add.reduceat(carrier[order,4]*carrier[order,6],starts)/mass,edge[eo,4],1e-12,1e-10,"aggregate weighted distance")

    support,support_sha=load_support(args.support_bin,material,child_count,
        parent_donors,parent_regions,parent_faces)
    if r.get("support_binary_sha256")!=support_sha:
        raise ValueError("semantic-support binary hash differs from current authenticated support")
    if mode=="solver_free_lp_feasibility" and r.get("semantic_support_sha256")!=support["typed_support_sha256"]:
        raise ValueError("semantic-support hash differs from current authenticated support")
    nonpreferred_p1_edges=check_edges(support,ec,ep,parent_regions)

    source_rows=read_rows(source_path);by_node={}
    for row in source_rows:
        if int(row["material_index"])!=material: continue
        node=int(row["object_local_node"]);donor=int(row["donor_body"]);m=float(row["mass_kg"])
        by_node.setdefault(node,[]).append((donor,m))
    splits={}
    for p,g in enumerate(globalrows):
        for node in pt[g]:
            owners=by_node[int(node)];total=sum(x[1] for x in owners)
            for donor,m in owners:
                splits[p,donor]=splits.get((p,donor),0)+parent_mass[p]*.25*m/total
    pd=load_numeric(D,"transport-source-parent-donor.csv",3)
    cd=load_numeric(D,"transport-child-source-donor.csv",3)
    if not np.array_equal(pd[:,:2],np.rint(pd[:,:2])) or not np.array_equal(cd[:,:2],np.rint(cd[:,:2])):
        raise ValueError("source-donor row identifiers are not integers")
    if len(splits)!=len(pd) or not np.all(pd[:,2]>0) or not np.all(cd[:,2]>0):
        raise ValueError("source-donor split domain mismatch")
    if set(map(tuple,pd[:,:2].astype(np.int64)))!=set(splits):
        raise ValueError("parent source-donor split identities differ")
    cd_keys=(cd[:,0].astype(np.uint64)<<32)|cd[:,1].astype(np.uint64)
    if len(np.unique(cd_keys))!=len(cd) or not set(map(int,np.unique(cd[:,1]))).issubset(set(map(int,np.unique(pd[:,1])))):
        raise ValueError("child source-donor identity rows are duplicated or unsupported")
    if np.any(cd[:,0]<0) or np.any(cd[:,0]>=child_count):
        raise ValueError("child source-donor row outside child domain")
    close(np.bincount(cd[:,0].astype(np.int64),weights=cd[:,2],minlength=child_count),
          child[:,2],0,1e-10,"child donor total")
    for p,d,m in pd:
        close(np.asarray([m]),np.asarray([splits[int(p),int(d)]]),0,1e-10,"source donor split")
    donor_error=0.
    for donor in np.unique(pd[:,1].astype(np.int64)):
        fraction=np.zeros(len(parent));mask=pd[:,1]==donor
        fraction[pd[mask,0].astype(np.int64)]=pd[mask,2]/parent_mass[pd[mask,0].astype(np.int64)]
        expected=np.bincount(ec,weights=edge[:,2]*fraction[ep],minlength=child_count)
        mask=cd[:,1]==donor;actual=np.zeros(child_count);actual[cd[mask,0].astype(np.int64)]=cd[mask,2]
        donor_error=max(donor_error,close(actual,expected,0,1e-10,"child source donor composition"))

    report={
        "status":f"PASS_INDEPENDENT_{role.upper()}_RECORDED_MASS_AND_CARRIER_FLOW",
        "receipt_mode":mode,"physical_admission":False,"native_steps":0,
        "child_node_path":str(node_path),"child_element_path":str(ele_path),
        "child_labels_path":str(labels_path),"child_mesh_sha256":child_mesh_sha,
        "counts":{"source_parents":len(parent),"children":len(child),"aggregate_flow_edges":len(edge),
                  "carrier_flow_edges":len(carrier),"carrierized_parents":carrierized},
        "source_mass_kg":float(parent_mass.sum()),"child_mass_kg":float(child[:,2].sum()),
        "density_min_kg_m3":float(density.min()),"density_max_kg_m3":float(density.max()),
        "maximum_carrier_distance_m":float(actual_distance.max()),
        "maximum_parent_mass_error_kg":parent_error,
        "maximum_source_carrier_mass_error_kg":carrier_mass_error,
        "maximum_source_carrier_measure_error_m3":carrier_measure_error,
        "maximum_child_mass_error_kg":child_error,"maximum_parent_measure_error_m3":measure_error,
        "maximum_source_donor_child_composition_error_kg":donor_error,
        "source_quadrature_base_sha256":q_sha,"source_carrier_packet_sha256":r["source_carrier_packet_sha256"],
        "source_carrier_array_sha256":r["source_carrier_array_sha256"],
        "allocation_result_sha256":sha(receipt_path),"semantic_support_file_sha256":support_sha,
        "nonpreferred_p1_but_onehop_permitted_edges":nonpreferred_p1_edges,
        "verifier_sha256":sha(Path(__file__)),"elapsed_s":time.monotonic()-start,
    }
    if mode=="solver_free_lp_feasibility":
        report["independent_claim"]="recorded held-plus-LP flow, carrier digests, mass/measure/donor closure; optimizer replay not run"
        report["optimizer_replay"]="NOT_RUN"
    out=D/"independent-recorded-allocation-verification.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
