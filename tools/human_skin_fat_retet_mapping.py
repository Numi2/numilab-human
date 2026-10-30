#!/usr/bin/env python3
"""Small, fail-closed source/operational mapping and selected-cell splice helpers."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

OPERATIONAL_DISTANCE_LIMIT_M = 0.0004
NODE_MAP_HEADER = [
    "joint_node_id","source_partition","source_tetgen_node_id",
    "registered_I_source_vertex_row","merge_peer_partition","merge_peer_tetgen_node_id",
    "registered_surface_layer","registered_surface_vertex_row","role",
    "current_x_m","current_y_m","current_z_m","donor_body","donor_binding",
    "donor_local_x_m","donor_local_y_m","donor_local_z_m","source_domain",
    "source_mapping_kind","source_vertex_row","source_face_row","source_bary0",
    "source_bary1","source_bary2","source_residual_x_m","source_residual_y_m",
    "source_residual_z_m","frame_owner_method","parent_registration_kind",
    "parent_tet_row","parent_bary0","parent_bary1","parent_bary2","parent_bary3",
    "parent_current_residual_x_m","parent_current_residual_y_m",
    "parent_current_residual_z_m","parent_registration_distance_m",
    "parent_registration_tie_count","incident_material_mask","geometric_donor_body",
    "anatomical_region_id","source_parent_node_row","operational_reference_tet_row",
    "operational_reference_bary0","operational_reference_bary1",
    "operational_reference_bary2","operational_reference_bary3",
    "operational_reference_residual_x_m","operational_reference_residual_y_m",
    "operational_reference_residual_z_m","operational_reference_distance_m",
    "source_parent_geometric_donor_body","source_parent_anatomical_region_id",
]

def signed_volume(points, tets):
    q=np.asarray(points,dtype=np.float64)[np.asarray(tets,dtype=np.int64)]
    return np.einsum("ij,ij->i",q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6.0

def outface(tet, opposite):
    f=[int(tet[j]) for j in range(4) if j!=opposite]
    if opposite%2: f[1],f[2]=f[2],f[1]
    return tuple(f)

def face_incidence(tets):
    inc=defaultdict(list)
    for ci,tet in enumerate(np.asarray(tets,dtype=np.int64)):
        for o in range(4):
            face=outface(tet,o)
            inc[tuple(sorted(face))].append((int(ci),face))
    return inc

def cycle(face):
    f=list(map(int,face));i=f.index(min(f))
    return tuple(f[i:]+f[:i])

def _check_face_incidence(tets):
    inc=face_incidence(tets)
    for key,hits in inc.items():
        if len(hits)>2: raise ValueError(f"nonmanifold face {key}")
        if len(hits)==2 and cycle(hits[0][1])==cycle(hits[1][1]):
            raise ValueError(f"same-wound shared face {key}")
    return inc


def validate_source330_cell_map_row(row, child_centroid, child_label, parent_labels,
                                    source_points, source_tets, tolerance=1e-10):
    """Check historical source registration; its authored residual is not op distance."""
    sr=int(row["selected_parent_tet_row"])
    if sr<0 or sr>=len(parent_labels): raise ValueError("source330 cell parent row out of domain")
    label=parent_labels[sr]
    expected=(int(label["material_index"]),int(label["geometric_donor_body"]),
              int(label["anatomical_region_id"]),int(label["stable_source_face_id"]))
    actual=tuple(map(int,child_label))
    mapped=(int(row["material_index"]),int(row["geometric_donor_body"]),
            int(row["anatomical_region_id"]),int(row["stable_source_face_id"]))
    if expected!=actual or mapped!=actual: raise ValueError("source330 cell parent semantic/stable-face mismatch")
    beta=np.asarray([float(row[f"centroid_barycentric{k}"]) for k in range(4)],dtype=np.float64)
    if not np.isfinite(beta).all() or beta.min()<-1e-8 or beta.max()>1+1e-8 or abs(float(beta.sum())-1)>1e-8:
        raise ValueError("source330 historical cell barycentric invalid")
    tet=np.asarray(source_tets[sr],dtype=np.int64)
    if signed_volume(source_points,np.asarray([tet]))[0]<=0: raise ValueError("source330 historical parent tet is nonpositive")
    registered=beta@np.asarray(source_points,dtype=np.float64)[tet]
    claimed=np.asarray([float(row[f"centroid_registered_reference_{a}_m"]) for a in "xyz"],dtype=np.float64)
    residual=np.asarray([float(row[f"centroid_reference_residual_{a}_m"]) for a in "xyz"],dtype=np.float64)
    observed=np.asarray(child_centroid,dtype=np.float64)-registered
    if not np.isfinite(claimed).all() or np.linalg.norm(registered-claimed)>tolerance:
        raise ValueError("source330 historical registered endpoint mismatch")
    if not np.isfinite(residual).all() or np.linalg.norm(observed-residual)>tolerance:
        raise ValueError("source330 historical residual vector mismatch")
    norm=float(np.linalg.norm(observed));declared=float(row["centroid_reference_residual_norm_m"])
    if not np.isfinite(declared) or abs(norm-declared)>tolerance: raise ValueError("source330 historical residual norm mismatch")
    kind=row["label_match_kind"]
    if kind not in ("exact_f32_parent_tet_geometry","parent_centroid_containment",
                    "bounded_closest_parent_tet_reference_registration"):
        raise ValueError("unsupported source330 historical registration kind")
    # This is source-history/provenance, not the operational-reference distance gate.
    max_text=row.get("reference_registration_max_distance_m","")
    if kind=="bounded_closest_parent_tet_reference_registration":
        if max_text in ("",None) or not np.isfinite(float(max_text)) or norm>float(max_text)+tolerance:
            raise ValueError("source330 bounded historical locator contradicts its own declared limit")
    dist_text=row.get("reference_registration_distance_m","")
    if dist_text not in ("",None) and abs(float(dist_text)-norm)>tolerance:
        raise ValueError("source330 historical locator distance differs from residual")
    return {"parent_tet_row":sr,"residual_norm_m":norm,"label_match_kind":kind}

def validate_historical_node_registration(row, source_points, source_tets, source_labels,
                                          current_point, tolerance=2e-8):
    """Historical source-parent registration: explicit residual, no op-distance bound."""
    tetrow=int(row["parent_tet_row"])
    if tetrow<0 or tetrow>=len(source_tets): raise ValueError("historical parent tet row invalid")
    label=source_labels[tetrow]
    source_donor=int(row["source_parent_geometric_donor_body"])
    source_region=int(row["source_parent_anatomical_region_id"])
    if int(label["geometric_donor_body"])!=source_donor or int(label["anatomical_region_id"])!=source_region:
        raise ValueError("historical parent donor/region mismatch")
    beta=np.asarray([float(row[f"parent_bary{k}"]) for k in range(4)],dtype=np.float64)
    residual=np.asarray([float(row[f"parent_current_residual_{a}_m"]) for a in "xyz"],dtype=np.float64)
    if not np.isfinite(beta).all() or beta.min()<-1e-8 or beta.max()>1+1e-8 or abs(float(beta.sum())-1)>1e-8:
        raise ValueError("historical parent barycentric invalid")
    if not np.isfinite(residual).all(): raise ValueError("historical parent residual nonfinite")
    distance=float(row["parent_registration_distance_m"])
    if not np.isfinite(distance) or distance<0 or abs(float(np.linalg.norm(residual))-distance)>2e-10:
        raise ValueError("historical parent residual/distance mismatch")
    tet=np.asarray(source_tets[tetrow],dtype=np.int64)
    if signed_volume(source_points,np.asarray([tet]))[0]<=0: raise ValueError("historical parent tet is nonpositive")
    reconstructed=beta@np.asarray(source_points,dtype=np.float64)[tet]+residual
    if not np.isfinite(reconstructed).all() or np.linalg.norm(reconstructed-np.asarray(current_point,dtype=np.float64))>tolerance:
        raise ValueError("historical parent registration does not reconstruct current point")
    identity=int(row.get("source_parent_node_row",-1))
    if identity>=0:
        if identity not in tet: raise ValueError("source-parent identity absent from historical tet")
        expected=np.asarray([1.0 if n==identity else 0.0 for n in tet])
        if not np.array_equal(beta,expected): raise ValueError("source-parent identity is not exact one-hot")
    return {"parent_tet_row":tetrow,"historical_distance_m":distance,"source_parent_node_row":identity}

def validate_operational_node_registration(row, node_row, points_current, points_reference,
        tets, materials, donors, regions, maximum_distance_m=OPERATIONAL_DISTANCE_LIMIT_M):
    """Check separate operational mapping through a positive incident candidate tet."""
    node=int(node_row);support=int(row["operational_reference_tet_row"])
    if support<0 or support>=len(tets): raise ValueError("operational support tet outside candidate domain")
    tet=np.asarray(tets[support],dtype=np.int64)
    if node not in tet: raise ValueError("operational support tet is not incident to its node")
    donor=int(row["geometric_donor_body"]);region=int(row["anatomical_region_id"])
    if int(donors[support])!=donor or int(regions[support])!=region:
        raise ValueError("operational support tuple differs from candidate node")
    if signed_volume(points_current,np.asarray([tet]))[0]<=0 or signed_volume(points_reference,np.asarray([tet]))[0]<=0:
        raise ValueError("operational support tet is not positive in both frames")
    beta=np.asarray([float(row[f"operational_reference_bary{k}"]) for k in range(4)],dtype=np.float64)
    residual=np.asarray([float(row[f"operational_reference_residual_{a}_m"]) for a in "xyz"],dtype=np.float64)
    distance=float(row["operational_reference_distance_m"])
    if not np.isfinite(beta).all() or beta.min()<-1e-8 or beta.max()>1+1e-8 or abs(float(beta.sum())-1)>1e-8:
        raise ValueError("operational barycentric invalid")
    if not np.isfinite(residual).all() or not np.isfinite(distance) or distance<0 or distance>maximum_distance_m:
        raise ValueError("operational residual exceeds its fixed bound or is nonfinite")
    if abs(float(np.linalg.norm(residual))-distance)>2e-10:
        raise ValueError("operational residual norm differs from recorded distance")
    target=np.asarray(points_current,dtype=np.float64)[node]
    current_base=beta@np.asarray(points_current,dtype=np.float64)[tet]
    if np.linalg.norm(current_base+residual-target)>2e-8:
        raise ValueError("operational current reconstruction mismatch")
    reference_residual=residual+np.sum(beta[:,None]*(
        np.asarray(points_current,dtype=np.float64)[tet]-
        np.asarray(points_reference,dtype=np.float64)[tet]),axis=0)
    reference_base=beta@np.asarray(points_reference,dtype=np.float64)[tet]
    ref_target=np.asarray(points_reference,dtype=np.float64)[node]
    if np.linalg.norm(reference_base+reference_residual-ref_target)>1e-8:
        raise ValueError("operational reference reconstruction mismatch")
    return {"support_tet_row":support,"distance_m":distance,"reconstruction_error_m":float(np.linalg.norm(current_base+residual-target))}

def parse_operational_node_map(path):
    with Path(path).open("r",encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=NODE_MAP_HEADER: raise ValueError("combined node map is not exact 54-column split schema")
        rows=list(reader)
    if any(None in row for row in rows): raise ValueError("malformed combined node map row")
    return rows



# The adapter contract below is intentionally stricter than the earlier prototype:
# original node rows stay byte-identical, retet groups are explicit, and physical
# O/I/D faces are merged against the retained outside region rather than replaced.

def _semantic_group_rows(material, donor, region):
    groups=defaultdict(int)
    for m,d,r in zip(material,donor,region): groups[(int(m),int(d),int(r))]+=1
    return {f"{m}/{d}/{r}":n for (m,d,r),n in sorted(groups.items())}

def _valid_role_face(role, face, incidence, materials):
    hits=incidence.get(tuple(sorted(map(int,face))),[])
    if role in ("O","D"):
        expected=15 if role=="O" else 16
        return len(hits)==1 and int(materials[hits[0][0]])==expected and cycle(face)==cycle(hits[0][1])
    if role=="I":
        return len(hits)==2 and {int(materials[x[0]]) for x in hits}=={15,16} and any(cycle(face)==cycle(x[1]) for x in hits)
    return False



def validate_combined_node_map(rows, points_current, points_reference, tets, materials,
        donors, regions, source_points, source_tets, source_labels,
        maximum_operational_distance_m=OPERATIONAL_DISTANCE_LIMIT_M,
        role_faces=None, pinned_source_node_rows=None):
    """Validate the native 54-column map's historical and operational domains separately."""
    p=np.asarray(points_current,dtype=np.float32);ref=np.asarray(points_reference,dtype=np.float64)
    t=np.asarray(tets,dtype=np.int64);m=np.asarray(materials,dtype=np.int64)
    d=np.asarray(donors,dtype=np.int64);r=np.asarray(regions,dtype=np.int64)
    if not np.array_equal(p,ref.astype(np.float32)): raise ValueError("candidate current/reference Float32 points differ")
    if len(t)!=len(m) or len(t)!=len(d) or len(t)!=len(r): raise ValueError("candidate cell labels differ in row count")
    if np.any(signed_volume(p,t)<=0) or np.any(signed_volume(ref,t)<=0): raise ValueError("candidate has nonpositive cells")
    incidence=face_incidence(t)
    around=[[] for _ in range(len(p))]
    for ci,tet in enumerate(t):
        for node in tet: around[int(node)].append(ci)
    bynode=defaultdict(list);owners=set();source_node_owners={}
    for row in rows:
        try:
            node=int(row["joint_node_id"])-1;source_id=int(row["source_tetgen_node_id"])
            part=row["source_partition"];role=row["role"]
        except Exception as e: raise ValueError("combined node map has an invalid node/source ID") from e
        if node<0 or node>=len(p) or source_id<=0 or part not in ("skin","fat"):
            raise ValueError("combined node map node/source identity outside domain")
        if (part,source_id) in owners: raise ValueError("combined node map repeats a TetGen source-node identity")
        owners.add((part,source_id))
        if role not in ("O","I","D","interior"): raise ValueError("combined node map declares unsupported role")
        layer=row["registered_surface_layer"];surface=int(row["registered_surface_vertex_row"])
        iface=int(row["registered_I_source_vertex_row"])
        peerpart=row["merge_peer_partition"];peertext=row["merge_peer_tetgen_node_id"]
        if role=="I":
            if layer!="I" or surface<0 or iface!=surface or peerpart!=("fat" if part=="skin" else "skin") or not peertext:
                raise ValueError("combined I row lacks reciprocal interface identity")
        else:
            if iface!=-1 or peerpart or peertext: raise ValueError("non-I row claims an interface merge")
            if role=="interior":
                if layer or surface!=-1: raise ValueError("interior row claims a registered surface")
            elif layer!=role or surface<0: raise ValueError("registered node role/layer/row mismatch")
            if role=="O" and part!="skin": raise ValueError("O node is not in skin partition")
            if role=="D" and part!="fat": raise ValueError("D node is not in fat partition")
        point=np.asarray([float(row[f"current_{a}_m"]) for a in "xyz"],dtype=np.float64)
        if not np.isfinite(point).all() or not np.array_equal(point.astype(np.float32),p[node]):
            raise ValueError("combined map current coordinate differs from candidate F32 node")
        donor=int(row["geometric_donor_body"]);region=int(row["anatomical_region_id"])
        if not around[node]: raise ValueError("combined map node has no incident candidate cell")
        incident_mask=0
        semantic_incident=False
        for cell in around[node]:
            incident_mask|=1<<int(m[cell])
            semantic_incident|=(int(d[cell])==donor and int(r[cell])==region)
        if incident_mask!=int(row["incident_material_mask"]): raise ValueError("combined map incident material mask differs from candidate")
        if not semantic_incident: raise ValueError("node donor/region is absent from all incident candidate cells")
        if any(not 0<=int(m[cell])<32 for cell in around[node]): raise ValueError("candidate material index is outside the native incident-mask domain")
        source_kind=row["source_mapping_kind"]
        source_domain=row["source_domain"]
        source_vertex=int(row["source_vertex_row"]);source_face=int(row["source_face_row"])
        source_bary=np.asarray([float(row[f"source_bary{k}"]) for k in range(3)],dtype=np.float64)
        source_residual=np.asarray([float(row[f"source_residual_{a}_m"]) for a in "xyz"],dtype=np.float64)
        frame_method=row["frame_owner_method"]
        if not np.isfinite(source_bary).all() or not np.isfinite(source_residual).all():
            raise ValueError("combined map source barycentric/residual is nonfinite")
        if source_domain=="derived_volume_interior_no_surface_ancestry":
            if role!="interior" or source_vertex!=-1 or source_face!=-1 or source_kind!="none" or frame_method!="derived_full_all_bodies_geometric_donor_pose" or np.any(source_bary!=0) or np.any(source_residual!=0):
                raise ValueError("explicit interior-no-source identity is inconsistent")
        else:
            direct=source_vertex>=0;embedded=source_face>=0
            if not source_domain or direct==embedded or (direct and (source_kind!="vertex" or source_face!=-1)) or (embedded and (source_kind!="face" or source_vertex!=-1)) or (source_kind not in ("vertex","face")):
                raise ValueError("combined map source mapping kind disagrees with source identity")
            if embedded and (source_bary.min()<-1e-8 or source_bary.max()>1+1e-8 or abs(float(source_bary.sum())-1)>1e-8):
                raise ValueError("combined map source-face barycentric is invalid")
            if not frame_method: raise ValueError("combined map source frame-owner method is missing")
        if row["parent_registration_kind"] not in ("source_parent_tet_reference_registration","inferred_current_parent_tet_same_material_donor_region"):
            raise ValueError("combined map parent-registration provenance is unsupported")
        # Source330 history. This is reconstruction/provenance only; its distance is
        # deliberately not compared to the 0.4 mm operational support limit.
        hist=validate_historical_node_registration(row,source_points,source_tets,source_labels,p[node])
        if int(row["parent_registration_tie_count"])<0: raise ValueError("historical source registration tie count is negative")
        identity=hist["source_parent_node_row"]
        if identity>=0:
            previous=source_node_owners.get(identity)
            if previous is not None and previous!=node: raise ValueError("source-parent original node injection is duplicated")
            source_node_owners[identity]=node
        op=validate_operational_node_registration(row,node,p,ref,t,m,d,r,maximum_operational_distance_m)
        bynode[node].append((row,role,part))
    if set(bynode)!=set(range(len(p))): raise ValueError("combined node map does not cover every child volume node")
    interface_pairs=0
    for node,entries in bynode.items():
        roles=[x[1] for x in entries];parts=[x[2] for x in entries]
        if roles==["I"] or roles==["I","I"]:
            if len(entries)!=2 or set(parts)!={"skin","fat"}: raise ValueError("I node lacks exactly reciprocal skin/fat rows")
            a,b=entries
            if a[0]["merge_peer_partition"]!=b[2] or b[0]["merge_peer_partition"]!=a[2] or int(a[0]["merge_peer_tetgen_node_id"])!=int(b[0]["source_tetgen_node_id"]) or int(b[0]["merge_peer_tetgen_node_id"])!=int(a[0]["source_tetgen_node_id"]):
                raise ValueError("I node source identities are not reciprocal")
            if any(a[0][k]!=b[0][k] for k in NODE_MAP_HEADER[6:]): raise ValueError("reciprocal I typed fields differ")
            interface_pairs+=1
        elif len(entries)!=1: raise ValueError("non-I volume node appears more than once")
    if role_faces is not None:
        for role in ("O","I","D"):
            expected=set(map(int,np.unique(np.asarray(role_faces.get(role,np.empty((0,3))),dtype=np.int64))))
            observed={node for node,entries in bynode.items() if any(entry[1]==role for entry in entries)}
            if expected!=observed:
                raise ValueError(f"combined node map {role} rows differ from role-face vertex coverage: missing={len(expected-observed)} extra={len(observed-expected)}")
    if pinned_source_node_rows is not None:
        expected=np.asarray(pinned_source_node_rows,dtype=np.int64)
        if expected.ndim!=1: raise ValueError("pinned source-node identity rows malformed")
        for node,source_node in enumerate(expected):
            if source_node<0: continue
            entries=bynode.get(node,[])
            if len(entries)!=1 and not (len(entries)==2 and entries[0][1]==entries[1][1]=="I"):
                raise ValueError("protected source node has malformed combined-map copies")
            if any(int(entry[0]["source_parent_node_row"])!=int(source_node) for entry in entries):
                raise ValueError(f"protected node {node} source330 identity changed")
    return {"volume_nodes":len(p),"combined_map_rows":len(rows),"shared_interface_nodes":interface_pairs,
            "historical_source_node_injections":len(source_node_owners),
            "max_historical_residual_m":max((float(e[0]["parent_registration_distance_m"]) for v in bynode.values() for e in v),default=0.0),
            "max_operational_residual_m":max((float(e[0]["operational_reference_distance_m"]) for v in bynode.values() for e in v),default=0.0)}
def _merge_role_faces_compact(base, retet, local_to_global, base_node_output,
                             full_tets, full_material, selected, retet_start):
    full_inc=face_incidence(full_tets)
    base_inc=face_incidence(base["volume_tetrahedra"])
    merged={};dropped={};added={}
    for role in ("O","I","D"):
        name=f"coarse_boundary_faces_{role}"
        if name not in base: continue
        old=np.asarray(base[name],dtype=np.int64)
        keep=[];keep_rows=[];drop=[]
        for row,face in enumerate(old):
            hits=base_inc.get(tuple(sorted(map(int,face))),[])
            if not _valid_role_face(role,face,base_inc,base["material_index"]):
                raise ValueError(f"base role {role} face {row} is not valid before splice")
            mapped=local_map=base_node_output[face]
            survives=(np.all(mapped>=0) and _valid_role_face(role,mapped,full_inc,full_material))
            if survives:
                keep.append(np.asarray(mapped,dtype=np.int64));keep_rows.append(row)
            else:
                if not any(int(h[0]) in selected for h in hits):
                    raise ValueError(f"retet invalidated outside-owned {role} face {row}")
                drop.append(row)
        local_name=f"coarse_boundary_faces_{role}"
        if local_name in retet:
            raw=np.asarray(retet[local_name],dtype=np.int64)
            if raw.ndim!=2 or raw.shape[1]!=3 or np.any(raw<0) or np.any(raw>=len(local_to_global)):
                raise ValueError(f"retet {role} face rows invalid")
            repl=local_to_global[raw]
        else:
            repl=np.empty((0,3),dtype=np.int64)
            if drop: raise ValueError(f"retet dropped {len(drop)} old {role} faces without replacement")
        kept_keys={tuple(sorted(map(int,f))):tuple(map(int,f)) for f in keep}
        newfaces=[];new_local_rows=[];seen=set()
        for i,face in enumerate(repl):
            k=tuple(sorted(map(int,face)))
            if k in seen: raise ValueError(f"retet repeats a {role} face")
            seen.add(k)
            if not _valid_role_face(role,face,full_inc,full_material):
                raise ValueError(f"retet {role} face has invalid incidence or orientation: {k}")
            if not any(int(hit[0])>=retet_start for hit in full_inc[k]):
                raise ValueError(f"retet {role} face is not owned by retet cell: {k}")
            if k in kept_keys:
                if cycle(kept_keys[k])!=cycle(face):
                    raise ValueError(f"retet reverses retained {role} face {k}")
                continue
            newfaces.append(np.asarray(face,dtype=np.int64));new_local_rows.append(i)
        allfaces=np.asarray(keep+newfaces,dtype=np.int64).reshape(-1,3)
        allkeys=[tuple(sorted(map(int,f))) for f in allfaces]
        if len(set(allkeys))!=len(allkeys): raise ValueError(f"merged {role} face list duplicates")
        merged[name]=allfaces;dropped[role]=drop;added[role]=len(newfaces)
        mapname=f"coarse_to_fine_face_{role}";baryname=f"coarse_to_fine_bary_{role}"
        if mapname in base or baryname in base or newfaces:
            if mapname not in base or baryname not in base:
                if newfaces: raise ValueError(f"new {role} faces require both fine-map arrays")
                continue
            bf=np.asarray(base[mapname]);bb=np.asarray(base[baryname])
            if bf.shape!=(len(old),) or bb.shape!=(len(old),3):
                raise ValueError(f"base {role} fine-map rows differ from faces")
            maps=[bf[keep_rows]];barys=[bb[keep_rows]]
            if newfaces:
                if mapname not in retet or baryname not in retet:
                    raise ValueError(f"replacement {role} faces lack fine-map ancestry")
                rf=np.asarray(retet[mapname]);rb=np.asarray(retet[baryname])
                if rf.shape!=(len(repl),) or rb.shape!=(len(repl),3):
                    raise ValueError(f"retet {role} fine-map row count mismatch")
                maps.append(rf[np.asarray(new_local_rows,dtype=np.int64)])
                barys.append(rb[np.asarray(new_local_rows,dtype=np.int64)])
            merged[mapname]=np.concatenate(maps).astype(bf.dtype,copy=False)
            merged[baryname]=np.concatenate(barys,axis=0).astype(bb.dtype,copy=False)
    return merged,{"retired_old_face_rows":dropped,"added_replacement_faces":added}

def assemble_selected_retet(base, selected_rows, retet):
    """Compact selected retets while retaining exact outside cells and explicit pins.

    A selected-only old node can retire. To retarget a protected canonical row, the
    retet packet supplies protected_node_local_rows; this records routing only, and
    does not claim anatomical target validity.
    """
    p0=np.asarray(base["points_current_world_m_f32"],dtype=np.float32)
    ref0=np.asarray(base["points_reference_world_m_f64"],dtype=np.float64)
    t0=np.asarray(base["volume_tetrahedra"],dtype=np.int64)
    m0=np.asarray(base["material_index"],dtype=np.int64)
    d0=np.asarray(base["geometric_donor_body"],dtype=np.int64)
    r0=np.asarray(base["anatomical_region_id"],dtype=np.int64)
    if not np.array_equal(p0,ref0.astype(np.float32)): raise ValueError("base current/reference F32 differ")
    if t0.ndim!=2 or t0.shape[1]!=4 or len(m0)!=len(t0) or len(d0)!=len(t0) or len(r0)!=len(t0):
        raise ValueError("base tet/label domains invalid")
    if np.any(signed_volume(p0,t0)<=0): raise ValueError("base has nonpositive cells")
    oldinc=_check_face_incidence(t0)
    sel=np.asarray(sorted(set(map(int,selected_rows))),dtype=np.int64)
    if len(sel)!=len(selected_rows) or not len(sel) or np.any(sel<0) or np.any(sel>=len(t0)):
        raise ValueError("selected source cell rows invalid or duplicated")
    selected=set(map(int,sel))
    outside=np.asarray([i for i in range(len(t0)) if i not in selected],dtype=np.int64)
    outside_set=set(map(int,outside))
    allowed={(int(m0[i]),int(d0[i]),int(r0[i])) for i in sel}

    rp=np.asarray(retet["points_current_world_m_f32"],dtype=np.float32)
    rt=np.asarray(retet["volume_tetrahedra"],dtype=np.int64)
    rm=np.asarray(retet["material_index"],dtype=np.int64)
    rd=np.asarray(retet["geometric_donor_body"],dtype=np.int64)
    rr=np.asarray(retet["anatomical_region_id"],dtype=np.int64)
    node_map=np.asarray(retet["retet_node_base_rows"],dtype=np.int64)
    if rp.ndim!=2 or rp.shape[1]!=3 or not np.isfinite(rp).all(): raise ValueError("retet points invalid")
    if rt.ndim!=2 or rt.shape[1]!=4 or not len(rt): raise ValueError("retet tetrahedra invalid")
    if rm.shape!=(len(rt),) or rd.shape!=(len(rt),) or rr.shape!=(len(rt),): raise ValueError("retet labels invalid")
    if node_map.shape!=(len(rp),) or np.any(node_map < -1) or np.any(node_map>=len(p0)):
        raise ValueError("retet node map invalid")
    mapped=node_map[node_map>=0]
    if len(np.unique(mapped))!=len(mapped): raise ValueError("retet maps duplicate base nodes")
    if np.any(rt<0) or np.any(rt>=len(rp)): raise ValueError("retet cell index invalid")
    got_groups={(int(mm),int(dd),int(rg)) for mm,dd,rg in zip(rm,rd,rr)}
    if got_groups!=allowed:
        raise ValueError(f"retet semantic groups differ from selected source cells: missing={sorted(allowed-got_groups)} extra={sorted(got_groups-allowed)}")
    if np.any(signed_volume(rp,rt)<=0): raise ValueError("retet has nonpositive F32 cells")
    used=set(map(int,rt.reshape(-1)))
    for role in ("O","I","D"):
        name=f"coarse_boundary_faces_{role}"
        if name in retet:
            faces=np.asarray(retet[name],dtype=np.int64)
            if faces.ndim!=2 or faces.shape[1]!=3 or np.any(faces<0) or np.any(faces>=len(rp)):
                raise ValueError(f"retet {role} face rows invalid")
            used.update(map(int,faces.reshape(-1)))
    if used!=set(range(len(rp))): raise ValueError("retet packet has unused local points")

    star=[set() for _ in range(len(p0))]
    outside_nodes=set()
    for ci,tet in enumerate(t0):
        for node in tet:
            star[int(node)].add(ci)
            if ci in outside_set: outside_nodes.add(int(node))
    for local,old in enumerate(node_map):
        if old>=0 and not np.array_equal(rp[local],p0[int(old)]):
            raise ValueError(f"retet moved base node {int(old)}; use an explicit new node plus retirement")

    pins=np.asarray(base.get("pinned_volume_node_rows",np.empty(0)),dtype=np.int64)
    if pins.ndim!=1 or len(np.unique(pins))!=len(pins) or np.any(pins<0) or np.any(pins>=len(p0)):
        raise ValueError("protected canonical node rows malformed")
    protected_local=np.asarray(retet.get("protected_node_local_rows",np.empty(0,dtype=np.int64)),dtype=np.int64)
    if protected_local.size and protected_local.shape!=(len(pins),):
        raise ValueError("protected target map must cover every canonical protected row")
    local_for_old={int(old):int(i) for i,old in enumerate(node_map) if old>=0}
    protected_targets=[];protected_modes=[]
    for k,old0 in enumerate(pins):
        old=int(old0)
        loc=int(protected_local[k]) if protected_local.size else -1
        if old in outside_nodes:
            if loc>=0 and (loc>=len(rp) or int(node_map[loc])!=old or not np.array_equal(rp[loc],p0[old])):
                raise ValueError(f"outside protected node {old} must retain exact base identity")
            protected_targets.append(("old",old));protected_modes.append("retained_original_node")
            continue
        if loc<0 and old in local_for_old: loc=local_for_old[old]
        if loc<0: raise ValueError(f"selected protected row {old} lacks explicit retet descendant mapping")
        if loc>=len(rp) or loc not in used: raise ValueError(f"protected target row invalid or unused for {old}")
        mapped_old=int(node_map[loc])
        if mapped_old>=0 and mapped_old!=old:
            raise ValueError(f"protected row {old} cannot be rebound to another original node")
        if mapped_old==old:
            if not np.array_equal(rp[loc],p0[old]): raise ValueError("retained protected source node moved")
            protected_targets.append(("old",old));protected_modes.append("retained_original_node")
        else:
            if old in set(map(int,mapped)):
                raise ValueError(f"retargeted protected row {old} still appears as an active old-node identity")
            protected_targets.append(("local",loc));protected_modes.append("explicit_retet_descendant")

    keep_old=outside_nodes|set(map(int,mapped))
    for mode,pair in zip(protected_modes,protected_targets):
        if mode=="retained_original_node": keep_old.add(int(pair[1]))
    keep=np.asarray(sorted(keep_old),dtype=np.int64)
    base_to_out=np.full(len(p0),-1,dtype=np.int64)
    base_to_out[keep]=np.arange(len(keep),dtype=np.int64)
    newlocals=np.flatnonzero(node_map<0)
    local_to_out=np.empty(len(rp),dtype=np.int64)
    for local,old in enumerate(node_map):
        if old>=0:
            outrow=int(base_to_out[int(old)])
            if outrow<0: raise ValueError(f"mapped retet base node {int(old)} was not retained")
            local_to_out[local]=outrow
    local_to_out[newlocals]=len(keep)+np.arange(len(newlocals),dtype=np.int64)
    if len(newlocals):
        oldkeys={tuple(np.asarray(x,dtype="<f4").view("<u4").tolist()) for x in p0}
        newkeys=set()
        for local in newlocals:
            key=tuple(np.asarray(rp[local],dtype="<f4").view("<u4").tolist())
            if key in oldkeys or key in newkeys: raise ValueError(f"new retet node {local} duplicates an existing F32 coordinate without explicit identity")
            newkeys.add(key)
    fullp=np.concatenate((p0[keep],rp[newlocals]),axis=0).astype(np.float32,copy=False)
    mapped_outside=base_to_out[t0[outside]]
    if mapped_outside.size and np.any(mapped_outside<0): raise ValueError("outside cell uses a retired base node")
    full_t=np.concatenate((mapped_outside,local_to_out[rt]),axis=0)
    full_m=np.concatenate((m0[outside],rm));full_d=np.concatenate((d0[outside],rd));full_r=np.concatenate((r0[outside],rr))
    if np.any(signed_volume(fullp,full_t)<=0): raise ValueError("spliced full F32 mesh has nonpositive cells")
    inc1=_check_face_incidence(full_t)
    oldcut={k for k,h in oldinc.items() if len(h)==2 and ((h[0][0] in selected)!=(h[1][0] in selected))}
    mapped_cut={tuple(sorted(map(int,base_to_out[np.asarray(k,dtype=np.int64)]))) for k in oldcut}
    noutside=len(outside)
    newcut={k for k,h in inc1.items() if len(h)==2 and ((h[0][0]<noutside)!=(h[1][0]<noutside))}
    if mapped_cut!=newcut: raise ValueError(f"selected/outside cut changed: missing={len(mapped_cut-newcut)} extra={len(newcut-mapped_cut)}")

    protected_out=[];protected_local_out=[]
    for mode,pair in zip(protected_modes,protected_targets):
        kind,row=pair
        protected_out.append(int(base_to_out[int(row)]) if kind=="old" else int(local_to_out[int(row)]))
        protected_local_out.append(-1 if kind=="old" else int(row))
    protected_out=np.asarray(protected_out,dtype=np.int64);protected_local_out=np.asarray(protected_local_out,dtype=np.int64)
    if len(np.unique(protected_out))!=len(protected_out): raise ValueError("protected canonical targets are not injective")
    if any(not np.any(full_t==node) for node in protected_out): raise ValueError("protected target is not active in output cells")

    result={"points_current_world_m_f32":fullp,"points_reference_world_m_f64":fullp.astype(np.float64),
      "volume_tetrahedra":full_t,"material_index":full_m,"geometric_donor_body":full_d,"anatomical_region_id":full_r,
      "outside_source_tet_rows":outside,"retet_source_tet_rows":sel,
      "retet_node_base_rows":node_map,"retet_node_output_rows":local_to_out,
      "base_node_output_rows":base_to_out,"retired_base_node_rows":np.flatnonzero(base_to_out<0).astype(np.int64),
      "selected_cut_face_keys":np.asarray(sorted(mapped_cut),dtype=np.int64).reshape(-1,3),
      "protected_base_node_rows":pins.copy(),"protected_node_output_rows":protected_out,
      "protected_node_local_rows":protected_local_out,
      "protected_node_mapping_kind":np.asarray(protected_modes,dtype="U32"),"pinned_volume_node_rows":protected_out.copy()}
    merged,face_report=_merge_role_faces_compact(base,retet,local_to_out,base_to_out,full_t,full_m,selected,noutside)
    result.update(merged)
    active=set(map(int,full_t.reshape(-1)))
    for role in ("O","I","D"):
        name=f"coarse_boundary_faces_{role}"
        if name in result: active.update(map(int,np.asarray(result[name]).reshape(-1)))
    if active!=set(range(len(fullp))): raise ValueError("compacted splice contains unused output nodes")
    group_before=defaultdict(float);group_after=defaultdict(float)
    v0=signed_volume(p0,t0);vnew=signed_volume(rp,rt)
    for ci in sel: group_before[(int(m0[ci]),int(d0[ci]),int(r0[ci]))]+=float(v0[ci])
    for ci,(mm,dd,rg) in enumerate(zip(rm,rd,rr)): group_after[(int(mm),int(dd),int(rg))]+=float(vnew[ci])
    group_report={f"{mm}/{dd}/{rg}":{"before_m3":group_before[(mm,dd,rg)],
      "after_m3":group_after[(mm,dd,rg)],"delta_m3":group_after[(mm,dd,rg)]-group_before[(mm,dd,rg)]}
      for mm,dd,rg in sorted(allowed)}
    report={"selected_source_cells":len(sel),"outside_source_cells":len(outside),"retet_cells":len(rt),
      "output_cells":len(full_t),"output_nodes":len(fullp),"retained_base_nodes":int(np.sum(base_to_out>=0)),
      "retired_base_nodes":int(np.sum(base_to_out<0)),"retired_base_node_count":int(np.sum(base_to_out<0)),"retired_base_node_rows":np.flatnonzero(base_to_out<0).astype(np.int64),
      "new_nodes":int(len(newlocals)),"cut_faces_rechecked":len(mapped_cut),"positive_output_cells":int(len(full_t)),
      "semantic_groups_selected":_semantic_group_rows(m0[sel],d0[sel],r0[sel]),
      "semantic_groups_retet":_semantic_group_rows(rm,rd,rr),"selected_group_volume_m3":group_report,
      "protected_targets":[{"base_node_row":int(old),"output_node_row":int(outrow),"mapping_kind":str(kind)}
        for old,outrow,kind in zip(pins,protected_out,protected_modes)],
      "protected_rows_retargeted":int(sum(x!="retained_original_node" for x in protected_modes)),
      "retired_protected_base_rows":[int(old) for old in pins if base_to_out[int(old)]<0],
      "face_replacement":face_report,"changed_outside_cells":0,"outside_coordinates_exact":True}
    return result,report
