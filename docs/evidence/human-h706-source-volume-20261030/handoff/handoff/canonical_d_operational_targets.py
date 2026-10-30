"""Validate the existing operational149 packet without changing source history.

This is preparation-only routing into native-domain-217. Physical volume,
mass, contact, and accepted-state admission remain separate mandatory gates.
"""
from pathlib import Path
import csv,hashlib,json
import numpy as np

HISTORICAL_SHA = "9dfc74113fc1b4b549d26437a20adefa05766f75c2f38b2853273ebcc0e6a24b"
SCHEMA = "numi.D.combined-operational-reference-preflight.v1"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def csv_rows(p):
    with Path(p).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def f32key(p):
    return np.asarray(p,dtype="<f4").reshape(3).tobytes()

def _array_from_csv(path,id_column,value_columns):
    rows=csv_rows(path)
    ids=[int(row[id_column]) for row in rows]
    if sorted(ids)!=list(range(len(rows))): raise ValueError("source table IDs are not a complete row domain")
    result=np.empty((len(rows),len(value_columns)),float)
    for row,i in zip(rows,ids):
        result[i]=[float(row[c]) for c in value_columns]
    return result,rows

# The supported assembly's authenticated construction sources. A reduction may
# choose coordinates from these domains; this does not authorize arbitrary edits.
POINT_SOURCE_SHA256={
    "D624":"c17794a1012f76924dcd9e4e754bb026c774deb5a3d37e2f1c79635ccfe9dc83",
    "D010":"7d8419bdc4ad1d3892a946c93ad7f0f9b2db2e4302f8f666480612166e25de2a",
}

def select_paired_inner_surface(proof,gates):
    """Bind the inner-surface pairing without changing the legacy I014 contract."""
    paired=proof.get("paired_inner_surface")
    if paired is None:
        # Legacy receipts retain their original gate name and shape. They do not
        # claim a current-I pairing or expose a paired surface hash.
        return {"path":None,"sha256":None,"gate_name":"changed-D-vs-I014"}
    if (not isinstance(paired,dict) or set(paired)!={"path","sha256"} or
        not isinstance(paired.get("path"),str)):
        raise ValueError("paired inner surface proof must contain only string path and sha256")
    path=Path(paired["path"])
    expected=paired["sha256"]
    if not isinstance(expected,str) or len(expected)!=64 or sha(path)!=expected:
        raise ValueError("paired current inner surface hash mismatch")
    gate_pair=gates.get("paired_inner_surface")
    if (not isinstance(gate_pair,dict) or set(gate_pair)!={"path","sha256"} or
        gate_pair.get("path")!=str(path) or gate_pair.get("sha256")!=expected):
        raise ValueError("paired current inner surface is not bound by the exact-gate receipt")
    return {"path":str(path),"sha256":expected,"gate_name":"changed-D-vs-current-I"}

def validate_current_paired_parent_report(proof,gates,deep_sha,paired_inner):
    """Authenticate the already-computed whole-candidate report behind a subset gate."""
    report_path=Path(gates.get("full_current_composite_gate_report_path",""))
    report_sha=gates.get("full_current_composite_gate_report_sha256")
    if (proof.get("full_current_composite_exact_report_path")!=str(report_path) or
        proof.get("full_current_composite_exact_report_sha256")!=report_sha or
        not isinstance(report_sha,str) or sha(report_path)!=report_sha):
        raise ValueError("current paired full-candidate report pin mismatch")
    report=json.loads(report_path.read_text())
    candidate=report.get("candidate",{})
    if (report.get("status")!="PASS" or candidate.get("path")!=str(Path(proof["candidate_path"])) or
        candidate.get("sha256")!=deep_sha or report.get("I050",{}).get("path")!=paired_inner["path"] or
        report.get("I050",{}).get("sha256")!=paired_inner["sha256"]):
        raise ValueError("current paired full-candidate report does not bind these inputs")
    required={"D_full_self","D_vs_I050","D_vs_all185_F32_F64"}
    parent_gates={item.get("name"):item for item in report.get("gates",[])}
    if not required.issubset(parent_gates):
        raise ValueError("current paired full-candidate report lacks required exact gates")
    for name in required:
        gate=parent_gates[name]
        pair_path=Path(gate["pairs_path"])
        if (gate.get("count")!=0 or gate.get("candidate_sha256",deep_sha)!=deep_sha or
            sha(pair_path)!=gate.get("pairs_sha256")):
            raise ValueError("current paired parent exact gate is not a pinned zero result")
        pairs=np.load(pair_path,allow_pickle=False)
        if pairs.size!=0 or pairs.shape!=(0,2):
            raise ValueError("current paired parent exact gate pair file is nonempty")
    return {"path":str(report_path),"sha256":report_sha,
            "candidate_sha256":deep_sha,"paired_inner_sha256":paired_inner["sha256"],
            "verified_zero_gate_names":sorted(required)}

def validate_point_preservation(packet,deep_path,points):
    """Independently check every reduced point against its exact source row."""
    if packet.get("mode")=="authenticated-unused-storage-row-compaction":
        return validate_storage_compaction(packet,deep_path,points)
    if packet.get("mode")!="authenticated-source-coordinate-subset":
        raise ValueError("unrecognized operational point preservation mode")
    sources=packet.get("sources",[])
    if len(sources)!=2 or [s.get("role") for s in sources]!=["D624","D010"]:
        raise ValueError("point preservation requires ordered D624/D010 source domains")
    source_points=[]
    for source in sources:
        path=Path(source["path"]);expected=POINT_SOURCE_SHA256[source["role"]]
        if source.get("sha256")!=expected or sha(path)!=expected:
            raise ValueError("point preservation authenticated source hash mismatch")
        with np.load(path,allow_pickle=False) as z:
            source_points.append(np.asarray(z["points_world_m"],np.float32))
    with np.load(deep_path,allow_pickle=False) as z:
        keys=("point_preservation_source_asset_index","point_preservation_source_vertex_row")
        if any(k not in z.files for k in keys):
            raise ValueError("point preservation lacks explicit candidate source-row maps")
        asset=np.asarray(z[keys[0]]);rows=np.asarray(z[keys[1]])
        emitted=np.asarray(z["points_world_m"],np.float32)
    points=np.asarray(points,np.float32)
    n=len(points)
    if (asset.shape!=(n,) or rows.shape!=(n,) or
        asset.dtype.kind not in "iu" or rows.dtype.kind not in "iu" or
        np.any(asset>1) or np.any(asset<0) or np.any(rows<0)):
        raise ValueError("point preservation source-row domain malformed")
    if (emitted.shape!=points.shape or points.ndim!=2 or points.shape[1]!=3 or
        not np.isfinite(points).all() or
        not np.array_equal(emitted.view(np.uint32),points.view(np.uint32))):
        raise ValueError("point preservation does not describe emitted Float32 points")
    counts=[]
    for domain,reference in enumerate(source_points):
        selected=np.flatnonzero(asset==domain)
        r=rows[selected]
        if np.any(r>=len(reference)):
            raise ValueError("point preservation source vertex row out of range")
        if not np.array_equal(points[selected].view(np.uint32),reference[r].view(np.uint32)):
            raise ValueError("point preservation changed an authenticated source coordinate")
        counts.append(len(selected))
    digest=hashlib.sha256(asset.astype("<i8").tobytes()+rows.astype("<i8").tobytes()).hexdigest()
    return {"mode":packet["mode"],"point_count":n,"source_domain_counts":counts,
            "source_point_map_sha256":digest,"all_emitted_coordinates_source_exact":True}

def validate_storage_compaction(packet,deep_path,points):
    """Verify a compact sibling that removes only authenticated unused storage rows."""
    expected_parent="5fb9e4265091af0583d443a7ac4768d777016a5aedd0d3a628d686040b2ea8ae"
    cert_path=Path(packet.get("certificate_path",""));map_path=Path(packet.get("row_map_path",""))
    if (not cert_path.is_file() or sha(cert_path)!=packet.get("certificate_sha256") or
        not map_path.is_file() or sha(map_path)!=packet.get("row_map_sha256")):
        raise ValueError("storage compaction certificate or row map hash mismatch")
    cert=json.loads(cert_path.read_text())
    source_path=Path(packet.get("source_candidate_path",""))
    if (cert.get("schema")!="numi.D.unused-output-storage-row-compaction.v1" or
        cert.get("status")!="PASS_STORAGE_COMPACTION_GEOMETRY_EQUIVALENCE_ONLY" or
        cert.get("physical_admission") is not False or cert.get("runtime")!=0 or
        cert.get("geometry_predicates_rerun") is not False or
        packet.get("source_candidate_sha256")!=expected_parent or
        cert.get("source_candidate_sha256")!=expected_parent or sha(source_path)!=expected_parent):
        raise ValueError("storage compaction is not bound to the authenticated prior candidate")
    compact_path=Path(deep_path)
    if (cert.get("compact_candidate_path")!=str(compact_path) or
        cert.get("compact_candidate_sha256")!=sha(compact_path) or
        cert.get("compact_candidate_sha256")!=packet.get("compact_candidate_sha256")):
        raise ValueError("storage compaction candidate hash/path mismatch")
    if cert.get("retained_point_coordinate_bits_identical") is not True or cert.get("ordered_face_coordinate_bits_identical") is not True or cert.get("face_order_and_orientation_preserved") is not True:
        raise ValueError("storage compaction certificate lacks exact geometry-equivalence claims")
    dropped=np.asarray(cert.get("unreferenced_source_output_vertex_rows",[]),dtype=np.int64)
    if not np.array_equal(dropped,np.asarray([72893,82725],dtype=np.int64)):
        raise ValueError("storage compaction removed-row inventory mismatch")
    with np.load(source_path,allow_pickle=False) as z:
        old={k:np.array(z[k],copy=True) for k in z.files}
    with np.load(compact_path,allow_pickle=False) as z:
        new={k:np.array(z[k],copy=True) for k in z.files}
    if set(old)!=set(new): raise ValueError("storage compaction NPZ member inventory changed")
    old_points=np.asarray(old["points_world_m"],np.float32);old_faces=np.asarray(old["faces"],np.int64)
    n=len(old_points)
    if not np.array_equal(np.setdiff1d(np.arange(n),np.unique(old_faces)),dropped):
        raise ValueError("declared removed rows are not exactly the unused source rows")
    with np.load(map_path,allow_pickle=False) as z:
        if set(z.files)!={"old_to_compact_vertex_row","compact_to_old_vertex_row"}:
            raise ValueError("storage compaction row-map members malformed")
        old_to_new=np.asarray(z["old_to_compact_vertex_row"],np.int64)
        new_to_old=np.asarray(z["compact_to_old_vertex_row"],np.int64)
    keep=np.setdiff1d(np.arange(n,dtype=np.int64),dropped)
    expected_map=np.full(n,-1,dtype=np.int64);expected_map[keep]=np.arange(len(keep),dtype=np.int64)
    if (not np.array_equal(old_to_new,expected_map) or not np.array_equal(new_to_old,keep) or
        not np.array_equal(old_to_new[new_to_old],np.arange(len(keep),dtype=np.int64))):
        raise ValueError("storage compaction old/new row map is stale or malformed")
    changed=set(cert.get("changed_candidate_members",[]))
    expected_changed={"points_world_m","faces","output_vertex_input_rows","output_vertex_input_parent_offsets","output_vertex_input_parent_rows","topology_patch_new_faces","topology_patch_boundary_loop_vertex_rows","operational_inserted_vertex_lineage_json","operational_reference_repair_json","operational_emission_json"}
    if changed!=expected_changed: raise ValueError("storage compaction changed-member inventory mismatch")
    if not np.array_equal(np.asarray(new["points_world_m"],np.float32).view(np.uint32),old_points[keep].view(np.uint32)):
        raise ValueError("storage compaction altered a retained point coordinate")
    mapped_faces=old_to_new[old_faces]
    if np.any(mapped_faces<0) or not np.array_equal(new["faces"],mapped_faces):
        raise ValueError("storage compaction face remap mismatch")
    if not np.array_equal(new["points_world_m"][new["faces"]].view(np.uint32),old_points[old_faces].view(np.uint32)):
        raise ValueError("storage compaction changed an ordered face coordinate")
    old_offsets=np.asarray(old["output_vertex_input_parent_offsets"],np.int64)
    old_rows=np.asarray(old["output_vertex_input_parent_rows"])
    lengths=np.diff(old_offsets)[keep]
    expected_offsets=np.empty(len(keep)+1,dtype=old_offsets.dtype);expected_offsets[0]=0
    np.cumsum(lengths,out=expected_offsets[1:])
    expected_parent_rows=np.concatenate([old_rows[old_offsets[r]:old_offsets[r+1]] for r in keep])
    if (not np.array_equal(new["output_vertex_input_rows"],old["output_vertex_input_rows"][keep]) or
        not np.array_equal(new["output_vertex_input_parent_offsets"],expected_offsets) or
        not np.array_equal(new["output_vertex_input_parent_rows"],expected_parent_rows)):
        raise ValueError("storage compaction changed the existing output-vertex source map")
    if not np.array_equal(new["topology_patch_new_faces"],old_to_new[old["topology_patch_new_faces"]]) or not np.array_equal(new["topology_patch_boundary_loop_vertex_rows"],old_to_new[old["topology_patch_boundary_loop_vertex_rows"]]):
        raise ValueError("storage compaction topology-patch vertex rows were not remapped")

    def remap_tree(obj,keys):
        if isinstance(obj,dict):
            out={}
            for k,v in obj.items():
                if k in keys:
                    def one(x):
                        if isinstance(x,list): return [one(y) for y in x]
                        if isinstance(x,int) and not isinstance(x,bool):
                            if x<0:return x
                            if x>=n or old_to_new[x]<0: raise ValueError("current JSON point reference targets removed row")
                            return int(old_to_new[x])
                        return x
                    out[k]=one(v)
                else: out[k]=remap_tree(v,keys)
            return out
        if isinstance(obj,list): return [remap_tree(x,keys) for x in obj]
        return obj
    for key,fields in (("operational_inserted_vertex_lineage_json",{"output_vertex_row","parent_candidate_face_vertices"}),
                       ("operational_reference_repair_json",{"operational_reference_vertex_row","new_vertex_row","parent_face_vertices","child_faces"})):
        expected=remap_tree(json.loads(str(old[key].item())),fields)
        if json.loads(str(new[key].item()))!=expected: raise ValueError("storage compaction JSON vertex references are stale: "+key)
    expected_emission=remap_tree(json.loads(str(old["operational_emission_json"].item())),{"new_vertex_rows"})
    expected_emission["output_points"]=len(keep)
    expected_emission["storage_unreferenced_rows_compacted"]=len(dropped)
    if json.loads(str(new["operational_emission_json"].item()))!=expected_emission:
        raise ValueError("storage compaction emission row metadata is stale")
    for key in old:
        if key not in changed:
            if (old[key].dtype.str!=new[key].dtype.str or old[key].shape!=new[key].shape or
                old[key].tobytes()!=new[key].tobytes()):
                raise ValueError("storage compaction changed an unrelated NPZ member: "+key)
    for key,digest in cert.get("per_face_arrays_sha256",{}).items():
        if key not in old or old[key].tobytes()!=new[key].tobytes() or hashlib.sha256(old[key].tobytes()).hexdigest()!=digest:
            raise ValueError("storage compaction changed face ancestry/identity: "+key)
    used=np.unique(new["faces"])
    if len(used)!=len(new["points_world_m"]) or cert.get("unused_compact_rows")!=[]:
        raise ValueError("storage compaction left unused output rows")
    return {"mode":packet["mode"],"point_count":len(keep),"source_candidate_sha256":expected_parent,
            "removed_unreferenced_source_rows":dropped.astype(int).tolist(),"all_retained_point_bits_identical":True,
            "all_ordered_face_coordinate_bits_identical":True,"existing_output_vertex_map_preserved":True,
            "compact_candidate_sha256":cert["compact_candidate_sha256"],"row_map_sha256":sha(map_path),
            "geometry_predicates_rerun":False}

def validate(proof_path,deep_path,surface,rows):
    proof=json.loads(Path(proof_path).read_text())
    if (proof.get("schema")!=SCHEMA or
        proof.get("status")!="PASS_LOCAL_STRUCTURAL_AND_CHANGED_FACE_EXACT_GATES_SOURCE_ONLY" or
        proof.get("physical_admission") is not False or proof.get("runtime")!=0 or
        proof.get("accepted_steps")!=0 or proof.get("GPU") is not False):
        raise ValueError("operational D proof is not a source-only local-gate PASS")
    deep_sha=sha(deep_path)
    if proof.get("candidate_sha256")!=deep_sha: raise ValueError("operational D candidate hash mismatch")
    history_path=Path(proof["preserved_historical_protected149_path"])
    if (proof.get("preserved_historical_protected149_sha256")!=HISTORICAL_SHA or
        sha(history_path)!=HISTORICAL_SHA): raise ValueError("immutable historical149 changed")
    history=csv_rows(history_path)
    current_path=Path(proof["operational_protected149_path"])
    if sha(current_path)!=proof["operational_protected149_sha256"] or rows!=csv_rows(current_path):
        raise ValueError("current149 rows do not match the pinned operational CSV")
    if (len(history)!=149 or len(rows)!=149 or
        {int(r["canonical_protected_row"]) for r in rows}!=set(range(149))):
        raise ValueError("canonical inventory is not149 unique rows")
    historical_by_row={int(r["canonical_protected_row"]):r for r in history}
    if set(historical_by_row)!=set(range(149)): raise ValueError("historical149 row IDs incomplete")
    # Historical fields stay intact. All changed reference data lives in the
    # explicit operational_reference_* columns of the existing CSV.
    for row in rows:
        old=historical_by_row[int(row["canonical_protected_row"])]
        if any(row.get(k)!=v for k,v in old.items()):
            raise ValueError("operational packet rewrote a historical canonical field")
    map_path=Path(proof["construction_map_path"])
    if sha(map_path)!=proof["construction_map_sha256"]: raise ValueError("operational construction map changed")
    m=json.loads(map_path.read_text())
    if (m.get("candidate_sha256")!=deep_sha or m.get("physical_admission") is not False or
        m.get("runtime")!=0 or m.get("outside_source_geometry_changed") is not False):
        raise ValueError("operational map candidate/source ownership mismatch")
    gate_path=Path(proof["changed_face_exact_gate_path"])
    if sha(gate_path)!=proof["changed_face_exact_gate_sha256"]:
        raise ValueError("operational changed-face exact receipt changed")
    gates=json.loads(gate_path.read_text())
    paired_inner=select_paired_inner_surface(proof,gates)
    paired_parent_report=None
    if paired_inner["sha256"] is not None:
        paired_parent_report=validate_current_paired_parent_report(proof,gates,deep_sha,paired_inner)
    expected_counts={"changed-D-self":0,paired_inner["gate_name"]:0,"changed-D-vs-all185":0}
    if proof.get("changed_face_gate_counts")!=expected_counts:
        raise ValueError("operational changed-face exact gates did not all pass")
    expected_names=set(expected_counts)
    if (gates.get("status")!="PASS_CHANGED_FACES_SELF_I_ALL185" or
        gates.get("candidate_sha256")!=deep_sha or
        gates.get("target_map_sha256")!=sha(map_path) or
        gates.get("physical_admission") is not False or gates.get("runtime")!=0):
        raise ValueError("operational changed-face receipt ownership/status mismatch")
    entries=gates.get("gates",[])
    if len(entries)!=3 or {g.get("name") for g in entries}!=expected_names:
        raise ValueError("operational changed-face gate inventory incomplete")
    if paired_inner["sha256"] is not None:
        inner_entries=[g for g in entries if g.get("name")=="changed-D-vs-current-I"]
        expected_pair={"path":paired_inner["path"],"sha256":paired_inner["sha256"]}
        if len(inner_entries)!=1 or inner_entries[0].get("paired_inner_surface")!=expected_pair:
            raise ValueError("current-I exact gate does not bind the authenticated paired surface")
    for entry in entries:
        if (entry.get("count")!=0 or entry.get("candidate_sha256")!=deep_sha or
            entry.get("changed_face_rows")!=gates.get("changed_face_rows") or
            sha(entry["pairs_path"])!=entry["pairs_sha256"]):
            raise ValueError("operational exact gate pair/count evidence mismatch")
    if gates.get("changed_face_rows")!=m.get("modified_output_face_rows"):
        raise ValueError("operational exact gate does not cover the construction changes")
    # These are changed-face gates, never a full-volume or physical admission.
    points=np.asarray(surface["points"],dtype=np.float32)
    faces=np.asarray(surface["faces"],dtype=np.int64)
    used=set(map(int,np.unique(faces)))
    old_candidate=Path(proof["input_candidate_path"])
    if sha(old_candidate)!=proof["input_candidate_sha256"]: raise ValueError("operational parent D changed")
    with np.load(old_candidate,allow_pickle=False) as z:
        old_points=np.asarray(z["points_world_m"],dtype=np.float32)
    point_packet=proof.get("point_preservation")
    if point_packet is None:
        if len(points)<len(old_points) or not np.array_equal(points[:len(old_points)],old_points):
            raise ValueError("operational D changed an existing point coordinate")
        point_preservation={"mode":"unchanged-input-point-prefix"}
    else:
        point_preservation=validate_point_preservation(point_packet,deep_path,points)
    hashes=m["source330_file_hashes"]
    for path,digest in hashes.items():
        if sha(path)!=digest: raise ValueError("immutable source330 table changed")
    base=Path(m["source330_mass_path"])
    node_path=Path(str(base)+".nodes.csv")
    weight_path=Path(str(base)+".weights.csv")
    tet_path=Path(str(base)+".tets.csv")
    label_path=Path(str(base)+".labels.csv")
    if any(str(p) not in hashes for p in [node_path,weight_path,tet_path,label_path]):
        raise ValueError("operational source330 table hash set is incomplete")
    nodes,_=_array_from_csv(node_path,"parent_node_index",["current_x_m","current_y_m","current_z_m"])
    weights,_=_array_from_csv(weight_path,"vertex",["w"+str(i) for i in range(86)])
    _,tet_rows=_array_from_csv(tet_path,"parent_tet_row",["parent_node0","parent_node1","parent_node2","parent_node3"])
    _,labels=_array_from_csv(label_path,"parent_tet_row",["material_index","geometric_donor_body","anatomical_region_id"])
    tet_by_id={int(t["parent_tet_row"]):t for t in tet_rows}
    label_by_id={int(t["parent_tet_row"]):t for t in labels}
    attachments=[r for r in rows if r["mapping_kind"]!="head_ring_boundary_target"]
    if len(attachments)!=14: raise ValueError("canonical D attachment count changed")
    routes={};ids=set();vertices=set()
    for row in attachments:
        canonical=int(row["canonical_protected_row"]);stable=int(row["stable_attachment_id"])
        donor=int(row["canonical_donor_body"]);lane=int(row["full86_confidence_lane"])
        if not 0<=lane<86 or int(row["full86_confidence_body"])!=donor:
            raise ValueError("canonical donor/lane identity mismatch")
        v=int(m["canonical14_current_rows"][str(stable)])
        if (v not in used or v in vertices or stable in ids or
            int(row["operational_reference_surface_vertex_row"])!=v or
            int(proof["canonical_attachment_rows"][str(stable)])!=v):
            raise ValueError("current canonical D row missing/duplicated/inconsistent")
        ids.add(stable);vertices.add(v)
        xyz=np.asarray([float(row["operational_reference_world_"+a+"_m"]) for a in "xyz"],np.float32)
        if row["operational_reference_candidate_sha256"]!=deep_sha or f32key(points[v])!=f32key(xyz):
            raise ValueError("canonical operational point does not match current D Float32 row")
        historical=np.asarray([float(row["target_world_"+a+"_m"]) for a in "xyz"],np.float32)
        displacement=float(np.linalg.norm(xyz.astype(float)-historical.astype(float)))
        if displacement>0.010: raise ValueError("operational target exceeds10mm declared construction bound")
        incident=np.flatnonzero(np.any(faces==v,axis=1))
        tri=points[faces[incident]].astype(float)
        twice_area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        edges=np.linalg.norm(np.roll(tri,-1,axis=1)-tri,axis=2)
        minimum_altitude=float(np.min(twice_area[:,None]/np.maximum(edges,1e-300)))
        if minimum_altitude<50e-6: raise ValueError("canonical surface star fails50um quality gate")
        r=m["canonical14_source_registration"][str(stable)]
        parent=int(r["source_parent_tet_row"])
        tet=tet_by_id[parent];label=label_by_id[parent]
        corners=np.asarray([int(tet["parent_node"+str(i)]) for i in range(4)])
        bary=np.asarray(r["source_registration_barycentric"],float)
        if (bary.shape!=(4,) or not np.isfinite(bary).all() or bary.min()<-1e-12 or
            bary.max()>1+1e-12 or abs(bary.sum()-1)>1e-10 or
            int(row["source330_parent_node_row"]) not in corners):
            raise ValueError("canonical registration lacks valid immutable parent-star ancestry")
        if corners.tolist()!=r["source_parent_node_rows"] or int(label["material_index"])!=16:
            raise ValueError("canonical source-parent identity/material mismatch")
        if int(label["geometric_donor_body"])!=int(r["source_tet_geometric_donor_body"]) or int(label["anatomical_region_id"])!=int(r["source_tet_anatomical_region_id"]):
            raise ValueError("canonical source-parent semantic labels mismatch")
        support=bary@nodes[corners]
        residual=float(np.linalg.norm(support-xyz.astype(float)))
        full86=bary@weights[corners]
        weight=float(full86[lane])
        if (not np.isfinite(full86).all() or full86.min()<-1e-8 or
            abs(full86.sum()-1)>2e-5 or residual>0.0004 or weight<0.85 or
            abs(weight-float(r["recomputed_full86_confidence_weight"]))>1e-10 or
            abs(residual-float(r["source_registration_residual_m"]))>1e-10):
            raise ValueError("canonical source registration/full86 recomputation failed")
        routes[canonical]={"surface_vertex_row":v,"stable_attachment_id":stable,
            "base_volume_node_row":int(row["base_volume_node_row"]),
            "source330_parent_node_row":int(row["source330_parent_node_row"]),
            "target_world_m_f32":xyz.tolist(),"historical_target_world_m_f32":historical.tolist(),
            "reference_displacement_m":displacement,"minimum_surface_altitude_m":minimum_altitude,
            "recomputed_full86_confidence_weight":weight,"source_registration_residual_m":residual,
            "proof_incident_face_count":len(incident),"proof_candidate":str(deep_path),
            "proof_candidate_sha256":deep_sha,"historical_mapping_kind":row["mapping_kind"]}
    heads=[r for r in rows if r["mapping_kind"]=="head_ring_boundary_target" and r["role"]=="D"]
    if len(heads)!=45: raise ValueError("D canonical head inventory changed")
    for row in heads:
        c=str(int(row["canonical_protected_row"]));v=int(m["D_head45_current_rows"][c]["candidate_vertex_row"])
        target=[float(row["target_world_"+a+"_m"])for a in "xyz"]
        if v not in used or f32key(points[v])!=f32key(target): raise ValueError("canonical D head coordinate changed")
    return {"path":str(proof_path),"sha256":sha(proof_path),"candidate":str(deep_path),
            "candidate_sha256":deep_sha,"attachment_routes":routes,
            "source_kind":"canonical-D-operational-reference-proof",
            "historical_protected149_sha256":HISTORICAL_SHA,
            "preserves_immutable_historical_rows":True,
            "point_preservation":point_preservation,
            "paired_inner_surface":paired_inner,
            "paired_inner_sha256":paired_inner["sha256"],
            "paired_parent_report":paired_parent_report,
            "physical_admission":False,"runtime":0}
