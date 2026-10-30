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

def validate_point_preservation(packet,deep_path,points):
    """Independently check every reduced point against its exact source row."""
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
    if proof.get("changed_face_gate_counts")!={"changed-D-self":0,"changed-D-vs-I014":0,"changed-D-vs-all185":0}:
        raise ValueError("operational changed-face exact gates did not all pass")
    gates=json.loads(gate_path.read_text())
    expected_names={"changed-D-self","changed-D-vs-I014","changed-D-vs-all185"}
    if (gates.get("status")!="PASS_CHANGED_FACES_SELF_I_ALL185" or
        gates.get("candidate_sha256")!=deep_sha or
        gates.get("target_map_sha256")!=sha(map_path) or
        gates.get("physical_admission") is not False or gates.get("runtime")!=0):
        raise ValueError("operational changed-face receipt ownership/status mismatch")
    entries=gates.get("gates",[])
    if len(entries)!=3 or {g.get("name") for g in entries}!=expected_names:
        raise ValueError("operational changed-face gate inventory incomplete")
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
            "physical_admission":False,"runtime":0}
