import copy
import argparse
import json
import shutil
import os
import csv
import importlib.util
from pathlib import Path
import tempfile
import numpy as np

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("producer_d",P/"export_refined_coarse_registered_volume_433.py")
producer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)
ROOT=None

SOURCE_ROLES={0:("D",0),1:("D",1),2:("D",2),3:("D",3),4:("D",4),5:("O",0)}
SOURCE_LABELS=[{"material_index":16,"geometric_donor_body":23,"anatomical_region_id":23,"stable_source_face_id":10}]
PARENT_TETS=np.asarray([[0,1,2,3]],dtype=np.int64)
SOURCE_FACES=[
 {"parent_d_face_row":0,"node_rows":(0,1,2),"parent_tet_row":0,"material_index":16,"geometric_donor_body":23,"anatomical_region_id":23,"stable_source_face_id":10},
 {"parent_d_face_row":1,"node_rows":(1,2,3),"parent_tet_row":0,"material_index":16,"geometric_donor_body":23,"anatomical_region_id":23,"stable_source_face_id":10},
 {"parent_d_face_row":2,"node_rows":(0,2,3),"parent_tet_row":0,"material_index":16,"geometric_donor_body":23,"anatomical_region_id":23,"stable_source_face_id":10},
]
PARENT_NODE_BY_CHILD=[1,2,3,4,5,-1]
OUTPUT_D={0:0,1:1,2:2,3:3,5:4}
CHILD_FACES=[
 {"node_rows":(0,1,5),"child_tet_row":0,"parent_tet_row":0,"material_index":16,"geometric_donor_body":23,"anatomical_region_id":23,"stable_source_face_id":99},
 {"node_rows":(0,3,4),"child_tet_row":0,"parent_tet_row":0,"material_index":16,"geometric_donor_body":23,"anatomical_region_id":23,"stable_source_face_id":99},
 {"node_rows":(0,1,2),"child_tet_row":0,"parent_tet_row":0,"material_index":16,"geometric_donor_body":23,"anatomical_region_id":23,"stable_source_face_id":10},
]
D_DOC={
 "schema":"numi-human.explicit-d-boundary-retirement.v1",
 "retired_parent_d_nodes":[{"volume_node_row":0,"parent_d_role_row":0}],
 "replaced_parent_d_faces":[
  {"parent_d_face_row":0,"source_stable_source_face_id":10,"source_material_index":16,"source_geometric_donor_body":23,"source_anatomical_region_id":23,"replacement_child_node_rows":[[0,1,5]],"replacement_boolean_face_rows":[[0]]},
  {"parent_d_face_row":2,"source_stable_source_face_id":10,"source_material_index":16,"source_geometric_donor_body":23,"source_anatomical_region_id":23,"replacement_child_node_rows":[[0,3,4]],"replacement_boolean_face_rows":[[1]]},
 ]
}


def make_boolean_fixture(tmp):
    tmp=Path(tmp)
    boolean=tmp/"boolean.npz";shell=tmp/"shell.npz";bank=tmp/"bank.npz";baseline=tmp/"baseline.npz";lineage=tmp/"boolean-lineage.json"
    points=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=np.float64)
    faces=np.asarray([[0,1,2],[0,2,3]],dtype=np.int64)
    np.savez(boolean,points_world_m_f64=points,faces_f64=faces,run_index=np.asarray([0,6],np.uint64),run_original_id=np.asarray([100],np.uint32),face_id=np.asarray([0,1],np.uint64))
    candidate_points=np.asarray([[0.,0.,0.],[1.,0.,0.],[.25,.25,.25],[0.,1.,0.],[0.,0.,1.],[0.,1.,0.]],dtype=np.float64)
    np.savez(shell,points_world_m_f64=points,faces_f64=faces,source_tet_rows=np.asarray([0,0],np.int64),source_face_global_nodes=np.asarray([[0,1,2],[0,2,3]],np.int64),source_face_kind=np.asarray(["existing_D_material_boundary","existing_D_material_boundary"]))
    np.savez(bank,points_world_f32=points.astype(np.float32),faces=np.asarray([[0,1,2]],np.int64),bank_stable_face_row=np.asarray([77],np.int64))
    np.savez(baseline,child_parent_tet_row=np.asarray([0],np.int64),parent_volume_node_row=np.arange(6,dtype=np.int64))
    pins={"source_shell_npz_sha256":producer.digest(shell),"bank_input_npz_sha256":producer.digest(bank),"frozen_current006_baseline_sha256":producer.digest(baseline)}
    doc={
     "schema":"numi-human.d-boolean-output-face-lineage.v1",
     "pins":pins,
     "run_sources":[{"run_original_id":100,"source_domain":"source_shell","source_mesh_sha256":pins["source_shell_npz_sha256"],"input_face_count":2}],
     "boolean_faces":[
      {"boolean_face_row":0,"run_original_id":100,"input_face_row":0,"parent_d_face_rows":[0]},
      {"boolean_face_row":1,"run_original_id":100,"input_face_row":1,"parent_d_face_rows":[2]},
     ],
     "child_face_lineage":[
      {"child_node_rows":[0,1,5],"boolean_face_rows":[0],"parent_d_face_rows":[0]},
      {"child_node_rows":[0,3,4],"boolean_face_rows":[1],"parent_d_face_rows":[2]},
     ]
    }
    lineage.write_text(producer.json.dumps(doc,sort_keys=True))
    return boolean,shell,bank,baseline,lineage,doc,candidate_points


def validate_nodes(doc=D_DOC,roles=SOURCE_ROLES,parent_nodes=PARENT_NODE_BY_CHILD,output_d=OUTPUT_D,protected={2}):
    return producer.validate_d_boundary_retirement_nodes(doc,roles,parent_nodes,output_d,protected)


def validate_faces(tmp,doc=D_DOC,source_faces=SOURCE_FACES,child_faces=CHILD_FACES,labels=SOURCE_LABELS,components=[0],lineage_doc=None):
    boolean,shell,bank,baseline,lineage_file,lineage_doc0,candidate_points=make_boolean_fixture(tmp)
    if lineage_doc is None:lineage_doc=lineage_doc0
    pins=lineage_doc["pins"]
    child_lineage,_=producer.validate_d_boolean_face_lineage(lineage_doc,boolean,shell,bank,baseline,pins,source_faces,labels,components,candidate_points)
    return producer.validate_d_boundary_face_retirement(doc,source_faces,child_faces,PARENT_NODE_BY_CHILD,SOURCE_ROLES,labels,components,child_lineage)


def expect_error(call,message):
    try:call()
    except ValueError as exc:assert message in str(exc),(message,str(exc))
    else:raise AssertionError("expected ValueError containing: "+message)


def test_new_d_node_source_residual_is_bounded_only_for_explicit_d_path():
    points=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(.25,.25,.2501)]
    parent_points=points[:4]
    parent_tets=np.asarray([[0,1,2,3]],np.int64)
    parent_rows=np.asarray([0,1,2,3,-1],np.int64)
    selected=[(0,(1.,0.,0.,0.)),(0,(0.,1.,0.,0.)),
              (0,(0.,0.,1.,0.)),(0,(0.,0.,0.,1.)),
              (0,(.25,.25,.25,.25))]
    roles={0:("D",0),1:("D",1),2:("D",2),3:("D",3),4:("D",4)}
    result=producer.validate_d_new_node_source_residuals(
        points,parent_points,parent_tets,parent_rows,selected,roles)
    assert result["new_d_boundary_node_count"]==1
    assert result["new_d_boundary_node_maximum_source_residual_m"] < 4.0e-4
    # An unchanged original node is not treated as a newly constructed D node.
    assert producer.validate_d_new_node_source_residuals(
        points,parent_points,parent_tets,np.asarray([0,1,2,3,4]),selected,roles
    )["new_d_boundary_node_count"]==0
    too_far=list(points);too_far[4]=(.25,.25,.2509)
    expect_error(lambda:producer.validate_d_new_node_source_residuals(
        too_far,parent_points,parent_tets,parent_rows,selected,roles),
        "exceeds the 0.4 mm original-source registration bound")


def test_source_protected_set_is_149():
    capture=ROOT/"native-physical-parent-mass-330"/"native-source-identity-control-002"
    def read(name):
        with (capture/name).open(newline="") as f:return list(csv.DictReader(f))
    tets=read("compiled-parent-cell-mass.csv.tets.csv");labels=read("compiled-parent-cell-mass.csv.labels.csv")
    roles=read("parent-layer-rows.csv");attachments=read("parent-deep-attachments.csv")
    parent_tets=np.asarray([[int(r["parent_node"+str(k)]) for k in range(4)] for r in tets],np.int64)
    role_by_node={int(r["volume_node_row"]):(r["role"],int(r["role_row"])) for r in roles}
    protected,summary=producer.derive_protected_parent_nodes(parent_tets,labels,role_by_node,attachments,producer.digest(capture/"parent-deep-attachments.csv"))
    assert len(protected)==149
    assert summary["head_boundary_nodes_by_role"]=={"O":45,"I":45,"D":45}
    assert summary["deep_attachment_nodes"]==14
    frozen006=ROOT/"reduced-physical-boundary-400"/"attempt-030-connected-constrained-edge-fit"/"run-006"/"candidate.npz"
    assert producer.digest(frozen006)=="888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63"
    with np.load(frozen006,allow_pickle=False) as z:
        pinned=np.asarray(z["parent_volume_node_row"],np.int64)
    rows=np.asarray(sorted(protected),np.int64)
    assert pinned.shape[0]>=7233 and np.array_equal(pinned[rows],rows)


def test_actual_saved_csg_tags_match_shell_and_native_bank_inputs():
    cont=ROOT/"reduced-physical-boundary-400"/"attempt-042-current006-d23-bone32-cavity"/"attempt-005-bone32-subtraction"/"continuation-005"
    csg=cont/"cavity-minus-bank32-f64-tagged.npz"
    shell=cont/"original-fat-cavity-shell.npz"
    replay=ROOT/"geometry-worker"/"matched-nesting-390"/"run-044-native-f32-replay-185"/"all185-source-order-native-f32-replay.npz"
    assert producer.digest(csg)=="06c52fa4a6c3797f04fc69db0b9aef73beb410fed10bb952e23cda07b52e0513"
    with np.load(csg,allow_pickle=False) as z:
        face_id=np.asarray(z["face_id"],np.int64)
        run_id=np.asarray(z["run_original_id"],np.int64)
        run_index=np.asarray(z["run_index"],np.int64)
        expanded=np.repeat(run_id,np.diff(run_index)//3)
        assert expanded.shape==face_id.shape==(5044,)
        shell_rows=face_id[expanded==62000502]
        bank_rows=face_id[expanded==80000048]
        assert len(shell_rows)==998 and shell_rows.min()==0 and shell_rows.max()==375
        assert len(bank_rows)==4046 and bank_rows.min()>=0 and bank_rows.max()<14510
        saved_stable=np.asarray(z["bank_source_stable_face_rows"],np.int64)
        saved_record=np.asarray(z["bank_source_record_indices"],np.int64)
    with np.load(shell,allow_pickle=False) as z:
        assert len(z["faces_f64"])==376
        assert np.asarray(z["source_tet_rows"]).shape==(376,)
        assert np.asarray(z["source_face_global_nodes"]).shape==(376,3)
    with np.load(replay,allow_pickle=False) as z:
        matches=np.flatnonzero(np.asarray(z["bank_source_id"])==80000048)
        assert len(matches)==1
        bi=int(matches[0]);fs,fe=map(int,z["bank_face_offsets"][bi:bi+2])
        assert fe-fs==14510
        assert np.array_equal(saved_stable,np.asarray(z["face_source_stable_row"][fs:fe],np.int64))
        assert np.array_equal(saved_record,np.asarray(z["face_source_record_index"][fs:fe],np.int64))


def test_actual_boolean_output_tags_are_geometrically_supported():
    cont=ROOT/"reduced-physical-boundary-400"/"attempt-042-current006-d23-bone32-cavity"/"attempt-005-bone32-subtraction"/"continuation-005"
    csg=cont/"cavity-minus-bank32-f64-tagged.npz"
    shell=cont/"original-fat-cavity-shell.npz"
    replay=ROOT/"geometry-worker"/"matched-nesting-390"/"run-044-native-f32-replay-185"/"all185-source-order-native-f32-replay.npz"
    with np.load(csg,allow_pickle=False) as z:
        points=np.asarray(z["points_world_m_f64"],np.float64)
        faces=np.asarray(z["faces_f64"],np.int64)
        tags=np.asarray(z["face_id"],np.int64)
        run_ids=np.asarray(z["run_original_id"],np.int64)
        run_offsets=np.asarray(z["run_index"],np.int64)
    face_runs=np.repeat(run_ids,np.diff(run_offsets)//3)
    with np.load(shell,allow_pickle=False) as z:
        shell_points=np.asarray(z["points_world_m_f64"],np.float64)
        shell_faces=np.asarray(z["faces_f64"],np.int64)
    with np.load(replay,allow_pickle=False) as z:
        bank={k:np.asarray(z[k]) for k in z.files}
    bank_id=80000048
    matches=np.flatnonzero(bank["bank_source_id"]==bank_id);assert len(matches)==1
    bi=int(matches[0]);bs,be=map(int,bank["bank_face_offsets"][bi:bi+2])
    tag_rows=np.asarray(bank["face_source_operand_face_row"][bs:be],np.int64)
    assert len(np.unique(tag_rows))==len(tag_rows)
    tag_to_local={int(tag):i for i,tag in enumerate(tag_rows)}
    bank_points=np.asarray(bank["points_world_f32"],np.float64)
    bank_faces=np.asarray(bank["faces"],np.int64)
    max_plane=0.0;counts={"source_shell":0,"bone_bank":0}
    for row,(run,tag) in enumerate(zip(face_runs,tags)):
        if int(run)==62000502:
            if int(tag)<0 or int(tag)>=len(shell_faces):raise AssertionError("shell face tag outside source shell")
            inp_points=shell_points;inp_tri=shell_faces[int(tag)];domain="source_shell"
        elif int(run)==bank_id:
            local=tag_to_local.get(int(tag))
            if local is None:raise AssertionError("bank output face tag absent from exact replay operand map")
            inp_points=bank_points;inp_tri=bank_faces[bs+local];domain="bone_bank"
        else:raise AssertionError(f"unexpected Boolean run ID {run}")
        receipt=producer.validate_tagged_output_triangle_geometry(row,points[faces[row]],inp_points,inp_tri)
        max_plane=max(max_plane,receipt["maximum_plane_residual_m"]);counts[domain]+=1
    assert counts=={"source_shell":998,"bone_bank":4046}
    assert max_plane<1.0e-14
    # An unrelated, parallel triangle is rejected even when labels/tags are otherwise valid.
    with np.load(shell,allow_pickle=False) as z:
        tri=np.asarray(z["faces_f64"][0],np.int64)
        input_points=np.asarray(z["points_world_m_f64"],np.float64)
    out=input_points[tri].copy();out[:,2]+=1.0e-6
    expect_error(lambda:producer.validate_tagged_output_triangle_geometry(999,out,input_points,tri),
                 "not geometrically supported")

def test_positive_retirement_and_face_lineage():
    retired=validate_nodes()
    assert retired=={0}
    with tempfile.TemporaryDirectory() as d:
        summary=validate_faces(d)
    assert summary["retired_parent_d_node_count"]==1
    assert summary["retired_parent_d_face_count"]==2
    assert summary["replacement_d_face_count"]==2
    assert summary["actual_boolean_face_ancestry_proof"] is True


def test_d_node_retirement_is_explicit_and_role_limited():
    doc=copy.deepcopy(D_DOC);doc["retired_parent_d_nodes"].append(dict(doc["retired_parent_d_nodes"][0]))
    expect_error(lambda:validate_nodes(doc=doc),"repeats a parent D source node")
    doc=copy.deepcopy(D_DOC);doc["retired_parent_d_nodes"][0]["parent_d_role_row"]=4
    expect_error(lambda:validate_nodes(doc=doc),"exact source D row")
    doc=copy.deepcopy(D_DOC);doc["retired_parent_d_nodes"][0]["volume_node_row"]=5;doc["retired_parent_d_nodes"][0]["parent_d_role_row"]=0
    expect_error(lambda:validate_nodes(doc=doc),"non-D source node")
    expect_error(lambda:validate_nodes(protected={0}),"protected head/deep attachment")
    expect_error(lambda:validate_nodes(parent_nodes=[1,2,3,4,-1,-1]),"dropped a non-retired original source node")
    doc=copy.deepcopy(D_DOC);doc["retired_parent_d_nodes"].append({"volume_node_row":1,"parent_d_role_row":1})
    expect_error(lambda:validate_nodes(doc=doc),"still retained in the volume")


def test_face_coverage_labels_and_ancestry_fail_closed():
    doc=copy.deepcopy(D_DOC);doc["replaced_parent_d_faces"].pop()
    with tempfile.TemporaryDirectory() as d:expect_error(lambda:validate_faces(d,doc=doc),"does not cover every changed Boolean/retired-node")
    doc=copy.deepcopy(D_DOC);doc["replaced_parent_d_faces"][0]["source_stable_source_face_id"]=999
    with tempfile.TemporaryDirectory() as d:expect_error(lambda:validate_faces(d,doc=doc),"stable-face history")
    doc=copy.deepcopy(D_DOC);doc["replaced_parent_d_faces"][0]["source_geometric_donor_body"]=999
    with tempfile.TemporaryDirectory() as d:expect_error(lambda:validate_faces(d,doc=doc),"material/donor/region")
    doc=copy.deepcopy(D_DOC);doc["replaced_parent_d_faces"].append(copy.deepcopy(doc["replaced_parent_d_faces"][0]))
    with tempfile.TemporaryDirectory() as d:expect_error(lambda:validate_faces(d,doc=doc),"mapped more than once")
    extra={"node_rows":(1,3,5),"child_tet_row":1,"parent_tet_row":0,"material_index":16,"geometric_donor_body":23,"anatomical_region_id":23,"stable_source_face_id":99}
    with tempfile.TemporaryDirectory() as d:expect_error(lambda:validate_faces(d,child_faces=CHILD_FACES+[extra]),"does not exactly cover changed D boundary")
    with tempfile.TemporaryDirectory() as d:
        boolean,shell,bank,baseline,lineage_file,ldoc,candidate_points=make_boolean_fixture(d)
        ldoc=copy.deepcopy(ldoc);ldoc["boolean_faces"][0]["run_original_id"]=999
        expect_error(lambda:producer.validate_d_boolean_face_lineage(ldoc,boolean,shell,bank,baseline,ldoc["pins"],SOURCE_FACES,SOURCE_LABELS,[0],candidate_points),"differs from the actual CSG NPZ")
    with tempfile.TemporaryDirectory() as d:
        boolean,shell,bank,baseline,lineage_file,ldoc,candidate_points=make_boolean_fixture(d)
        ldoc=copy.deepcopy(ldoc);ldoc["child_face_lineage"][0]["parent_d_face_rows"]=[2]
        expect_error(lambda:producer.validate_d_boolean_face_lineage(ldoc,boolean,shell,bank,baseline,ldoc["pins"],SOURCE_FACES,SOURCE_LABELS,[0],candidate_points),"CSR differs")


def test_boolean_geometry_and_parent_row_domains_fail_closed():
    with tempfile.TemporaryDirectory() as d:
        boolean,shell,bank,baseline,lineage_file,doc,candidate_points=make_boolean_fixture(d)
        doc=copy.deepcopy(doc);doc["boolean_faces"][0]["parent_d_face_rows"]=[999999]
        expect_error(lambda:producer.validate_d_boolean_face_lineage(
            doc,boolean,shell,bank,baseline,doc["pins"],SOURCE_FACES,SOURCE_LABELS,[0],candidate_points),
            "invalid or repeated 330 D source-face ancestry")
    with tempfile.TemporaryDirectory() as d:
        boolean,shell,bank,baseline,lineage_file,doc,candidate_points=make_boolean_fixture(d)
        doc=copy.deepcopy(doc)
        with np.load(boolean,allow_pickle=False) as z:
            arrays={k:np.asarray(z[k]) for k in z.files}
        arrays["points_world_m_f64"]=arrays["points_world_m_f64"].copy()
        arrays["points_world_m_f64"][2,2]=1.0e-6
        np.savez(boolean,**arrays)
        doc["boolean_faces"][0]["parent_d_face_rows"]=[0]
        doc["pins"]=dict(doc["pins"])
        doc["pins"]["boolean_output_npz_sha256"]=producer.digest(boolean)
        expected=dict(doc["pins"])
        expect_error(lambda:producer.validate_d_boolean_face_lineage(
            doc,boolean,shell,bank,baseline,expected,SOURCE_FACES,SOURCE_LABELS,[0],candidate_points),
            "not geometrically supported")
    with tempfile.TemporaryDirectory() as d:
        boolean,shell,bank,baseline,lineage_file,doc,candidate_points=make_boolean_fixture(d)
        doc=copy.deepcopy(doc)
        doc["child_face_lineage"][0]["child_node_rows"]=[0,5,1]
        # Reversed winding is not the exact oriented emitted-F32 facet.
        expect_error(lambda:producer.validate_d_boolean_face_lineage(
            doc,boolean,shell,bank,baseline,doc["pins"],SOURCE_FACES,SOURCE_LABELS,[0],candidate_points),
            "exact oriented emitted-F32 Boolean facet")
def test_actual_d_outward_face_orientation_must_match_boolean_child_order():
    child=[dict(face) for face in CHILD_FACES]
    child[0]["node_rows"]=(0,5,1)  # same vertices, reversed winding
    with tempfile.TemporaryDirectory() as d:
        expect_error(lambda:validate_faces(d,child_faces=child),
                     "orientation differs from the actual outward D boundary facet")


def test_frozen006_refined_d_facets_are_retained_by_exact_geometry_and_530_534_ancestry():
    capture=ROOT/"native-physical-parent-mass-330"/"native-source-identity-control-002"
    baseline=ROOT/"reduced-physical-boundary-400"/"attempt-030-connected-constrained-edge-fit"/"run-006"/"candidate.npz"
    def read(path):
        with path.open(newline="") as f:return list(csv.DictReader(f))
    node_rows=read(capture/"compiled-parent-cell-mass.csv.nodes.csv")
    tet_rows=read(capture/"compiled-parent-cell-mass.csv.tets.csv")
    label_rows=read(capture/"compiled-parent-cell-mass.csv.labels.csv")
    layer_rows=read(capture/"parent-layer-rows.csv")
    tets=np.asarray([[int(r["parent_node"+str(k)]) for k in range(4)] for r in tet_rows],np.int64)
    labels=[{k:int(r[k]) for k in ("material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id")} for r in label_rows]
    roles={int(r["volume_node_row"]):(r["role"],int(r["role_row"])) for r in layer_rows}
    source_faces=producer.collect_parent_d_faces(tets,labels,roles)
    components=producer.parent_tet_source_components(tets,labels)
    frozen=producer.collect_frozen006_d_face_lineage(baseline,source_faces,labels)
    with np.load(baseline,allow_pickle=False) as z:
        b={k:np.asarray(z[k]) for k in z.files}
    points=b["points_current_world_m_f32"].astype(np.float64)
    child_faces=[]
    for rec in frozen["facet_records"]:
        parent=int(rec["parent_tet_row"]);label=labels[parent]
        child_faces.append({"node_rows":rec["child_node_order"],"child_tet_row":rec["child_tet_row"],
            "parent_tet_row":parent,**label})
    doc={"retired_parent_d_nodes":[],"replaced_parent_d_faces":[]}
    result=producer.validate_d_boundary_face_retirement(
        doc,source_faces,child_faces,b["parent_volume_node_row"],roles,labels,components,{},
        candidate_points=points,frozen_baseline_d_faces=frozen)
    assert result["retained_frozen_current006_d_facet_count"]==4949
    assert result["covered_original_330_d_source_face_count"]==4777
    assert result["replacement_d_face_count"]==0
    assert frozen["summary"]["authenticated_530_534_source_face_ancestry"] is True


def test_exporter_full_d_retirement_integration():
    # Full exporter wiring against current006 ancestry. The identity CSG shell
    # is synthetic test input; this exercises exact-facet binding only.
    capture=ROOT/"native-physical-parent-mass-330"/"native-source-identity-control-002"
    baseline=ROOT/"reduced-physical-boundary-400"/"attempt-030-connected-constrained-edge-fit"/"run-006"/"candidate.npz"
    baseline_dir=baseline.parent
    bank=ROOT/"geometry-worker"/"matched-nesting-390"/"run-044-native-f32-replay-185"/"all185-source-order-native-f32-replay.npz"
    def read(path):return producer.read_csv(path)
    node_rows=read(capture/"compiled-parent-cell-mass.csv.nodes.csv")
    tet_rows=read(capture/"compiled-parent-cell-mass.csv.tets.csv")
    label_rows=read(capture/"compiled-parent-cell-mass.csv.labels.csv")
    weight_rows=read(capture/"compiled-parent-cell-mass.csv.weights.csv")
    layer_rows=read(capture/"parent-layer-rows.csv")
    attachment_rows=read(capture/"parent-deep-attachments.csv")
    parent_points=np.asarray([[producer.f32(float(r[k])) for k in ("current_x_m","current_y_m","current_z_m")]
                              for r in node_rows],np.float64)
    parent_tets=np.asarray([[int(r["parent_node"+str(k)]) for k in range(4)] for r in tet_rows],np.int64)
    source_labels=[{k:int(r[k]) for k in ("material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id")}
                   for r in label_rows]
    source_weights=np.asarray([[float(r["w"+str(k)]) for k in range(86)] for r in weight_rows],np.float64)
    role_by_node={int(r["volume_node_row"]):(r["role"],int(r["role_row"])) for r in layer_rows}
    protected,_=producer.derive_protected_parent_nodes(
        parent_tets,source_labels,role_by_node,attachment_rows,
        producer.digest(capture/"parent-deep-attachments.csv"))
    source_faces=producer.collect_parent_d_faces(parent_tets,source_labels,role_by_node)
    components=producer.parent_tet_source_components(parent_tets,source_labels)
    frozen=producer.collect_frozen006_d_face_lineage(baseline,source_faces,source_labels)
    with np.load(baseline,allow_pickle=False) as z:
        base={k:np.asarray(z[k]) for k in z.files}
    points=base["points_reference_world_m_f64"].astype(np.float32).astype(np.float64)
    assert np.array_equal(points,base["points_current_world_m_f32"].astype(np.float64))
    parent_to_child={int(old):child for child,old in enumerate(base["parent_volume_node_row"]) if old>=0}
    base_facets_by_child={}
    for rec in frozen["facet_records"]:
        for node in rec["child_node_order"]:
            base_facets_by_child.setdefault(int(node),[]).append(rec)
    candidates=[]
    for old,(role,role_row) in sorted(role_by_node.items()):
        child=parent_to_child.get(old)
        if role!="D" or old in protected or child is None:
            continue
        incident=[f for f in source_faces if old in f["node_rows"]]
        current_faces=base_facets_by_child.get(child,[])
        if not incident or not current_faces:
            continue
        if np.linalg.norm(points[child]-parent_points[old])>=3.9e-4:
            continue
        if any(set(rec["parent_d_face_rows"])!={int(f["parent_d_face_row"])}
               for rec in current_faces for f in incident
               if int(rec["parent_tet_row"])==int(f["parent_tet_row"])):
            continue
        # A small Float32-representable move makes this a real changed-facet
        # integration case while retaining positive child tetrahedra.
        incident_tets=np.flatnonzero(np.any(base["volume_tetrahedra"]==child,axis=1))
        for delta in (1.0e-6,5.0e-7,1.0e-7,5.0e-8):
            for axis in range(3):
                for sign in (1.0,-1.0):
                    trial=points.copy()
                    trial[child,axis]=np.float32(points[child,axis]+sign*delta)
                    if np.array_equal(trial[child],points[child]):
                        continue
                    q=trial[base["volume_tetrahedra"]]
                    vol=np.einsum("ij,ij->i",q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6.0
                    if np.all(vol>0.0):
                        candidates.append((old,role_row,child,incident,current_faces,trial,delta,axis,sign))
                        break
                if candidates:break
            if candidates:break
        if candidates:break
    assert candidates,"no unprotected current006 D source node has a bounded positive-cell test move"
    retired_node,retired_role_row,retired_child,incident_source_faces,changed_baseline_faces,points,delta,axis,sign=candidates[0]
    # Select a source tet and one-hot original-node barycentrics for the retired
    # source identity; the moved point remains within the existing 0.4 mm bound.
    source_face=incident_source_faces[0]
    source_tet=int(source_face["parent_tet_row"])
    corner=int(np.flatnonzero(parent_tets[source_tet]==retired_node)[0])
    beta=[1.0 if k==corner else 0.0 for k in range(4)]
    parent_node_by_child=base["parent_volume_node_row"].copy()
    parent_node_by_child[retired_child]=-1
    with tempfile.TemporaryDirectory(prefix="d-retirement-export-") as temp:
        temp=Path(temp)
        candidate_npz=temp/"candidate.npz";candidate_node=temp/"candidate.node"
        candidate_map=temp/"candidate-node-map.csv";ancestry=temp/"inserted-node-ancestry.csv"
        shell_path=temp/"shell.npz";boolean_path=temp/"boolean.npz"
        lineage_path=temp/"boolean-lineage.json";retirement_path=temp/"d-retirement.json"
        out=temp/"export"
        arrays={k:np.asarray(v) for k,v in base.items()}
        arrays["points_reference_world_m_f64"]=points
        arrays["points_current_world_m_f32"]=points.astype(np.float32)
        arrays["parent_volume_node_row"]=parent_node_by_child
        np.savez_compressed(candidate_npz,**arrays)
        with candidate_node.open("w") as f:
            f.write(f"{len(points)} 3 0 0\n")
            for i,xyz in enumerate(points):
                f.write(str(i+1)+" "+" ".join(format(float(x),".17g") for x in xyz)+"\n")
        shutil.copyfile(baseline_dir/"coarse-node-to-fine-surface.csv",candidate_map)
        existing_ancestry=read(baseline_dir/"inserted-node-ancestry.csv")
        with ancestry.open("w",newline="") as f:
            fieldnames=["volume_node_row","parent_tet_row"]+["parent_bary"+str(k) for k in range(4)]
            writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader()
            for row in existing_ancestry:
                writer.writerow({k:row[k] for k in fieldnames})
            writer.writerow({"volume_node_row":retired_child,"parent_tet_row":source_tet,
                **{"parent_bary"+str(k):format(beta[k],".17g") for k in range(4)}})
        # Bind only the current006 D facets incident to the changed node.
        changed_records=sorted({int(rec["baseline_d_face_row"]):rec for rec in changed_baseline_faces}.values(),
                               key=lambda rec:int(rec["baseline_d_face_row"]))
        source_row_by_baseline={int(rec["baseline_d_face_row"]):int(rec["parent_d_face_rows"][0])
                                for rec in frozen["facet_records"]}
        source_shell_faces=np.asarray([rec["child_node_order"] for rec in changed_records],np.int64)
        source_parent_rows=np.asarray([source_row_by_baseline[int(rec["baseline_d_face_row"])]
                                       for rec in changed_records],np.int64)
        source_shell_tets=np.asarray([rec["child_tet_row"] for rec in changed_records],np.int64)
        source_shell_nodes=np.asarray([source_faces[int(row)]["node_rows"] for row in source_parent_rows],np.int64)
        np.savez_compressed(shell_path,points_world_m_f64=points,faces_f64=source_shell_faces,
            source_face_global_nodes=source_shell_nodes,source_tet_rows=source_shell_tets,
            source_face_kind=np.asarray(["existing_D_material_boundary"]*len(changed_records)))
        run_id=62000502
        np.savez_compressed(boolean_path,points_world_m_f64=points,faces_f64=source_shell_faces.copy(),
            run_index=np.asarray([0,3*len(source_shell_faces)],np.uint64),
            run_original_id=np.asarray([run_id],np.uint32),
            face_id=np.arange(len(source_shell_faces),dtype=np.uint64))
        inputs={
            "parent_packet_sha256":producer.digest(capture/"compiled-parent-cell-mass.csv"),
            "parent_nodes_sha256":producer.digest(capture/"compiled-parent-cell-mass.csv.nodes.csv"),
            "parent_tets_sha256":producer.digest(capture/"compiled-parent-cell-mass.csv.tets.csv"),
            "parent_labels_sha256":producer.digest(capture/"compiled-parent-cell-mass.csv.labels.csv"),
            "parent_weights_sha256":producer.digest(capture/"compiled-parent-cell-mass.csv.weights.csv"),
            "parent_layer_rows_sha256":producer.digest(capture/"parent-layer-rows.csv"),
            "parent_deep_attachments_sha256":producer.digest(capture/"parent-deep-attachments.csv"),
            "frozen_current006_baseline_sha256":producer.digest(baseline),
            "native_poses_sha256":producer.digest(capture/"registered-skin-poses-100.json"),
            "candidate_npz_sha256":producer.digest(candidate_npz),
            "candidate_node_sha256":producer.digest(candidate_node),
            "candidate_node_map_sha256":producer.digest(candidate_map),
            "inserted_node_ancestry_sha256":producer.digest(ancestry),
            "o_boundary_retirement_map_sha256":None,
            "boolean_output_npz_sha256":producer.digest(boolean_path),
            "source_shell_npz_sha256":producer.digest(shell_path),
            "bank_input_npz_sha256":producer.digest(bank),
        }
        bool_pins={k:v for k,v in inputs.items() if k!="o_boundary_retirement_map_sha256"}
        shell_run={"run_original_id":run_id,"source_domain":"source_shell",
            "source_mesh_sha256":inputs["source_shell_npz_sha256"],"input_face_count":len(source_shell_faces)}
        boolean_faces=[];child_lineage=[];replacement_by_source={}
        for i,rec in enumerate(changed_records):
            brow=int(rec["baseline_d_face_row"]);prow=source_row_by_baseline[brow]
            nodes=list(map(int,rec["child_node_order"]))
            boolean_faces.append({"boolean_face_row":i,"run_original_id":run_id,
                "input_face_row":i,"parent_d_face_rows":[prow]})
            child_lineage.append({"child_node_rows":nodes,"boolean_face_rows":[i],
                "parent_d_face_rows":[prow]})
            source=source_faces[prow]
            entry=replacement_by_source.setdefault(prow,{"parent_d_face_row":prow,
                "source_stable_source_face_id":int(source["stable_source_face_id"]),
                "source_material_index":int(source["material_index"]),
                "source_geometric_donor_body":int(source["geometric_donor_body"]),
                "source_anatomical_region_id":int(source["anatomical_region_id"]),
                "replacement_child_node_rows":[],"replacement_boolean_face_rows":[]})
            entry["replacement_child_node_rows"].append(nodes)
            entry["replacement_boolean_face_rows"].append([i])
        boolean_doc={"schema":"numi-human.d-boolean-output-face-lineage.v1",
            "pins":bool_pins,"run_sources":[shell_run],"boolean_faces":boolean_faces,
            "child_face_lineage":child_lineage}
        lineage_path.write_text(json.dumps(boolean_doc,sort_keys=True)+"\n")
        retirement_doc={"schema":"numi-human.explicit-d-boundary-retirement.v1",
            "pins":dict(inputs,boolean_face_lineage_sha256=producer.digest(lineage_path)),
            "retired_parent_d_nodes":[{"volume_node_row":retired_node,"parent_d_role_row":retired_role_row}],
            "replaced_parent_d_faces":list(replacement_by_source.values())}
        retirement_path.write_text(json.dumps(retirement_doc,sort_keys=True)+"\n")
        args=argparse.Namespace(capture=capture,out=out,candidate_npz=candidate_npz,
            candidate_node=candidate_node,candidate_node_map=candidate_map,
            inserted_node_ancestry=ancestry,o_boundary_retirement_map=None,
            d_boundary_retirement_map=retirement_path,d_boolean_output_npz=boolean_path,
            d_boolean_source_shell_npz=shell_path,d_boolean_bank_input_npz=bank,
            d_boolean_face_lineage=lineage_path,d_retirement_baseline_npz=baseline,
            maximum_mass_transport_distance_m=0.02,candidate_operational_reference=False)
        result=producer.export(args)
        d=result["d_boundary_retirement"]
        assert result["physical_admission"] is False and result["native_steps"]==0
        assert d["retired_parent_d_node_count"]==1
        assert d["retained_frozen_current006_d_facet_count"]==4949-len(changed_records)
        assert d["covered_original_330_d_source_face_count"]==4777
        assert d["replacement_d_face_count"]==len(changed_records)
        assert d["child_faces_bound_to_exact_oriented_f32_boolean_facets"]==len(changed_records)
        assert d["preexisting_current006_d_nodes_unchanged_and_exempt"]>0
        assert d["preexisting_current006_d_nodes_changed_and_checked_as_new"]==0
        assert d["new_d_boundary_node_count"]==1
        assert d["new_d_boundary_node_maximum_source_residual_m"]<=d["new_d_boundary_node_maximum_source_residual_limit_m"]
        assert delta<=1.0e-6 and abs(float(points[retired_child,axis])-float(base["points_current_world_m_f32"][retired_child,axis]))<1.1e-6



def test_only_exactly_unchanged_preexisting_current006_nodes_are_exempt():
    beta=(0.25,0.25,0.25,0.25)
    parent_nodes=np.asarray([-1,0,-1],np.int64)
    baseline_points=np.asarray([[.25,.25,.25],[0,0,0],[.25,.25,.25]],np.float32)
    ancestry_rows=[{"volume_node_row":"0","parent_tet_row":"0",**{"parent_bary"+str(k):str(beta[k]) for k in range(4)}},
                   {"volume_node_row":"2","parent_tet_row":"0",**{"parent_bary"+str(k):str(beta[k]) for k in range(4)}}]
    candidate_ancestry={0:(0,beta),2:(0,beta)}
    unchanged=producer.unchanged_preexisting_derived_node_rows(
        parent_nodes,baseline_points,ancestry_rows,parent_nodes,baseline_points.astype(np.float64),candidate_ancestry)
    assert unchanged=={0,2}
    moved=baseline_points.astype(np.float64);moved[2,0]=np.float32(moved[2,0]+5.0e-4)
    moved_exempt=producer.unchanged_preexisting_derived_node_rows(
        parent_nodes,baseline_points,ancestry_rows,parent_nodes,moved,candidate_ancestry)
    assert moved_exempt=={0}
    wrong_ancestry={0:(0,beta),2:(1,beta)}
    reparented_exempt=producer.unchanged_preexisting_derived_node_rows(
        parent_nodes,baseline_points,ancestry_rows,parent_nodes,baseline_points.astype(np.float64),wrong_ancestry)
    assert reparented_exempt=={0}
    parent_points=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]
    parent_tets=[(0,1,2,3)]
    roles={0:("D",0),2:("D",1)}
    selected=[(0,beta),(0,beta),(0,beta)]
    expect_error(lambda:producer.validate_d_new_node_source_residuals(
        moved,parent_points,parent_tets,parent_nodes,selected,roles,
        preexisting_derived_node_rows=moved_exempt),"exceeds the 0.4 mm original-source registration bound")


def test_source_component_change_rejected():
    labels=[dict(SOURCE_LABELS[0]),dict(SOURCE_LABELS[0])]
    child=[dict(face) for face in CHILD_FACES]
    child[0]["parent_tet_row"]=1
    with tempfile.TemporaryDirectory() as d:
        expect_error(lambda:validate_faces(d,child_faces=child,labels=labels,components=[0,1]),"source-connected parent-tet component")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence-root",type=Path,required=True,
                    help="physical-transfer evidence root containing the pinned current006 and 330/530/534 inputs")
    args=ap.parse_args()
    ROOT=args.evidence_root.resolve()
    tests=[test_new_d_node_source_residual_is_bounded_only_for_explicit_d_path,
           test_source_protected_set_is_149,test_actual_saved_csg_tags_match_shell_and_native_bank_inputs,
           test_actual_boolean_output_tags_are_geometrically_supported,test_positive_retirement_and_face_lineage,
           test_d_node_retirement_is_explicit_and_role_limited,test_face_coverage_labels_and_ancestry_fail_closed,
           test_boolean_geometry_and_parent_row_domains_fail_closed,
           test_actual_d_outward_face_orientation_must_match_boolean_child_order,
           test_frozen006_refined_d_facets_are_retained_by_exact_geometry_and_530_534_ancestry,
           test_exporter_full_d_retirement_integration,test_only_exactly_unchanged_preexisting_current006_nodes_are_exempt,
           test_source_component_change_rejected]
    for test in tests:
        test();print(test.__name__+"=PASS")
    print("d_boundary_retirement_501=PASS assertions=enabled tests="+str(len(tests))+" physical_admission=0")
