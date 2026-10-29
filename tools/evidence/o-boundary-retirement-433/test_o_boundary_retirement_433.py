import importlib.util
from pathlib import Path
import copy

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("producer433",P/"export_refined_coarse_registered_volume_433.py")
producer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)

SOURCE_ROLES={0:("O",0),1:("O",1),2:("O",2),3:("O",3)}
SOURCE_FACES=[
    {"parent_o_face_row":0,"node_rows":(0,1,2),"parent_tet_row":0,"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":700},
    {"parent_o_face_row":1,"node_rows":(0,2,3),"parent_tet_row":0,"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":700},
    {"parent_o_face_row":2,"node_rows":(0,1,3),"parent_tet_row":0,"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":700},
    {"parent_o_face_row":3,"node_rows":(1,2,3),"parent_tet_row":0,"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":700},
]
DOC={
    "schema":"numi-human.explicit-o-boundary-retirement.v1",
    "retired_parent_o_nodes":[{"volume_node_row":0,"parent_o_role_row":0}],
    "replaced_parent_o_faces":[
        {"parent_o_face_row":0,"source_stable_source_face_id":700,"source_material_index":15,"source_geometric_donor_body":65,"source_anatomical_region_id":66,"replacement_child_node_rows":[[1,2,4]]},
        {"parent_o_face_row":1,"source_stable_source_face_id":700,"source_material_index":15,"source_geometric_donor_body":65,"source_anatomical_region_id":66,"replacement_child_node_rows":[[2,3,4]]},
        {"parent_o_face_row":2,"source_stable_source_face_id":700,"source_material_index":15,"source_geometric_donor_body":65,"source_anatomical_region_id":66,"replacement_child_node_rows":[[1,3,4]]},
    ],
}
PARENT_NODE_BY_CHILD=[0,1,2,3,-1]
OUTPUT_O={1:0,2:1,3:2,4:3}
CHILD_FACES=[
    {"node_rows":(1,2,4),"child_tet_row":5,"parent_tet_row":1,"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":901},
    {"node_rows":(2,3,4),"child_tet_row":6,"parent_tet_row":1,"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":901},
    {"node_rows":(1,3,4),"child_tet_row":7,"parent_tet_row":1,"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":901},
    {"node_rows":(1,2,3),"child_tet_row":8,"parent_tet_row":1,"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":700},
]
PARENT_TETS=[[0,1,2,3],[1,2,3,4],[10,11,12,13]]
PARENT_LABELS=[
    {"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":700},
    {"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":901},
    {"material_index":15,"geometric_donor_body":65,"anatomical_region_id":66,"stable_source_face_id":901},
]


def validate(doc=DOC, child_faces=CHILD_FACES, roles=SOURCE_ROLES,
             parent_nodes=PARENT_NODE_BY_CHILD, output_o=OUTPUT_O, labels=PARENT_LABELS):
    retired=producer.validate_o_boundary_retirement_nodes(doc,roles,parent_nodes,output_o)
    components=producer.parent_tet_source_components(PARENT_TETS,labels)
    summary=producer.validate_o_boundary_face_retirement(
        doc,SOURCE_FACES,child_faces,parent_nodes,roles,labels,components)
    return retired,summary


def expect_error(call, message):
    try:
        call()
    except ValueError as exc:
        assert message in str(exc), (message,str(exc))
    else:
        raise AssertionError("expected ValueError containing: "+message)


def test_pins_are_exact():
    expected={"parent_tets_sha256":"a"*64,"candidate_npz_sha256":"b"*64}
    producer.validate_o_boundary_retirement_pins({"pins":expected},expected)
    bad=dict(expected);bad["candidate_npz_sha256"]="c"*64
    expect_error(lambda:producer.validate_o_boundary_retirement_pins({"pins":bad},expected),"pins do not match")


def test_positive():
    retired,summary=validate()
    assert retired=={0}
    assert summary=={
        "retired_parent_o_node_count":1,"retired_parent_o_face_count":3,
        "replacement_o_face_count":3,"retained_parent_o_face_count":1,
        "source_connected_material_donor_region_proof":True,"physical_admission":False,
    }


def test_negative_node_contracts():
    doc=copy.deepcopy(DOC);doc["retired_parent_o_nodes"]=[]
    expect_error(lambda:validate(doc=doc),"explicitly name")
    doc=copy.deepcopy(DOC);doc["retired_parent_o_nodes"][0]["parent_o_role_row"]=2
    expect_error(lambda:validate(doc=doc),"exact source O row")
    doc=copy.deepcopy(DOC);doc["retired_parent_o_nodes"][0]["volume_node_row"]=4
    expect_error(lambda:validate(doc=doc,roles={**SOURCE_ROLES,4:("I",0)}),"non-O source node")
    expect_error(lambda:validate(output_o={**OUTPUT_O,0:4}),"remains in the candidate O boundary")
    expect_error(lambda:validate(output_o={1:0,2:1,4:2}),"unretired parent O node lost")
    expect_error(lambda:validate(parent_nodes=[1,2,3,-1]),"dropped an original O source node")


def test_negative_face_coverage():
    doc=copy.deepcopy(DOC);doc["replaced_parent_o_faces"].pop()
    expect_error(lambda:validate(doc=doc),"does not cover every removed")
    doc=copy.deepcopy(DOC);doc["replaced_parent_o_faces"][0]["replacement_child_node_rows"]=[[1,2,9]]
    expect_error(lambda:validate(doc=doc),"not an actual new O boundary face")
    extra=dict(CHILD_FACES[0],node_rows=(2,3,5),child_tet_row=12)
    expect_error(lambda:validate(child_faces=CHILD_FACES+[extra],
        parent_nodes=[0,1,2,3,-1,-1],output_o={**OUTPUT_O,5:4}),"does not cover every new")
    doc=copy.deepcopy(DOC);doc["replaced_parent_o_faces"][0]["source_stable_source_face_id"]=999
    expect_error(lambda:validate(doc=doc),"stable-face history")
    doc=copy.deepcopy(DOC);doc["replaced_parent_o_faces"][0]["source_geometric_donor_body"]=66
    expect_error(lambda:validate(doc=doc),"source material/donor/region")
    doc=copy.deepcopy(DOC);doc["replaced_parent_o_faces"].append({
        "parent_o_face_row":3,"source_stable_source_face_id":700,"source_material_index":15,
        "source_geometric_donor_body":65,"source_anatomical_region_id":66,
        "replacement_child_node_rows":[[1,2,4]]})
    expect_error(lambda:validate(doc=doc),"face that was not retired")


def test_negative_source_component_and_semantics():
    disconnected=[dict(row,parent_tet_row=2) if row["node_rows"]==(1,2,4) else row for row in CHILD_FACES]
    expect_error(lambda:validate(child_faces=disconnected),"source-connected parent-tet component")
    changed_labels=copy.deepcopy(PARENT_LABELS)
    changed_labels[1]["geometric_donor_body"]=66
    expect_error(lambda:validate(labels=changed_labels),"changed its source material/donor/region")


if __name__=="__main__":
    tests=[test_pins_are_exact,test_positive,test_negative_node_contracts,test_negative_face_coverage,test_negative_source_component_and_semantics]
    for test in tests:
        test()
        print(test.__name__+"=PASS")
    print("o_boundary_retirement_433=PASS assertions=enabled tests="+str(len(tests))+" physical_admission=0")
