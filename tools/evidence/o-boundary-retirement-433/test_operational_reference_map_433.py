#!/usr/bin/env python3
"""Verify the opt-in candidate self-tet map against run005 real sidecars."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, pathlib
import numpy as np

HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("producer433",HERE/"export_refined_coarse_registered_volume_433.py")
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b"") : h.update(block)
    return h.hexdigest()

def rows(path):
    with pathlib.Path(path).open(newline="") as f: return list(csv.reader(f))

def expect_error(fn, needle):
    try: fn()
    except ValueError as exc: assert needle in str(exc),(needle,str(exc))
    else: raise AssertionError("expected ValueError containing "+needle)

def test_helper_contracts():
    labels=[{"geometric_donor_body":7,"anatomical_region_id":21}]
    tets=[(0,1,2,3),(1,2,3,4)]
    child=[dict(labels[0]),dict(labels[0])]
    incident=[[0],[0,1],[0,1],[0,1],[1]]
    fields=producer.candidate_operational_reference_fields(0,0,0,labels,tets,child,incident)
    assert fields==[0,0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,7,21]
    producer.validate_candidate_operational_reference_fields(fields,0,0,0,labels,tets,child,incident)
    bad=list(fields);bad[1]=1
    expect_error(lambda:producer.validate_candidate_operational_reference_fields(bad,0,0,0,labels,tets,child,incident),"not incident")
    bad=list(fields);bad[2:6]=[0.25]*4
    expect_error(lambda:producer.validate_candidate_operational_reference_fields(bad,0,0,0,labels,tets,child,incident),"one-hot")
    bad=list(fields);bad[10]=8
    expect_error(lambda:producer.validate_candidate_operational_reference_fields(bad,0,0,0,labels,tets,child,incident),"source-parent donor/region")
    bad=list(fields);bad[6]=1e-12
    expect_error(lambda:producer.validate_candidate_operational_reference_fields(bad,0,0,0,labels,tets,child,incident),"zero residual")
    bad=list(fields);bad[0]=4
    expect_error(lambda:producer.validate_candidate_operational_reference_fields(bad,0,0,0,labels,tets,child,incident),"node identity")

def validate_real(default_map, operational_map, candidate_path, parent_capture, receipt_path):
    baseline=rows(default_map);current=rows(operational_map)
    assert len(baseline[0])==42 and len(current[0])==54
    assert current[0]==baseline[0]+producer.OPERATIONAL_REFERENCE_MAP_FIELDS
    assert len(baseline)==len(current) and len(current)>1
    for old,new in zip(baseline[1:],current[1:]):
        assert len(old)==42 and len(new)==54 and new[:42]==old,"legacy columns changed under opt-in"
    with np.load(candidate_path,allow_pickle=False) as z:
        node_parent=np.asarray(z["parent_volume_node_row"],dtype=np.int64)
        tets=np.asarray(z["volume_tetrahedra"],dtype=np.int64)
        material=np.asarray(z["material_index"],dtype=np.int64)
        donor=np.asarray(z["geometric_donor_body"],dtype=np.int64)
        region=np.asarray(z["anatomical_region_id"],dtype=np.int64)
    assert node_parent.shape==(len(node_parent),) and len(node_parent)==7233
    assert sorted(node_parent.tolist())==list(range(7233))
    assert tets.shape==(28632,4) and material.shape==donor.shape==region.shape==(len(tets),)
    parent_labels={int(r["parent_tet_row"]):r for r in csv.DictReader((pathlib.Path(parent_capture)/"compiled-parent-cell-mass.csv.labels.csv").open(newline=""))}
    unique={};copies=0
    for row in current[1:]:
        node=int(row[0])-1;assert 0<=node<len(node_parent)
        if node in unique:
            prior=unique[node]
            assert row[0]==prior[0] and row[3]==prior[3] and row[6:]==prior[6:],"reciprocal I copies disagree outside their partition identities"
            copies+=1;continue
        unique[node]=row
        source_node,op_cell=int(row[42]),int(row[43])
        assert source_node==int(node_parent[node])
        assert 0<=op_cell<len(tets)
        tet=tets[op_cell].tolist();assert tet.count(node)==1
        expected=[1.0 if n==node else 0.0 for n in tet]
        assert [float(x) for x in row[44:48]]==expected,"operational registration is not exact one-hot self-tet"
        assert row[48:52]==["0.0","0.0","0.0","0.0"],"operational self-tet has nonzero residual/distance"
        historical_tet=int(row[29]);parent=parent_labels[historical_tet]
        assert int(row[52])==int(parent["geometric_donor_body"])
        assert int(row[53])==int(parent["anatomical_region_id"])
        assert donor[op_cell]==int(row[40]) and region[op_cell]==int(row[41])
        assert material[op_cell] in (15,16)
    assert len(unique)==len(node_parent) and copies==2411
    source_rows={int(r[42]) for r in unique.values()}
    assert source_rows==set(range(len(node_parent))),"original source-node injection is incomplete"
    receipt=json.loads(pathlib.Path(receipt_path).read_text())
    op=receipt["candidate_operational_reference_map"]
    assert receipt["registration_domain"]=="candidateOperationalReference"
    assert op["node_map_sha256"]==sha(operational_map)
    assert op["candidate_npz_sha256"]==sha(candidate_path)
    assert op["immutable_source_parent_packet_sha256"]==sha(pathlib.Path(parent_capture)/"compiled-parent-cell-mass.csv")
    assert op["source_parent_node_identity_rows"]==len(node_parent) and op["physical_admission"] is False
    registry=json.loads((pathlib.Path(receipt_path).parent/"source-domain-registry.json").read_text())
    assert registry["registration_domain"]=="candidateOperationalReference"
    assert registry["candidate_operational_reference_map"]["node_map_sha256"]==sha(operational_map)
    # A wrong support index must be caught before use, and the map pin changes.
    child_tets=[tuple(map(int,tet)) for tet in tets]
    child_labels=[{"geometric_donor_body":int(donor[i]),"anatomical_region_id":int(region[i])}
                   for i in range(len(tets))]
    incident=[[] for _ in range(len(node_parent))]
    for cell,tet in enumerate(child_tets):
        for node in tet: incident[node].append(cell)
    first=next(iter(unique.values()))
    node=int(first[0])-1;parent_tet=int(first[29])
    fields=[int(first[42]),int(first[43]),*[float(x) for x in first[44:54]]]
    producer.validate_candidate_operational_reference_fields(fields,node,int(node_parent[node]),parent_tet,
        list(parent_labels.values()),child_tets,child_labels,incident)
    bad=list(fields);bad[1]=len(tets)
    expect_error(lambda:producer.validate_candidate_operational_reference_fields(bad,node,int(node_parent[node]),parent_tet,
        list(parent_labels.values()),child_tets,child_labels,incident),"not incident")
    bad=list(fields);bad[10]=int(bad[10])+1
    expect_error(lambda:producer.validate_candidate_operational_reference_fields(bad,node,int(node_parent[node]),parent_tet,
        list(parent_labels.values()),child_tets,child_labels,incident),"source-parent donor/region")
    assert sha(operational_map)!=sha(default_map)
    # The unchanged legacy export must be exactly the same bytes as the prior run.
    result={"default_map_sha256":sha(default_map),
            "default_map_bytes_identical_to_run015":True,
            "operational_map_sha256":sha(operational_map),"candidate_sha256":sha(candidate_path),
            "parent_packet_sha256":op["immutable_source_parent_packet_sha256"],
            "node_count":len(node_parent),"tet_count":len(tets),"map_rows":len(current)-1,
            "unique_volume_nodes":len(unique),"shared_I_copies":copies,"source_parent_injection_rows":len(source_rows),
            "self_tet_one_hot_rows":len(unique),"zero_residual_rows":len(unique),"wrong_map_out_of_range_rejected":True,
            "scope":"source-only parser/control; positive cells do not imply mass or physical admission"}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--default-map",type=pathlib.Path,required=True)
    ap.add_argument("--baseline-map",type=pathlib.Path,required=True)
    ap.add_argument("--operational-map",type=pathlib.Path,required=True)
    ap.add_argument("--candidate",type=pathlib.Path,required=True)
    ap.add_argument("--parent-capture",type=pathlib.Path,required=True)
    ap.add_argument("--receipt",type=pathlib.Path,required=True)
    a=ap.parse_args()
    test_helper_contracts()
    assert sha(a.default_map)==sha(a.baseline_map),"default-off map bytes changed from run015"
    result=validate_real(a.default_map,a.operational_map,a.candidate,a.parent_capture,a.receipt)
    result["baseline_map_sha256"]=sha(a.baseline_map)
    result["default_map_sha256"]=sha(a.default_map)
    result["helper_wrong_map_cases"]=5
    print(json.dumps(result,indent=2,sort_keys=True))
    print("operational_reference_map_433=PASS parser_control_only=1 physical_admission=0 native_steps=0")

if __name__=="__main__": main()
