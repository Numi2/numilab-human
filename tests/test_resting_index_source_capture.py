"""Index-only repairs must not replace actual GPU coordinates with a replay."""
import copy
import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location("index_source_cli",
    Path(__file__).resolve().parents[1]/"tools/verify_resting_anatomy.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

def fixture(tmp_path):
    p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[2,2,0]],dtype=np.float32)
    row=dict(stable_id=1,first_vertex=10,vertex_count=5,first_binding=0,binding_count=1,
             positions=p,normals=np.tile(np.array([0,0,1],dtype=np.float32),(5,1)),
             binding_indices=np.zeros((5,4),dtype=np.uint32),
             weights=np.tile(np.array([1,0,0,0],dtype=np.float32),(5,1)),
             faces=np.array([[0,1,2],[0,2,3]],dtype=np.int64))
    base=dict(records=[(0,1,10,5,0,6,1,1)],binding_start=0,raw=b"x"*36,row=row)
    candidate=copy.deepcopy(base)
    candidate["row"]["faces"]=np.array([[0,1,3],[1,2,3]],dtype=np.int64)
    # Deliberate tiny GPU/replay difference: the audit must keep GPU bytes.
    captured=row["positions"][:4].copy()
    captured[2,0]=np.nextafter(captured[2,0],np.float32(np.inf))
    key=(51005,1);surfaces={key:(captured,row["faces"].astype(np.uint32),None,
                               np.arange(110,114,dtype=np.uint64))}
    paths=[]
    for name,data in [("base",b"base"),("candidate",b"candidate")]:
        path=tmp_path/name;path.write_bytes(data);paths.extend([path,hashlib.sha256(data).hexdigest()])
    tables={str(paths[0]):base,str(paths[2]):candidate}
    return key,surfaces,paths,tables

def run(key,surfaces,paths,tables):
    with patch("numilab_human.passive_attachment_composition._read_nhtiss4",
               side_effect=lambda p:tables[str(p)]),          patch("numilab_human.passive_attachment_composition._nhtiss_row_arrays",
               side_effect=lambda table,rec:table["row"]):
        return cli.apply_index_source(surfaces,key,*paths,{})

def test_changed_diagonal_uses_captured_float32_corners_without_cpu_replay(tmp_path):
    key,surfaces,paths,tables=fixture(tmp_path);before=copy.deepcopy(surfaces[key])
    with patch("numilab_human.passive_attachment_composition._native_blended_rest_positions",
               side_effect=AssertionError("CPU replay must not run")):
        result=run(key,surfaces,paths,tables)
    assert surfaces[key][0].tobytes()==before[0].tobytes()
    assert surfaces[key][3].tobytes()==before[3].tobytes()
    assert not np.array_equal(surfaces[key][1],before[1])
    assert result["source_capture_vertex_namespace_offset"]==110
    assert not result["actual_native_emission_verified"]
    assert not result["anatomical_admission"]

@pytest.mark.parametrize("change",["positions","binding_indices","weights","rest_records",
                                  "new_slot","wrong_base_faces","wrong_capture_ids"])
def test_any_material_change_or_incomplete_source_join_rejects(tmp_path,change):
    key,surfaces,paths,tables=fixture(tmp_path);base=tables[str(paths[0])]
    candidate=tables[str(paths[2])];row=candidate["row"]
    if change=="positions":row[change][0,0]=np.nextafter(np.float32(0),np.float32(1))
    elif change=="binding_indices":row[change][0,0]=1
    elif change=="weights":row[change][0,0]=np.nextafter(np.float32(1),np.float32(0))
    elif change=="rest_records":candidate["raw"]=b"y"*36
    elif change=="new_slot":row["faces"][0]=[0,1,4]
    elif change=="wrong_base_faces":base["row"]["faces"][0]=[0,1,3]
    elif change=="wrong_capture_ids":surfaces[key][3][1]=120
    with pytest.raises(ValueError):
        run(key,surfaces,paths,tables)
