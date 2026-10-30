import importlib.util
from pathlib import Path
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/quality-warning-connected-union-038/merge-run004-run005-O/compose_run004_run005_o.py')
spec=importlib.util.spec_from_file_location('compose_o',S); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def test_disjoint_retired_patches_preserve_tetra_boundaries():
    tet0=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
    tet1=tet0+4
    base=np.vstack((tet0,tet1))
    before=m.directed_boundary(base)
    mandatory=[{'patch_id':'005-a','retired_input_face_rows':[0],'new_faces_input_point_rows':base[[0]].tolist()}]
    additions=[{'patch_id':'004-b','retired_input_face_rows':[4],'new_faces_input_point_rows':base[[4]].tolist()}]
    merged,keep,retired,selected,skipped=m.merge_face_operations(base,mandatory,additions)
    assert len(selected)==2 and not skipped and retired=={0,4}
    assert m.directed_boundary(merged)==before
def test_shared_retired_rows_are_skipped_and_mandatory_overlap_fails():
    base=np.array([[0,1,2],[3,4,5]],np.int64)
    mandatory=[{'patch_id':'005-a','retired_input_face_rows':[0],'new_faces_input_point_rows':base[[0]].tolist()}]
    additions=[{'patch_id':'004-conflict','retired_input_face_rows':[0],'new_faces_input_point_rows':base[[0]].tolist()},
               {'patch_id':'004-disjoint','retired_input_face_rows':[1],'new_faces_input_point_rows':base[[1]].tolist()}]
    merged,keep,retired,selected,skipped=m.merge_face_operations(base,mandatory,additions)
    assert [x['patch_id'] for x in selected]==['005-a','004-disjoint']
    assert [x['patch_id'] for x in skipped]==['004-conflict']
    try:m.merge_face_operations(base,mandatory+mandatory,[])
    except ValueError:pass
    else:raise AssertionError('overlapping mandatory rows must fail')
