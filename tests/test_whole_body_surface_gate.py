"""The inspection profile must select only proved embedded source copies."""
import copy
import io
import json
import tarfile

import pytest

from numilab_human.whole_body_surface_gate import selected_surface_rows,census_rows
from numilab_human.surface_topology_repair import BASE_SHA


def test_selected_view_refuses_bad_copy_or_hidden_id_drift():
    from numilab_human.model import REPOSITORY_ROOT
    root=REPOSITORY_ROOT/'Docs/media/source-topology-repair-20260929'
    candidates=json.loads((root/'source-audit.json').read_text())['rows']
    selection=json.loads((root/'inspection-selection.final.json').read_text())
    sources=[{'stable_id':i,'source_layer_code':1,'source_member_id':f'FJ{i}',
              'source_object_name':None,'status':'closed_embedded_source_candidate',
              'closed_embedded_surface_candidate':True} for i in range(1,580)]
    chosen=selected_surface_rows(sources,candidates,selection)
    assert len(chosen)==579
    assert sum(r['visible_stable_id']>579 for r in chosen)==11
    assert all(r['selected_closed_embedded_surface_candidate'] for r in chosen)
    wrong=copy.deepcopy(selection);wrong['hidden_anatomy_stable_ids'].pop()
    with pytest.raises(Exception,match='inspection'):
        selected_surface_rows(sources,candidates,wrong)
    wrong_candidates=copy.deepcopy(candidates)
    target=next(c for c in wrong_candidates if c['executed_FP32_embedding']['self_intersection_free'])
    target['executed_FP32_embedding']['count']=1
    with pytest.raises(Exception,match='source-chain'):
        selected_surface_rows(sources,wrong_candidates,selection)


def test_rehashed_archive_row_cannot_impersonate_complete_census(tmp_path):
    from numilab_human.model import REPOSITORY_ROOT
    source=REPOSITORY_ROOT/'Docs/media/whole-body-embeddedness-20260929/indexed-census.tar.gz'
    output=tmp_path/'changed.tar.gz'
    with tarfile.open(source,'r:gz') as old,tarfile.open(output,'w:gz') as new:
        for member in old.getmembers():
            data=old.extractfile(member).read()
            if member.name=='rows/001.json':data+=b' '
            member.size=len(data)
            new.addfile(member,io.BytesIO(data))
    with pytest.raises(Exception,match='row hash'):
        census_rows(output,BASE_SHA)
