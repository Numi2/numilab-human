from copy import deepcopy
import json
from pathlib import Path

import pytest

from numilab_human import right_eye_laterality_cleanup as cleanup
from numilab_human.model import ImportError as HumanImportError


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'Docs/media/right-choroid-laterality-20260929/candidate/right-choroid-laterality-candidate.nhanatomy'
POSE = ROOT/'Docs/media/right-choroid-laterality-20260929/native/myosim-fullbody-articulated-bodyparts-bones-source-torso-anatomy-focus-body-20.torso-anatomy-poses.json'
PUBLIC = ROOT/'Docs/media/right-eye-laterality-20260929/candidate'


def test_published_right_eye_source_face_exclusions_are_exact():
    payload = PUBLIC/cleanup.PAYLOAD_NAME
    manifest = json.loads((PUBLIC/cleanup.MANIFEST_NAME).read_text())
    audited = cleanup.audit_output(PARENT,POSE,payload,manifest)
    assert audited['passed']
    assert audited['all_first_599_surfaces_retained_byte_identically']
    assert audited['three_geometry_failures_remain']

    altered = deepcopy(manifest)
    altered['candidates'][0]['excluded_original_source_face_ids'].append(0)
    with pytest.raises(HumanImportError, match='exclusion identity'):
        cleanup.audit_output(PARENT,POSE,payload,altered)
