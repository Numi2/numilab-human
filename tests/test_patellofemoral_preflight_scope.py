import json
from pathlib import Path

import pytest

from tools.verify_patellofemoral_preflight_scope import parse


def test_native_preflight_scope_rejects_forged_pair_localization():
    receipt = json.loads((Path(__file__).parents[1] /
        'Docs/media/patellofemoral-preflight-scope-20260930/receipt.json').read_text())
    output = receipt['executions']['left']['stdout']
    measured = parse(output)
    assert measured['patellar_pair']['samples'] == '11586'

    forged = output.replace('slave_surface=FMC_@_PTC_ContactFaces samples=11586',
                            'slave_surface=FMC_@_PTC_ContactFaces samples=1', 1)
    with pytest.raises(RuntimeError, match='broad femoral sample identity'):
        parse(forged)

    forged = output.replace('peak_normal_force_n=1900.6557478842808',
                            'peak_normal_force_n=1', 1)
    with pytest.raises(RuntimeError, match='duplicated prescribed-closure response'):
        parse(forged)
