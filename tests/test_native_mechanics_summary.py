from pathlib import Path
import pytest
from numilab_human.model import ImportError
from numilab_human.native_force_convergence import _native_metrics


@pytest.mark.parametrize('kind', ['myosim_articulated_marker_visual',
    'myosim_articulated_bodyparts_bone_visual', 'myosim_articulated_mechanics'])
def test_native_result_kind_is_preserved(tmp_path: Path, kind: str) -> None:
    path = tmp_path / 'native.txt'
    flags = (' rendering_performed=false visual_coverage_qualified=false'
             if kind.endswith('mechanics') else '')
    path.write_text(kind + '=ok' + flags + ' persistent_completed_steps=64\n')
    result = _native_metrics(path)
    assert result[kind] == 'ok'
    assert result['persistent_completed_steps'] == '64'


@pytest.mark.parametrize('text', [
    '',
    'myosim_articulated_marker_visual=failed',
    'myosim_articulated_marker_visual=ok x=1 x=2',
    'myosim_articulated_marker_visual=ok\nmyosim_articulated_marker_visual=ok',
    'myosim_articulated_marker_visual=ok\nmyosim_articulated_mechanics=ok',
    'myosim_articulated_marker_visual=ok myosim_articulated_mechanics=ok',
    'myosim_articulated_marker_visual=ok device="unterminated',
    'myosim_articulated_mechanics=ok',
    'myosim_articulated_mechanics=ok rendering_performed=true visual_coverage_qualified=false',
    'myosim_articulated_mechanics=ok rendering_performed=false visual_coverage_qualified=true',
])
def test_ambiguous_or_false_visual_claim_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / 'native.txt'
    path.write_text(text)
    with pytest.raises(ImportError):
        _native_metrics(path)
