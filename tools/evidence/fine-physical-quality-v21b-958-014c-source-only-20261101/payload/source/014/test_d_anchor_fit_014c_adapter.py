import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/"batch_inner_endpoint_quotient_014c_strict_D_volume_root.py"
spec=importlib.util.spec_from_file_location("d014c_adapter",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_d_fit_anchor_key_adapter():
    import numpy as np
    x=np.asarray([.1,.2,.3],np.float32)
    assert np.array_equal(m.fitted_anchor_world("D",{"anchor_point_f32":x}),x)
    try:m.fitted_anchor_world("D",{"anchor_world_m_f32":x})
    except ValueError as e:assert "anchor_point_f32" in str(e)
    else:raise AssertionError("D accepted the wrong helper return field")
def test_legacy_I_fit_key_is_unchanged():
    import numpy as np
    x=np.asarray([.1,.2,.3],np.float32)
    assert np.array_equal(m.fitted_anchor_world("I",{"anchor_world_m_f32":x}),x)
