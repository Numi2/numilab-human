import json,sys
from pathlib import Path
import numpy as np
from preserve_muscle_cells import det,det_derivative
from material_context import sha

def run(asset,run):
    seed=np.random.default_rng(591)
    x=seed.normal(size=(4,3));t=np.array([[0,1,2,3]]);n=seed.normal(size=(4,3));q=seed.normal(size=4)
    eps=1e-4
    analytic=(det_derivative(x,n,t)@q)[0]
    numerical=((det(x+n*q[:,None]*.001*eps,t)-det(x-n*q[:,None]*.001*eps,t))/(2*eps*det(x,t)))[0]
    np.testing.assert_allclose(analytic,numerical,rtol=1e-6,atol=1e-9)
    baseline=json.loads((asset/'result.json').read_text());fit=json.loads((asset/'cell-fit.json').read_text())
    assert fit['source_pair_count']==fit['registered_pair_count']==0
    assert baseline['muscles']['29']['crossing_counts']['51005/30']==199
    for sid in ('29','30'):
        rows=fit['sides'][sid]['trials'][-1]
        assert all(row['inverted']==0 and row['self_pairs']==0 for row in rows)
        assert abs(rows[0]['volume_fraction']-1)<1e-7
    actual=json.loads((run/'actual.json').read_text());parity=json.loads((run/'parity.json').read_text())
    assert parity['exact_columns']==77 and not parity['different_columns'] and parity['trace_rows']==8
    for sample in ('entrance','accepted-8'):
        for row in actual[sample]['sides'].values():
            assert row['bilateral_pairs']==row['self_pairs']==row['inverted']==0
            assert row['minimum_determinant_ratio']>0
    for delta in json.loads((run/'geometry-recovery.json').read_text()):
        assert sha(run/delta['delta'])==delta['delta_sha256']
    print('Passed directional determinant, reference-volume, native registration, conditional mass and eight-step physics regression checks.')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
