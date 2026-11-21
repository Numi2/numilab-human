import numpy as np
from local_pressure_equilibrium import ColumnLaw
from material_normal_response import compare_virtual_work
def run():
    xy=np.array([[-.1,-.15],[.11,-.17],[-.12,.17],[.09,.12]])
    geom=dict(xyz=np.c_[xy,[0,.0002,.0001,.00005]],area_m2=np.array([.007,.006,.009,.008]),layer_heights_m=np.array([[.003,.012],[.002,.015],[.004,.008],[.003,.011]]),missing=np.empty((0,5)))
    mats=[dict(shear_pa=13255.,log_bulk_pa=658333.,minimum_J=.35),dict(shear_pa=551.7,log_bulk_pa=5333.,minimum_J=.35)]
    for mode in ('confined','traction_free'):
        for m in mats:compare_virtual_work(m,mode)
        model=ColumnLaw(geom,*mats,mode,np.zeros(2))
        q=np.array([.0012,.0002,-.0003])
        w,k,p,j,valid=model.evaluate(q);assert valid and (p>0).all()
        units=np.eye(3)*1e-10
        fd=np.column_stack([(model.evaluate(q+d)[0]-model.evaluate(q-d)[0])/2e-10 for d in units])
        np.testing.assert_allclose(k,fd,rtol=2e-5,atol=.0001)
        inactive=model.evaluate(np.array([-.01,0.,0.]))
        assert not np.any(inactive[0]) and not np.any(inactive[1])
        qc,row=model.equilibrium(1.,q)
        assert row['qualified_within_sampled_material_and_coverage'] and row['maximum_balance_error']<1e-8
        overload=model.evaluate(np.array([.1,0.,0.]))
        assert not overload[-1]
    print('Material virtual work, contact activation, tangent, pressure balance and invalid compression passed.')
if __name__=='__main__':run()
