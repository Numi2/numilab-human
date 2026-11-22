"""Balance total load and two moments using a continuous geometric correction to the initial contact patch."""
import csv,json,shutil,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from fit_native_bed_barrier_load import force
from screen_registered_dense_posture import load_geometry
from qualify_loaded_pose_registration import anchors
from recover_material_thickness import audit
from untangle_registered_donor_field import differentials
def run(full,placement,seed,band,out):
    x,tgt,owner,t,tag,outer,_,pins=load_geometry(full,placement);src=json.loads((seed/'result.json').read_text())
    assert src['source']==pins and src['failed']==0
    q=x+np.load(seed/'displacement.npy');j=json.loads(band.read_text());dt=j['dt_s'];w=j['band_m']
    env=json.loads((full/'launch.json').read_text())['environment'];m=np.empty(len(t))
    for name,mat in [('SKIN',15),('FAT',16)]:
        key='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_'+name+'_CHILD_MASS_CSV'
        m[tag==mat]=np.loadtxt(env[key],skiprows=1,delimiter=',',usecols=2)
    mass=np.bincount(t.ravel(),weights=np.repeat(m/4,4),minlength=len(x))
    com=np.array(j['accounted_com_xy_m']);weight=j['weight_N'];ids=np.flatnonzero(outer&(q[:,2]<3*w))
    arm=q[:,:2]-com;scale=np.maximum(abs(arm[ids]).max(0),.01)
    basis=np.column_stack((np.ones(len(q)),arm/scale))
    # A shared affine displacement across the patch preserves rigid in-plane
    # coordinates. Its C1 taper vanishes before the nearest deep enthesis.
    fall=np.clip((q[:,2]-.001)/.009,0.,1.);blend=1-fall*fall*(3.-2.*fall)
    basis*=blend[:,None]*1.e-6
    def loads(g):
        active=(g<w);f=np.zeros_like(g)
        f[active]=force(g[active],mass[ids][active],dt,w)
        return f
    def residual(par):
        g=q[ids,2]+basis[ids]@par;f=loads(g)
        return np.r_[f.sum()/weight-1,(f@arm[ids])/(weight*scale)]
    opt=least_squares(residual,np.zeros(3),xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=60)
    if not opt.success or max(abs(residual(opt.x)))>1e-7:raise ValueError(('load balance',opt.message,residual(opt.x)))
    y=q.copy();y[:,2]=(q[:,2]+basis@opt.x).astype('f4');active=np.flatnonzero(outer&(y[:,2]<w))
    assert y[:,2].min()>1e-9 and np.isin(active,ids).all()
    rf=loads(y[ids,2]);cpf=rf@y[ids,:2]/rf.sum()
    assert abs(rf.sum()/weight-1)<1e-4 and np.linalg.norm(cpf-com)<1e-5
    lock=np.array([row[1] for row in anchors(env).values()]);assert np.array_equal(y[lock],q[lock])
    ref=np.concatenate([differentials(x,t[i:i+32768])[0] for i in range(0,len(t),32768)])
    J,S,D,bad=audit(y,t,ref);out.mkdir(exist_ok=False)
    if bad.any():raise ValueError(('invalid emitted cells',bad.sum(),np.flatnonzero(bad)[:10]))
    np.save(out/'displacement.npy',y-x);np.savez_compressed(out/'construction-boundary.npz',node=lock,axis=np.ones((len(lock),3),bool),value=y[lock],trial_contact=active)
    exact=force(y[active,2],mass[active],dt,w)
    b=dict(j,nodes=active.tolist(),gaps=y[active,2].tolist(),minimum_m=float(y[active,2].min()),maximum_m=float(y[active,2].max()),
           predicted_normal_N=float(rf.sum()),individual_force_N=exact.tolist(),node_mass_kg=mass[active].tolist(),center_of_pressure_xy_m=cpf.tolist(),
           geometric_parameters_um=opt.x.tolist(),maximum_adjustment_m=float(abs(y[:,2]-q[:,2]).max()),force_error_N=float(rf.sum()-weight))
    (out/'load.json').write_text(json.dumps(b,indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(dict(source=pins,failed=0,active=len(active),minimum_relative_volume=float(J.min()),maximum_relative_volume=float(J.max()),
        selected=src['selected'],note='Full native force and interval checks required.'),indent=2)+'\n')
    print(len(active),rf.sum(),cpf,b['maximum_adjustment_m'],flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
