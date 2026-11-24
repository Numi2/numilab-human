"""Check static support feasibility using native mass moments and the registered contact surface."""
import json,re,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from compile_connected_joint import digest

def run(job,fit,out):
    scene=fit/'scene';lines=(scene/'stdout.log').read_text().splitlines()
    ledger=json.loads(next(l.split('=',1)[1] for l in lines if l.startswith('initial_gravity_total=')))
    installed=next(l for l in lines if l.startswith('registered_loaded_material_install=PASS'))
    val=lambda name:float(re.search(r' '+name+r'=([^ ]+)',installed)[1])
    plane,margin=val('fixed_bed_offset_m'),val('ipc_width_m')
    com=np.asarray(ledger['center_m']);weight=ledger['mass_kg']*9.81
    b=np.load(job/'registered-boundary/registered-boundary.npz')
    raw=np.loadtxt(fit/'current.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    vertices=b['O_node_joint'];x=raw[vertices];gap=x[:,2]-plane
    if (gap<0).any() or (gap<margin).sum()!=int(val('outer_contact_nodes')):raise ValueError('Native contact classification differs')
    active=x[gap<margin];dis=np.linalg.norm(active[:,:2]-com[:2],axis=1)
    order=np.argsort(gap)
    def supported(k):
        if k<3:return False
        try:h=ConvexHull(x[order[:k],:2])
        except QhullError:return False
        return bool(np.all(h.equations[:,:2]@com[:2]+h.equations[:,2]<=0))
    lo=0;hi=len(x)
    if not supported(hi):raise ValueError('Center of mass outside entire footprint')
    while hi-lo>1:
        mid=(hi+lo)//2
        if supported(mid):hi=mid
        else:lo=mid
    # A rigid placement can remove the overturning moment only on a supporting
    # plane whose contact polygon contains the vertical projection of the COM.
    hull=ConvexHull(x);best=None
    for corners,plane4 in zip(hull.simplices,hull.equations):
        normal=plane4[:3]
        if normal[2]>=0:continue
        points=x[corners];a=points[0];u,v=points[1:]-a
        dist=(com-a)@normal;point=com-dist*normal
        mat=np.column_stack((u,v));uv=np.linalg.lstsq(mat,point-a,rcond=None)[0]
        fractions=np.r_[1-uv.sum(),uv]
        if np.min(fractions)<-1e-11:continue
        angle=float(np.rad2deg(np.arccos(np.clip(-normal[2],-1,1))))
        if best is None or angle<best['tilt_degrees']:
            best=dict(tilt_degrees=angle,support_nodes=vertices[corners].tolist(),gravity_fractions=fractions.tolist(),
                      vertical_after_rotation=(-normal).tolist(),com_above_support_m=-dist)
    result=dict(native_launch=digest(scene/'launch.json'),position_sha256=digest(fit/'current.node'),gravity=ledger,
                bed_z_m=plane,activation_width_m=margin,outer_nodes=len(x),contact_nodes=vertices[gap<margin].tolist(),
                com_to_only_contact_horizontal_m=float(dis[0]) if len(dis)==1 else None,
                gravity_moment_about_only_contact_Nm=float(weight*dis[0]) if len(dis)==1 else None,
                undeformed_gap_needed_for_com_in_footprint_m=float(gap[order[hi-1]]),candidate_rigid_support=best,
                interpretation='Static support geometry only. Reaction magnitudes from the contact constitutive law and deformable equilibrium still require native solution.')
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*(Path(a).resolve()for a in sys.argv[1:]))
