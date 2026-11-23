"""Solve the external support wrench on the registered material skin at rest."""
import csv, ctypes, hashlib, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from probe_registered_layers import measure
from material_context import sha

SOURCE = 'https://www.wimed.it/wp-content/uploads/2024/08/WI_ST_SYSTAM_MATERASSO-VISCOFLEX-SERIE-P161M.pdf'
STRAIN = [0., .25, .40, .65]
BLUE = np.array([0., 1500., 2050., 3900.])
BLUE_TOL = np.array([0., 300., 350., 800.])
CORE = np.array([0., 2700., 3400., 7300.])

class Footprint:
    def __init__(self, lib, v, f, com, pitch):
        self.v, self.f, self.com, self.pitch = v, np.ascontiguousarray(f, dtype='i4'), com, pitch
        self.fn = ctypes.CDLL(str(lib)).bed_boundary_columns
        d = np.ctypeslib.ndpointer(dtype='float64',flags='C_CONTIGUOUS')
        i = np.ctypeslib.ndpointer(dtype='int32',flags='C_CONTIGUOUS')
        self.fn.argtypes = [d,ctypes.c_int,i,ctypes.c_int,d,ctypes.c_int,i,d,d]
        self.fn.restype = ctypes.c_int
    def sample(self, angle):
        rot = Rotation.from_rotvec([*angle,0.]).as_matrix()
        v = np.ascontiguousarray((self.v-self.com) @ rot.T + self.com)
        lo,hi = np.floor(v[:,:2].min(0)/self.pitch), np.ceil(v[:,:2].max(0)/self.pitch)
        axes=[(np.arange(l,h)+.5)*self.pitch for l,h in zip(lo,hi)]
        xy=np.ascontiguousarray(np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2))
        n=len(xy); index=np.empty(n,'i4'); z=np.empty(n); b=np.empty((n,3))
        if self.fn(v,len(v),self.f,len(self.f),xy,n,index,z,b):raise ValueError('invalid registered boundary')
        hit=index>=0
        cloud=np.c_[xy[hit],z[hit]]
        if abs((b[hit,:,None]*v[self.f[index[hit]]]).sum(1)-cloud).max()>1.e-8:raise ValueError('face ancestry mismatch')
        return cloud
    def solve(self, height, pressures, weight):
        # Iterate the exact world-vertical raycast after rotating the actual body.
        # Continued slope is for locating the root. Only measured-domain solutions pass.
        angles=np.zeros(2); drop=.08
        def pressure(e):
            return np.interp(e,STRAIN,pressures) + np.maximum(e-.65,0.)*(pressures[3]-pressures[2])/.25
        for step in range(12):
            points=self.sample(angles); rel=points[:,:2]-self.com[:2]
            def wrench(x):
                strain=(x[0]-points[:,2]-x[1]*rel[:,1]+x[2]*rel[:,0])/height
                load=self.pitch**2*pressure(strain)
                return np.array([load.sum()-weight,load@rel[:,1],-load@rel[:,0]])
            residual=wrench([drop,0.,0.])
            if abs(residual).max()<.04:break
            change=least_squares(wrench,[drop,0.,0.],bounds=([0.,-.15,-.15],[.4,.15,.15]),xtol=1e-13,ftol=1e-13,gtol=1e-12)
            drop=change.x[0]; angles+=change.x[1:]
        strain=np.maximum((drop-points[:,2])/height,0.); active=strain>0.
        return dict(height_m=height,pressures_pa=pressures.tolist(),pitch_m=self.pitch,rotation_vector_xy_rad=angles.tolist(),descent_m=drop,
                    reaction_N=residual[0]+weight,moments_about_com_Nm=residual[1:].tolist(),iterations=step+1,
                    peak_strain=float(strain.max()),peak_pressure_pa=float(pressure(strain).max()),outside_measured_range_area_m2=float(np.count_nonzero(strain>.65)*self.pitch**2),
                    contact_area_m2=float(active.sum()*self.pitch**2),admissible=bool(abs(residual).max()<.04 and strain.max()<=.65))

def main(base,case,opening,out):
    row=next(csv.DictReader((opening/'resting-surface-audit.csv').open()))
    if int(row['step'])!=0:raise ValueError('missing opening COM')
    com=np.array([float(row['body_com_'+k+'_m']) for k in 'xyz']); weight=float(row['represented_body_mass_kg'])*9.81
    def grab(k,v,f):
        if k=='O': np.savez(out/'opening-outer.npz',vertices=v,triangles=f)
    model=measure(case,base/'bed-load-paths.json',opening,grab)
    mesh=np.load(out/'opening-outer.npz'); results=[]
    for pitch,height,law in [(.006,.142,BLUE),(.003,.142,BLUE),(.006,.142,BLUE-BLUE_TOL),(.006,.142,BLUE+BLUE_TOL),(.006,.142,CORE),(.006,.2,BLUE)]:
        s=Footprint(base/'bed_boundary_columns.dylib',mesh['vertices'],mesh['triangles'],com,pitch)
        result=s.solve(height,law,weight);results.append(result);print(result,flush=True)
    result=dict(definition='Static distributed external reaction of an idealized uniform foam on the exposed lower surface. Body joint and internal tissue coordinates fixed. Nominal thickness is the product total; component layouts, shear and hysteresis are not replicated.',source=SOURCE,source_boundary=model['physical_reference'],pose=model['target_pose_receipt'],native_mass_and_com=sha(opening/'resting-surface-audit.csv'),com_m=com.tolist(),gravity_N=weight,mesh=sha(out/'opening-outer.npz'),cases=results)
    (out/'support-wrench.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main(*(Path(x).resolve() for x in sys.argv[1:]))
