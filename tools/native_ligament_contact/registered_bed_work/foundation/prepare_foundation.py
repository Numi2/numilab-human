"""Compose the existing two tissue laws and the measured foam curve at equal traction."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from material_context import sha
from fit_physical_support import BLUE,STRAIN,SOURCE

def prepare(opening,out):
    source=json.loads((opening/'launch.json').read_text()); law,=[Path(k) for k in source['pins'] if k.endswith('/curves.json')]
    if sha(law)!=source['pins'][str(law)]: raise ValueError('material changed')
    props=json.loads(law.read_text())['material']['materials']
    def stress(s,m):return m['mu_pa']*(1/s-s)-m['log_volume_coefficient_pa']*np.log(s)/s
    cap=min(stress(2/3,m) for m in props); p=np.linspace(0,cap,257)
    for corner in BLUE[1:-1]:p[np.argmin(abs(p-corner))]=corner
    foam=.2*np.interp(p,BLUE,STRAIN)
    tissue=np.array([[1-brentq(lambda s:stress(s,m)-pp,2/3,1.) for m in props] for pp in p])
    table=np.column_stack((tissue,foam,p)).astype('f4')
    if np.min(np.diff(table,axis=0))<=0 or not (p[-1]<BLUE[-1]):raise ValueError('nonmonotone material')
    col=opening/'registered-bed-columns.csv';g=np.genfromtxt(col,delimiter=',',names=True);g=g[g['area_m2']>0.]
    bal=next(csv.DictReader((opening/'resting-surface-audit.csv').open()))
    w=float(bal['represented_body_mass_kg'])*9.81
    stock=np.array([g['skin_m'],g['fat_m']]).T
    def shortening(k):return np.einsum('ij,ij->i',stock,table[k,:2])+table[k,2]
    limit=g['z_m']+stock@table[-1,:2]+table[-1,2]
    def forces(top):
        depth=np.maximum(top-g['z_m'],0.)
        lo=np.zeros(len(g),dtype='i4');hi=lo+256
        for _ in range(8):
            mid=(lo+hi)//2;sel=shortening(mid)<depth
            lo=np.where(sel,mid,lo);hi=np.where(sel,hi,mid)
        s=(depth-shortening(lo))/(shortening(hi)-shortening(lo))
        return g['area_m2']*(table[lo,3]+s*(table[hi,3]-table[lo,3]))
    max_force=forces(limit.min()).sum()
    # Choose a fixed horizontal top. Every subsequent load follows displacement.
    if max_force < w:raise ValueError(('insufficient calibrated footprint',float(max_force),w))
    top=brentq(lambda h:forces(h).sum()-w,float(g['z_m'].min()),float(limit.min()))
    f=forces(top); area=g['area_m2']; maxp=(f/area).max()
    moment=[f@(g['y_m']-float(bal['body_com_y_m'])),-f@(g['x_m']-float(bal['body_com_x_m']))]
    spec=dict(foam_height_m=.2,free_surface_z_m=top,backing_z_m=top-.2,pressure_limit_pa=cap,
              source=SOURCE,compression_fraction=STRAIN,pressure_pa=BLUE.tolist(),material=sha(law),opening=sha(col),
              body_weight_N=w,projected_skin_reaction_N=f.sum(),unbalanced_xy_moment_Nm=moment,
              peak_pressure_Pa=maxp,peak_foam_strain=float(np.interp(maxp,BLUE,STRAIN)),minimum_stock_margin_m=float(limit.min()-top),
              scope='Reduced skin/fat serial compression and a uniform independent-column foam foundation. The native skeleton is released freely. Bed thickness is a prototype parameter; viscoelastic hysteresis is uncalibrated.')
    def scalar(x):return float(x).hex()+'f'
    shader='constant float foamSurfaceZ='+scalar(np.float32(top))+';\nconstant float4 seriesBed[257]={\n'+',\n'.join('float4('+','.join(map(scalar,row))+')' for row in table)+'\n};\n'
    (out/'NumiHumanFoamFoundation.metalinc').write_text(shader)
    (out/'foundation.json').write_text(json.dumps(spec,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':prepare(*(Path(s).resolve() for s in sys.argv[1:]))
