"""Compare the native material law, external impulse and registered endpoint at matched poses."""
import csv,json,re,sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from material_context import sha

def main(root):
    def read(p):return json.loads((root/p).read_text())
    spec=read('foundation.json');case=read('admitted-32/terminal-bed-certificate.json')
    cfg=read('admitted-32/launch.json');law,=[Path(k) for k in cfg['pins'] if k.endswith('/curves.json')]
    if cfg['pins'][str(law)]!=sha(law) or sha(law)!=spec['material']:raise ValueError('changed source material')
    mat=json.loads(law.read_text())['material']['materials']
    pfile=root/'owner/NumiHumanFoamFoundation.metalinc'
    rows=re.findall(r'float4\(([^)]+)\)',pfile.read_text())
    tab=np.array([[float.fromhex(c.removesuffix('f')) for c in r.split(',')] for r in rows])
    if tab.shape!=(257,4) or np.any(np.diff(tab,axis=0)<=0):raise ValueError('incomplete or nonmonotone series table')
    probes=np.r_[tab[:,3],np.diff(tab[:,3])*.371+tab[:-1,3]]
    errors=[]
    for p in probes:
        comp=[]
        for m in mat:
            def res(s):return m['mu_pa']*(1/s-s)-m['log_volume_coefficient_pa']*np.log(s)/s-p
            comp.append(1-brentq(res,.66,1.))
        exact=[*comp,spec['foam_height_m']*np.interp(p,spec['pressure_pa'],spec['compression_fraction'])]
        interp=[np.interp(p,tab[:,3],tab[:,i]) for i in range(3)]
        errors.append(np.abs(np.subtract(exact,interp)))
    maxerr=np.max(errors,axis=0)
    if maxerr[0]>1e-7 or maxerr[1]>4e-6 or maxerr[2]>2e-8:raise ValueError(('material interpolation',maxerr))
    def impulse(run):
        rows=list(csv.DictReader((root/run/'resting-com-support-impulses.csv').open()))
        return sum(float(r['normal_impulse_ns']) for r in rows)/.002,sum(float(r['normal_impulse_ns'])>0 for r in rows)
    f,count=impulse('admitted-32');old,oldcount=impulse('foam-17')
    if case['checked_columns']!=49294 or case['count_beyond_material_domain'] or abs(case['normal_reaction_at_endpoint_N']-f)>.5:raise ValueError('contact reaction or endpoint changed')
    if not np.isclose(spec['body_weight_N'],read('support-wrench.json')['gravity_N']):raise ValueError('different body mass')
    def rejected(run):
        f=read(run+'-source/witness.json')
        if f['depth_m']<=f['allowed_depth_m'] or read(run+'/execution.json')['exit_code']==0:raise ValueError('invalid failure classification')
        return f
    res=dict(device='Apple M6',material_interpolation_max_absolute=dict(dermis_strain=maxerr[0],fat_strain=maxerr[1],foam_m=maxerr[2]),
             mass_kg=spec['body_weight_N']/9.81,force_N_at_opening=spec['projected_skin_reaction_N'],opening_xy_moment_Nm=spec['unbalanced_xy_moment_Nm'],
             legacy_plane_gate=dict(accepted_steps=17,measured_rows=oldcount,impulse_over_dt_N=old,first_rejection=rejected('foam-32')),
             pressure_gate=dict(accepted_capture=case['step'],measured_rows=count,impulse_over_dt_N=f,reconstructed_pressure_integral_N=case['normal_reaction_at_endpoint_N'],limit_crossings_at_capture=case['count_beyond_material_domain'],first_rejection=rejected('motion-16')),
             step_s=.002,scope='Short coupled reduced-contact release. Contact force at an accepted endpoint and the preceding step impulse can differ during motion. Static outer-volume wrench feasibility does not establish native dense FEM equilibrium.')
    (root/'qualification.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
