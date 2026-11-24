"""Replay measured support traction against the full tissue's confined uniaxial law."""
import csv,json,sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from compile_connected_joint import digest


def compressed_stretch(pressure,mu,lam):
    if pressure == 0:
        return 1.
    def stress(stretch):
        return mu*(stretch-1./stretch) + lam*np.log(stretch)/stretch
    return brentq(lambda stretch:stress(stretch)+pressure,1e-8,1.,xtol=1e-14)


def compare(j,out):
    p=next(parent for parent in j.parents if (parent/'motion-16/step-3999.receipt.json').exists())
    case=p/'motion-16';fit=p/'registration-strain.json'
    params=json.loads(fit.read_bytes())['materials']
    launch=json.loads((case/'launch.json').read_bytes())
    dt=float(launch['argv'][launch['argv'].index('--muscle-step-seconds')+1])
    receipt=json.loads((case/'step-3999.receipt.json').read_bytes())
    if receipt['physical_endpoint']!='accepted':raise ValueError('Missing accepted load')
    step=receipt['accepted_step'];sections_file=case/'resting-local-bed-sections.csv';impulse_file=case/'resting-com-support-impulses.csv'
    with sections_file.open() as f:sections={int(r['contact_index']):r for r in csv.DictReader(f)}
    samples=[]
    with impulse_file.open() as f:
        for r in csv.DictReader(f):
            if int(r['accepted_step'])!=step+1:continue
            row=int(r['contact_index']);sec=sections[row];area=float(sec['projected_area_m2'])
            if int(sec['source_geometry_index'])!=int(r['source_geometry_index']) or area<=0:raise ValueError('Unmatched contact area')
            force=float(r['normal_impulse_ns'])/dt
            if force<0:raise ValueError('Tensile bed load')
            if force==0:continue
            pressure=force/area
            data=dict(row=row,body=int(r['body_index']),area_m2=area,native_normal_force_N=force,mean_pressure_Pa=pressure)
            for name,role in (('skin','15'),('fat','16')):
                a=params[role]['registered'];mu=a['source_mu_pa'];lam=a['source_logJ_coefficient_pa']
                stretch=compressed_stretch(pressure,mu,lam)
                data[name]=dict(column_modulus_Pa=float(sec[name+'_modulus_pa']),confined_tangent_Pa=lam+2*mu,
                    unconfined_tangent_Pa=mu*(3*lam+2*mu)/(lam+mu),full_confined_stretch=stretch)
            samples.append(data)
    if len({r['row']for r in samples})!=len(samples) or not samples:raise ValueError('Unmatched load rows')
    result=dict(initial_accepted_step=step,step=step+1,step_s=dt,registered_parameters=digest(fit),source_launch=digest(case/'launch.json'),
            source_impulses=digest(impulse_file),source_areas=digest(sections_file),receipt=digest(case/'step-3999.receipt.json'),
            recorded_force_N=sum(s['native_normal_force_N'] for s in samples),
            load_on_rows_below_J_035_N=sum(s['native_normal_force_N']for s in samples if s['fat']['full_confined_stretch']<.35),
            minimum_confined_fat_stretch=min(s['fat']['full_confined_stretch']for s in samples),
            rows=samples,scope='Matched accepted support rows; static homogeneous laterally confined uniaxial probe using mean pressure on each region. No alternative dynamics or applied force.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:v for k,v in result.items()if k in ('step','recorded_force_N','minimum_confined_fat_stretch','load_on_rows_below_J_035_N')})
if __name__=='__main__':compare(*(Path(a).resolve()for a in sys.argv[1:]))
