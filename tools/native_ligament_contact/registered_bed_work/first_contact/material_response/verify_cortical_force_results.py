"""Compare native loads to the fully constrained path, both body moments and a finite energy derivative."""
from pathlib import Path
import json,sys
import numpy as np
from compile_connected_joint import digest

def run(parent,wide,fine,lib,out):
    material=json.loads((parent/'registered-ligaments/registration.json').read_text())['fibers'];result={}
    for label,p in [('quarter',wide),('fine',fine)]:
        q=json.loads((p/'reference.json').read_text())['records'];measured=np.fromfile(p/'measured.bin','<f4').reshape(len(q),3,4)
        expected=np.array([r['output'] for r in q]);diff=abs(measured-expected)
        selected=np.fromfile(p/'measured.bin.features.bin','<u4')
        if selected.shape!=(len(q),):raise ValueError('Missing GPU feature selection')
        wrong=[row['fiber']+'/'+row['probe'] for row,i in zip(q,selected) if i!=row['expected_range']]
        failures=[r['fiber']+'/'+r['probe'] for r,v in zip(q,measured) if not np.isfinite(v).all()]
        result[label]=dict(input=digest(p/'reference.json'),output=digest(p/'measured.bin'),trials=len(q),nonfinite=failures,wrong_features=wrong,maximum_force_error_n=float(np.nanmax(diff[:,0,:3])),maximum_length_error_m=float(np.nanmax(diff[:,0,3])),maximum_torque_error_nm=float(np.nanmax(diff[:,1:,:3])),maximum_tension_error_n=float(np.nanmax(diff[:,1,3])),native=(p/'native.log').read_text().strip())
    q=json.loads((fine/'reference.json').read_text())['records'];M=np.fromfile(fine/'measured.bin','<f4').reshape(len(q),3,4);table={(r['fiber'],r['probe']):np.asarray(r['output']) for r in q}
    for fiber in sorted({r['fiber'] for r in q}):
        m=material[fiber];k=m['stiffness_n_per_m'];L0=m['reference_length_m'];c4=.001;tm=.05
        def energy(q):
            strain=q[0,3]/L0-1.;u=np.clip(strain,0,tm);extra=max(0.,strain-tm)
            # Integral of the native exponential / linear extension law with stable near-slack arithmetic.
            return k*L0**2*((np.expm1(c4*u)-c4*u)/c4**2+extra*np.expm1(c4*tm)/c4+.5*extra**2*np.exp(c4*tm))
        ref=M[[i for i,s in enumerate(q) if s['fiber']==fiber and s['probe']=='reference'][0]]
        grad={}
        for mode,step,reference in [('translate',5e-6,-ref[0,:3]),('rotate',np.deg2rad(.005),ref[2,:3])]:
            force=-np.array([(energy(table[fiber,mode+'_1_'+a])-energy(table[fiber,mode+'_-1_'+a]))/(2*step) for a in 'xyz'])
            grad[mode]=dict(energy_derivative_force_or_torque=force.tolist(),native=reference.tolist(),max_absolute_error=float(np.max(abs(force-reference))))
        result[fiber]=grad
    result['metallib']=digest(lib)
    result['passed']=all(not result[k]['nonfinite'] and not result[k]['wrong_features'] and result[k]['maximum_length_error_m']<5e-8 and result[k]['maximum_force_error_n']<.01 and result[k]['maximum_torque_error_nm']<.001 for k in ('quarter','fine')) and all(v['translate']['max_absolute_error']<.02 and v['rotate']['max_absolute_error']<.002 for k,v in result.items() if '/' in k)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:result[k] for k in ['quarter','fine','passed']});return result['passed']
if __name__=='__main__':sys.exit(0 if run(*map(Path,sys.argv[1:])) else 1)
