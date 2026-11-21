"""Normal nominal stress for the exact isotropic energy used by both native materials."""
from pathlib import Path
import hashlib,json,re
import numpy as np

def load(path):
    text=path.read_text()
    text=re.sub(r'//[^\n]*','',text)
    compact=''.join(text.split())
    expr='energy=0.5*shear*(I1()-3-2*log(J()))+0.5*bulk*pow(log(J()),2);'
    if expr not in compact:raise ValueError('Material energy requires a different normal condensation')
    def coefficient(name):
        m=re.search(r'parameter\s+'+name+r'\s*:\s*MPa\s*=\s*([.\dEe+-]+)',text)
        if not m:raise ValueError(name)
        return float(m.group(1))*1e6
    J=float(re.search(r'minimum_J\s*=\s*([\d.]+)',text).group(1))
    if f'valid=J()-{J};' not in compact:raise ValueError('Validity gate differs')
    return dict(shear_pa=coefficient('shear'),log_bulk_pa=coefficient('bulk'),minimum_J=J,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def make_chart(material,mode,count=12001):
    mu,K=material['shear_pa'],material['log_bulk_pa']
    # Parameterize compression by axial stretch. No nominal tensile pressure.
    stretch=np.geomspace(1.,.002,count)
    if mode=='confined':a2=np.ones(count)
    elif mode=='traction_free':
        # For transverse traction zero, mu*(a^2-1)+K*log(a^2*stretch)=0.
        a2=np.ones(count)
        for _ in range(20):
            a2-=(mu*(a2-1)+K*np.log(a2*stretch))/(mu+K/a2)
        assert np.max(abs(mu*(a2-1)+K*np.log(a2*stretch)))<1e-6
    else:raise ValueError(mode)
    J=a2*stretch
    pressure=mu*(1/stretch-stretch)-K*np.log(J)/stretch
    energy=.5*mu*(2*a2+stretch**2-3-2*np.log(J))+.5*K*np.log(J)**2
    if np.any(np.diff(pressure)<=0) or energy.min() < -1e-10:raise ValueError('Bad constitutive chart')
    return pressure,1-stretch,J,energy

def compare_virtual_work(material,mode):
    p,c,J,e=make_chart(material,mode)
    numerical=np.gradient(e,c,edge_order=2)
    selected=(p>10)&(J>material['minimum_J']+.01)
    error=float((abs(p[selected]-numerical[selected])/p[selected]).max())
    if error>5e-4:raise ValueError(('Energy mismatch',error))
    return error
