"""Measure elastic response of the exact source and trial cavity on matched fields."""
import json,sys
from pathlib import Path
import numpy as np
from prepare_indentation import soft_response

def compare(material):
    a=np.load('cavity.npz');p=json.load(open('proposals.json'))[0];t=a['local'][np.searchsorted(a['rows'],p['old'])];new=np.searchsorted(a['ids'],p['new'])
    x,y,z=a['xyz'];G=json.loads(material.read_bytes())['materials']['16']['registered'];mu,bulk=G['source_mu_pa'],G['source_logJ_coefficient_pa'];vid=np.unique(t)
    fixed=np.array([[1.06,.012,0],[0,.98,.03],[-.018,0,1.01]])
    def response(T,pos):
        base=x[T[:,1:]]-x[T[:,:1]];F=np.linalg.solve(base,pos[T[:,1:]]-pos[T[:,:1]])
        det=np.linalg.det(base)/6;j=np.linalg.det(F);q=np.log(j)
        e=det@(.5*mu*((F*F).sum((1,2))-3)-mu*q+.5*bulk*q*q)
        forces,_=soft_response(x,pos,T,np.full(len(T),mu),np.full(len(T),bulk))(pos)
        f=np.zeros_like(x);np.add.at(f,T.ravel(),forces.reshape(-1,3));return e,f
    report={}
    for key,pos in [('affine',x@fixed+.11),('registered',y),('first_accepted',z)]:
        e,f=response(t,pos);E,F=response(new,pos)
        report[key]=dict(source_energy_J=e,trial_energy_J=E,relative_energy_change=(E-e)/e,relative_nodal_force_change=np.linalg.norm((F-f)[vid])/np.linalg.norm(f[vid]),max_nodal_force_delta_N=np.linalg.norm((F-f)[vid],axis=1).max(),net_internal_force_N=np.linalg.norm(F.sum(0)))
    Path('patch-response.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':compare(Path(sys.argv[1]))
