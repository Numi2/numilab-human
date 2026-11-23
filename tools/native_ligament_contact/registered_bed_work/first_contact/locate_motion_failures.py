"""Trace every failed endpoint material check to its installed cell and source ancestry."""
import json
import sys
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from supported_material_admission import geom, installed_topology, material_state, stress_limits


def locate(run, out):
    e=json.loads((run/'launch.json').read_text())
    p=e['environment'];x,y,t,role,_=inputs(run)
    t,role=installed_topology(t,role,p)
    props,files=materials(Path(e.get('compiled_material_owner',Path(e['argv'][0]).parent)))
    limits=np.asarray(stress_limits(files))
    z=np.loadtxt(run/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    cp=Path(p['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    if sha(cp)!=p['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']:raise ValueError('Changed cell composition')
    with np.load(cp) as c:
        retain=c['retained_input_cell_rows'];off=c['replacement_parent_offsets'];source=c['replacement_parent_cell_rows']
    label=Path(p['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_CHILD_LABELS'])
    if sha(label)!=p['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_CHILD_LABELS_SHA256']:raise ValueError('Changed donor labels')
    lab=np.loadtxt(label,delimiter=',',skiprows=1,usecols=(0,1,2,3),dtype='i4')
    found=[];inverted=0
    for i in range(0,len(t),32768):
        tt=t[i:i+32768];roles=role[i:i+len(tt)]
        X=x[tt].astype(float);Y=y[tt].astype(float);Z=z[tt]
        dt,quality=geom(z,tt)
        J,s,stress=material_state(X,Z,props[roles-15])
        bad=(dt<=1e-18)|(quality<1e-5)|(J<.4)|(J>2.5)|(s[:,0]>1.5)|(s[:,-1]<2/3)|(stress>limits[roles-15])
        inverted+=int((dt<=0).sum())
        for j in np.flatnonzero(bad):
            k=i+j
            parents=retain[k:k+1] if k<len(retain) else source[off[k-len(retain)]:off[k-len(retain)+1]]
            if np.any(lab[parents,1]!=roles[j]):raise ValueError('Material ancestry changed')
            E=X[j,1:]-X[j,:1];G=np.linalg.inv(E).T
            normals=np.r_[G,-G.sum(0,keepdims=True)]
            disp=Z[j]-Y[j];v,q=geom(x.astype(float),tt[j:j+1])
            found.append(dict(installed_row=int(k),source_rows=parents.tolist(),material=int(roles[j]),
                donors=np.unique(lab[parents,2]).tolist(),nodes=tt[j].tolist(),center_m=X[j].mean(0).tolist(),
                minimum_reference_altitude_m=float(1/np.linalg.norm(normals,axis=1).max()),quality_start=float(q[0]),
                quality_end=float(quality[j]),volume_end_m3=float(dt[j]),J=float(J[j]),principal_stretches=s[j].tolist(),
                peak_stress_Pa=float(stress[j]),max_relative_displacement_m=float(np.linalg.norm(disp-disp.mean(0),axis=1).max())))
    out.write_text(json.dumps(dict(frame=sha(run/'physical-accepted.node'),inverted=inverted,failures=found),indent=2)+'\n')


if __name__=='__main__':locate(*(Path(v).resolve() for v in sys.argv[1:]))
