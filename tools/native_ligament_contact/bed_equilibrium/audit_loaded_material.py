"""Evaluate current source material laws under the native bed rows' measured normal loads."""
import csv,json,sys
from pathlib import Path
import numpy as np
from material_normal_response import make_chart
from material_context import sha
def run(root,contract,out):
    def read(n):
        with (root/n).open() as f:return list(csv.DictReader(f))
    cfg=json.loads((root/'launch.json').read_text());args=cfg['argv'];dt=float(args[args.index('--muscle-step-seconds')+1])
    sec=read('resting-local-bed-sections.csv');obs=read('resting-com-support-impulses.csv');export=read('resting-com-momentum-diagnostic.csv')
    ids=[int(s['contact_index']) for s in sec];assert ids==list(range(len(ids)))
    frames=[int(s['accepted_step']) for s in export];map_frame={s:i for i,s in enumerate(frames)}
    forces=np.full((len(frames),len(ids)),np.nan)
    for r in obs:
        i=map_frame[int(r['accepted_step'])];j=int(r['contact_index'])
        if r['row_class']!='external_support_contact' or int(r['source_geometry_index'])!=int(sec[j]['source_geometry_index']) or not np.isnan(forces[i,j]):raise ValueError('Contact row identity mismatch')
        forces[i,j]=float(r['normal_impulse_ns'])/dt
    if np.any(~np.isfinite(forces)) or np.min(forces)<0:raise ValueError('Incomplete force raster')
    total=[float(r['normal_force_last_physical_step_n']) for r in export]
    err=float(np.max(np.abs(forces.sum(1)-total)))
    if err>.005:raise ValueError(('Root-vs-segment impulse mismatch',err))
    mats=json.loads(contract.read_text())['materials']
    def vector(k):return np.array([float(r[k]) for r in sec])
    area=vector('projected_area_m2');press=forces/area;last=np.array(frames)*dt>frames[-1]*dt-2
    res=dict(accepted_pose_samples=len(frames),rows=len(ids),section_mean_nominal_pressure_peak_pa=float(press.max()),reported_row_force_crosscheck_max_n=err,final_two_seconds_mean_total_force_n=float(forces[last].sum(1).mean()),measured_max_single_section_load_n=float(forces.max()),modes={})
    for mode in ('confined','traction_free'):
        data=[];invalid=np.zeros_like(press,dtype=bool);capacity=None
        for layer,mat in zip(('skin','fat'),mats):
            pp,compression,J,energy=make_chart(mat,mode)
            if np.max(press)>pp[-1]:raise ValueError('Measured load outside inverse stress chart')
            chosen=np.interp(press,pp,J);axial=np.interp(press,pp,compression)
            invalid|=chosen<=mat['minimum_J']
            if layer=='fat' and J[-1]<mat['minimum_J']:capacity=float(np.interp(mat['minimum_J'],J[::-1],pp[::-1]))
            data.append(dict(layer=layer,minimum_J=float(chosen.min()),maximum_mean_thickness_loss_fraction=float(axial.max()),invalid_section_samples=int((chosen<=mat['minimum_J']).sum()),invalid_last_two_seconds=int((chosen[last]<=mat['minimum_J']).sum()),threshold=mat['minimum_J'],max_qualification_pressure_pa=float(np.interp(mat['minimum_J'],J[::-1],pp[::-1])) if J[-1]<mat['minimum_J'] else None))
        res['modes'][mode]=dict(materials=data,invalid_witness_rows=np.unique(np.nonzero(invalid)[1]).tolist(),late_mean_force_on_invalid_sections_n=float(np.where(invalid,forces,0.)[last].sum(1).mean()))
        bodies={int(row['body_index']):row['body_name'] for row in read('resting-body-motion-partition.csv')}
        bodyrow={int(r['contact_index']):int(r['body_index']) for r in obs}
        parts=[]
        for j in np.unique(np.nonzero(invalid)[1]):
            parts.append(dict(contact_index=int(j),body_index=bodyrow[j],body_name=bodies[bodyrow[j]],source_geometry_index=sec[j]['source_geometry_index'],registered_projected_area_cm2=area[j]*1e4,peak_pressure_pa=float(press[:,j].max()),minimum_area_for_peak_fat_pressure_cm2=float(forces[:,j].max()/capacity)*1e4))
        res['modes'][mode]['contact_rows_exceeding_material_threshold']=parts
    res['sources']={p.name:sha(p) for p in (contract,root/'launch.json',root/'resting-local-bed-sections.csv',root/'resting-com-support-impulses.csv',root/'resting-com-momentum-diagnostic.csv',root/'resting-body-motion-partition.csv')}
    res['scope']='Counterfactual uniform nominal pressure over each entire registered row area from observed accepted forces. The fine mesh has different geometry, and row means do not resolve local contact concentrations.'
    out.write_text(json.dumps(res,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
