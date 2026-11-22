"""Measure directional stretches and constitutive stress on the installed, mapped tissue elements."""
import json
from pathlib import Path
import sys
import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha

from supported_material_admission import installed_topology, material_state as state

def audit(root):
    run=root/'release-balance-both-four';x,y,t,tags,pins=inputs(run);x=x.astype(float)
    launch=json.loads((run/'launch.json').read_text());env=launch['environment']
    proposal=Path(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    if sha(proposal)!=env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']:raise ValueError('Changed tetrahedral connectivity')
    t,tags=installed_topology(t,tags,env)
    prop,mat=materials(root/'release-balance-both/native-owner');records={}
    files={
        'paired_material_loaded':'paired-fat-material/current.node','two_compression_cells_loaded':'release-balance/current.node',
        'third_cell_repaired_loaded':'release-balance-both/current.node','batch_loaded_rejected':'registered-footprint-batch/current.node',
        'paired_material_accepted_3':'paired-fat-three-resumed/physical-accepted.node','third_cell_repaired_accepted_4':'release-balance-both-four/physical-accepted.node'}
    for name,relative in files.items():
        path=root/relative;pin=sha(path)
        if 'accepted' in name:
            ep=json.loads((path.parent/'verified-endpoint.json').read_text())
            if ep['physical_nodes_sha256']!=pin:raise ValueError('Unverified accepted pose')
        else:
            ids=json.loads((path.parent/'native-owner/identities.json').read_text())
            if ids['loaded_pose_override']['sha256']!=pin:raise ValueError('Unknown loaded pose')
        q=np.loadtxt(path,skiprows=1,usecols=(1,2,3));sv=np.empty((len(t),3));peak=np.empty(len(t));vol=np.empty(len(t))
        for i in range(0,len(t),32768):
            sl=slice(i,i+32768);c=t[sl];X=x[c]
            vol[sl]=np.linalg.det(X[:,1:]-X[:,:1])/6
            _,sv[sl],peak[sl]=state(X,q[c],prop[tags[sl]-15])
        bad=(sv[:,0]>1.5)|(sv[:,2]<2/3)
        extreme=(sv[:,0]>3)|(sv[:,2]<1/3)
        stats={}
        for tag,label in ((15,'dermis'),(16,'adipose')):
            sel=tags==tag;ids=np.flatnonzero(sel)
            ix=int(ids[np.argmax(sv[sel,0])]);low=int(ids[np.argmin(sv[sel,2])]);high=int(ids[np.argmax(peak[sel])])
            stats[label]=dict(cells=int(sel.sum()),reference_volume_m3=float(vol[sel].sum()),outside_existing_registration_stretch_envelope=int((sel&bad).sum()),
               affected_reference_volume_m3=float(vol[sel&bad].sum()),severe_distortion_count=int((sel&extreme).sum()),
               outside_native_component_stress_cap=int(np.sum(peak[sel]>5.e6)),max_stress_Pa=float(peak[high]),maximum_stress_row=high,
               max_stretch=float(sv[ix,0]),max_stretch_row=ix,min_stretch=float(sv[low,2]),min_stretch_row=low,
               upper_stretch_p99_by_cell=float(np.quantile(sv[sel,0],.99)),lower_stretch_p01_by_cell=float(np.quantile(sv[sel,2],.01)))
        records[name]=dict(pose_sha256=pin,regions=stats,extreme_rows=np.flatnonzero(extreme).tolist() if extreme.sum()<=256 else None)
    return dict(reference=pins,remesh_sha256=sha(proposal),material_sha256=mat,phases=records,
        scope='Principal stretches from the actual physical reference to each loaded or accepted state. The 2/3 to 1.5 envelope is the existing engineering registration check. The 1/3 to 3 interval reports severe distortion. Neither is a clinical acceptance range. Stress uses the current two constitutive laws and maximum absolute first Piola component.')

if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1]).resolve()),indent=2,allow_nan=False))
