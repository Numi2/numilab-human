"""Evaluate the actual release material columns under a rigid normal translation using measured GPU witnesses."""
import csv,json,struct
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from compile_connected_joint import digest
def evaluate(folder):
    cfg=json.load(open(folder/'launch.json'));s=folder/'native-traces';wfile=s/'resting-support-selected-skin-witness.csv';afile=s/'resting-local-bed-sections.csv'
    entries=[r for r in csv.DictReader(wfile.open())if int(r['pre_dynamics_step_index'])==0 and int(r['accepted_step'])==1];areas=list(csv.DictReader(afile.open()))
    unique={}
    for r in entries:
        idx=int(r['region'])
        if unique.setdefault(idx,r)!=r:raise ValueError('Conflicting witnesses in the same release')
    entries=[unique[k] for k in sorted(unique)]
    if len(entries)!=32 or len(areas)!=32:raise ValueError('Missing native sections or witnesses')
    packet=Path(cfg['argv'][cfg['argv'].index('--support-contact-payload')+1]);h=struct.unpack_from('<8s4I32s7f',packet.read_bytes());n=np.array(h[9:12]);n/=np.linalg.norm(n);o=np.array(h[6:9])
    pts=[];compliance=[];stock=[];witness=[]
    for i,(r,a) in enumerate(zip(entries,areas)):
        if (int(r['region']),int(a['contact_index']))!=(i,i)or r['source_geometry_index']!=a['source_geometry_index']or any(int(r[k])for k in ('region_error','global_error','nonfinite_jacobian_count')):raise ValueError('Material section and GPU witness disagree')
        pts.append([float(r['selected_point_world_'+xyz+'_m'])for xyz in 'xyz'])
        compliance.append(sum(float(a[k+'_compliance_m_per_n'])for k in ['skin','fat']));stock.append(sum(float(a[k+'_thickness_m'])for k in ['skin','fat']));witness.append(int(r['selected_skin_vertex_local']))
    pts=np.array(pts);g=(pts-o)@n;c=np.array(compliance);stock=np.array(stock);grav=next(json.loads(l.partition('=')[2])for l in (folder/'stdout.log').read_text().splitlines()if l.startswith('initial_gravity_total='))
    # The native articulated world's gravity is -9.81 m/s² along Z for this registered scene.
    momentum=list(csv.DictReader((s/'resting-com-momentum-diagnostic.csv').open()));dt=float(cfg['argv'][cfg['argv'].index('--muscle-step-seconds')+1])
    row=momentum[0];gravity_impulse=np.array([float(row['gravity_impulse_'+axis+'_ns'])for axis in 'xyz']);accel=gravity_impulse/(dt*float(row['segment_steps'])*float(row['total_body_mass_kg']));normal_target=float(-accel@n*grav['mass_kg'])
    force=lambda dz:np.maximum(dz-g,0)/c
    bound=float(np.min(g+stock));limit=sum(force(bound));delta=brentq(lambda x:np.sum(force(x))-normal_target,float(g.min())-1e-8,float(g.max()+max(stock)+np.max(c)*normal_target));loads=force(delta);normals=loads[:,None]*n
    center=np.array(grav['center_m']);torque=np.sum(np.cross(pts-center,normals),axis=0)
    return dict(launch=digest(folder/'launch.json'),bed=digest(packet),witness=digest(wfile),sections=digest(afile),initial_gap_min_m=float(g.min()),inclination_deg=float(np.degrees(np.arccos(n[2]))),normal=n.tolist(),initial_load_n=float(force(0).sum()),required_normal_load_n=normal_target,full_stock_bound_m=bound,load_at_stock_bound_n=float(limit),translation_toward_plane_m=float(delta),max_fraction_of_stock=float(np.max(np.maximum(delta-g,0)/stock)),within_stock=bool(delta<bound),active_regions=int((loads>0).sum()),net_bed_moment_about_com_nm=torque.tolist(),normal_load_centroid_world_m=np.average(pts,axis=0,weights=loads).tolist(),weighted_loads_n=loads.tolist(),scope='Existing 32 condensed normal section reactions with the original skin witnesses and zero starting velocity. A normal-load balance alone does not certify joint or angular equilibrium.')
if __name__=='__main__':
 r=evaluate(Path('bed-engagement'));Path('registered-bed-release-loads.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k in ['required_normal_load_n','full_stock_bound_m','load_at_stock_bound_n','translation_toward_plane_m','max_fraction_of_stock','active_regions','net_bed_moment_about_com_nm']})
