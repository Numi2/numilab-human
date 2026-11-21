"""Recover the supported reference load at the recorded moving contact points."""
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def vec(r, prefix, suffix):
    return np.array([float(r[prefix+a+suffix]) for a in 'xyz'])
def key(r):
    return int(r['accepted_step']),int(r['contact_index']),int(r['source_geometry_index']),int(r['body_index'])

def calculate(run, reference):
    names=('resting-com-support-impulses.csv','resting-support-point-motion.csv','resting-com-momentum-diagnostic.csv','resting-local-bed-sections.csv','resting-coupled.csv')
    impulses, positions, whole, sections, phys=(rows(run/n) for n in names)
    argv=json.loads((run/'launch.json').read_text())['argv']
    dt=float(argv[argv.index('--muscle-step-seconds')+1])
    expected=int(argv[argv.index('--muscle-step-count')+1])
    if int(phys[-1]['step'])!=expected or int(whole[-1]['accepted_step'])!=expected:raise ValueError('Missing physical endpoint')
    for filename in (names[0],names[2],names[3],names[4]):
        if sha(run/filename)!=sha(reference/filename):raise ValueError(f'Observer changed the matched reference trace in {filename}')
    points={key(r):r for r in positions}
    if len(points)!=len(positions):raise ValueError('Duplicate accepted point observations')
    sections={(int(r['contact_index']),int(r['source_geometry_index'])):r for r in sections}
    com={int(r['accepted_step']):r for r in whole if (expected-int(r['accepted_step']))*dt<2}
    samples={step:[] for step in com}
    records={k:[] for k in sections}
    for r in impulses:
        step,region,source,body=key(r)
        if step not in com or r['row_class']!='external_support_contact':continue
        p=points[key(r)]
        if int(p['pre_step_index'])+1!=step:raise ValueError('Stale contact position')
        load=vec(r,'impulse_world_','_ns')/dt
        fn=float(r['normal_impulse_ns'])/dt
        xyz=vec(p,'point_world_','_m')
        if not np.all(np.isfinite([load,xyz])) or abs(load[2]-fn)>1e-5 or fn<0:raise ValueError('Invalid flat-bed traction')
        row=sections[region,source]
        elastic=fn*(float(row['skin_compliance_m_per_n'])+float(row['fat_compliance_m_per_n']))
        records[region,source].append((fn,*xyz,elastic))
        samples[step].append((load,xyz,fn))
    by_step=[];site=[]
    for step,r in sorted(com.items()):
        entries=samples[step]
        if len(entries)!=len(sections):raise ValueError(f'Support row coverage at {step}')
        force,xyz,normal=map(np.asarray,zip(*entries))
        center=np.average(xyz[:,:2],axis=0,weights=normal)
        c=vec(r,'com_','_m')
        # The stream measures the point used in the final substep of each segment.
        # COM is the accepted endpoint. The wrench about the world origin uses no COM estimate.
        net=force.sum(axis=0);moment=np.cross(xyz,force).sum(axis=0)
        by_step.append(dict(step=step,total_force_N=net.tolist(),world_moment_Nm=moment.tolist(),
            normal_CoP_xy_m=center.tolist(),CoP_to_endpoint_COM_xy_m=float(np.linalg.norm(center-c[:2]))))
    for (region,source),values in records.items():
        if len(values)!=len(by_step):raise ValueError('Section sampling changed')
        arr=np.asarray(values);f=arr[:,0];avg=float(f.mean())
        weighted=np.average(arr[:,1:4],weights=f,axis=0).tolist() if f.sum()>0 else None
        site.append(dict(region=region,source_geometry_index=source,mean_normal_N=avg,
            force_weighted_world_position_m=weighted,mean_reference_elastic_compression_m=float(arr[:,4].mean())))
    return dict(scope='Late sampled ground wrench of the 32-section native model. Each sample uses the physical substep impulse and its pre-step material point. The loaded section positions are transfer references, not applied forces on the full volume.',
        sha256={n:sha(run/n) for n in names},source_launch_sha256=sha(run/'launch.json'),dt_s=dt,
        accepted_s=float(phys[-1]['time_s']),terminal_pleural_pa=float(phys[-1]['pleural_pa']),samples=len(by_step),
        mean_ground_force_N=np.mean([r['total_force_N'] for r in by_step],axis=0).tolist(),
        max_CoP_to_endpoint_COM_xy_m=max(r['CoP_to_endpoint_COM_xy_m'] for r in by_step),
        source_controls_identical=True,sections=sorted(site,key=lambda r:-r['mean_normal_N']),sampled_wrenches=by_step)

if __name__=='__main__':
    run,ref,out=map(Path,sys.argv[1:])
    out.write_text(json.dumps(calculate(run,ref),indent=2,allow_nan=False)+'\n')
