"""Publish reproducible registrations and the physical witnesses that govern native admission."""
import csv
import json
from pathlib import Path
import shutil
import sys
from archive_cavity_trial import archive_files, sha

def read(path):
    return json.loads(path.read_bytes())

def native(scene):
    failure=read(scene/'failure.json') if (scene/'failure.json').exists() else {}
    accepted=list(csv.DictReader((scene/'native-traces/resting-coupled.csv').open()))
    work=[]; selected=[]; floors=[]
    for line in (scene/'stderr.log').open():
        name, _, value=line.partition('=')
        if name=='matter_fgmres_active_work':
            work.append(json.loads(value))
        elif name=='fem_newton_iterate':
            d=json.loads(value)
            if d['iteration']==15 or d['stage']=='final_line_search':
                selected.append(dict(iteration=d['iteration'],stage=d['stage'],alpha=d['alpha'][0],
                                     object_alpha=d['object_line_search'][0][0],residual=d['fgmres'][0]['nonlinear'][1]))
        elif name=='fem_deformable_limit_trace':
            d=json.loads(value)
            minima=sorted(d.get('minimum_lane_bounds',[]),key=lambda x:x['minimum_bound'])
            records=d.get('contacts',[])
            if minima:
                winner=minima[0]
                records=[r for r in records if r['contact_slot']==winner['contact_slot']]
            elif records:
                records=sorted(records,key=lambda r:r['lane_bound_after'])
                records=[r for r in records if r['lane_bound_after'] in d['line_search_alpha']]
            floors.append(dict(minima=minima[:3],records=records[:1],active_counts=d['active_counts'],iteration=d['iteration']))
    return dict(launch_sha256=sha(scene/'launch.json'),execution=read(scene/'execution.json'),
                accepted_steps=failure.get('stand_completed_steps'),trace_rows=len(accepted),
                diagnostics=failure.get('matter_diagnostics'),work=work,limits=selected,physical_contact=floors)

def run(out):
    sources=['register_witness_batch.py','protect_cortical_interface.py','trace_contact_floor.py','audit_registered_release.py','inspect_witness_neighborhood.py','rebuild_limiting_boundary_cells.py','preserve_contact_registration.py']
    files={n:out/n for n in sources+['coupled-trial-recovery.json','held-cortex.npz','registered-release-witnesses.json','physical-neighborhood.json','limiting-cavities/result.json','limiting-cavities/patches.npz','limiting-cavities/ancestry.json']}
    reports={'limiting_material': read(out/'physical-neighborhood.json'),'candidate_cavity_changes':read(out/'limiting-cavities/result.json')}
    for name in ('witness-batch','cortical-fit'):
        for s in ['result.json','quality.json','layers.json','bone-overlap/result.json','candidate.npz']:
            files[f'{name}/{s}']=out/name/s
        reports[name]={k:read(out/name/(k+'.json')) for k in ('result','quality')}
        reports[name]['layer_pairs']=read(out/name/'layers.json')['pair_counts']
    for s in ['current.node','build.json','app/numi-human-native']:
        files['cortical-fit/'+s]=out/'cortical-fit'/s
    for name in ('registered-star/retry','cortical-fit/scene','contact-witness'):
        scene=out/name
        reports[name]=native(scene)
        if name.startswith('registered-star'):continue
        for s in ('execution.json','failure.json','stderr.log','stdout.log','launch.json','native-traces/resting-coupled.csv'):
            if (scene/s).exists():files[name+'/'+s]=scene/s
    receipt=out/'contact-registration-recovery.json'
    if not receipt.exists():
        archive_files(files,'cortical-contact-registration-'+sha(out/'contact-witness/launch.json')[:16]+'.tar.zst',receipt,
            'Matched full-scene trials, bone-constrained and rejected registration, complete contact witnesses and source. Parent recovery identifies the input volume and donor masses.')
    staging=out/'contact-delivery';sub=staging/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response'
    evidence=staging/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261124'
    sub.mkdir(exist_ok=True,parents=True);evidence.mkdir(exist_ok=True,parents=True)
    for s in sources:shutil.copy2(out/s,sub/s)
    shutil.copy2(receipt,evidence/receipt.name)
    (evidence/'cortical-contact-registration.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    run(Path(sys.argv[1]).resolve())
