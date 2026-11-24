"""Capture the decisive contact at the end of the complete 64-region solve."""
import json,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha
from preserve_contact_registration import native
def run(p):
    s=p/'limiting-contact64';control=p/'incident64-scene'
    trace=native(s);ref=native(control)
    if json.loads((s/'launch.json').read_bytes())['parent_launch']!=sha(control/'launch.json'):
        raise ValueError('Wrong matched source')
    raw=[]
    for line in (s/'stderr.log').open():
        if line.startswith('fem_deformable_limit_trace='):
            data=json.loads(line.split('=',1)[1]);raw.append(dict(root=data['root'],iteration=data['iteration'],rows=data['recorded_count'],minima=len(data['minimum_lane_bounds'])))
    files={'previous.json':p/'incident64-recovery.json'}
    for n in ['preserve_contact_outcome.py','preserve_contact_registration.py','trace_contact_floor.py']:
        files[n]=p/n
    for n in ['launch.json','execution.json','failure.json','stderr.log','stdout.log','native-traces/resting-coupled.csv']:
        f=s/n
        if f.is_file():files[n]=f
    receipt=p/'contact64-recovery.json'
    if not receipt.exists():archive_files(files,'coupled-feasibility-witness-'+sha(s/'launch.json')[:16]+'.tar.zst',receipt,
        'Fresh full-scene contact trace on the tested wide shared-node registration, preserving both solver and physical acceptance behavior.')
    stage=p/'contact64-delivery';code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response'
    evidence=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261124'
    code.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    for n in ['preserve_contact_outcome.py','preserve_contact_registration.py']:shutil.copy2(p/n,code/n)
    report=dict(raw_trace_summary=raw,control_diagnostics=ref['diagnostics'],traced_run=trace,conclusion=dict(
        accepted_steps=trace['accepted_steps'],initial_support='One outer node within bed-contact range. CoM horizontal offset 87.27 mm. No balanced starting checkpoint.',
        anatomy='All starting cell intervals pass on the retained material pose. Nonzero cortical overlap remains.'))
    (evidence/'coupled-feasibility-witness.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    shutil.copy2(receipt,evidence/receipt.name)
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
