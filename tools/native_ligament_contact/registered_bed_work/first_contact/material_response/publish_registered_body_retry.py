"""Publish the reproduced coupled step result and independent recovery references."""
import csv,json,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(job,scene):
    read=lambda p:json.loads(p.read_bytes())
    execution=read(scene/'execution.json')
    cfg=read(scene/'launch.json')
    old=job/'certified-heart-runtime/scene'
    if cfg['parent_launch']!=sha(old/'launch.json'):raise ValueError('Launch mismatch')
    prev=read(job/'current-cardiac-certificate-recovery.json')
    files={'retry_registered_body.py':job/'retry_registered_body.py','publish_registered_body_retry.py':Path(__file__).resolve()}
    refs={}
    for p in scene.rglob('*'):
        if not p.is_file():continue
        key='scene/'+str(p.relative_to(scene))
        if prev['verified_member_sha256'].get(key)==sha(p):
            refs[key]={'sha256':sha(p),'url':prev['url'],'archive_sha256':prev['archive_sha256']}
        else:files[key]=p
    ext=scene/'prior-recovery.json'
    ext.write_text(json.dumps(refs,indent=2)+'\n')
    files['prior-recovery.json']=ext
    journal=job/'storage/restorable-closed-viewer-outputs.jsonl'
    if journal.exists():files['storage/restorable-closed-viewer-outputs.jsonl']=journal
    for p in (job/'storage').glob('historical-viewer-products-*.json'):files['storage/'+p.name]=p
    receipt=job/'registered-body-retry-recovery.json'
    if not receipt.exists():archive_files(files,'registered-body-retry-'+sha(scene/'launch.json')[:14]+'.tar.zst',receipt,'Native all-tissue release, exact commands, traces, failures and preverified archive references.')
    r=read(receipt)
    out=job/'registered-retry-delivery'
    code=out/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';code.mkdir(parents=True,exist_ok=True)
    dest=out/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123';dest.mkdir(parents=True,exist_ok=True)
    for n in ['retry_registered_body.py','publish_registered_body_retry.py']:shutil.copy2(job/n,code/n)
    trace=scene/'native-traces/resting-coupled.csv'
    states=list(csv.DictReader(trace.open())) if trace.exists() else []
    logs=(scene/'stdout.log').read_text().splitlines()+(scene/'stderr.log').read_text().splitlines()
    markers=('resting_initialization=','resting_FEM_bed_release_initialization=','human_contact_audit_layout ','human_compact_immutable_contacts ','resting_vascular_external_pressure=','resting_fem_skin_bed_source=','persistent_stand_failure','physical_step_failure','material_cell_witness ','persistent_stand_progress','resting_tissue_shared_runtime=','resting_coupled_','human_execution_stage=authoritative_segment')
    data={'device':'Apple M6','execution':execution,'parent_launch_sha256':cfg['parent_launch'],'launch_sha256':sha(scene/'launch.json'),'model_binary':cfg['pins'][cfg['argv'][0]],'coupled_rows':len(states),'coupled_times_seconds':[s.get('time_s') for s in states],'diagnostics':[s for s in logs if s.startswith(markers)],'stderr_tail':(scene/'stderr.log').read_text().splitlines()[-25:],'stdout_tail':(scene/'stdout.log').read_text().splitlines()[-12:],'recovery':{k:r[k] for k in ['url','size_bytes','archive_sha256']}}
    (dest/'registered-coupled-native-retry.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
