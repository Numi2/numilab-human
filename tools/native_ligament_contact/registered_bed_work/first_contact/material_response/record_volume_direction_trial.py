"""Keep the measured effect of the interior-direction trial beside its matched control."""
from pathlib import Path
import json,shutil,csv,sys
from archive_cavity_trial import archive_files,sha

def run(job):
    read=lambda p:json.loads(p.read_bytes())
    base=job/'live-coupled-retry';test=job/'interior-direction-scene';trial=job/'interior-volume-direction'
    prev=read(job/'current-cardiac-certificate-recovery.json')
    ctrl=read(job/'registered-body-retry-recovery.json')
    rec=job/'interior-direction-recovery.json'
    if rec.exists():raise ValueError('Already published')
    files={k:job/k for k in ['retry_registered_body.py','build_interior_volume_direction.py','record_volume_direction_trial.py','registered-body-retry-recovery.json','current-cardiac-certificate-recovery.json']}
    for k in ['build.json','metal/NumiMatter.metallib','metal/src/metal/fem.metalinc']:files['build/'+k]=trial/k
    refs={}
    for p in test.rglob('*'):
        if not p.is_file():continue
        k='scene/'+str(p.relative_to(test));h=sha(p)
        prior=next((r for r in [ctrl,prev] if r['verified_member_sha256'].get(k)==h),None)
        if prior:refs[k]=dict(sha256=h,url=prior['url'],archive_sha256=prior['archive_sha256'])
        else:files[k]=p
    rr=job/'interior-direction-member-recovery.json';rr.write_text(json.dumps(refs,indent=2)+'\n');files[rr.name]=rr
    archive_files(files,'registered-interior-direction-'+sha(test/'launch.json')[:14]+'.tar.zst',rec,'Matched complete native M6 trial. Identical registered state, Newton search target inside unchanged physical material limits.')
    r=read(rec)
    out=job/'interior-direction-trial-delivery';e=out/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123';code=out/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response'
    e.mkdir(parents=True,exist_ok=True);code.mkdir(parents=True,exist_ok=True)
    for n in ['retry_registered_body.py','build_interior_volume_direction.py','record_volume_direction_trial.py']:shutil.copy2(job/n,code/n)
    def results(scene):
        logs=(scene/'stderr.log').read_text().splitlines()
        fail=read(scene/'failure.json') if (scene/'failure.json').exists() else None
        stdout=(scene/'stdout.log').read_text().splitlines()
        states=list(csv.DictReader((scene/'native-traces/resting-coupled.csv').open()))
        return dict(launch=sha(scene/'launch.json'),execution=read(scene/'execution.json'),failure=None if fail is None else {k:fail[k] for k in ['stand_completed_steps','matter_object_index','matter_failing_index','matter_diagnostics','common_status','respiratory_candidate_status']},coupled_times=[s['time_s'] for s in states],stages=[s for s in stdout if 'human_execution_stage=authoritative_segment_' in s],active_work=[json.loads(s.split('=',1)[1]) for s in logs if s.startswith('matter_fgmres_active_work=')])
    value=dict(device='Apple M6',control=results(base),trial=results(test),build=read(trial/'build.json'),archive={k:r[k] for k in ['url','archive_sha256','size_bytes']})
    (e/'registered-interior-direction-comparison.json').write_text(json.dumps(value,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
