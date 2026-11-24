"""Publish conservative connected registration comparisons and their coupled native result."""
from pathlib import Path
import json,sys,shutil,csv
from archive_cavity_trial import sha,archive_files

def run(j,t):
    read=lambda p:json.loads(p.read_bytes())
    scene=t/'scene';exec_result=read(scene/'execution.json')
    runs=['connected-loaded-strain','feasible-loaded-strain','conditioned-loaded-strain','cortex-conditioned-strain']
    names=['measure_initial_tissue_strain.py','isolate_registered_stretch.py','fit_connected_loaded_strain.py','fit_feasible_loaded_strain.py','hold_regressed_cortical_interfaces.py','audit_loaded_registration_fit.py','test_conditioned_physical_body.py','audit_fixed_registration_strain.py','measure_registered_layer_stretch.py','record_connected_registration.py']
    files={n:j/n for n in names};files.update({n:j/n for n in ['prescribed-stretch-bounds.json','registered-layer-stretch.json','registered-initial-strain.json','registered-stretch-patch/map.json','registered-stretch-patch/interior-patch.npz','held-loaded-cortex/map.json','held-loaded-cortex/nodes.npy','persistent-witness-recovery.json','storage/restorable-closed-viewer-outputs.jsonl']})
    for p in (j/'storage').glob('historical-viewer-products-*.json'):files[str(p.relative_to(j))]=p
    for n in runs:
        for rel in ['result.json','candidate-positions.npz','interval-audit.json','loaded-layers.json','bone-overlap/result.json']:
            p=j/n/rel
            if p.exists():files[str(p.relative_to(j))]=p
    refs={};prev=[read(j/n) for n in ['persistent-witness-recovery.json','interior-direction-recovery.json','registered-body-retry-recovery.json','current-cardiac-certificate-recovery.json']]
    for p in t.rglob('*'):
        if not p.is_file():continue
        k=str(p.relative_to(t))
        if k.startswith('app/') and p.suffix not in ('.inc',) and p.name!='numi-human-native':continue
        if k.startswith('app/') and p.suffix=='.inc' and p.name not in ('NumiHumanLoadedSkinPoseData.inc','NumiHumanCorticalSkinBinding.inc'):continue
        dig=sha(p);r=next((r for r in prev if r['verified_member_sha256'].get(k)==dig),None)
        if r:refs[k]=dict(sha256=dig,url=r['url'],archive_sha256=r['archive_sha256'])
        else:files['native/'+k]=p
    lookup=j/'conditioned-native-prior-files.json';lookup.write_text(json.dumps(refs,indent=2)+'\n');files[lookup.name]=lookup
    rc=j/'conditioned-native-recovery.json'
    if not rc.exists():archive_files(files,'connected-strain-native-'+sha(scene/'launch.json')[:14]+'.tar.zst',rc,'Bilateral connected material registration with native executable, variant inputs, layer and all 185 cortical checks, control and reprojection. Shape and motion qualification are distinct.')
    receipt=read(rc);out=j/'conditioned-native-delivery'
    code=out/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';e=out/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    code.mkdir(parents=True,exist_ok=True);e.mkdir(parents=True,exist_ok=True)
    for n in names:shutil.copy2(j/n,code/n)
    raw=read(t/'bone-overlap/result.json')['results'];base=read(j/'interval-certified-scene/bone-audit/result.json')['results']
    overlap={k:dict(before_mm3=sum(v[k] for v in base.values()),after_mm3=sum(v[k] for v in raw.values()),positive_bones=sum(v[k]>1e-6 for v in raw.values())) for k in ['skin','fat']}
    summary=read(t/'result.json');native=read(scene/'failure.json') if (scene/'failure.json').is_file() else None
    logs=(scene/'stderr.log').read_text().splitlines()
    record=dict(device='Apple M6',build=read(t/'build.json'),launch_sha256=sha(scene/'launch.json'),initial_stretch=read(j/'registered-initial-strain.json'),initial_anatomical_location=read(j/'registered-stretch-patch/map.json'),fitted=summary,prescribed_stretch=read(j/'prescribed-stretch-bounds.json'),surface_stretch=read(j/'registered-layer-stretch.json'),severe_whole_before=summary['history'][0]['severe']+summary['unaffected_severe'],severe_whole_after=summary['history'][-1]['severe']+summary['unaffected_severe'],interval_gate=read(t/'interval-audit.json'),tissue_layers=read(t/'loaded-layers.json')['pair_counts'],cortical_overlap=overlap,execution=exec_result,partial_work=[json.loads(s.split('=',1)[1]) for s in logs if s.startswith(('matter_solver_domain_norms=','matter_fgmres_active_work='))],native_failure={k:native[k] for k in ['matter_object_index','matter_failing_index','matter_diagnostics','stand_completed_steps','common_status','common_residual','respiratory_accepted_status']} if native else None,physiology_times=[s.get('time_s') for s in csv.DictReader((scene/'native-traces/resting-coupled.csv').open())],native_end=[s for s in (scene/'stdout.log').read_text().splitlines() if s.startswith(('resting_integrated_body=','human_execution_stage=authoritative_segment_','resting_initialization=','resting_bed_contact_owner='))],archive={k:receipt[k] for k in ['url','archive_sha256','size_bytes']})
    linear=[json.loads(s.split("=",1)[1]) for s in logs if s.startswith("fem_true_residual_cycle=")]
    record["linear_and_nonlinear_progress"]=[]
    for iteration in sorted(set(r["newton_iteration"] for r in linear)):
        samples=[r["environments"][0] for r in linear if r["newton_iteration"]==iteration]
        useful=[r for r in samples if r["rhs_l2"]>0]
        if useful:
            record["linear_and_nonlinear_progress"].append(dict(iteration=iteration,rhs_l2=useful[0]["rhs_l2"],best_fresh_linear_l2=min(r["true_l2"] for r in useful)))
    (e/'connected-registered-tissue-and-native-motion.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
