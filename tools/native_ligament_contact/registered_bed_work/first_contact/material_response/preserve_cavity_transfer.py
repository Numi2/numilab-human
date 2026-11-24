"""Publish measured constitutive, mass and native replay evidence for the connected repair."""
import json, shutil, sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(job,out):
    result=out/'relative';rd=lambda f:json.loads(f.read_bytes())
    archive=out/'transfer-recovery.json'
    scripts=['repair_loaded_cavities.py','transfer_cavities.py','rebuild_cavity_support.py','balance_cavity_credits.py','rebuild_cavity_ledger.py','audit_final_credits.py','install_cavity_mesh.py','run_cavity_scene.py','trace_first_root.py','preserve_cavity_transfer.py']
    files={s:out/s for s in scripts}
    for k in ['fit/patches.npz','fit/ancestry.json','fit/result.json','volume/overlay.npz','volume/transfer.json','binding/tables.json','binding/build.json','binding/contact/result.json','binding/contact/material-contact.npz','binding/cortex/result.json','binding/cortex/registered-cortical-facets.npz','material-p1/result.json','material-p1/current-paired-volume-state.json','relative/packets/result.json','relative/flows/result.json','relative/native-replay/result.json','relative/native-replay/launch.json','relative/volume/native-import.json','relative/final-credits.json','relative/scene/launch.json','relative/scene/execution.json','matched-root/launch.json','matched-root/failure.json','matched-root/execution.json','matched-root/stderr.log','scene/stderr.log','scene/execution.json']:
        files[k]=out/k
    for base in ['relative/flows','relative/packets','relative/scene/native-traces']:
        directory=out/base
        for p in directory.iterdir():
            if p.name.endswith('.npz') or ('resting-coupled.csv'==p.name or 'registered-volume-fractional-mass-' in p.name):files[str(p.relative_to(out))]=p
    for k in ['relative/scene/failure.json','relative/scene/stdout.log','relative/scene/stderr.log']:
        if (out/k).exists():files[k]=out/k
    for p in [job/'conditioned-native-recovery.json',job/'deep-fat-material-scene-recovery.json',job/'persistent-witness-recovery.json']:
        files['inputs/'+p.name]=p
    if not archive.exists():
        archive_files(files,'registered-cavity-transfer-'+sha(result/'scene/launch.json')[:16]+'.tar.zst',archive,'Connected reference and starting-pose cavity edits, overlap map, donor flows, custody and matched complete native scene result.')
    dest=out/'delivery';code=dest/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';docs=dest/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261124'
    code.mkdir(parents=True,exist_ok=True);docs.mkdir(parents=True,exist_ok=True)
    for s in scripts:shutil.copy2(out/s,code/s)
    compact=rd(out/'volume/transfer.json');compact.pop('cavities')
    def motion(scene):
        execution=rd(scene/'execution.json')
        failed=rd(scene/'failure.json') if (scene/'failure.json').exists() else {}
        stderr=(scene/'stderr.log').read_text().splitlines()
        def entry(k):
            return next((json.loads(l.split('=',1)[1]) for l in stderr if l.startswith(k+'=')),{})
        back=entry('fem_backtracking_trials');probes=back.get('trials',[])
        samples=[json.loads(l.split('=',1)[1]) for l in stderr if l.startswith('fem_newton_iterate=')]
        finals=[v['alpha'][0] for v in samples if v['stage']=='final_line_search']
        return dict(execution=execution,accepted_steps=failed.get('stand_completed_steps'),physical_status=failed.get('matter_code'),physical_diagnostics=failed.get('matter_diagnostics'),active_linear_work=entry('matter_fgmres_active_work'),final_selected_alpha=finals[-1] if finals else None,last_limiting_global_tet=next((v['first_bad_tet'] for v in reversed(probes) if v.get('first_bad_tet',4294967295)!=4294967295),None),launch=sha(scene/'launch.json'))
    reg=rd(out/'fit/result.json');r=rd(archive)
    report=dict(device='Apple M6',recovery=dict(url=r['url'],sha256=r['archive_sha256']),geometry=compact,stretch_by_material={k:{n:v[n] for n in ['start','remaining']} for k,v in reg['materials'].items()},material_import=rd(result/'volume/native-import.json'),per_role_quadrature_audit=rd(result/'final-credits.json'),native_ledger=rd(result/'native-replay/result.json')['receipts'],maximum_thickness_change_m=rd(out/'binding/build.json')['maximum_sample_thickness_change_m'],control=motion(out/'matched-root'),coupled=motion(result/'scene'),first_full_import_failure='One carrier failed the 1e-10 relative-only closure. Corrected and revalidated local feasible flow with its true source mass.')
    (docs/'connected-cavity-transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(archive,docs/'connected-cavity-transfer-recovery.json')
    print(report['coupled'],flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
