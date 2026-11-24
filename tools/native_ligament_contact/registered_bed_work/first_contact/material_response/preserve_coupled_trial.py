"""Preserve the complete nonlinear trial, including the alternatives that failed to advance motion."""
import csv,json,re,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(out):
    rd=lambda p:json.loads(p.read_bytes())
    source=['register_limiting_star.py','test_registered_star.py','build_volume_path.py','check_determinant_path.py','identify_material_witness.py','fit_bonded_volume.py','relax_internal_registration.py','preserve_coupled_trial.py']
    files={n:out/n for n in source}
    for n in ['transfer-recovery.json','path-math.json','volume-path/build.json','volume-path/metal/NumiMatter.metallib','volume-path/metal/src/metal/fem.metalinc','registered-star/build.json','registered-star/app/numi-human-native','registered-star/actual-transport.npz','registered-star/current.node','registered-star/candidate.npz','registered-star/quality.json','registered-star/result.json','registered-star/layers.json','registered-star/bone-overlap/result.json','internal-fit/result.json','internal-fit/candidate.npz','bonded-fit/result.json','bonded-fit/candidate.npz']:
        files[n]=out/n
    def metrics(scene):
        err=(scene/'stderr.log').read_text().splitlines();trace=list(csv.DictReader((scene/'native-traces/resting-coupled.csv').open()))
        t=[float(s['time_s']) for s in trace]
        fail=rd(scene/'failure.json') if (scene/'failure.json').exists() else {}
        work=[json.loads(l.split('=',1)[1]) for l in err if l.startswith('matter_fgmres_active_work=')]
        final=[json.loads(l.split('=',1)[1]) for l in err if l.startswith('fem_newton_iterate=')]
        final=[x for x in final if x['stage']=='final_line_search']
        return dict(execution=rd(scene/'execution.json'),trace_records=len(t),start=min(t,default=None),end=max(t,default=None),accepted_steps=fail.get('stand_completed_steps'),diagnostics=fail.get('matter_diagnostics'),root_solver_work=work,final_search_fractions=[v['alpha'][0] for v in final],compiled_span=next((l for l in err if l.startswith('registered_material_span ')),None),launch=sha(scene/'launch.json'))
    records={}
    for k,folder in [('remesh','relative/scene'),('path_trial','path-scene'),('material_registration','registered-star/retry')]:
        s=out/folder;records[k]=metrics(s)
        if k=='remesh':continue
        for name in ['launch.json','execution.json','failure.json','stdout.log','stderr.log','native-traces/resting-coupled.csv']:
            p=s/name
            if p.exists():files[folder+'/'+name]=p
    for name in ['launch.json','execution.json','stderr.log']:
        files['registered-star/scene/'+name]=out/'registered-star/scene'/name
    cp=out/'coupled-trial-recovery.json'
    if not cp.exists():archive_files(files,'coupled-material-star-'+sha(out/'registered-star/retry/launch.json')[:16]+'.tar.zst',cp,'Matched whole-body scene after conservative tissue transfer, two pose counterfactuals, determinant-path trial and registered material-star intervention.')
    info=rd(cp);stage=out/'trial-delivery';e=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261124'
    code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';e.mkdir(parents=True,exist_ok=True);code.mkdir(parents=True,exist_ok=True)
    for s in source:shutil.copy2(out/s,code/s)
    shutil.copy2(cp,e/'coupled-material-star-recovery.json')
    span=records['material_registration']['compiled_span'];fit=rd(out/'registered-star/result.json');match=re.search(r'element_offset=(\d+) element_count=(\d+)',span or '')
    if not match or int(match[1])!=fit['provisional_offset'] or int(match[2])!=rd(out/'volume/transfer.json')['cells']:raise ValueError('Source cell not verified in shared native world')
    report=dict(registered_cell=fit['witness']-int(match[1]),registration=fit,quality=rd(out/'registered-star/quality.json'),initial_layer_intersections=rd(out/'registered-star/layers.json')['pair_counts'],runtime_comparison=records,determinant_trial_math=rd(out/'path-math.json'),recovery=dict(url=info['url'],archive_sha256=info['archive_sha256']))
    (e/'coupled-material-star.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
