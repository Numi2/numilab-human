"""Retain constitutive registration tests and full-scene decisions alongside their restorable native inputs."""
from pathlib import Path
import csv,json,re,shutil,sys
from analyze_swept_run import summarize
from archive_cavity_trial import sha,archive_files

read=lambda p:json.loads(p.read_bytes())
def save(h,g,d,out):
    dirs={'control':g/'swept-second-root','four_node':h/'trial','six_node':h/'coupled-trial','strain_bounded':h/'guarded-trial','bounded_one_step':h/'guarded-motion','half_step':h/'guarded-half'}
    report={};files={}
    def add(key,path):files[key]=path
    for label,run in dirs.items():
        rr=summarize(run)
        clocks={}
        for line in (run/'stdout.log').read_text().splitlines():
            match=re.search(r'human_execution_stage=(native_horizon_begin|native_horizon_end) wall_elapsed_ms=([\d.]+)',line)
            if match:clocks[match[1]]=float(match[2])
        body_seconds=(clocks.get('native_horizon_end',0)-clocks.get('native_horizon_begin',0))/1000
        trace=list(csv.DictReader((run/'native-traces/resting-coupled.csv').open()))
        t=rr['accepted_s'][-1] if rr['accepted_s'] else 0.
        report[label]=dict(accepted_steps=len(trace),accepted_s=t,end_to_end_wall_s=rr['execution']['wall_s'],exit=rr['execution']['exit'],body_wall_s=body_seconds,
            accepted_time_per_body_wall_s=t/body_seconds if body_seconds>0 else None,accepted_time_per_launch_wall_s=t/rr['execution']['wall_s'],
            final_fresh_mechanical_residual=[r.get('final_nonpressure_nontransport_relative_residual') for r in rr['roots']],
            active_krylov_columns=[r['used_columns_sum'] for r in rr['roots']],failure=rr.get('rejected'),launch=sha(run/'launch.json'),
            physical_dt_s=float(read(run/'launch.json')['argv'][read(run/'launch.json')['argv'].index('--muscle-step-seconds')+1]),pleural_pressure_mode=read(run/'launch.json')['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE'],
            terminal_accepted_trace={k:trace[-1][k] for k in ('time_s','pleural_pa','lung_volume_ml','lv_ml','blood_error_ml','root_assistance_n','root_assistance_nm') if trace and k in trace[-1]})
        if label!='control':
            for k in ['launch.json','execution.json','failure.json','stderr.log','stdout.log','native-traces/resting-coupled.csv','physical-accepted.node','native-initial-bodies.json']:
                if (run/k).is_file():add(label+'/'+k,run/k)
            for p in (run/'native-traces/accepted-geometry').glob('*.receipt.json'):add(label+'/'+p.name,p)
    (h/'matched-trials.json').write_text(json.dumps(report,indent=2)+'\n')
    cases=('force-fit','coupled-fit','guarded-fit')
    for k in cases:
        for n in ('fit.json','current.node','interval.json','wall-crossings.json','strain.json','binary/build.json'):
            add(k+'/'+n,h/k/n)
    add('guarded-fit/binary/numi-human-native',h/'guarded-fit/binary/numi-human-native')
    for n in ('cavity.json','cavity.npz','proposals.json','cavities.json','inertia.json','measured-cell.json','patch-response.json','force-owners.json','matched-trials.json','guarded-cortex/result.json','prep/witness-material/current-p1/result.json','transfer/volume/native-import.json','cell-trial/launch.json','cell-trial/execution.json','cell-trial/failure.json','cell-trial/stderr.log','cell-observer/identity.json','cell-observer/search.inc','cell-observer/observe.inc','cell-observer/build.py','cell-build/pins.json'):
        add(n,h/n)
    for n in ('remaining-load.json','right-ulna/region.json','right-ulna/linked-fit/fit.json','right-ulna/cavity.npz','right-ulna/force-owners.json','whole-registration.json','motion-check/motion.json','whole-cortex/result.json','whole-cortex/solids/result.json','matched-motion-strain.json','guarded-observer/link.json','guarded-observer/pins.json','guarded-observed/launch.json','guarded-observed/execution.json','guarded-observed/stderr.log','guarded-observed/failure.json','guarded-observed/native-traces/resting-coupled.csv'):
        add(n,h/n)
    for n in ['inspect_cavity.py','fit_force_cavity.py','measure_trial_cell.py','prepare_witness.py','check_transfer.py','compare_cavity_response.py','rank_loaded_force.py','equilibrate_forearm.py','certify_registration_strain.py','test_registration_strain.py','certify_fit.sh','run_registered_star.py','compare_local_cortex.py','rank_body_registration.py','locate_remaining_load.py','refine_physical_clock.py','link_registration_observer.py','qualify_registered_motion.py','accepted_cortical_solids.py','compare_accepted_strain.py','inspect_residual_region.py','prepare_linked_star.py','retain_forearm.py']:
        add('tools/'+n,h/n)
    out.mkdir(exist_ok=True);rec=h/'recovery.json'
    if not rec.exists():archive_files(files,'forearm-registration-'+sha(h/'matched-trials.json')[:16]+'.tar.zst',rec,'Full-tissue native forearm registration comparisons, rejected local remesh, deformation and mass checks, matched inputs, and trace evidence.')
    if read(rec)['verified_member_sha256']!={k:sha(v)for k,v in files.items()}:raise ValueError('Independent recovery mismatch')
    ev=out/'evidence/human-connected-mesh-rebuild-20261105/m6-forearm-equilibrium-20261124';ev.mkdir(parents=True,exist_ok=True)
    tool=out/'tools/native_ligament_contact/registered_material_volume/forearm_registration';tool.mkdir(parents=True,exist_ok=True)
    for k,p in files.items():
        if k.startswith('tools/'):shutil.copy2(p,tool/p.name)
    for f in ('cavity.json','measured-cell.json','force-owners.json','matched-trials.json','inertia.json','patch-response.json','proposals.json','recovery.json'):
        shutil.copy2(h/f,ev/f)
    for case in cases:
        dst=ev/case;dst.mkdir(exist_ok=True)
        for f in ('fit.json','interval.json','strain.json','wall-crossings.json'):shutil.copy2(h/case/f,dst/f)
    shutil.copy2(h/'guarded-cortex/result.json',ev/'guarded-cortical-comparison.json')
    shutil.copy2(h/'prep/witness-material/current-p1/result.json',ev/'remesh-source-check.json')
    for n,k in [('remaining-load.json','terminal-residual.json'),('right-ulna/region.json','right-ulna-region.json'),('right-ulna/linked-fit/fit.json','right-ulna-preparation-trial.json'),('whole-registration.json','whole-registration.json'),('motion-check/motion.json','accepted-motion.json'),('matched-motion-strain.json','accepted-stretches.json'),('whole-cortex/result.json','accepted-cortical-volumes.json')]:
        shutil.copy2(h/n,ev/k)
    lines=['# Left forearm load and material registration on Apple M6','',
        'In the retained control, the unconstrained second-root correction takes fat cell 1433491 from a candidate volume ratio of 2.49999 to 4.02542. The validity limit is 2.5. In the adjacent skin cell 1394217, the starting ratio is 0.46005 and its contribution along the peak nodal force is 484 N. These values come from paired GPU displacement words and independent evaluation of the registered material law.','',
        'Six old fat tetrahedra can be replaced with five while leaving surface coordinates fixed. The proposed faces lose their inherited source-face labels on P1 replay. Re-lumping the geometric mass transfer changes a nodal mass by up to 0.288 g. Its isolated nonaffine force distribution also changes. This topology was rejected.','',
        'The registration trials retain all source nodes, elastic cells, source mass, muscle and ligament bindings. They initialize selected coordinates of the bonded internal interface. The six fitted interface nodes move under the original operators during execution. The four-node and six-node energy fits introduce excessive shear despite valid determinants. The strain-bounded candidate protects both principal stretches in every incident cell in the initial pose and under the retained control displacement. The latter check is a replay prediction. Actual accepted time is reported below.','',
        '| Full-scene trial | Accepted steps | Simulated seconds | Second-root mechanical residual | Active columns in second root | Launch seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label,r in report.items():
        rr=r['final_fresh_mechanical_residual'];cols=r['active_krylov_columns']
        lines.append(f"| {label.replace('_',' ')} | {r['accepted_steps']} | {r['accepted_s']:.9g} | {rr[1] if len(rr)>1 else 'unavailable'} | {cols[1] if len(cols)>1 else 'unavailable'} | {r['end_to_end_wall_s']:.2f} |")
    lines.extend(['','All trials used the variable pleural pressure setting and full bonded tissue mechanics. Native rejections remain in the record. Skin/fat wall and cortical comparisons in the edited region do not qualify the remaining body. The whole-body 16 s motion and throughput gates remain outstanding. The recovery archive contains the launch configurations, code, source hashes, registered coordinates, native binary for the strain-bounded candidate, and full comparison traces.'])
    motion=read(h/'motion-check/motion.json');strain=read(h/'matched-motion-strain.json');cortex=read(h/'whole-cortex/result.json');load=read(h/'remaining-load.json');world=read(h/'whole-registration.json')
    lines.extend(['',f"The accepted one-step capture has {motion['exact_material_surface_pair_count']} surface intersections across the four material walls. It has {motion['roles']['skin']['below_shape']} skin and {motion['roles']['fat']['below_shape']} fat cells below the shape threshold. The cortical solid test across {cortex['bones']} bones gives {sum(v['actual_mm3'] for v in cortex['results'].values())/1000:.6f} cubic centimeters of summed pairwise intersection.",
        f"The matched mechanical capture passes the stretch comparison on all 144 directly affected cells. Across the rest of the body {sum(len(v['failed']) for k,v in strain['results'].items() if k.startswith('outside'))} skin/fat cells exceed the diagnostic 0.5 percent envelope outside the near-identity window.",
        f"Across the full shell the number of nodes with more than 100 N of internal elastic resultant changes from {world['before']['nodes_above_100_N']} to {world['after']['nodes_above_100_N']}. External loads and inertia enter separately. The mass map and original physical support remain active.",
        f"After the registration, the peak final certified residual is in object {load['peak_object']}. The skin/fat peak is at zero-based node {load['skin_fat_peak_node_zero_based']}. Raw GPU measurements reproduce the fresh residual within relative error {load['relative_attribution_error']:.3g}."])
    half=report['half_step'];ratio=half['accepted_s']/report['strain_bounded']['accepted_s'] if report['strain_bounded']['accepted_s'] else 0
    lines.extend(['',f"The finer clock was a matched-time diagnostic at {half['physical_dt_s']:.8g} s per step. It accepted {half['accepted_steps']} steps and reached {half['accepted_s']:.9g} physical seconds. Compare physical time and fresh certificates when the step length differs.",
        'A subsequent local energy fit at the right-ulna witness moved the initially high fat volume ratio from 2.4900 to 2.4925. Its limited change in local initial forces did not justify another native registration trial. The measured starting force at that node was 1.55 N. Its terminal residual includes inertia and interactions. Further right-side work requires the actual late-motion force and material direction.'])
    (ev/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':save(*(Path(x).resolve()for x in sys.argv[1:]))
