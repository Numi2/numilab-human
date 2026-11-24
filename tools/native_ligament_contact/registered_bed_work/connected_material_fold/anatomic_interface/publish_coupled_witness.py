"""Publish the physical cavity repair with the complete coupled outcome and matched controls."""
import csv,json,re,shutil
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(work):
    q=work.parent;stage=work/'witness-material'
    read=lambda p:json.loads(p.read_bytes())
    trial=work/'native-witness-05';cap=work/'accepted-material-07';failed=read(trial/'failure.json')
    with (trial/'native-traces/resting-coupled.csv').open()as f:trace=list(csv.DictReader(f))
    ends=read(trial/'execution.json');loops=(trial/'stdout.log').read_text()
    get=lambda name:float(re.search(name+r' wall_elapsed_ms=([\d.e+-]+)',loops)[1])
    duration=(get('native_horizon_end')-get('native_horizon_begin'))/1000
    geometry=read(stage/'volume/result.json');state=read(stage/'recorded-mass/native-replay/result.json');motion=read(stage/'accepted-motion.json') if (stage/'accepted-motion.json').exists()else None
    if geometry['exit'] or state['exit'] or not trace or read(work/'segment-math-test/result.json')['gpu_false_acceptances']:raise ValueError('Missing preparation evidence')
    ctrl=read(work/'native-motion.json')
    newton=[]
    for line in (trial/'stderr.log').read_text().splitlines():
        if line.startswith('matter_fgmres_active_work='):
            d=json.loads(line.split('=',1)[1]);newton.append({k:d[k]for k in ['root','used_columns_sum','cycles_with_work','budget_exhausted_solves','active_newton_begins','minimum_newton_norm','last_newton_norm','last_newton_threshold']})
    record=dict(mesh=geometry['inputs']['mesh.ele'],source_replay=sha(stage/'recorded-mass/native-replay/result.json'),native=geometry['import_report'],scope='Mixed-source resting reference adult. Full coupled native release under gravity. A resting equilibrium, sustained speed, zero whole-body penetrations and long-duration integrated physiology remain unqualified.',native_device=ctrl['device'],variable_pleural_pressure=read(trial/'launch.json')['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1',motion=motion,registered_material_mass_transfer=read(stage/'transfer-check.json'),mesh_resolution=read(stage/'resolution.json'),step_test=dict(physical_steps_requested=4,accepted_steps=int(trace[-1]['step']),accepted_seconds=float(trace[-1]['time_s']),whole_launch_wall_s=ends['wall_s'],native_horizon_wall_s=duration,whole_launch_factor=float(trace[-1]['time_s'])/ends['wall_s'],native_horizon_factor=float(trace[-1]['time_s'])/duration,launch=sha(trial/'launch.json'),failure_object=failed['matter_object_index'],failure_diagnostics=failed['matter_diagnostics'],active_solver_work=newton),prior_motion=ctrl['failure'],before_altitude_um=read(work/'witness-cavity/star.json')['baseline_minimum_altitude_um'],after_altitude_um=read(work/'witness-cavity/star.json')['minimum_altitude_um'],segment_experiment=dict(math=read(work/'segment-math-test/result.json'),whole_scene=read(work/'native-interval-04/execution.json'),failure=read(work/'native-interval-04/failure.json')['matter_diagnostics'],accepted_steps=read(work/'native-interval-04/failure.json')['segment_start'],selected_for_release=False),rigid_overlap_registered_pose=read(work/'verified-material.json')['remaining_summed_bone_overlap_mm3'],parent=read(work/'native-recovery.json')['url'])
    quality=read(stage/'motion-quality.json')
    record['accepted_cell_quality']=dict(cells=len(quality['witnesses']),regions=quality['regions'],submicron_at_start=quality['submicron_at_start'])
    record['accepted_bone_overlap']=read(stage/'accepted-bones.json')
    cap_log=(cap/'stdout.log').read_text()
    wall=lambda tag:float(re.search(tag+r' wall_elapsed_ms=([\d.e+-]+)',cap_log)[1])
    elapsed=(wall('native_horizon_end')-wall('native_horizon_begin'))/1000
    whole=read(cap/'execution.json')['wall_s']
    record['successful_single_step_timing']=dict(simulated_s=motion['accepted_s'],native_horizon_wall_s=elapsed,native_horizon_factor=motion['accepted_s']/elapsed,whole_launch_wall_s=whole,whole_launch_factor=motion['accepted_s']/whole)
    report=work/'coupled-witness.json';report.write_text(json.dumps(record,indent=2)+'\n')
    holder=work/'witness-recovery';receipt=work/'witness-archive.json'
    if not (holder/'reconstruction.json').exists():raise ValueError('Unverified native inputs')
    files={str(p.relative_to(work)):p for p in holder.iterdir()if p.is_file()}
    def add(f):
        if f.is_file()and not f.is_symlink():files[str(f.relative_to(work))]=f
    new_scripts=['admit_witness_material','bind_cavity_sidecars','bind_material_surfaces','bind_native_material','build_admissible_segment','capture_accepted_material_step','condition_motion_witness','fit_material_cavity_center','measure_material_resolution','map_accepted_quality','measure_accepted_bones','prepare_witness_recovery','publish_coupled_witness','rebuild_current_material_basis','replay_native_credits','route_local_carriers','run_current_coupled','run_segment_comparison','triangulate_witness_star','verify_first_motion','verify_material_star_transfer','verify_segment_extrema']
    for name in new_scripts:add(work/(name+'.py'))
    for f in [report,stage/'transfer-check.json',stage/'accepted-motion.json',stage/'motion-walls.json',stage/'motion-quality.json',stage/'accepted-bones.json',stage/'motion-bone-volumes/result.json',cap/'accepted-registered-bone-pose.json',stage/'resolution.json',work/'segment-math-test/result.json',work/'admissible-segment/build.json',work/'admissible-segment/src/metal/fem.metalinc',work/'native-recovery.json']:add(f)
    for f in (work/'witness-cavity').glob('*'):add(f)
    for folder in [trial,cap,work/'native-interval-04',work/'native-interval-03',work/'native-interval-02',work/'accepted-material-06']:
        for name in ['execution.json','launch.json','stdout.log','stderr.log','failure.json','native-traces/resting-coupled.csv','native-traces/resting-surface-audit.csv','native-traces/resting-com-momentum-diagnostic.csv','native-initial-bodies.json']:add(folder/name)
    # Keep the exact terminal native capture once. Incomplete trial dumps are reproducible from their launch and logs.
    for name in ['physical-accepted.node','physical-reference.node']:add(cap/name)
    for f in (cap/'native-traces/accepted-geometry').glob('*'):add(f)
    if not receipt.exists():archive_files(files,'native-coupled-material-witness-'+sha(report)[:16]+'.tar.zst',receipt,'Full native source-carrier admission, material-star construction with unchanged exterior, force-preserving source credits, Metal trial evidence and accepted motion audit. Previous source revisions are linked; large changed inputs use independently checksum-tested standard Zstandard patches against named parent hashes.')
    dst=work/'witness-delivery';source=dst/'tools/native_ligament_contact/registered_bed_work/connected_material_fold/anatomic_interface';ev=dst/'evidence/human-connected-mesh-rebuild-20261105/m6-native-conservative-material-replay';ev.mkdir(parents=True,exist_ok=True);source.mkdir(parents=True,exist_ok=True)
    for file in (report,receipt):shutil.copy2(file,ev/file.name)
    for name in new_scripts:
        if name in ('build_admissible_segment','run_segment_comparison','verify_segment_extrema','triangulate_witness_star'):continue
        shutil.copy2(work/(name+'.py'),source/(name+'.py'))
if __name__=='__main__':run(Path(__file__).resolve().parent)
