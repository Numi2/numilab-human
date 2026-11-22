"""Publish the fixed-reference native comparison and corrected initial pose audit."""
import json,sys,shutil
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def read(p):return json.loads(p.read_text())
def publish(root):
    case=root/'bonded-equilibrium';d=case/'held-volume';r=read(d/'result.json')
    fit=read(d/'fit.json');m=read(d/'verified-mass.json');frames=read(d/'frame-check.json')
    trial=r['held_parent_volume'];endo={k:v for k,v in r['measured_endpoints'].items() if k.startswith('held-volume-')}
    source=['admissible.py','relax.py','prepare.py','verify_cavity_mass.py','cavity_mass.py','check_trial.py','observe_release.py','run_held.sh','qualify_held.sh','finish_held.sh','capture_opening.py','opening.py','compare_opening.py','bank_motion.py','source_wrist_comparison.py','release_cell_loading.py','six_summary.py','extend_held.sh','check_release_frames.py','publish_held.py']
    evidence=['fit.json','preload.npz','preparation.json','current.node','layers.json','verified-mass.json','opening-recovery.json','opening.json','frame-check.json','executed-parity.json','executed-envelope/initial-envelope.json','executed-envelope/physical-banks.json','executed-envelope/target-witnesses.jsonl','result.json','observed-owner/identities.json','observed-owner/numi-human-native','observed-owner/NumiHumanLoadedSkinPoseData.inc']
    for n in ('executed-to-terminal.json','bank-motion.json','source-attribution.full.json','six-summary.json','cell-loading.json'):
        if (d/n).is_file():evidence.append(n)
    members={p:case/p for p in source}
    members.update({'held-volume/'+p:d/p for p in evidence})
    members['free-interface/recovery.json']=case/'free-interface/recovery.json'
    for run in [d/'three',d/'accepted-1',d/'accepted-2',d/'accepted-4',d/'accepted-5',d/'six']:
        if not run.exists():continue
        for k in ['execution.json','launch.json','stdout.log','stderr.log','failure.json','admission.json','terminal.receipt.json','native-initial-bodies.json','registered-initial-bodies.json','native-traces/resting-coupled.csv','endpoint-recovery.json','verified-endpoint.json','motion.json']:
            if (run/k).is_file():members['held-volume/'+run.name+'/'+k]=run/k
    receipt=d/'recovery.json';archive_files(members,'fixed-release-volume-'+sha(d/'result.json')[:12]+'-'+sha(d/'six-summary.json')[:12]+'.tar.zst',receipt,'Fixed-volume fitted tissue, source and native binary, executed poses, mechanical and geometric checks, native release and restorable accepted frames.')
    stage=root/'held-publication';prev=root/'interface-publication'
    note=next(prev.rglob('free-interface/README.md')).parent
    target=stage/note.parent.relative_to(prev)/'held-volume';target.mkdir(parents=True,exist_ok=True)
    tools=stage/next(prev.rglob('bonded_equilibrium/relax.py')).parent.relative_to(prev);tools.mkdir(parents=True,exist_ok=True)
    for name in source:shutil.copyfile(case/name,tools/name)
    for f in ['fit.json','preparation.json','verified-mass.json','frame-check.json','executed-parity.json','result.json','recovery.json','executed-to-terminal.json','bank-motion.json','source-attribution.full.json','six-summary.json','cell-loading.json']:
        if (d/f).exists():shutil.copyfile(d/f,target/f)
    banks=read(d/'executed-envelope/physical-banks.json')
    lines=['# Fixed-volume comparison during supported release','',
        f"The previous fitted release allowed the limiting fat cell to reach 32.889% of the original starting volume. This preparation holds the previous volume while fitting the complete connected stars of the affected remeshed material nodes. It moved {fit['moved']} existing nodes, preserved both external material surfaces and all 14 attachment sites, and changed the limiting cell’s initial volume by {(fit['witness_initial_to_previous_volume']-1)*100:.6f}%. All {m['cells']:,} installed cells passed preparation. The maximum installed nodal versus cell mass-transport difference was {m['max_nodal_transport_error_kg_m']:.4g} kg m.",'',
        f"The native M6 run accepted {trial['accepted']} of three steps, corresponding to {trial['seconds']*1e6:.5f} µs in {trial['wall_s']:.3f} wall seconds. All release comparisons used the same bed, gravity, constitutive laws, recruitment bounds and numerical acceptance tolerances.",'']
    if 'diagnostics' in trial:lines+=[f"The next candidate was rejected at native index {trial['index']} with diagnostics {trial['diagnostics']}. Shared state and respiration rollback passed.",'']
    for k,a in endo.items():
        v=a['volume'];s=a['audit'];mo=a['motion']
        lines += [f"At {s['simulated_s']*1e6:.5f} µs the minimum volume ratios were {v['min_J_parent']:.9g} relative to the previous starting shape and {v['min_J_new']:.9g} relative to the fitted shape. The respective counts at or below 0.35 were {v['parent_floor_failures']} and {v['new_floor_failures']}. The bed reaction was {mo['bed_normal_reaction_n']:.3f} N against {mo['weight_n']:.3f} N of weight. Upward tissue force on the 811.448 N skeleton was {a['terminal_skeleton_upward_N']:.3f} N. Throughput including launch was {s['end_to_end_factor']:.5g}×; the measured native horizon ran at {mo['native_horizon_factor']:.5g}×.".replace('×; the','×. The'),'',
            f"The accepted endpoint had {s['physical_skin_tissue_pairs']:,} exterior tissue intersections, {s['skin_self_intersections']} skin self-intersections and {s['layer_crossing_pairs']} layer crossings. Element stretch counts are listed separately in result.json.",'']
    lines += [f"The native registration snapshot and actual release root differ by {frames['source_registration_root_error_m']*1000:.6f} mm. The observed starting poses were emitted directly from the 157-body Metal articulated result used by the renderer. Root discrepancy after correction was {frames['executed_initial_root_error_m']:.3g} m. The exact executed-pose opening mesh had {banks['physical_nonocular_pairs']:,} skin/tissue crossings across {banks['checked_banks']} banks, including {banks['bone_pairs']} skin/bone pairs.",'']
    if (d/'executed-to-terminal.json').exists():
        p=read(d/'executed-to-terminal.json');lines+=[f"Of the opening triangle pairs, {p['persisted']:,} also intersected at the verified endpoint. {p['cleared']} cleared and {p['new']} appeared. The opening uses the source pack and executed body transforms. The terminal mesh uses the accepted GPU vertex buffer. Bank displacement is independently quantified in bank-motion.json.",'']
    attribution=read(d/'source-attribution.full.json')['banks']
    def count(key):return sum(a[key] for a in attribution)
    lines+=[f"The two extensor digitorum and two brachioradialis surfaces crossed the original skin in {count('source_triangle_pairs'):,} pairs affecting {count('source_muscle_faces'):,} muscle triangles. In the registered skin these became {count('registered_reference_pairs'):,} pairs across {count('registered_reference_muscle_faces'):,} muscle triangles. Exactly {count('common_registered_installed_pairs'):,} of these registered pairs appeared in the installed initial skin. The input surfaces already conflict. The additional conflicts arose during registration, before bed loading. Pair counts between different skin topologies cannot be subtracted to count physical defects. Muscle triangles retain common identities throughout this comparison.",'']
    longer=read(d/'six-summary.json')
    lines+=[f"The six-step request accepted {longer['accepted']} steps in {longer['wall_s']:.3f} seconds including native launch. The first {longer['prefix_identical']} accepted skeletal load samples and respiratory trace rows reproduced the short run. Accepted time was {longer['accepted_s']*1e6:.4f} µs.",'']
    if 'rejection_code' in longer:lines+=[f"At the next step the native solver rejected object {longer['object']}, index {longer['index']}, code {longer['rejection_code']}. The common coordinates and respiratory state rolled back. The native diagnostics were {longer['diagnostics']}.",'']
    loading=read(d/'cell-loading.json')
    low,high=loading['before_failure']
    lines+=[f"The two native volume witnesses were both subcutaneous fat tetrahedra with all four corners on the bonded interface in anatomical donor {low['geometric_donor_body']}. Their centers were {1000*loading['witness_centers_separation_m']:.2f} mm apart. Reference altitudes were {low['minimum_material_altitude_um']:.2f} and {high['minimum_material_altitude_um']:.2f} µm. By step three the second was already {high['accepted_J']:.7f} times its material-reference volume, with a largest principal stretch of {max(high['accepted_principal_stretches']):.3f}. Step four exceeded the unchanged material determinant limit of 2.5 during line search.",'']
    lines+=['Use the coupled compact scene for cardiopulmonary regression. The dense forearms require a joint source and registration repair against the executed skeleton. Posterior tissue requires a material-interface and load-transfer repair covering both opposite volume failures. Global tolerances, cosmetic offsets and a longer release cannot establish either result.','']
    lines+=['These captures cover a short release under gravity. Supported equilibrium, continuous motion with clear anatomy, breathing and cardiac cycles, the gas-response intervention and the end-to-end throughput targets remain to be demonstrated.','']
    (target/'README.md').write_text('\n'.join(lines))
if __name__=='__main__':publish(Path(sys.argv[1]).resolve())
