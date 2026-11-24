"""Publish the recompiled material binding, its real native outcome and joint-source qualification."""
from pathlib import Path
import csv,json,re,shutil
from archive_cavity_trial import archive_files,sha
def read(p):return json.loads(p.read_bytes())
def save(w):
    stage=w/'coupled-publication';out=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-native-fat-cavity-20261124';tool=stage/'tools/native_ligament_contact/registered_material_volume/support_observation/right_shoulder'
    for p in (out,tool):p.mkdir(exist_ok=True,parents=True)
    files={};case=w/'native-cavity-bound';old=w.parent/'posterior-interface-082/native-pair';reg=w/'native-binding'
    for n in ('admit_native_cavity.py','bind_native_cavity_topology.py','evaluate_native_cavity.py','fit_registered_assembly.py','transfer_shoulder_fit.py','choose_emitted_shoulder.py','retain_coupled_cavity.py'):
        shutil.copy2(w/n,tool/n);files['source/'+n]=w/n
    for n in ('native-cavity/execution.json','native-cavity/launch.json','native-cavity/stderr.log','native-cavity/stdout.log',
              'native-binding/build.json','native-binding/before-direction.json','native-binding/numi-human-native','native-binding/libmetalrobo.dylib',
              'native-binding/NumiHumanLoadedSkinPoseData.inc','native-binding/NumiHumanCorticalSkinBinding.inc','native-binding/numi_human.mm','native-binding/runtime.mm','native-binding/app-runtime.mm',
              'material-import/result.json','material-ancestry/result.json','material-ancestry/carrier-replay/native-replay/result.json',
              'joint-guarded/result.json','transfer-witness/witness.json','source-choices/results.json','joint-emitted/results.json','final-contacts.json','final-layers.json'):
        files[n]=w/n
    for n in ('execution.json','launch.json','stderr.log','stdout.log','force-observer.f32','failure.json','physical-accepted.node','physical-posterior-bed-history.csv',
              'material-check.json','support-check.json','matched.json','wall-crossings.json','native-traces/resting-coupled.csv'):
        if (case/n).exists():files['native/'+n]=case/n
    for p in (case/'native-traces/accepted-geometry').glob('*'):
        if p.is_file():files['native/accepted/'+p.name]=p
    for p in (w/'joint-emitted').glob('*.npz'):files['joint-emitted/'+p.name]=p
    a=read(case/'execution.json');b=read(reg/'build.json');an=read(w/'joint-emitted/results.json');lay=read(w/'final-layers.json');ct=read(w/'final-contacts.json')
    match=read(case/'matched.json')if (case/'matched.json').exists()else None
    fail=read(case/'failure.json')if (case/'failure.json').exists()else {}
    q=read(w/'material-import/result.json')['quality'];e=read(w/'material-ancestry/result.json');sub=e['roles']
    verified={(row['muscle'],row['material'],row['stage'])for row in lay['rows'] if row['mm3']>=0 and row['solid_partitions']>0}
    for item in ct['indeterminate']:
        if item['kind']!=0 or item['bank'] not in (15,16) or not all((sid,item['bank'],stage) in verified for sid in item['affected'] for stage in ('before','after')):raise ValueError('Unresolved surrounding material')
    if e['carrier_exit'] or not b['direction']['compiled']:raise ValueError('Missing native mass proof, topology or surroundings')
    d=dict(parent='512db954265c72bb2797d30c0cce566f40193083',control=sha(old/'launch.json'),native=sha(case/'launch.json'),binding=sha(reg/'build.json'),cells=b['direction']['cells'],nodes=b['direction']['nodes'],registration=dict(donor=b['donor'],source_parent=b['source_tet'],distance_m=b['source_distance_m'],unchanged_cortical_facets=b['protected_cortical_facets']),mass=sha(w/'material-ancestry/carrier-replay/native-replay/result.json'),execution=a,failure={k:fail.get(k)for k in ['segment_start','matter_object_index','matter_diagnostics']},motion=match,source_fractions=an['chosen'],registered_pair_sum_mm3=dict(before=an['reference_pair_sum_mm3'],after=an['emitted_pair_sum_mm3']),jointly_admissible_choices=an['admissible_combinations'],surrounding_pairs=len(ct['pairs']),layer_volumes_measured_from_tetrahedra=ct['indeterminate'],deltoid_fat_mm3=next(s['mm3']for s in lay['rows']if s['muscle']==79 and s['stage']=='after'and s['material']==16))
    if not match and not fail:d['termination']=(case/'stderr.log').read_text().splitlines()[-1]
    dest=out/'results.json';dest.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');files['results.json']=dest
    desc=[f"The native executable reimported {d['cells']:,} physical skin/fat cells and {d['nodes']:,} nodes. The additional interior fat node was registered against source tetrahedron {b['source_tet']} on donor body {b['donor']}. All retained nodes, cortical faces, tendon endpoints and bed contacts keep their original indices. The compiled GPU owner checks every tetrahedron and material against the exact new input."]
    if match:
        res=match['new'];p=res['material'];sp=res['support'];term=res['terminal'];rx=re.search(r'body_real_time_factor=([^ ]+)',term or '')
        desc.append(f"The variable-pleural-pressure run accepted {p['accepted_step']} steps, covering {p['accepted_s']:.10g} s. Its cold duration was {a['wall_s']:.3f} s. The terminal material scan found {p['roles']['skin']['below_shape']} skin and {p['roles']['fat']['below_shape']} fat cells below shape, {sum(r['inverted'] for r in p['roles'].values())} inverted cells and {p['exact_material_surface_pair_count']} crossing pairs on the four registered tissue walls. Terminal bed reaction was {sp['accepted_normal_force_N']:.4f} N against {sp['normal_weight_N']:.4f} N normal weight. Native body timing follows in the matched receipt.")
    else:desc.append('The run ended before the complete matched motion audit. The retained launch, logs and failure output identify its physical and numerical extent.')
    desc.append(f"The six-muscle registration uses emitted Float32 source coordinates and independent per-muscle displacements. Its summed overlapping pair volumes changed from {an['reference_pair_sum_mm3']/1000:.4f} to {an['emitted_pair_sum_mm3']/1000:.4f} cm³, with all six muscle boundaries and tetrahedra checked across the source, skeletal reference, opening and retained endpoint. Spinal deltoid/fat still overlaps by {d['deltoid_fat_mm3']/1000:.4f} cm³. Shoulder force ownership remains with the existing native scene.")
    desc.append('Neither sixteen seconds of supported motion, whole-body clearance, nor 0.1× end-to-end throughput has been demonstrated. This comparison retains the physical bed, constitutive laws, accept/reject rules and variable pleural pressure.')
    doc=out/'README.md';doc.write_text('## Coupled native material qualification on SSH Numi\n\n'+'\n\n'.join(desc)+'\n');files['README.md']=doc
    recovery=out/'recovery.json'
    if not recovery.exists():archive_files(files,'native-fat-cavity-and-source-fit-'+sha(dest)[:16]+'.tar.zst',recovery,'Real full-scene material binding and independent registered shoulder source selection, including failed startup and matched result.')
    if read(recovery)['verified_member_sha256']!={k:sha(v)for k,v in files.items()}:raise ValueError('GitHub recovery mismatch')
    print('verified',len(files),flush=True)
if __name__=='__main__':save(Path.cwd())
