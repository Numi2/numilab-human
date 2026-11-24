"""Retain the actual native search, tissue transfer and the joint registered fit with scope."""
import csv,json,os,shutil,sys
from pathlib import Path
import numpy as np
from archive_cavity_trial import archive_files,sha


def read(p):return json.loads(p.read_bytes())


def native(case):
    cfg=read(case/'launch.json');done=read(case/'execution.json')
    with (case/'native-traces/resting-coupled.csv').open()as f:
        rows=list(csv.DictReader(f))
    work=[json.loads(s.split('=',1)[1])for s in (case/'stderr.log').read_text().splitlines() if s.startswith('matter_fgmres_active_work=')]
    f=case/'failure.json';failure=read(f)if f.exists()else None
    return dict(launch=sha(case/'launch.json'),execution=done,accepted_s=float(rows[-1]['time_s'])if rows else 0.,completed_steps=int(rows[-1]['step']) if rows else 0,
                variable_pleural=cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE'],active_columns=[r['used_columns_sum'] for r in work],
                failure=None if failure is None else {'root':failure['segment_start'],'object':failure['matter_object_index'],'diagnostics':failure['matter_diagnostics']},
                physiology={k:float(rows[-1][k])for k in ['lung_volume_ml','pleural_pa','PaO2_mmhg','PaCO2_mmhg','root_assistance_n']} if rows else {})


def save(w):
    stage=w/'second-publication';out=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-shoulder-and-material-motion-20261124'
    tool=stage/'tools/native_ligament_contact/registered_material_volume/support_observation/right_shoulder'
    out.mkdir(parents=True,exist_ok=True);tool.mkdir(parents=True,exist_ok=True)
    files={}
    for n in ('fit_registered_assembly.py','transfer_shoulder_fit.py','rebuild_tissue_source_mapping.py','repair_skin_fat_quality.py','refine_skin_fat_edges.py','compile_tissue_repairs.py','build_search_guard.py','search_endpoint.metalinc','native_search_trial.py','test_search_endpoint.cpp','retain_physical_checkpoint.py'):
        shutil.copy2(w/n,tool/n);files['source/'+n]=w/n
    base=w.parent/'posterior-interface-082/native-third-matched';trial=w/'native-search-retry'
    mat=read(w/'material-import/result.json');fit=read(w/'joint-constrained/result.json');record=read(w.parent/'posterior-interface-082/native-pair/support-check.json')
    ledger=dict(parent='de66f0f3b0eec514f70f3fbe9684110780b26f44',control=native(base),experiment=native(trial),stopped=read(w/'native-search/execution.json'),
                tissue=mat,joint=fit,source_transfer=read(w/'transferred/results.json'),material_ancestry=read(w/'material-ancestry/result.json'),solver_unit=(w/'search-unit.log').read_text().strip(),matched_weight_N=record['normal_weight_N'],matched_support_N=record['accepted_normal_force_N'])
    tests=ledger['experiment'];control=ledger['control']
    if tests['variable_pleural']!='1' or control['variable_pleural']!='1' or mat['exit'] or not mat['native']['native_imported']:raise ValueError('Missing native admission or pressure')
    if ledger['material_ancestry']['carrier_exit'] or read(w/'material-ancestry/carrier-replay/native-replay/result.json')['exit']:raise ValueError('Rejected carrier replay')
    for n in ('source','input_mass'):assert len(mat[n])==64
    if mat['native']['inverted'] or mat['native']['below_shape'] or abs(mat['mass_change_kg'])>1e-10:raise ValueError('Unqualified material transfer')
    for label,case in [('trial',trial),('control',base)]:
        for f in ('launch.json','execution.json','failure.json','stdout.log','stderr.log','native-traces/resting-coupled.csv'):
            if (case/f).exists():files[label+'/'+f]=case/f
    for d in ('skin-fat-cavities','nested-tissue','compensated-tissue','material-import','joint-fit','joint-constrained','transferred'):
        for f in (w/d).iterdir():
            if f.is_file() and f.suffix in ('.json','.npz'):files[d+'/'+f.name]=f
    for n in ('native-import.json','native-import.stderr','overlay.npz','source-overlay.npz'):files['material-import/volume/'+n]=w/'material-import/volume'/n
    for n in ('search-build/build.json','search-build/NumiMatter.metallib','search-build/src/metal/fem.metalinc','search-unit.log','native-search/execution.json','emitted-contacts.json','emitted-layers.json','material-ancestry/result.json','material-ancestry/p1.log','material-ancestry/carrier.log','material-ancestry/carrier-replay/native-replay/result.json','material-ancestry/carrier-replay/flows/result.json'):
        files[n]=w/n
    for n in ('reclaim.log','old-clones.log'):
        f=w/n
        if f.exists():files['storage/'+n]=f
    for f in (w/'archived-closed').glob('*.json'):files['storage/closed/'+f.name]=f
    result=out/'results.json';result.write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
    q=mat['quality']['accepted_replay'];h0,h1=fit['history'][0],fit['history'][-1]
    lines=[f"The matched control reached {control['accepted_s']:.9g} s using {control['active_columns']} Krylov columns across the attempted roots. The new trial reached {tests['accepted_s']:.9g} s using {tests['active_columns']}. Its complete process took {tests['execution']['wall_s']:.2f} wall seconds. Both used variable pleural pressure and the same full-body model, tendon bindings and GPU load coupling."]
    if tests['failure']:lines.append(f"The trial rejected at root {tests['failure']['root']} with diagnostics {tests['failure']['diagnostics']}. The solver experiment is excluded from the production default.")
    (out/'README.md').write_text('## Whole-body tissue and native trial on SSH Numi\n\n'+' '.join(lines)+f'''

The native material importer verified {mat["native"]["cells"]:,} cells and {mat["native"]["nodes"]:,} nodes. In the reference and opening pose the skin and fat have zero inverted or below-shape cells. Applying the same retained motion to the reconstructed volume leaves {q["15"]["low_shape"]} skin and {q["16"]["low_shape"]} fat cells below the shape threshold. The admitted cavities retain reference-material identity and a conservative parent-to-child mass and virtual-work check. The unchanged simulator supplied the saved physical coordinate samples. The new material import has not passed a new coupled motion run.

The six-muscle constrained fit included immutable bones, neighboring muscles, physical skin and fat, source tendon footprints, volume and local tetrahedral orientation. The sum of measured pairwise intersections changed from {h0["sum_pair_mm3"]/1000:.5f} to {h1["sum_pair_mm3"]/1000:.5f} cm³ over {len(fit["history"])-1} accepted authoring iterations. This sum may count the same space in multiple pairs. It describes registration in a retained frame. Current control bed force was {record["accepted_normal_force_N"]:.3f} N versus {record["normal_weight_N"]:.3f} N normal weight.

The native source-carrier replay also passed after integrating the appended fat node and six replacement cells. Cell-level source identities and the physical mass of each carrier were conserved. Transferring the six-muscle fit back to the emitted source admitted only 1/16 of the authoring displacement. The independently measured spinal deltoid/fat overlap remained 42.965 cm³.\n\nThe complete model has not qualified supported rest, zero material overlaps, continuous breaths and beats, or 0.1× throughput. The measured trial is too short for an integrated steady-state benchmark.
''')
    files['results.json']=result;files['README.md']=out/'README.md'
    rec=out/'recovery.json'
    if not rec.exists():archive_files(files,'shoulder-tissue-physics-'+sha(result)[:16]+'.tar.zst',rec,'Physical skin/fat transfer, six-region constrained fit and full coupled GPU trial with matched control. Includes stopped and failed candidates.')
    if read(rec)['verified_member_sha256']!={n:sha(f) for n,f in files.items()}:raise ValueError('Independent recovery failure')
    print('verified',len(files),'files',flush=True)
if __name__=='__main__':save(Path.cwd())
