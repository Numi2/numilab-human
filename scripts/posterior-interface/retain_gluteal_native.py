"""Publish matched results, material ownership and exact recovery data for the native experiment."""
import json
import shutil
import sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha


def read(p):
    return json.loads(p.read_bytes())


def retain(w,j,work):
    case=w/'native-pair';build=w/'native-build';old=j/'compensated-start-two'
    sys.path.insert(0,str(j/'supported-contact-081'))
    from retain_material_extremum import records
    out=w/'publication/evidence/human-connected-mesh-rebuild-20261105/m6-native-posterior-interface-20261124'
    tools=w/'publication/scripts/posterior-interface'
    out.mkdir(parents=True,exist_ok=True);tools.mkdir(parents=True,exist_ok=True)
    methods=['posterior_interface.py','gluteal_interface.metalinc','build_gluteal_interface.py','run_gluteal_interface.py','check_gluteal_work.py',
             'test_gluteal_kernel.mm','verify_gluteal_gpu.py','measure_loaded_shoulder.py','measure_shoulder_footprint.py','qualify_gluteal_prefix.py','check_gluteal_bones.py','retain_gluteal_native.py']
    for f in methods:shutil.copyfile(w/f,tools/f)
    states=['registered-registered_gluteal','registered-all','stress_free-registered_gluteal','stress_free-all']
    patch={a:read(w/a/'response.json') for a in states}
    supported=read(case/'support-check.json');mat=read(case/'material-check.json');bones=read(case/'bone-solids/result.json')
    incident=read(w/'loaded-shoulder.json');foot=read(w/'footprint.json')
    r=read(case/'matched.json');initial=read(build/'build.json')
    attempts={k:records(p) for k,p in [('third',w/'native-third-matched'),('third_control',j/'supported-contact-081/matched-control-restored')]}
    if attempts['third']['completed_steps']!=2 or r['control_launch']!=sha(old/'launch.json') or mat['receipt_sha256']!=incident['accepted_receipt']:
        raise ValueError('Source acceptance mismatch')
    volume=sum(d['skin']+d['fat'] for d in bones['results'].values())
    gpu=read(w/'kernel-check/gpu.json');mass=read(work/'posterior-material-registration-078/coupled-material/mass.json')
    result=dict(host='SSH numi',operator=initial['operator'],pins=initial['pins'],source_control='1b83ccf91c89a2687de9cd2ef3c1f3f44bfb932f',
                deep_fat_mass_kg_in_donor_inertias=mass['total_kg'],offline=patch,native=r,support=supported,material=mat,
                bone_skin_fat_overlap_mm3=volume,shoulder=incident,footprint=foot,steps=attempts,
                kernel_check=gpu,energy_check=read(w/'work-check.json'),
                variable_external_pressure=read(case/'launch.json')['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1',
                qualification='Experimental near-reference elastic coupling. Full support, equilibrium, fidelity, sustained physiology, motion and speed remain unqualified.')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'README.md').write_text(f'''# Posterior tissue forces in the native Human scene

The native Apple GPU experiment joins 51 registered deep-fat nodes to muscle material parents with reactions on the pelvis and both femurs. The same runtime applies both sides before generalized-load projection. The reference operator uses 2,256 tetrahedral integration cells and the existing passive muscle compliance. It is a corotational energy with a condensed reference tangent. The original donor inertias still carry {1000*mass["total_kg"]:.2f} g of reconstructed fat. Large-strain material response and independent tissue inertia remain outside this approximation.

## Matched physical result

| Measure | Result on SSH Numi |
| --- | --- |
| Accepted time | Two 7.8125 µs steps, 15.625 µs total |
| Body speed, including the coupled body clock | 0.000000221× |
| Cold process | {read(case/'execution.json')['wall_s']:.2f} s |
| Normal bed force and normal weight | {supported["accepted_normal_force_N"]:.3f} N / {supported["normal_weight_N"]:.3f} N |
| Third step equilibrium residual | {attempts["third"]["roots"][-1]["final_newton_norm"]:.8f}, versus {attempts["third_control"]["roots"][-1]["final_newton_norm"]:.8f} in the control |
| Third-step Krylov columns | {attempts["third"]["roots"][-1]["used_columns"]}, versus {attempts["third_control"]["roots"][-1]["used_columns"]} in the control |
| Material surfaces at step two | {mat["exact_material_surface_pair_count"]} intersecting pairs across the four material walls |
| Skin and fat | {mat["roles"]["skin"]["inverted"]+mat["roles"]["fat"]["inverted"]} inverted cells, {mat["roles"]["skin"]["below_shape"]} skin and {mat["roles"]["fat"]["below_shape"]} fat cells below the shape threshold |
| Actual skin and fat intersection with 185 cortical banks | {volume/1000:.4f} cm³ |
| Largest change in accepted skin/fat node position against control | {r["accepted_frame_max_change_m"]*1e6:.4f} µm |

The passing short run used the full shared tissue solver, the fixed bed and variable external vascular pressure. The two GPU force observations balance to within {max(v["imbalance_N"] for v in r["force_owner"]):.2e} N. The terminal net tissue reaction was about 0.00017 N in magnitude. The failed third step preserved two accepted steps. The results include separate physiological and mechanical clocks. Native presentation and sustained performance are unqualified.

## Where the load goes

In the earlier patch, fixed cut nodes carried almost all of an imposed bed load. Here the cut is traction-free. A unit perturbation at the two currently loaded sites in the registered nine-site gluteal footprint gives {patch["registered-registered_gluteal"]["work_compliance_m_per_N"]*1e3:.3f} mm/N of work-conjugate compliance with the three skeletal donors held. All {patch["registered-registered_gluteal"]["free_nodes"]:,} material nodes are free. The full measured footprint gives {patch["registered-all"]["work_compliance_m_per_N"]*1e3:.3f} mm/N on the opening prestressed geometry. This negative direction rejects that local equilibrium approximation.

Skin vertex {incident["dominant_skin_node"]} carried {100*incident["fraction"]:.2f}% of the terminal microstep normal impulse. Its eight neighboring surface edges span {100*min(foot["edge_length_m"]):.1f}–{100*max(foot["edge_length_m"]):.1f} cm. The source maps the nearest right infraspinatus to the humerus and scapula. At this vertex the bed-normal ray traverses about 0.025 mm of infraspinatus and first reaches scapula about 29.6 mm below the skin. Neighboring rays traverse several different tissues. The accepted vertex is {foot["native_gap_m"]*1e9:.1f} nm above the registered plane at the endpoint. Terminal impulses refer to the preceding microstep. The clipped contact area at the endpoint is zero. A frozen 1 mm plane advance gives {foot["samples"][4]["projected_contact_area_m2"]*1e6:.1f} mm² over the same eight triangles.

Treat the initial shoulder pressure as a localized transient. The next physical registration needs the distributed skin, fat, shoulder-muscle and cortical interfaces under an actual gravity preload. A single vertical spring at the peak node would infer a load route across several tissues.

## Reproduction

The verified archive contains the exact two-step and three-step launches, runtime modules, generated sources, native captures, physical state, independent tests, trial receipts and the earlier rejected local witnesses. The build script reuses the pinned Metal pipeline and existing Human adapter. The opt-in operator fingerprint covers its reference geometry, donors, force law and matrix. The runtime verifies the matrix digest at load. The normal production model remains the control.

The independent small-deformation and combined-rotation tests directly invoke the compiled Metal kernel and compare all 51 scattered tissue forces and all three skeletal wrenches against virtual work. The largest relative error in these two loaded tests is {max(s["relative_norm"] for s in gpu if s["case"]in ("bed","combined")):.2e}. Rigid motion tests use absolute errors near the zero-force state. The clipped-face test conserves projected area and first moment. An initial three-step command requested a capture at step two with incompatible submission cadence. It exited during setup. The retained physical comparison uses the separate, matched three-step launch.
''')
    files={'summary/results.json':out/'results.json','summary/README.md':out/'README.md'}
    for f in methods:files['source/'+f]=w/f
    for f in ('NumiMatter.metallib','numi_human.mm','libmetalrobo.dylib','material.f32','build.json','src/metal/numi_human.metalinc'):
        files['build/'+f]=build/f
    assets=work/'posterior-material-registration-078/coupled-material'
    for f in ('mass.json','motion-replay.json','material/parents.npz','material/reference-tangent.npz','material/reference-tangent.json','material/gradient.npz'):
        files['material/'+f]=assets/f
    for q in [*states,'current-port','kernel-check']:
        for f in (w/q).iterdir():
            if f.is_file():files[q+'/'+f.name]=f
    for folder in ('native-pair','native-third','native-third-matched'):
        d=w/folder
        for f in d.iterdir():
            if f.is_file() and f.name!='registered-skin-native.csv':
                files[folder+'/'+f.name]=f
        for n in ('resting-coupled.csv','resting-surface-audit.csv','resting-com-support-impulses.csv'):
            f=d/'native-traces'/n
            if f.exists():files[folder+'/native-traces/'+n]=f
        for f in (d/'native-traces/accepted-geometry').glob('step-*'):
            files[folder+'/accepted-geometry/'+f.name]=f
    files['native-pair/bone-solids.json']=case/'bone-solids/result.json'
    for f in ('loaded-shoulder.json','footprint.json','work-check.json'):files[f]=w/f
    for prefix,d in [('two',old),('three',j/'supported-contact-081/matched-control-restored')]:
        for f in ('launch.json','execution.json','stderr.log','stdout.log'):
            files['controls/'+prefix+'/'+f]=d/f
    rec=out/'recovery.json'
    if not rec.exists():
        archive_files(files,'gluteal-native-'+sha(out/'results.json')[:18]+'.tar.zst',rec,
                      'Matched native tissue and skeletal force owner, complete accepted frame, source, GPU tests and rejected equilibrium.')
    if read(rec)['verified_member_sha256']!={k:sha(p) for k,p in files.items()}:
        raise ValueError('Independent archive restore differs')
    print('verified',len(files),'members',flush=True)

if __name__=='__main__':retain(*(Path(s).resolve()for s in sys.argv[1:]))
