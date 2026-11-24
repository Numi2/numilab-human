"""Record the full-scene accepted step and its first subsequent, rolled-back defect."""
from pathlib import Path
import csv,json,re,shutil
from archive_cavity_trial import archive_files, sha
import numpy as np

def run(work,case):
    q=work.parent
    native=work/case
    read=lambda p:json.loads(p.read_bytes())
    stderr=(native/'stderr.log').read_text()
    stdout=(native/'stdout.log').read_text()
    failure=read(native/'failure.json')
    with (native/'native-traces/resting-coupled.csv').open()as f: coupled=list(csv.DictReader(f))
    accepted=float(coupled[-1]['time_s'])
    e=re.search(r'resting_fem_failure_accepted_witness global_tet=(\d+) rejected_J=([\d.e+-]+) accepted_J=([\d.e+-]+).* local_tet=(\d+)',stderr)
    if not e or 'registered_loaded_material_install=PASS' not in stdout or 'registered_volume_fractional_mass=PASS' not in stdout:raise ValueError('Missing integrated material or rollback witness')
    raw=np.loadtxt(q/'qualified-material/mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    idx=int(e[4]);tet=raw[idx,:4]-1
    get=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))[tet]
    a=get(q/'qualified-material/mesh.node');b=get(work/'selected-pose/current.node')
    det=lambda p:np.linalg.det(p[1:]-p[0])
    faces=[np.delete(a,i,0)for i in range(4)]
    alt=abs(det(a))/np.array([np.linalg.norm(np.cross(f[1]-f[0],f[2]-f[0]))for f in faces])
    end=read(native/'execution.json');start=float(re.search(r'native_horizon_begin wall_elapsed_ms=([\d.e+-]+)',stdout)[1]);finish=float(re.search(r'native_horizon_end wall_elapsed_ms=([\d.e+-]+)',stdout)[1])
    controlled=read(q.parent.parent.parent/'ligament-equilibrium-070/validated-body-controls.json')['supported_control']
    cfg=read(native/'launch.json')
    result=dict(executable=sha(Path(cfg['argv'][0])),library=sha(work/'binding/runtime/libmetalrobo.dylib'),metal=sha(Path(cfg['environment']['NUMI_MATTER_METALLIB_OVERRIDE'])),launch=sha(native/'launch.json'),device=read(q/'gravity-contact/geometry.json')['device'],variable_pleural_pressure=cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1',registered_material_cells=len(raw), accepted_steps=int(coupled[-1]['step']),accepted_seconds=accepted,wall_seconds=end['wall_s'],body_window_s=(finish-start)/1000,accepted_seconds_per_body_wall_second=1000*accepted/(finish-start),end_to_end_including_rejected_attempt=accepted/end['wall_s'],failure=dict(matter_object=failure['matter_object_index'],tet=idx,material=int(raw[idx,4]),initial_J=det(b)/det(a),accepted_J=float(e[3]),rejected_J=float(e[2]),min_reference_altitude_m=float(min(alt)),physical_nodes=tet.tolist()),surface_check_scope='Four layer surfaces are intersection-free at the registered starting pose. Bone overlap remains. Through-motion crossings and equilibrium remain unqualified.',historical_comparison=dict(accepted_s=controlled['accepted_s'],factor=controlled['end_to_end_factor'],full_skin_fat_volume=controlled['full_skin_fat_volume'],endpoint_physical_nonocular_pairs=controlled['endpoint_physical_nonocular_pairs']),material_archive=read(work/'recovery.json')['url'])
    report=work/'native-motion.json';report.write_text(json.dumps(result,indent=2)+'\n')
    receipt=work/'native-recovery.json'
    files=[report,work/'recovery.json',*(work/f for f in ['bind_native_material.py','bind_material_surfaces.py','run_current_coupled.py','publish_native_coupling.py'])]
    for folder in [native,work/'binding']:
        files.extend(f for f in folder.rglob('*')if f.is_file()and not f.is_symlink()and f.suffix!='.o')
    if not receipt.exists():
        archive_files({str(f.relative_to(work)):f for f in files},'native-accepted-connected-material-'+sha(report)[:16]+'.tar.zst',receipt,'Exact integrated native build, launch, material-bound surface tables, accepted native output and frozen second-step geometry failure. Reference inputs and recorded physical mass are in the linked predecessor archive.')
    dst=work/'native-delivery'
    src=dst/'tools/native_ligament_contact/registered_bed_work/connected_material_fold/anatomic_interface'
    evidence=dst/'evidence/human-connected-mesh-rebuild-20261105/m6-native-conservative-material-replay'
    src.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    for f in files[:6]:
        if f.suffix=='.py':shutil.copy2(f,src/f.name)
    for f in [report,receipt]:shutil.copy2(f,evidence/f.name)
if __name__=='__main__':
    run(Path(__file__).resolve().parent,'native-integrated-01')
