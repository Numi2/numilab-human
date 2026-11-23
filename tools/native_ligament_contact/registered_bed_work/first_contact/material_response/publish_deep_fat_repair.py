from pathlib import Path
import sys,json,shutil
from archive_cavity_trial import archive_files,sha
def run(job):
    scripts=['trace_fat_crossing_history.py','screen_fat_source_replay.py','separate_deep_fat_stars.py','polish_fat_cell_stars.py','audit_moved_layer_stars.py','balance_repaired_fat_mass.py','emit_repaired_fat_volume.py','publish_deep_fat_repair.py']
    files={s:job/s for s in scripts+['deep-fat-quality-global.json','registered-frame-recovery.json']}
    for name in ['deep-fat-source-history','deep-fat-source-replay','deep-fat-joint-fit','deep-fat-cavity-fit','deep-fat-quality-fit','deep-fat-credits','deep-fat-native']:
        for f in (job/name).glob('*'):
            if f.is_file(): files[str(f.relative_to(job))]=f
    path=job/'deep-fat-recovery.json'
    if not path.exists():archive_files(files,'source-separated-deep-fat-'+sha(job/'deep-fat-native/mesh.node')[:12]+'.tar.zst',path,'Shoulder fat surface repair, constrained tissue mass, exact two-pose scans and native material verification.')
    rd=lambda n:json.loads((job/n).read_bytes());receipt=rd(path.name)
    stage=job/'deep-fat-delivery';code=stage/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    for d in [code,ev]:d.mkdir(exist_ok=True,parents=True)
    for s in scripts:shutil.copy2(job/s,code/s)
    fit=rd('deep-fat-quality-fit/result.json');result=dict(qualification='Native material reader; independent exact full-layer checks in the source and initial physical pose. No elapsed dynamics.',native_volume=rd('deep-fat-native/result.json'),mass_routing=rd('deep-fat-credits/result.json'),moved_nodes=len(__import__('numpy').load(job/'deep-fat-quality-fit/candidate.npz')['nodes']),maximum_displacement_um=fit['max_move_um'],initial_deep_fat_self_pairs=13,other_bone_tissue_contacts='Independent anatomical clearance still requires checking',recovery={k:receipt[k] for k in ['url','archive_sha256','size_bytes']})
    (ev/'deep-fat-connected-physical-repair.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
