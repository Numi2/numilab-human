"""Preserve verified native preflight and distinguish later physical admission."""
import json,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(p):
    def rd(s):return json.loads((p/s).read_bytes())
    scripts=['bind_cortical_role_frames.py','build_role_face_consumers.py','compile_cortical_skin_contact.py','build_cortical_skin_registration.py','publish_native_material_preflight.py']
    files=scripts+['surface-recovery.json','physical-weight-frames/I-nodes.csv','physical-weight-frames/D-nodes.csv','physical-weight-frames/combined-node-map.csv','physical-weight-frames/result.json','contact-quadrature/material-contact.npz','contact-quadrature/result.json','physical-interface/build.json','physical-roles/build.json','physical-roles/app/numi-human-native','physical-roles/app/NumiHumanCorticalSkinBinding.inc','physical-roles/app/NumiHumanPhysicalSkinAuthoringRoleConsumer.hpp']
    for folder in ['scene-physical-interfaces','scene-physical-roles']:
        files += [folder+'/'+f for f in ['launch.json','execution.json','stderr.log','stdout.log']]
    archived=p/'preflight-recovery.json'
    if not archived.exists():
        archive_files({f:p/f for f in files},'native-connected-skin-preflight-'+sha(p/'scene-physical-roles/launch.json')[:16]+'.tar.zst',archived,'Exact connected skin/fat contact binding and native executable, verified full86 surface preflight, and unresolved physical carrier support domain.')
    orig=rd('surface-recovery.json');rec=rd('preflight-recovery.json');out=(p/'scene-physical-roles/stdout.log').read_text().splitlines();err=(p/'scene-physical-roles/stderr.log').read_text().splitlines();inter=(p/'scene-physical-interfaces/stderr.log').read_text().splitlines()
    result=dict(device='SSH Numi Apple M6',baseline_recovery=orig['archive_sha256'],native_run=rd('scene-physical-roles/execution.json'),native_preflight=next(s for s in out if s.startswith('registered_volume_native_preflight=PASS')),ligaments=next(s for s in err if s.startswith('registered_ligament_source_geometry=')),blocking_guard=next(s for s in err if s.startswith('myosim_articulated_visual=failed')),previous_guard=next(s for s in inter if s.startswith('myosim_articulated_visual=failed')),source_frame_corrections=rd('physical-weight-frames/result.json')['dominant_frame_updates'],cortical_contact=rd('contact-quadrature/result.json'),native_accepted_steps=0,qualification='The native scene validates topology, anatomical sources, all registered node frames, and 86-body role weights. Cortical nodal samples are compiled from certified material and skeletal inputs. The later runtime contact installation and force transmission through motion have not executed in this launch.',recovery={k:rec[k] for k in ('url','archive_sha256','size_bytes')})
    dest=p/'preflight-delivery';code=dest/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=dest/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123';code.mkdir(parents=True,exist_ok=True);ev.mkdir(parents=True,exist_ok=True)
    for file in scripts:shutil.copy2(p/file,code/file)
    (ev/'connected-skin-fat-native-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
