"""Archive the original mass-source replay, its registered frame binding and the layer rejection."""
import json, shutil, sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(job):
    tools=['capture_parent_placement.py','bind_original_material_placement.py','bind_registered_donor_coordinates.py','audit_registered_layers.py','publish_registered_mass_frames.py']
    names=tools+['inspect_coarse_reference.py','qualify_native_parent_roundtrip.py','accepted-material-mass-recovery.json','registered-donor-cook/frame-registration.json','registered-donor-cook/build.json']
    for folder in ['coarse-reference-audit','parent-roundtrip','material-source-placement','independent-parent','registered-donor-cook']:
        names += [str(f.relative_to(job)) for f in (job/folder/'scene').glob('*') if f.is_file() and f.suffix in {'.json','.log'}]
    names += [str(f.relative_to(job)) for f in (job/'connected-layer-crossings').glob('*.json')]
    names += ['registered-donor-cook/app/'+s for s in ['NumiHumanPhysicalSkinReplacement.hpp','numilab_human_myosim_visual_probe.mm','numi-human-native']]
    receipt=job/'registered-frame-recovery.json'
    if not receipt.exists():archive_files({s:job/s for s in names},'registered-skeletal-tissue-frames-'+sha(job/names[-1])[:14]+'.tar.zst',receipt,'Verified physical tissue source replay, registered donor coordinates, native scene logs, source code and whole-layer intersection witnesses.')
    info=json.loads(receipt.read_bytes())
    read=lambda f:json.loads((job/f).read_bytes())
    dest=job/'registered-frame-delivery'
    code=dest/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response'
    evidence=dest/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    for d in (code,evidence):d.mkdir(exist_ok=True,parents=True)
    for s in tools:shutil.copy2(job/s,code/s)
    layers={s:read('connected-layer-crossings/'+s+'.json') for s in ['reference','posed']}
    result=dict(device='SSH Numi Apple M6',material_mass_frames=read('registered-donor-cook/frame-registration.json'),surfaces=layers,rejection=next(s for s in (job/'registered-donor-cook/scene/stderr.log').read_text().splitlines() if s.startswith('myosim_articulated_visual=failed')),accepted_steps=0,recovery={k:info[k] for k in ('url','archive_sha256','size_bytes')})
    (evidence/'source-tissue-mass-registered-skeletal-frames.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
