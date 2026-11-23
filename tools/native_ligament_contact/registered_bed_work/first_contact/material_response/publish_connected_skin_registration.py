"""Archive the complete connected skin/fat registration and the exact integrated trial."""
import json,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(job):
    def rd(p):return json.loads((job/p).read_bytes())
    scripts=['compile_material_boundary_lineage.py','resolve_material_surface_rows.py','emit_connected_material_nodes.py','emit_connected_surface_tables.py','build_embedded_skin_registration.py','launch_material_registration.py','emit_connected_facet_uses.py','register_loaded_cortical_interfaces.py','rebind_authored_surface_parents.py','publish_connected_skin_registration.py']
    files=scripts+['pose-recovery.json','connected-boundary/result.json','connected-boundary/boundary-registration.npz','registered-boundary/registered-boundary.npz','registered-boundary/result.json','joint-node-source/result.json','joint-node-source/I-new.csv','joint-node-source/D-new.csv','loaded-cortex/registered-cortical-facets.npz','loaded-cortex/result.json','connected-face-uses-fast/result.json','embedded-app/build.json','embedded-app/app/NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp','embedded-app/app/numi-human-native','joint-node-rows/result.json','connected-surface-tables/result.json']
    for folder in ['scene-joined-map','scene-connected-surfaces','scene-authored-parents','scene-connected-facets']:
        files+=[folder+'/'+key for key in ['launch.json','stderr.log','execution.json']]
    catalog=rd('connected-face-uses-fast/result.json')
    for v in catalog['files'].values():
        p=Path(v['path'])
        if sha(p)!=v['sha256']:raise ValueError('Stale physical sidecar')
        files.append(str(p.relative_to(job)))
    recent=rd('joint-node-source/result.json');surf=rd('registered-boundary/result.json');walls=rd('loaded-cortex/result.json');stages=[]
    for folder in ['scene-joined-map','scene-connected-surfaces','scene-authored-parents','scene-connected-facets']:
        lines=(job/folder/'stderr.log').read_text().splitlines();stages.append(dict(launch=sha(job/folder/'launch.json'),exit=rd(folder+'/execution.json')['exit'],native=next(v for v in lines if v.startswith('registered_volume_native_preflight='))))
    if recent['created_nodes']<=0 or recent['retired_historical_parent_nodes'] or any(v.get('unresolved_count',0)>0 for v in surf['layers'].values()) or 'external child facet' not in stages[-1]['native']:raise ValueError('Unmatched integration outcome')
    rec=job/'surface-recovery.json'
    if not rec.exists():
        archive_files({k:job/k for k in files},'connected-surface-registration-'+sha(job/'scene-connected-facets/launch.json')[:16]+'.tar.zst',rec,'Source-bound O/I/D table generation, current material joint and facet IDs, body-contact quadrature and the full native scene through strict external-facet admission.')
    a=rd('surface-recovery.json');dest=job/'surface-delivery';code=dest/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';ev=dest/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123'
    code.mkdir(parents=True,exist_ok=True);ev.mkdir(parents=True,exist_ok=True)
    for f in scripts:shutil.copy2(job/f,code/f)
    report=dict(native_stages=stages,device='SSH Numi Apple M6',node_mapping=recent,registered_surface=surf['layers'],emitted_face_uses=catalog['face_uses'],loaded_cortical_interfaces=walls,prior_static_state=rd('pose-recovery.json')['url'],current_scene_accepted_roots=0,remaining='The registered volume reader has consumed the connected source-backed surfaces and joint-node map. Its external-boundary classification still supports only registered O/D surfaces and historical head walls. The current scene requires independently validated cortical interfaces connected to the physical contact path.',recovery={k:a[k] for k in ['url','archive_sha256','size_bytes']})
    (ev/'connected-skin-fat-native-registration.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
