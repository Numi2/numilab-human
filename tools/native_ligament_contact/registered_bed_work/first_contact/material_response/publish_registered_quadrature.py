"""Archive the restored authoring source and native evidence at the last verified stage."""
import json,shutil,sys
from pathlib import Path
from archive_cavity_trial import archive_files,sha
def run(p):
    read=lambda f:json.loads((p/f).read_bytes())
    scripts=['rebuild_connected_p1.py','run_material_support.py','compose_registered_carriers.py','solve_connected_carrier_flow.py','bind_connected_quadrature.py','register_connected_cell_labels.py','bind_cell_label_incidence.py','publish_registered_quadrature.py']
    folders=['material-p1','adjacent-carriers','connected-quadrature','connected-child-labels','cell-label-incidence']
    stages=['scene-material-p1','scene-material-carriers','scene-material-labels','scene-material-incidence'];files={s:p/s for s in scripts}
    for d in folders:
        for f in (p/d).rglob('*'):
            if f.is_file():files[str(f.relative_to(p))]=f
    for d in stages:
        for name in ['launch.json','stdout.log','stderr.log','execution.json']:files[d+'/'+name]=p/d/name
    files['preflight-recovery.json']=p/'preflight-recovery.json';rec=p/'registered-quadrature-recovery.json'
    if not rec.exists():archive_files(files,'registered-quadrature-'+sha(p/'scene-material-incidence/launch.json')[:16]+'.tar.zst',rec,'Electronically reproduced physical role registration and quadrature, conserved point and cell totals, native eligibility gates, and subsequent recorded-flow rejection before any native step.')
    archive=read(rec.name);controls={d:next(s for s in (p/d/'stderr.log').read_text().splitlines() if s.startswith('myosim_articulated_visual=failed')) for d in stages}
    ev=dict(device='SSH Numi Apple M6',source=read('preflight-recovery.json')['archive_sha256'],p1={k:{i:v[i] for i in ['cells','points','reprojected','identical_retained_cells']} for k,v in read('material-p1/result.json')['roles'].items()},carrier_flow={k:{i:v[i] for i in ['carrier_count','cells','flow_edges','maximum_source_error_kg','maximum_receiver_error_kg','unresolved_parents','joint_allocation']} for k,v in read('adjacent-carriers/result.json').items()},quadrature={k:{i:v[i] for i in ['carriers','preserved_primary','rebound_primary','maximum_flow_distance_m','out_of_radius_edges']} for k,v in read('connected-quadrature/result.json')['roles'].items()},cell_labels=read('connected-child-labels/result.json')['roles'],incidence_nodes=read('cell-label-incidence/result.json')['updated_incident_nodes'],native_stages=controls,qualification='Source quadrature, current P1 input and native node eligibility passed. All current receiver and source masses close independently. An updated recorded allocation is still required to establish native physical admission. No motion or end-to-end throughput measured.',native_accepted_steps=0,recovery={k:archive[k] for k in ('url','archive_sha256','size_bytes')})
    root=p/'registered-quadrature-delivery';code=root/'tools/native_ligament_contact/registered_bed_work/first_contact/material_response';data=root/'evidence/human-connected-mesh-rebuild-20261105/m6-first-contact-20261123';code.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    for s in scripts:shutil.copy2(p/s,code/s)
    (data/'connected-material-source-quadrature.json').write_text(json.dumps(ev,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
