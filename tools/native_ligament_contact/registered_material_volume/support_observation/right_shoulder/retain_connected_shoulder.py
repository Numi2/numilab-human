"""Retain source-constrained material registration and its measured relationship to native bed load."""
from pathlib import Path
import json,shutil
from archive_cavity_trial import archive_files,sha

def read(p):return json.loads(p.read_bytes())
def publish(w):
    stage=w/'source-publication'
    evidence=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-shoulder-cortex-and-load-20261124'
    tool=stage/'tools/native_ligament_contact/registered_material_volume/support_observation/right_shoulder'
    evidence.mkdir(parents=True,exist_ok=True);tool.mkdir(parents=True,exist_ok=True)
    paths={}
    for n in ['locate_attachment_conflicts.py','cortical_registration_metric.py','coupled_shoulder_poses.py','restore_shoulder_volume.py','fit_registered_assembly.py','transfer_shoulder_fit.py','choose_emitted_shoulder.py','trace_registered_shoulder_load.py','observe_native_support_column.py','qualify_native_support_column.py','retain_connected_shoulder.py']:
        paths['source/'+n]=w/n;shutil.copy2(w/n,tool/n)
    folders=['cortical-joint','cortical-emitted','cortical-bounded','bounded-emitted','physical-joint','physical-emitted','volume-emitted']
    for folder in folders:
        for f in (w/folder).iterdir():
            if f.suffix in ('.npz','.json'):paths[str(f.relative_to(w))]=f
    for n in ('attachment-conflicts.json','cortical-layers.json','bounded-layers.json','physical-choices/results.json','physical-layers.json',
              'physical-contacts.json','physical-volume-choices/results.json','volume-contacts.json','volume-layers.json','volume-attachment.json','bed-corridor.json','native-column.json','closed-registration-clones.log'):
        paths[n]=w/n
    for n in ('launch.json','execution.json','stdout.log','stderr.log','physical-posterior-tissue-cells.csv','physical-posterior-tissue-objects.csv','physical-posterior-bed-history.csv','loaded-scapular-bounds.txt','native-traces/resting-coupled.csv'):
        paths['native-observer/'+n]=w/'native-shoulder-footprint'/n
    rec=read(w/'volume-emitted/results.json');old=read(w/'attachment-conflicts.json');new=read(w/'volume-attachment.json');mech=read(w/'native-column.json');corr=read(w/'bed-corridor.json')
    mat=read(w/'volume-layers.json');pose=read(w/'physical-joint/result.json')
    if not pose['joint_source_pose_constraints'] or rec['emitted_pair_sum_mm3']>rec['reference_pair_sum_mm3']:raise ValueError('Missing source-space acceptance')
    if any(c['minimum_motion_J']<=0 or any(z['surface_crossing_pairs'] or z['nonpositive'] for z in c['stages'].values())for c in rec['metrics'].values()):raise ValueError('Invalid source or posed material')
    initial=[r for r in old['cases'] if r['site_kind']=='fixed_material_vertex'and r['obstacle_kind']==51004];finalcs=[r for r in new['cases'] if r['site_kind']=='fixed_material_vertex'and r['obstacle_kind']==51004]
    if any(r['obstacle_kind']==0 and r['site_kind']!='fixed_material_vertex'for r in old['cases']+new['cases']):raise ValueError('Tendon endpoint in skin or fat')
    base=sum(corr['samples']['native'][0]['segments'][i]['end_mm']-corr['samples']['native'][0]['segments'][i]['start_mm'] for i in range(len(corr['samples']['native'][0]['segments'])) if not corr['samples']['native'][0]['segments'][i]['registered_occupants'])
    summary=dict(source=sha(w/'physical-joint/result.json'),native=mech,source_selection=rec,source_layer_results=mat['rows'],material_proxy_inside_cortex_before=len(initial),material_proxy_inside_cortex_after=len(finalcs),
        deepest_proxy_mm_before=max(r['distance_inside_mm']for r in initial),deepest_proxy_mm_after=max(r['distance_inside_mm']for r in finalcs),
        true_tendon_manifest=old['manifest'],actual_tendon_endpoints_and_distributed_sites_unchanged=True,dominant_receiver_fraction=corr['fraction'],registered_unassigned_mm=base,candidate_registered_unassigned_mm=corr['samples']['source_candidate'][0]['unassigned_mm'])
    result=evidence/'results.json';result.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');paths['results.json']=result
    tissue=next(r['mm3']/1000 for r in mat['rows']if r['material']==16 and r['muscle']==79 and r['stage']=='after')
    count=mech['coverage'][0]['cells'];gap=mech['coverage'][0]['unassigned_mm'];L=corr['samples']['native'][0]['depth_mm']
    text=f"""## Native bed reaction and connected shoulder registration

The matched whole-body trial measured a terminal bed reaction of {corr['skin_bed_N']:.4f} N. Skin node {mech['receiver']} carried {100*corr['fraction']:.2f}% of the normal impulse. Along its bed normal the first scapular surface was {L:.3f} mm below skin. The terminal observer checked {len(mech['observer_objects'])} native continuum objects, reporting {count} additional tetrahedra intersecting the central line. After combining those materials with the registered skin, fat and muscle intervals, {gap:.3f} mm remained unassigned along that line. The four offset rays and every measured interval are in the recovery archive. This local test does not resolve oblique force paths or unmodeled anatomical tissues.

The prior fit froze {len(initial)} inferred muscle surface vertices more than 0.05 mm inside a cortical bone. The source tendon sites, force-distribution records and their moment arms retain their original bytes. The authoring repair allows those inferred volume vertices to move toward their measured cortical exits. It checks source, skeletal and supported poses while solving for the displacement and restores intermediate material volume before choosing the six muscles jointly.

Tested summed overlap across the six muscle regions changed from {rec['reference_pair_sum_mm3']/1000:.4f} to {rec['emitted_pair_sum_mm3']/1000:.4f} cm³ in emitted coordinates. Multiple overlapping structures can contribute to this sum. Spinal deltoid and fat still share {tissue:.4f} cm³. The candidate remains a registration result. The actual new native material in the coupled run is the conservatively remapped adipose cavity recorded in the preceding commit. Supported resting equilibrium, whole-body clearance and the 16-second performance target remain open.

BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International. Original OBJ headers are preserved in the recovered source. [Archive licensing](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)
"""
    doc=evidence/'README.md';doc.write_text(text);paths['README.md']=doc;recovery=evidence/'recovery.json'
    if not recovery.exists():archive_files(paths,'connected-shoulder-load-'+sha(result)[:16]+'.tar.zst',recovery,'Joint posed material registration, real native shoulder column observation, all candidate outcomes and restored source.')
    if read(recovery)['verified_member_sha256']!={k:sha(v)for k,v in paths.items()}:raise ValueError('Recovery mismatch')
    print('verified',len(paths),'files')
if __name__=='__main__':publish(Path.cwd())
