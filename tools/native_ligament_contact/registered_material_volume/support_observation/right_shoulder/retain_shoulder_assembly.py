"""Publish reproducible multi-pose shoulder material and its surrounding-body decision."""
import json,shutil,sys,zipfile,hashlib
from pathlib import Path
import numpy as np
from archive_cavity_trial import sha,archive_files


def read(p):return json.loads(p.read_bytes())


def retain(w):
    evidence=w/'publication/evidence/human-connected-mesh-rebuild-20261105/m6-connected-shoulder-volumes-20261124'
    tools=w/'publication/tools/native_ligament_contact/registered_material_volume/support_observation/right_shoulder'
    evidence.mkdir(parents=True,exist_ok=True);tools.mkdir(parents=True,exist_ok=True)
    reg=read(w/'registered/registration.json');src=read(w/'atlas/recovery.json');lib=w/'official-isa.zip'
    if sha(lib)!=src['archive_sha256']:raise ValueError('Atlas hash mismatch')
    with zipfile.ZipFile(lib)as z:
        for member in src['members'].values():
            if hashlib.sha256(z.read(member['member'])).hexdigest()!=member['member_sha256']:raise ValueError('Unmatched source member')
    cert=read(w/'certified/result.json');near=read(w/'surrounding.json');layers=read(w/'layers.json')
    if set(map(int,cert))!={75,79,83,85,87,89} or any(v['native_motion_min_J']<=0 or any(k['nonpositive']for k in v['stages'].values())for v in cert.values()):raise ValueError('Unqualified shoulder material')
    if not all(r['after_mm3']>=0 for r in near['pairs']) or len(layers['rows'])!=24:raise ValueError('Missing occupancy results')
    if {(i['kind'],i['bank'])for i in near['indeterminate']}!={(0,15),(0,16)}:raise ValueError('Unresolved neighboring solids')
    files={}
    for name in ('prepare_shoulder.py','recover_atlas_shoulders.py','trace_registrations.py','compare_original.py','repair_registered_surface.py',
                 'prepare_winding_material.py','assemble_material.py','measure_assembly_contacts.py','measure_material_layers.py','certify_material_motion.py','repair_material_cavities.py','retain_shoulder_assembly.py'):
        shutil.copy2(w/name,tools/name);files['source/'+name]=w/name
    for d in ('atlas','original-comparison','local-repair','winding','common-material','motion','certified'):
        for p in (w/d).glob('*'):
            if p.is_file() and p.suffix in ('.npz','.json','.obj'):files[d+'/'+p.name]=p
    for p in (w/'common-material').glob('cells-*/material.npz'):files[str(p.relative_to(w))]=p
    for p in (w/'registered').glob('mesh-*/source.poly'):files[str(p.relative_to(w))]=p
    for name in ['source-lock.json','surrounding.json','layers.json','registered/neighbors.npz','registered/registration.json','stages/summary.json']:files[name]=w/name
    resid=read(w.parent/'posterior-interface-082/native-third-matched/failure.json')
    support=read(w.parent/'posterior-interface-082/native-pair/support-check.json')
    def getpair(i,k,b):return next(r for r in near['pairs'] if (r['muscle'],r['kind'],r['bank'])==(i,k,b))
    mb=next((w/'wheel-cache').glob('libigl*.whl'))
    summary=dict(archive=src,independently_verified_complete_archive=sha(lib),offline_compiler_wheel=dict(name=mb.name,sha256=sha(mb)),
        repaired_cells={k:{s:d[s]for s in ('cells','faces','native_motion_min_J','stages')}for k,d in cert.items()},
        repair=read(w/'common-material/results.json'),adjacent_pair=getpair(83,51005,87),worsened_cortex=getpair(75,51004,10),layers=layers['rows'],
        native_control=dict(loaded_N=support['accepted_normal_force_N'],weight_N=support['normal_weight_N'],time_s=support['time_s'],
                            failed_root=resid['segment_start'],physical_residual=resid['matter_diagnostics'][0]),
        neighbor_pairs=len(near['pairs']),parent_payload=reg['surface'],parent_tendon=reg['tendon'])
    target=evidence/'results.json';target.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    r=summary['adjacent_pair'];bad=summary['worsened_cortex'];n=sum(s['cells']for s in cert.values())
    (evidence/'README.md').write_text(f'''# Registered right shoulder material on Apple M6

Six actuator-linked source surfaces can now be represented by {n:,} tetrahedra. Each has a closed two-manifold boundary and positive cell volume in the atlas, skeletal reference, supported opening and reconstructed accepted endpoint. The lowest certified determinant ratio along the opening-to-endpoint chord is {min(x["native_motion_min_J"]for x in cert.values()):.6f} relative to the atlas material. The actual retained native prefix lasted {support["time_s"]*1e6:.3f} µs. This is an offline replay of its poses.

Three stored surfaces were self-intersecting before posing. The acromial deltoid, subscapularis and teres major now use the checksum-verified BodyParts3D members. The preparation resolves the positive-winding material boundary and records birth faces, barycentric ancestry, singular vertex-link repairs and changes to the inferred registration weights. Existing scapular and humeral donor identities are retained. The other three muscles use their existing source surfaces. The combined setup retains tendon endpoints and the earlier back registration. Rebuilding a muscle boundary does not authorize adding its volume a second time to the body's inertial mass.

The infraspinatus–teres major intersection decreased from {r["before_mm3"]/1000:.5f} to {r["after_mm3"]/1000:.6f} cm³. The acromial deltoid–scapula intersection increased from {bad["before_mm3"]:.2f} to {bad["after_mm3"]:.2f} mm³. The separate physical tetrahedron audit measures all six muscles against both skin and fat. The spinal deltoid still shares 43.36 cm³ with fat. The anatomical assembly fails whole-body admission. Neither the six shoulder volumes nor their inferred bindings were activated as additional native force owners.

The retained native control supplied {support["accepted_normal_force_N"]:.2f} N against {support["normal_weight_N"]:.2f} N normal weight, and its next physical root rejected at residual {resid["matter_diagnostics"][0]:.7f}. Supported rest, long-duration motion and performance remain open.

## Reproduction on SSH Numi

Restore the payload identified in recovery.json. Use the established Human preparation environment and source-module paths. libigl 2.6.3 is used only by the offline source repair and geometric audit. The wheel hash is in results.json. Run prepare_shoulder, trace_registrations, recover_atlas_shoulders and compare_original against the pinned scene and atlas. Run repair_registered_surface and prepare_winding_material for the retained comparisons, then assemble_material, certify_material_motion and repair_material_cavities. Run measure_assembly_contacts and measure_material_layers on the same capture. The release retains the original geometry, failed candidates and the certified arrays. The native solver and acceptance thresholds are unchanged.

{src["attribution"]}.
''')
    files.update({'results.json':target,'README.md':evidence/'README.md'})
    dest=evidence/'recovery.json'
    if not dest.exists():archive_files(files,'common-shoulder-'+sha(target)[:16]+'.tar.zst',dest,'Six pose-checked shoulder muscle volumes, authenticated atlas members and failed whole-body geometry comparisons.')
    if read(dest)['verified_member_sha256']!={n:sha(p)for n,p in files.items()}:raise ValueError('Independent restoration failed')
    print('restored',len(files),'entries',n,'cells',flush=True)
if __name__=='__main__':retain(Path.cwd())
