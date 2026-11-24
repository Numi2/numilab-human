"""Preserve the source-derived material and the coupled registration witnesses."""
import json
from pathlib import Path
import shutil
import sys
from archive_cavity_trial import archive_files,sha


def read(p):return json.loads(p.read_bytes())


def retain(w):
    out=w/'publication/evidence/human-connected-mesh-rebuild-20261105/m6-loaded-right-shoulder-20261124'
    tools=w/'publication/tools/native_ligament_contact/registered_material_volume/support_observation/right_shoulder'
    out.mkdir(parents=True,exist_ok=True);tools.mkdir(parents=True,exist_ok=True)
    files={}
    for name in ('prepare_infraspinatus.py','repair_infraspinatus.py','assess_scapular_interface.py','partition_infraspinatus.py',
                 'fit_scapular_material.py','shoulder_neighbors.py','registration_stages.py','fit_surrounding_tissues.py','retain_scapular_material.py'):
        shutil.copyfile(w/name,tools/name);files['source/'+name]=w/name
    for d in ('sourced','repaired','scapular-partition','connected-fit','surrounding-fit'):
        for f in (w/d).iterdir():
            if f.is_file():files[d+'/'+f.name]=f
    for f in ('cortex.json','neighbors.json','registration-stages.json'):
        files[f]=w/f
    native=w.parent/'posterior-interface-082';frame=native/'native-pair'
    for f in ('loaded-shoulder.json','footprint.json'):files['native/'+f]=native/f
    for f in ('launch.json','native-initial-bodies.json','native-traces/accepted-geometry/step-2.receipt.json'):
        files['native/'+f]=frame/f
    q=read(w/'repaired/quality.json');s=read(w/'registration-stages.json');c=read(w/'cortex.json')
    part=read(w/'scapular-partition/partition.json');n=read(w/'neighbors.json');fit=read(w/'connected-fit/fit.json')
    total=read(w/'surrounding-fit/fit.json')
    pairs={tuple(z['pair']):z for z in s['pairwise'] if 83 in z['pair']}
    ledger=dict(known_parent_commit='4fea5cbf9b365d1d590c364a2932a15448129233',registration=read(w/'sourced/qualification.json'),
                repaired=q,cortical=c,partition=part,connected_fit=fit,surrounding=n,stages=s,broad_fit=total)
    target=out/'results.json';target.write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
    r=part['stages']['accepted'];R=q['after']['accepted'];cortex=c['results']['accepted']['interactions']
    b={s['bank']:s['overlap_mm3'] for s in cortex}
    assert R['inverted']==0 and q['faces_preserved']==866 and q['motion_min_J_bound']>0 and len(s['pairwise'])==28 and len(n['surrounding'])==74
    assert any(v['kind']==51005 and v['bank']==75 and v['after']>1 and v['before']<1e-7 for v in n['surrounding'])
    body=f'''# Right shoulder material and registration decision

The complete native frame in the preceding experiment placed 67% of its terminal bed impulse on one overlying skin node. The reference muscle there is the right infraspinatus, stable ID 83, BodyParts3D member FJ1500 and MyoSim actuator INFSP. Its existing source donors are the right humerus and scapula. This package preserves surface identity, bindings and the active muscle owner.

## Material construction

TetGen ran on the exact welded source surface on SSH Numi. One interior tetrahedron inverted under the existing donor registration. Replacing three tetrahedra with two restores positive orientation across all {q["cells"]:,} cells. All {q["faces_preserved"]} original boundary triangles and all {q["vertices"]} material vertices are preserved. The lowest determinant ratio along the straight interpolation from the opening state to the two-step accepted state was bounded below by {q["motion_min_J_bound"]:.6f}. This covers 15.625 µs. Surface volume agrees with the captured muscle to better than 0.0001 cm³. Determinant positivity alone does not qualify the tissue law or attachment.

## Contact and surrounding tissues

All 185 cortical banks were covered by the distance and solid tests, with spatial exclusion for separated banks. At the accepted endpoint the muscle penetrates the scapula by {b[10]/1000:.4f} cm³ and the humerus by {b[12]/1000:.4f} cm³.

A direct partition preserves the source-face ancestry of the resulting cortical interfaces. It splits off {min(r["component_volumes_mm3"]):.3f} mm³ and displaces {r["displaced_mm3"]:.3f} mm³ of muscle. The partition is retained for attribution.

A connected fit conserved total muscle volume and reduced both bone intersection volumes below 3e-9 mm³. Its maximum material motion was {fit["peak_registration_mm"]:.3f} mm. Checking 74 neighboring displayed solids exposed a new 1.39 mm³ acromial deltoid intersection and increased spinal deltoid and teres major intersection volumes. This disqualifies the bone-only fit.

A second fit considered surrounding fixed tissues together and retained positive relative material determinants. Its largest point correction was {total["peak_correction_mm"]:.2f} mm. It still left {total["final_geometry_mm3"]["(51005, 153)"]:.1f} mm³ of ascending trapezius overlap and increased total volume by {total["final_volume_error_mm3"]:.1f} mm³. This candidate also failed admission. Both fits are authoring experiments.

## Registration ownership

The eight-bank comparison preserves the stored source coordinates, neutral skeletal reconstruction, supported opening reconstruction and actual accepted native triangles as separate stages. Pairwise volumes may double-count occupied space. They are not tissue mass estimates.

| Infraspinatus partner | Stored mesh coordinates | Neutral skeletal reconstruction | Accepted native geometry |
| --- | ---: | ---: | ---: |
| Right ascending trapezius | {pairs[83,153]["stored"]/1000:.3f} cm³ | {pairs[83,153]["skeletal_reference"]/1000:.3f} cm³ | {pairs[83,153]["accepted"]/1000:.3f} cm³ |
| Right teres major | {pairs[83,87]["stored"]/1000:.3f} cm³ | {pairs[83,87]["skeletal_reference"]/1000:.3f} cm³ | {pairs[83,87]["accepted"]/1000:.3f} cm³ |

The trapezius mismatch is already substantial in the skeletal reconstruction. The teres major overlap also exists in the stored mesh inputs. The results include separate current body and muscle owners, rotations and reconstruction errors. The two trapezius surfaces differ from the simple donor reconstruction at the accepted state by up to 41 µm. The accepted column uses the actual native surfaces.

Further shoulder qualification requires a common material registration for these interacting structures with the humeral and scapular attachments preserved. Carry the right infraspinatus volume as a reference asset. Keep the current simulator as the control until the combined material, geometry, mass transfer and load response pass. None of the shoulder candidates in this archive altered the native run.
'''
    (out/'README.md').write_text(body)
    files['results.json']=target;files['README.md']=out/'README.md'
    dest=out/'recovery.json'
    if not dest.exists():archive_files(files,'loaded-scapular-material-'+sha(target)[:18]+'.tar.zst',dest,'Right infraspinatus volume and neighboring bone and muscle intersections in the accepted native frame, including rejected registrations.')
    if read(dest)['verified_member_sha256']!={k:sha(v)for k,v in files.items()}:raise ValueError('Restore mismatch')
    print(len(files),'verified',flush=True)

if __name__=='__main__':retain(Path.cwd())
