"""Publish the registered atlas cross-section and reversible native binding changes with rejection witnesses."""
from pathlib import Path
import json,shutil
import numpy as np
from archive_cavity_trial import sha,archive_files
from assemble_material import frames
from repair_registered_surface import state
from registration_stages import parts
from prepare_shoulder import read

def retain(w):
    stage=w/'atlas-publication';tool=stage/'tools/native_ligament_contact/registered_material_volume/support_observation/right_shoulder'
    out=stage/'evidence/human-connected-mesh-rebuild-20261105/m6-supported-scapular-source-20261124'
    out.mkdir(exist_ok=True,parents=True);tool.mkdir(exist_ok=True,parents=True);files={}
    for name in ('recover_scapular_source_column.py','rebind_scapular_charts.py','retain_scapular_atlas.py'):
        shutil.copy2(w/name,tool/name);files['code/'+name]=w/name
    case=w/'native-cavity-bound';cfg=read(case/'launch.json');args=cfg['argv'];original=Path(args[args.index('--soft-tissue-payload')+1]);pose=frames(case)[-1];bed=read(case/'support-check.json')
    normal=np.array(bed['bed_normal']);offset=bed['source_plane_offset_m']
    ref=read(w/'source-support-column.json');results={}
    if ref['ambiguous'] or ref['frame']['max_vertex_residual_m']>1e-6:raise ValueError('Unresolved atlas registration')
    for mode in ('scapular-charts','scapular-common-charts'):
        m=w/mode;report=read(m/'result.json');dst=m/original.name;after=dst.read_bytes();baseline=original.read_bytes();recon=bytearray(baseline)
        changed=np.flatnonzero(np.frombuffer(baseline,'u1')!=np.frombuffer(after,'u1'))
        if not len(changed)or report['reference']!=sha(original)or report['candidate']!=sha(dst):raise ValueError('Changed native bank or empty patch')
        recon_array=np.frombuffer(recon,'u1');delta=np.frombuffer(after,'u1')[changed];recon_array[changed]=delta
        if bytes(recon)!=after:raise ValueError('Unrecoverable native payload')
        packed=m/'source-delta.npz';np.savez_compressed(packed,offsets=changed,new=delta,old=np.frombuffer(baseline,'u1')[changed]);files[mode+'/source-delta.npz']=packed;files[mode+'/result.json']=m/'result.json'
        summary={}
        for sid in (153,155):
            vals=[]
            for p in (original,dst):
                x,wgt,o,f,a=parts(p,sid);xx=state(x,wgt,a,o,pose).astype('f4').astype(float);distance=xx@normal-offset;v=np.unique(f)
                vals.append(dict(gap_m=distance[v].min(),below_bed_vertices=int(np.sum(distance[v]<0))))
            r=report['metrics'][str(sid)];summary[sid]=dict(native=r['control'],rebound=r['scapula'],bed=vals,
                skin_fat=[a for a in r['surroundings']if a['kind']==0],unresolved=[a for a in r['surroundings'] if a.get('unresolved')])
        results[mode]=dict(payload=report['candidate'],bytes_changed=len(changed),patch=sha(packed),coverage=summary)
    result=out/'results.json';result.write_text(json.dumps(dict(original=sha(original),native=ref['native'],atlas=ref,variants=results),indent=2,allow_nan=False)+'\n')
    for f in ('bed-corridor.json','native-column.json','source-support-column.json','closed-branch-clones.log'):files[f]=w/f
    nat=read(w/'bed-corridor.json');length=nat['samples']['native'][0];new=results['scapular-common-charts']['coverage'][153];a,b=new['rebound']['depth_crossings_mm'][:2];deep=new['bed'][-1]['gap_m']*1000
    doc=out/'README.md';doc.write_text(f"""## Source-backed posterior shoulder registration

Skin node {nat['node']} carried {nat['fraction']*100:.2f}% of the terminal bed impulse. The first scapular surface was {length['depth_mm']:.3f} mm inside the registered skin. Direct comparison against all 13,118 corresponding vertices registered the source scapula to the accepted native frame with maximum error {1e6*ref['frame']['max_vertex_residual_m']:.3f} µm. The source query screened {ref['members_scanned']:,} atlas members and tested {ref['bounds_candidates']} possible intersections.

The unmodified atlas places the ascending trapezius between 10.86 and 20.49 mm along the measured load normal. The infraspinatus occupies the final 3.88 mm before the scapula. In the running model the trapezius begins behind the first cortical surface and the infraspinatus covers about 0.025 mm before it. This comparison uses the measured scapula frame. The other muscles have their own anatomical motion and registration.

Two reversible trials use the existing NHTISS4 format. Replacing only the scapular donor introduced folds. Transforming the complete donor set in the same source chart placed the ascending trapezius between {a:.2f} and {b:.2f} mm, with zero ascending-trapezius self-intersections in the three posed checks. Its minimum bed clearance became {deep:.3f} mm. It also increased overlap with the existing three-dimensional skin and fat. Both trials were rejected. The current coupled mechanics, repaired back and accepted skin/fat asset keep their existing owners.

For the next physical revision, the posterior muscle surfaces, local skin/fat boundaries and supported posture must be fitted together against the scapula and bed. Retain source mass, muscle and tendon attachments, the actual bed normal and measured force balance. A normal-line measurement does not determine three-dimensional force transmission.

BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International. Original headers are preserved. [Archive attribution and licensing](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)
""")
    files['README.md']=doc;files['results.json']=result;rec=out/'recovery.json'
    if not rec.exists():archive_files(files,'scapula-registered-atlas-'+sha(result)[:16]+'.tar.zst',rec,'Verified source registrations, native occupancy, two reversible rejected native binding changes.')
    if read(rec)['verified_member_sha256']!={k:sha(p)for k,p in files.items()}:raise ValueError('Restore failed')
    print('Archive verified',len(files))
if __name__=='__main__':retain(Path.cwd())
