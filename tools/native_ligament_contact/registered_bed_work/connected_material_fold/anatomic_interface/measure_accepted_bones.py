"""Measure cortical overlap using matching accepted rigid and deformable coordinates."""
from pathlib import Path
import json
import numpy as np
from compile_connected_joint import digest
from measure_bone_tissue_overlap import run as audit


def run(work):
    study = work / 'witness-material'
    volume = study / 'volume'
    native = work / 'accepted-material-07'
    receipt = native / 'native-traces/accepted-geometry/step-1.receipt.json'
    frame = json.loads(receipt.read_bytes())
    motion = json.loads((study / 'accepted-motion.json').read_bytes())
    nodes = native / 'physical-accepted.node'
    if frame['accepted_step'] != 1 or digest(nodes) != motion['file'] or any(r['inverted'] for r in motion['roles'].values()):
        raise ValueError('Require the audited, oriented terminal material mesh')
    rows = frame['accepted_body_poses']
    if [r['body_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Incomplete native articulated poses')
    # Registered rows are the subset used by the viewer. A bank can also belong
    # to an articulated body outside that subset; its accepted transform exists.
    for row in frame['accepted_registered_body_poses']:
        i = row['body_index']
        for key in ('position_m','quaternion_xyzw'):
            if not np.array_equal(rows[i][key],row[key]):
                raise ValueError('Viewer and native articulated transform differ')
    j = work.parents[2]
    baseline = json.loads((j/'distal-registered-native/target-pose.json').read_bytes())['all_bodies']
    if len(rows) != len(baseline):
        raise ValueError('Pose inventory mismatch')
    delta = [np.linalg.norm(np.subtract(r['position_m'],b['current']['position_world_m'])) for r,b in zip(rows,baseline)]
    pose = native/'accepted-registered-bone-pose.json'
    pose.write_text(json.dumps(dict(receipt_sha256=digest(receipt),all_bodies=[dict(body_index=r['body_index'],current=dict(position_world_m=r['position_m'],orientation_world_xyzw=r['quaternion_xyzw'])) for r in rows]))+'\n')
    inputs = study/'motion-bone-input'
    inputs.mkdir(exist_ok=True)
    for file,target in [('mesh.node',nodes),('mesh.ele',volume/'mesh.ele')]:
        link=inputs/file
        if not link.exists():
            link.symlink_to(target)
        if link.resolve() != target.resolve():
            raise ValueError('Stale accepted material input')
    out = study / 'motion-bone-volumes'
    if not (out/'result.json').exists():
        audit(j.parent/'ligament-equilibrium-070',out,pose,None,inputs)
    bank = json.loads((out/'result.json').read_bytes())
    if bank['capture'] != digest(pose) or bank['mesh'] != digest(nodes) or bank['elements'] != digest(volume/'mesh.ele'):
        raise ValueError('Stale cortical measurement')
    prior = json.loads((work/'verified-material.json').read_bytes())['remaining_summed_bone_overlap_mm3']
    report = dict(receipt=digest(receipt),banks=len(bank['results']),native_body_count=len(rows),registered_body_count=len(frame['accepted_registered_body_poses']),maximum_body_translation_from_start_m=max(delta),accepted_s=motion['accepted_s'],tissues={})
    for k in ('skin','fat'):
        stats=bank['summary'][k]
        if stats['unresolved']:
            raise ValueError('Indeterminate cortical intersections')
        summed=sum(r[k] for r in bank['results'].values())
        report['tissues'][k]=dict(sum_mm3=summed,change_from_start_mm3=summed-prior[k],positive_banks=stats['positive_banks'],largest_banks_mm3=stats['largest_mm3'])
    report['scope']='Pairwise cortical-solid and material-solid intersection volumes at the accepted endpoint. Different pairs can occupy the same volume.'
    (study/'accepted-bones.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)

if __name__=='__main__':
    run(Path(__file__).resolve().parent)
