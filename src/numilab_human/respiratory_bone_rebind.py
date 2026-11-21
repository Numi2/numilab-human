"""Rebind unchanged respiratory geometry after a source-declared shoulder repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numilab_human.capsular_material_registration import material_bank, read_material_banks

def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def rebind(parent, candidate, anatomy, receipt):
    old, new = read_material_banks(parent), read_material_banks(candidate)
    doc = json.loads(receipt.read_text())
    bound = doc['functional_bindings']
    prov = doc['provenance']
    declared = new[0]['source']['connected_girdle_parent']
    if old[1] != declared['bone_sha256'] or old[1] != bound['bones_payload_sha256'] or old[1] != prov['bones_payload_sha256']:
        raise ValueError('Bone parent, receipt or provenance mismatch')
    if file_hash(anatomy) != bound['anatomy_payload_sha256'] or old[2].keys() != new[2].keys():
        raise ValueError('Anatomy payload or registered bone inventory changed')
    ribs = bound['rib_bone_stable_ids']
    sternum = bound['sternum_bone_stable_ids']
    thorax = set(ribs + sternum)
    if len(ribs) != 24 or len(sternum) != 2 or len(thorax) != 26 or not thorax <= old[2].keys():
        raise ValueError('Invalid respiratory bone inventory')
    changed = []
    for sid, record in old[2].items():
        a, r, v, f = material_bank(old, sid, record[0])
        b, s, w, g = material_bank(new, sid, record[0])
        keys = ('core_body_index', 'source_record_index', 'member_id', 'member_sha256', 'myosim_body')
        if r != s or any(a[k] != b[k] for k in keys) or not np.array_equal(f, g):
            raise ValueError(f'Changed binding, rest frame or connectivity on bone {sid}')
        if v.tobytes() != w.tobytes():
            if sid in thorax or a['myosim_body'] not in {'clavicle_r', 'clavicle_l'}:
                raise ValueError(f'Unexpected geometry change on bone {sid}')
            changed.append(dict(stable_id=sid, member_id=a['member_id'], core_body=record[0]))
    if sorted(changed,key=lambda r:r['stable_id']) != sorted(declared['changed_anchors'],key=lambda r:r['stable_id']):
        raise ValueError('Changed bones disagree with the source manifest')
    attachments = []
    for key, row in bound.items():
        if not key.endswith('_motion_attachment') or not isinstance(row, dict):
            continue
        channel = row['existing_rib_channel']
        sid = channel['stable_id']
        a = material_bank(new, sid, channel['body_index'])[0]
        if (sid not in ribs or row['bones_payload_sha256'] != old[1] or
            row['anatomy_payload_sha256'] != bound['anatomy_payload_sha256'] or
            (channel['source_member_id'],channel['source_member_sha256']) != (a['member_id'],a['member_sha256'])):
            raise ValueError(f'Lost respiratory attachment identity at {key}')
        row['bones_payload_sha256'] = new[1]
        attachments.append(key)
    expected = {f'intercostal_313_rib{sid}_motion_attachment' for sid in (48, 49, 50)}
    if not expected <= set(attachments):
        raise ValueError('Incomplete source intercostal attachment inventory')
    bound['bones_payload_sha256'] = new[1]
    prov['bones_payload_sha256'] = new[1]
    prov['bones_payload'] = str(candidate)
    proof = dict(parent_receipt_sha256=file_hash(receipt),parent_bones_sha256=old[1],bones_sha256=new[1],
                 anatomy_sha256=bound['anatomy_payload_sha256'],changed_bones=changed,identical_respiratory_banks=sorted(thorax),
                 identical_registered_banks=len(old[2])-len(changed),material_attachments=sorted(attachments))
    prov['respiratory_bone_girdle_rebind'] = proof
    return doc, proof

def write_bundle(doc, proof, parent_receipt, output, verification):
    common=doc['provenance']['cardiac_geometry_binding']['common_field']
    sidecars={}
    for field in ('map','domain_boxes','polynomials'):
        data=common[field]
        relative=Path(data['path'])
        if relative.is_absolute() or '..' in relative.parts or not relative.name:
            raise ValueError(f'Invalid cardiac sidecar path for {field}')
        source=parent_receipt.parent/relative
        if file_hash(source)!=data['sha256']:
            raise ValueError(f'Cardiac sidecar identity changed for {field}')
        dest=output.parent/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            with tempfile.TemporaryDirectory(dir=dest.parent) as scratch:
                staged=Path(scratch)/'sidecar'
                subprocess.run(['/bin/cp','-c','-p',source,staged],check=True)
                if file_hash(staged)!=data['sha256']:raise ValueError('Cardiac sidecar copy mismatch')
                os.link(staged,dest)
        if file_hash(dest)!=data['sha256']:raise ValueError(f'Conflicting cardiac sidecar for {field}')
        sidecars[str(relative)]=data['sha256']
    proof['cardiac_sidecars']=sidecars
    for path,data in ((output,doc),(verification,proof)):
        with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as stream:
            staged=Path(stream.name)
            try:
                json.dump(data,stream,separators=(',',':'),sort_keys=True,allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.replace(staged,path)
            finally:
                staged.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('parent','candidate','anatomy','receipt','output','verification'):
        parser.add_argument(arg,type=Path)
    args = parser.parse_args()
    doc, proof = rebind(args.parent,args.candidate,args.anatomy,args.receipt)
    write_bundle(doc,proof,args.receipt,args.output,args.verification)
if __name__ == '__main__':
    main()
