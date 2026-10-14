#!/usr/bin/env python3
"""Scoped passive intercostal source-facet repair using existing NHA and receipts."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from numilab_human import resting_anatomy_interface_patch as codec

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def repair(payload, receipt_path, manifest_path, output):
    source_sha=sha(payload);receipt_sha=sha(receipt_path)
    receipt=json.loads(receipt_path.read_text());manifest=json.loads(manifest_path.read_text()) if manifest_path else None
    if receipt['payload']['sha256']!=source_sha:
        raise ValueError('receipt does not bind source NHA')
    if manifest is not None and manifest['receipt']['sha256']!=receipt_sha:
        raise ValueError('manifest does not bind source receipt')
    header,rows=codec.parse_payload(payload)
    if codec.encode_payload(header,rows)!=payload.read_bytes():
        raise ValueError('existing codec roundtrip changed source bytes')
    original=copy.deepcopy(rows);repairs={}
    for sid in (313,315,317):
        r=rows[sid]
        if (r['body_index'],r['layer'],r['flags'])!=(20,1,0):
            raise ValueError('unexpected intercostal identity/owner/layer')
        r['faces'],repairs[str(sid)]=codec.remove_isolated_opposed_fragments(r['vertices6'],r['faces'])
        if not repairs[str(sid)]['removed_source_face_ids']:
            raise ValueError('expected declared source fragments absent')
    if output.exists():
        raise FileExistsError('preserve immutable attempt directory')
    output.mkdir(parents=True)
    (output/'tool-source.py').write_bytes(Path(__file__).read_bytes())
    (output/'codec-source.py').write_bytes(Path(codec.__file__).read_bytes())
    dst=output/'resting-thorax.nhanatomy';dst.write_bytes(codec.encode_payload(header,rows))
    h2,out=codec.parse_payload(dst)
    assert h2[:4]==header[:4] and h2[5:]==header[5:] and list(out)==list(original)
    for sid,r in original.items():
        assert np.array_equal(r['vertices6'],out[sid]['vertices6'])
        assert (r['body_index'],r['layer'],r['flags'])==(out[sid]['body_index'],out[sid]['layer'],out[sid]['flags'])
        if sid not in (313,315,317):assert np.array_equal(r['faces'],out[sid]['faces'])
    evidence={'status':'PASS_SCOPED_SOURCE_PREPARATION_REPAIR_NOT_ANATOMY_ADMISSION',
              'source_payload':{'path':str(payload),'sha256':source_sha},
              'source_receipt':{'path':str(receipt_path),'sha256':receipt_sha},
              'source_manifest':{'path':str(manifest_path),'sha256':sha(manifest_path)} if manifest_path else None,
              'tool_sha256':sha(__file__),'codec_sha256':sha(codec.__file__),
              'output_payload':{'path':str(dst),'sha256':sha(dst)},'surfaces':repairs,
              'all_vertex_normal_bytes_owners_and_unrelated_faces_unchanged':True,
              'physical_mass_and_circulation_state_added':False,
              'candidate_admission':False}
    report=output/'source-fragment-repair-report.json'
    report.write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    receipt['payload'].update(path=str(dst),sha256=sha(dst),index_count=h2[4])
    receipt['functional_bindings']['anatomy_payload_sha256']=sha(dst)
    declaration={'method':'remove isolated opposed source fragments','source_payload_sha256':source_sha,
                 'report_path':str(report),'report_sha256':sha(report),'source_coordinates_modified':False,
                 'vertices_normals_attachments_rest_frames_preserved':True,
                 'physiology_and_physical_mass_owners_unchanged':True,
                 'qualification':'source preparation only; strict rest/motion/native geometry pending'}
    receipt['provenance']['local_intercostal_source_fragment_repair']=declaration
    for sid,r in repairs.items():
        source = receipt['provenance']['source_id_map'][sid]
        if source.get('repair') is None:
            source['repair'] = {}
        if not isinstance(source['repair'], dict):
            raise ValueError('existing repair metadata must be null or a mapping')
        source['repair']['isolated_opposed_fragments']=r
    rp=output/'resting-anatomy-receipt.json';rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    if manifest is not None:
        manifest.update(payload=receipt['payload'],receipt={'path':str(rp),'sha256':sha(rp)},
                        functional_bindings=receipt['functional_bindings'],qualification=receipt['qualification'],
                        source_surfaces=receipt['provenance']['source_id_map'])
        mp=output/'resting-anatomy-manifest.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'report_sha256':sha(report),'payload_sha256':sha(dst),'receipt_sha256':sha(rp),
                      'removed_faces':{sid:len(r['removed_source_face_ids']) for sid,r in repairs.items()},
                      'native_admission':False}))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('payload','receipt','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    a=p.parse_args();repair(a.payload,a.receipt,a.manifest,a.output)
if __name__=='__main__':main()
