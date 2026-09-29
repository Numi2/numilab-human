"""Bind complete compiled-surface censuses to the selected native Human view.

This gate can admit only a per-mesh, source-support-preserving embeddedness
candidate. It does not infer cross-surface non-overlap, component containment,
physical organ volume, clinical anatomy, or tissue mechanics.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

from . import model as human
from . import lung_envelope as lung
from . import surface_topology_repair as repair
from . import surface_topology_audit as candidate_audit

SCHEMA='numi.human.whole-body-selected-surface-geometry-gate.v1'
LAYERS={1:'organ',2:'vessel',3:'nerve',4:'airway',5:'pulmonary_artery',
        6:'pulmonary_vein',7:'lung_lobe',8:'pleura',9:'cavity_reference',
        10:'duct',11:'neural_region_reference',12:'ventricular_region_reference',
        13:'junction_reference',14:'ocular_region_reference',15:'ocular_muscle_reference'}


def require(ok,message):
    if not ok:raise human.ImportError('whole-body selected surface gate: '+message)


def census_rows(folder,payload_sha):
    row_names=[f'rows/{i:03}.json' for i in range(1,580)]
    names={'identity.json','summary.json',*row_names}
    if folder.is_file():
        with tarfile.open(folder,'r:gz') as archive:
            entries=archive.getmembers()
            members={m.name:m for m in entries}
            require(len(entries)==len(members)==581 and set(members)==names
                    and sum(m.size for m in entries)<16*1024*1024
                    and all(m.isfile() and m.size<1024*1024 for m in entries),
                    'incomplete or unsafe census archive')
            raw={name:archive.extractfile(members[name]).read() for name in names}
        read=lambda name:json.loads(raw[name])
        digest=lambda name:hashlib.sha256(raw[name]).hexdigest()
    else:
        require(folder.is_dir(),'missing census directory')
        read=lambda name:human.read_json(folder/name)
        digest=lambda name:human.sha256(folder/name)
    summary=read('summary.json')
    identity=read('identity.json')
    require(summary['surface_count']==579 and summary['payload_sha256']==payload_sha
            and identity['payload_sha256']==payload_sha
            and summary['identity_sha256']==digest('identity.json'),
            'census payload or identity')
    require(set(summary['row_files'])=={name.removeprefix('rows/') for name in row_names},
            'incomplete census rows')
    rows=[]
    for i,name in enumerate(row_names,1):
        require(digest(name)==summary['row_files'][name.removeprefix('rows/')],
                'census row hash drift')
        row=read(name)
        require(row['stable_id']==i and row['schema'] in
                ('numi.human.compiled-surface-embeddedness.v1',
                 'numi.human.compiled-quotient-surface-embeddedness.v1'),
                'census row stable identity')
        rows.append(row)
    require(dict(sorted(Counter(x['status'] for x in rows).items()))==summary['status_counts']
            and sum(x['triangle_count'] for x in rows)==summary['triangles_audited']
            and sum(x['closed_embedded_surface_candidate'] for x in rows)==summary['closed_embedded_surface_candidate_count'],
            'census summary replay')
    return summary,identity,rows,digest('summary.json')


def selected_surface_rows(quotient_rows,candidates,selection):
    """One visible source representation per stable source ID."""
    require(len(quotient_rows)==579
            and [r['stable_id'] for r in quotient_rows]==list(range(1,580)),
            'complete ordered compiled source rows')
    require(len(candidates)==19 and len({c['source_stable_id'] for c in candidates})==19
            and len({c['candidate_stable_id'] for c in candidates})==19,
            'unique derived candidate lineage')
    require(all(type(c['source_stable_id']) is int and 1<=c['source_stable_id']<=579
                for c in candidates)
            and sorted(c['candidate_stable_id'] for c in candidates)==list(range(580,599)),
            'candidate/source stable ID range')
    by_source={c['source_stable_id']:c for c in candidates}
    chosen=[c for c in candidates if c['executed_FP32_embedding']['self_intersection_free']]
    refused=[c for c in candidates if not c['executed_FP32_embedding']['self_intersection_free']]
    require(len(chosen)==11 and len(refused)==8
            and sorted(selection['visible_repaired_source_ids'])==sorted(c['source_stable_id'] for c in chosen)
            and sorted(selection['retained_raw_source_ids'])==sorted(c['source_stable_id'] for c in refused)
            and selection['hidden_anatomy_stable_ids']==sorted(
                [c['source_stable_id'] for c in chosen]+[c['candidate_stable_id'] for c in refused]),
            'inspection does not select exactly the audited source copies')
    result=[]
    for row in quotient_rows:
        sid=row['stable_id'];candidate=by_source.get(sid)
        if candidate and candidate['executed_FP32_embedding']['self_intersection_free']:
            require(candidate['passed_source_chain_and_topology']
                    and candidate['executed_FP32_embedding']['topology']['closed_oriented_manifold_candidate']
                    and candidate['executed_FP32_embedding']['count']==0,
                    'selected copy lacks source-chain or exact embedding proof')
            status='source_support_derived_embedded_candidate'
            visible_id=candidate['candidate_stable_id']
            passed=True
        else:
            status=row['status'];visible_id=sid
            passed=row['closed_embedded_surface_candidate']
        result.append({'source_stable_id':sid,'visible_stable_id':visible_id,
                       'source_member_id':row.get('source_member_id'),
                       'source_object_name':row.get('source_object_name'),
                       'source_layer_code':row['source_layer_code'],
                       'source_original_status':row['status'],
                       'selected_status':status,
                       'selected_closed_embedded_surface_candidate':passed,
                       'improved_over_compiled_source':passed and not row['closed_embedded_surface_candidate'],
                       'clinical_anatomy':False,'physical_volume':False,'mechanics':False})
    require(len(result)==579 and len({r['visible_stable_id'] for r in result})==579,
            'selected source representation coverage')
    return result


def audit(baseline_payload,indexed_dir,quotient_dir,candidate_payload,candidate_report,
          selection_path,native_evidence_dir):
    baseline_payload=baseline_payload.resolve();candidate_payload=candidate_payload.resolve()
    require(human.sha256(baseline_payload)==repair.BASE_SHA,'baseline payload identity')
    original,raw_identity,raw_rows,raw_summary_sha=census_rows(indexed_dir,repair.BASE_SHA)
    quotient,qidentity,qrows,quotient_summary_sha=census_rows(quotient_dir,repair.BASE_SHA)
    _,predecessors=repair.source_specs(baseline_payload)
    require(raw_identity['source_manifest_hashes']==qidentity['source_manifest_hashes']==predecessors,
            'census predecessor source provenance')
    for identity in [raw_identity,qidentity]:
        for filename,digest in identity['predicate_source_sha256'].items():
            require(human.sha256(Path(__file__).with_name(filename))==digest,
                    'census predicate program drift')
    require(original['surface_count']==quotient['surface_count']==579,'census scope mismatch')
    _,base_header,base_records,base_vertices,base_indices=lung.decode(baseline_payload)
    require(base_header[:2]==(5,579),'baseline ABI')
    for i,(a,b,r) in enumerate(zip(raw_rows,qrows,base_records,strict=True),1):
        fv,nv,fi,ni=map(int,[r[1],r[2],r[3],r[4]])
        require(a['stable_id']==b['stable_id']==i
                and a['geometry_sha256']==b['geometry_sha256']
                and a['geometry_sha256']==hashlib.sha256(
                    base_vertices[fv:fv+nv,:3].tobytes()+base_indices[fi:fi+ni].tobytes()).hexdigest(),
                'raw/quotient geometry identity mismatch')
        require(a['source_body_index']==b['source_body_index']==int(r[0])
                and a['source_layer_code']==b['source_layer_code']==int(r[6]),
                'raw/quotient source semantics mismatch')
    _,candidate_header,candidate_records,candidate_vertices,candidate_indices=lung.decode(candidate_payload)
    require(candidate_header[:2]==(5,598) and candidate_header[4:]==base_header[4:]
            and candidate_records[:579].tobytes()==base_records.tobytes()
            and candidate_vertices[:len(base_vertices)].tobytes()==base_vertices.tobytes()
            and candidate_indices[:len(base_indices)].tobytes()==base_indices.tobytes(),
            'candidate payload changed baseline geometry')
    proof=human.read_json(candidate_report)
    require(proof['payload_sha256']==human.sha256(candidate_payload)
            and proof['candidate_count']==19 and len(proof['rows'])==19
            and proof['raw_579_geometry_byte_identical'] is True,
            'derived candidate proof identity')
    published_receipt=candidate_payload.parents[1]/'receipt.json'
    if published_receipt.exists():
        receipt=human.read_json(published_receipt)
        require(receipt['published_payload_sha256']==proof['payload_sha256']
                and receipt['files'][str(candidate_payload.relative_to(published_receipt.parent))]['sha256']==proof['payload_sha256']
                and receipt['files']['source-audit.json']['sha256']==human.sha256(candidate_report)
                and receipt['files']['inspection-selection.final.json']['sha256']==human.sha256(selection_path)
                and qidentity['source_repair_manifest_sha256']==human.sha256(
                    candidate_payload.with_name(repair.MANIFEST)),
                'published candidate receipt drift')
        for path,digest in receipt['executed_source_sha256'].items():
            require(human.sha256(human.REPOSITORY_ROOT/path)==digest,
                    'published repair proof source drift')
    selection=human.read_json(selection_path)
    require(selection['source_payload_sha256']==proof['payload_sha256']
            and selection['source_audit_sha256']==human.sha256(candidate_report),
            'candidate inspection selection drift')
    selected=selected_surface_rows(qrows,proof['rows'],selection)
    native=[]
    for profile in ['source-rest','projected-neutral','torso-flexion','brain','gut']:
        folder=native_evidence_dir/profile/'views'
        packs=list(folder.glob('*.mrvpack'))
        poses=list(folder.glob('*.torso-anatomy-poses.json'))
        require(len(packs)==len(poses)==1,'native inspection profile evidence')
        mask=1024 if profile=='brain' else (1 if profile=='gut' else 32767)
        report=candidate_audit.audit_native(candidate_payload,packs[0],poses[0],mask,
                                            selection['hidden_anatomy_stable_ids'])
        require(report['passed'] and report['surface_count']==598,
                'native selected geometry ownership')
        native.append({'profile':profile,'visual_packet_sha256':report['native_pack_sha256'],
                       'pose_snapshot_sha256':report['snapshot_sha256'],
                       'visible_layer_mask':mask,'all_598_surfaces_retained':True})
    layers=defaultdict(lambda:{'total':0,'original_embedded':0,'selected_embedded':0,'improved':0})
    for q,row in zip(qrows,selected,strict=True):
        layer=LAYERS[row['source_layer_code']];r=layers[layer]
        r['total']+=1;r['original_embedded']+=int(q['closed_embedded_surface_candidate'])
        r['selected_embedded']+=int(row['selected_closed_embedded_surface_candidate'])
        r['improved']+=int(row['improved_over_compiled_source'])
    counts=dict(sorted(Counter(row['selected_status'] for row in selected).items()))
    return {'schema':SCHEMA,'baseline_payload_sha256':repair.BASE_SHA,
            'candidate_payload_sha256':proof['payload_sha256'],
            'indexed_census_sha256':raw_summary_sha,
            'quotient_census_sha256':quotient_summary_sha,
            'candidate_audit_sha256':human.sha256(candidate_report),
            'selection_sha256':human.sha256(selection_path),
            'source_surface_count':579,'compiled_indexed_embedded_candidate_count':original['closed_embedded_surface_candidate_count'],
            'compiled_quotient_embedded_candidate_count':quotient['closed_embedded_surface_candidate_count'],
            'selected_embedded_candidate_count':sum(r['selected_closed_embedded_surface_candidate'] for r in selected),
            'selected_improved_source_count':sum(r['improved_over_compiled_source'] for r in selected),
            'selected_unqualified_source_ids':[r['source_stable_id'] for r in selected
                                               if not r['selected_closed_embedded_surface_candidate']],
            'selected_status_counts':counts,'layer_counts':dict(sorted(layers.items())),
            'rows':selected,'native_profiles':native,
            'cross_surface_overlap':'not_checked','component_containment':'not_checked',
            'clinical_anatomy':False,'physical_volume':False,'mechanics':False,
            'boundary':'All 579 spatial source representations, five actual native packet profiles, and 19 source-chain repairs are bound. Only per-surface exact compiled-coordinate embeddedness is counted. Disjoint anatomical tissue, containment, organ-specific physical volumes, clinical placement and mechanics remain open.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline-payload','indexed-dir','quotient-dir','candidate-payload',
                 'candidate-report','selection','native-evidence-dir','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=audit(a.baseline_payload,a.indexed_dir,a.quotient_dir,a.candidate_payload,
                 a.candidate_report,a.selection,a.native_evidence_dir)
    human.write_json(a.output,result)
    print(json.dumps({'selected_embedded_candidate_count':result['selected_embedded_candidate_count'],
                      'selected_improved_source_count':result['selected_improved_source_count'],
                      'unqualified':len(result['selected_unqualified_source_ids'])}),flush=True)


if __name__=='__main__':main()
