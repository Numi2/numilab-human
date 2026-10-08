#!/usr/bin/env python3
"""Read-only validation of the selected v8 source-proven reciprocal lobe map."""
from __future__ import annotations
import importlib.util
import json
import hashlib
from pathlib import Path

E=Path('/Users/n/numi-human-resting-evidence-20261005')
HERE=E/'native-lung-transformed-pose-audit-runner-1095'
OUT=HERE/'attempt-002-v8-bridge'
RUNNER=HERE/'audit_lung_cycle_1095.py'
EXPECTED_RUNNER_SHA='5327057df228384b2fdbc180fbb73c5c9428a697d7fbe776d3d6f74f3361be49'
NHA=E/'native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8/final/resting-thorax.nhanatomy'
NHA_SHA='1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc'
REPORT=E/'native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8/current-reciprocal-map-report-v2.json'
REPORT_SHA='6d19c8e7e7011ffc01d6ca575f2ae4ef701485c64e5c187cbe0117535a36b151'
FACE_MAP=E/'native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8/current-reciprocal-face-map-v2.jsonl'
FACE_MAP_SHA='8f66e5bf2b929bd343dea5f3868fa6348b24f44706daa4d62388276ddb1d7d56'
EDGE_MAP=E/'native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8/current-reciprocal-edge-map-v2.jsonl'
EDGE_MAP_SHA='45060d64f69361d285118dc0ee0ed39ed1502bc6f8d8a2b7bab56eb1659c1089'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runner=load(RUNNER,'audit_cycle_v8_bridge_validation')
    if EXPECTED_RUNNER_SHA and sha(RUNNER)!=EXPECTED_RUNNER_SHA:
        raise SystemExit('runner source pin mismatch')
    for path,want,label in ((NHA,NHA_SHA,'final NHA'),(REPORT,REPORT_SHA,'bridge report'),
                            (FACE_MAP,FACE_MAP_SHA,'face map'),(EDGE_MAP,EDGE_MAP_SHA,'edge map')):
        if not path.is_file() or sha(path)!=want:
            raise SystemExit(label+' hash mismatch: '+str(path))
    base=runner.load_module(runner.BASE,'auditbase_v8_bridge_validation')
    runner.load_native_adapters(base)
    parser=runner.load_module(base.PARSER,'parser_v8_bridge_validation')
    final_rows=parser.parse_payload(NHA)[1]
    lineage=runner.load_module(runner.LINEAGE_V2,'lineage_v2_bridge_validation')
    loaded=lineage.load_v2_bridge(base=base,report_path=REPORT,final_nha_path=NHA,
                                  final_nha_sha=NHA_SHA,final_rows=final_rows)
    pairs=[]
    for pair in loaded['declared_pairs']:
        key=tuple(pair); m=loaded['pairs'][key]
        pairs.append({'pair':pair,'face_pair_count':m['map_count'],'edge_count':m['edge_count'],
                      'shared_vertex_count':len(m['shared_vertices']),
                      'shared_edge_count':len(m['shared_edges'])})
    report={
        'schema':'numi.human.lung-current-row-bridge-reader-validation.v1',
        'status':'PASS_reader_validation_only',
        'scope':'Validates the v2 source-proven current-row reciprocal face and edge maps against the exact selected NHA and pinned 1055 source lineage. This is source-map validation only; it does not classify transformed native-pose intersections or qualify a native scan.',
        'runner':{'path':str(RUNNER),'sha256':sha(RUNNER)},
        'reader':{'path':str(runner.LINEAGE_V2),'sha256':sha(runner.LINEAGE_V2)},
        'inputs':{
            'final_nha':{'path':str(NHA),'sha256':sha(NHA)},
            'bridge_report':{'path':str(REPORT),'sha256':sha(REPORT)},
            'face_map':{'path':str(FACE_MAP),'sha256':sha(FACE_MAP)},
            'edge_map':{'path':str(EDGE_MAP),'sha256':sha(EDGE_MAP)},
            'lineage_inputs':loaded['inputs']},
        'pair_declaration_count':len(pairs),
        'mapped_face_pair_total':sum(p['face_pair_count'] for p in pairs),
        'mapped_edge_total':sum(p['edge_count'] for p in pairs),
        'pairs':pairs,
        'requirements_checked':['all ten pair declarations including six explicit zeros',
            'exact final Float32 triangle bits and opposite winding',
            'exact reciprocal mapped-edge incidence and equal boundary sets',
            '1055 source overlay face row + source ID ancestry resolution',
            '1083 paired two-parent diagonal-flip ledger replay'],
    }
    out=OUT/'reader-validation.json'
    out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'report':str(out),'sha256':sha(out),
                      'mapped_face_pair_total':report['mapped_face_pair_total'],
                      'mapped_edge_total':report['mapped_edge_total'],'pairs':pairs},indent=2))

if __name__=='__main__': main()
