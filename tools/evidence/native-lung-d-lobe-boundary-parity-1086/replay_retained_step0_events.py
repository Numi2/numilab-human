#!/usr/bin/env python3
"""Replay retained 1078 step-0 D/lobe witnesses through fresh 1085 adapter."""
import argparse, collections, hashlib, importlib.util, json
from fractions import Fraction
from pathlib import Path

E=Path('/Users/n/numi-human-resting-evidence-20261005')
OUT=None
BASE=E/'native-lung-transformed-pose-audit-1079/audit_lung_interfaces_1042_pinned.py'
WRAPPER=E/'native-lung-transformed-pose-audit-1079/audit_lung_interfaces_1079.py'
RUN=E/'final-native-scene-preflight-936/skin-927-lung-1078-viewer-018-v015-attempt1/native-run'
SCAN=E/'native-lung-transformed-pose-audit-1079/native-1078-step0-corrected-cross-003'
NHA=E/'native-lung-conditioned-final-compose-1078/final/resting-thorax.nhanatomy'
ADAPTER=Path(__file__).resolve().with_name('native_feature_index_1085.py')
EXPECTED_COUNTS={305:4754,306:4978,307:2662,308:656,309:0}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def load(p,n):
 spec=importlib.util.spec_from_file_location(n,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def record(base,pose,owner,fi):
 row=pose[owner];ids=tuple(int(x) for x in row['f'][fi]);tri=tuple(base.pkey(row['v'][i,:3]) for i in ids)
 lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
 return tri,lo,hi,fi,ids

def main():
 global OUT
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--out',required=True,type=Path,help='fresh output directory outside the source repository')
 OUT=ap.parse_args().out.resolve()
 if OUT.exists(): raise SystemExit(f'refuse existing output {OUT}')
 OUT.mkdir(parents=True)
 b=load(BASE,'scanner1042_replay1085')
 w=load(WRAPPER,'wrapper1079_replay1085')
 ctx=w.load_context(b)
 owner=b.load(b.OWNER,'owner937_replay1085')
 parser=b.load(b.PARSER,'parser_replay1085')
 _,nha_rows=parser.parse_payload(NHA)
 pose,pack=b.row_pose(RUN,0,nha_rows)
 maps={sid:b.native_map(sid,{'pairs':ctx['map_doc']['pairs']},pose) for sid in b.LOBES}
 if any(not m['valid'] or m['db']!=m['lb'] for m in maps.values()):raise RuntimeError('native D/lobe map invalid')
 lobe_maps=b.native_lobe_maps(ctx['lineage'],pose)
 adapter=load(ADAPTER,'feature_index1085_replay')
 adapter.install(b)
 before={str(p):sha(p) for p in [BASE,WRAPPER,ADAPTER,SCAN/'unclassified-cross-intersections.jsonl',NHA,pack,pack.with_suffix('.receipt.json')]}
 counts=collections.Counter();labels=collections.Counter();old=collections.Counter();event_total=0
 with (SCAN/'unclassified-cross-intersections.jsonl').open() as src,(OUT/'replay-events.jsonl').open('x') as sink:
  for line_no,line in enumerate(src):
   event=json.loads(line)
   owners=event['owners']
   if 311 not in owners: continue
   sid=owners[0] if owners[1]==311 else owners[1]
   if sid not in EXPECTED_COUNTS: raise RuntimeError(f'unexpected D pair {owners}')
   if owners != [sid,311]: raise RuntimeError(f'expected corrected lobe,D order, got {owners}')
   points=tuple(tuple(Fraction(int(n),int(d)) for n,d in point) for point in event['points_exact_lattice_rational'])
   ra=record(b,pose,sid,int(event['faces'][0])); rb=record(b,pose,311,int(event['faces'][1]))
   label=b.classify(sid,311,ra,rb,points,pose,maps,lobe_maps)
   counts[sid]+=1;labels[sid,label]+=1;old[event['class']]+=1;event_total+=1
   sink.write(json.dumps({'line':line_no,'owners':owners,'faces':event['faces'],'old_label':event['class'],'new_label':label},sort_keys=True,separators=(',',':'))+'\n')
 expected={k:v for k,v in EXPECTED_COUNTS.items() if v}
 if dict(counts)!=expected:raise RuntimeError(f'event counts differ: {dict(counts)}')
 for sid,n in expected.items():
  got=labels[sid,'exact_fullunion_boundary_contact']
  if got!=n or sum(v for (s,_),v in labels.items() if s==sid)!=n:raise RuntimeError(f'D{sid} accepted labels differ: {labels}')
 after={str(p):sha(p) for p in [BASE,WRAPPER,ADAPTER,SCAN/'unclassified-cross-intersections.jsonl',NHA,pack,pack.with_suffix('.receipt.json')]}
 if before!=after:raise RuntimeError('pinned input changed during replay')
 report={'schema':'numi.human.native-lung-diaphragm-lobe-adapter-replay.v1','status':'complete_retained_event_replay_source_rule_aligned','scope':'Existing corrected native step-0 event ledger only; no new geometry scan or changed legacy audit labels.','adapter':{'path':str(ADAPTER),'sha256':sha(ADAPTER),'source_helper_revision':b.native_source_contact_rule_revision},'inputs':{p:{'sha256_before':h,'sha256_after':after[p]} for p,h in before.items()},'old_event_labels':dict(old),'event_total':event_total,'events_per_lobe':dict(sorted(counts.items())),'new_labels_per_lobe':{str(sid):{label:n for (s,label),n in labels.items() if s==sid} for sid in b.LOBES},'native_maps_valid_and_boundaries_equal':True,'event_replay_file':{'path':str(OUT/'replay-events.jsonl'),'sha256':sha(OUT/'replay-events.jsonl'),'lines':event_total},'qualification':'Replay of existing exact witnesses through the composed adapter. The original scan and its labels/gates remain unchanged; this does not waive non-boundary intersections or qualify other poses.'}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'events':event_total,'per_lobe':dict(sorted(counts.items())),'labels':report['new_labels_per_lobe'],'report':str(OUT/'report.json'),'report_sha256':sha(OUT/'report.json')},sort_keys=True))

if __name__=='__main__':main()
