#!/usr/bin/env python3
"""Bound source-facet projection and shared strict preparation across retained N27 poses."""
import hashlib,json,sys,time,gc,signal
from pathlib import Path
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
ROOT=Path('/Users/n/numi-human-final-integration-publish-1397')
O=A/'full-anatomy-delivery-audit-1378/intercostal-retained-motion-preparation-028-001'
SOURCE=A/'liver-island-retirement-candidate-1384-002/resting-thorax.nhanatomy'
REPAIRED=A/'intercostal-source-fragment-repair-028-003/resting-thorax.nhanatomy'
REPAIR=REPAIRED.parent/'source-fragment-repair-report.json'
CAPTURE=A/'accepted-capture-inputs-1398-001.json'
PINS={str(SOURCE):'cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4',
      str(REPAIRED):'50ab8248d2e0f55dd6cb8b24afc34c8844061d5841ecdf89d7daf03288816bfd',
      str(REPAIR):'9327e129cc630c24d2abe1e316fe0077568729f2600dee40359da50e910f1bd6',
      str(CAPTURE):'3df61c8bf202e343a9124588fa94be17e9876db17d1a6ef5dd5fa615aaa7b815'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run():
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('600s bounded preparation')));signal.alarm(600)
 for p,h in PINS.items():assert sha(p)==h,p
 sys.path.insert(0,str(ROOT/'src'))
 from numilab_human import resting_anatomy_interface_patch as codec
 from numilab_human import resting_anatomy_verify as av
 t=time.monotonic();h,rows=codec.parse_payload(SOURCE);_,new=codec.parse_payload(REPAIRED)
 data=json.loads(REPAIR.read_text());manifest=json.loads(CAPTURE.read_text())
 raw=SOURCE.read_bytes();records=[codec.RECORD.unpack_from(raw,codec.HEADER.size+k*codec.RECORD.size) for k in range(h[2])]
 offsets={int(x[5]):int(x[1]) for x in records};del raw
 retained={}
 for sid in (313,315,317):
  removed=set(data['surfaces'][str(sid)]['removed_source_face_ids'])
  retained[sid]=np.asarray([i for i in range(len(rows[sid]['faces'])) if i not in removed],dtype=np.int64)
  assert np.array_equal(new[sid]['vertices6'],rows[sid]['vertices6'])
  assert np.array_equal(new[sid]['faces'],rows[sid]['faces'][retained[sid]])
 for mod in (codec,av):(O/Path(mod.__file__).name).write_bytes(Path(mod.__file__).read_bytes())
 report={'status':'RUNNING_STRICT_RETAINED_MOTION_PREPARATION','script_sha256':sha(__file__),'input_pins':PINS,
         'codec_sha256':sha(codec.__file__),'verifier_sha256':sha(av.__file__),
         'scope':'Exact captured source-face projection and preparation only. No self/pair/containment/native admission.',
         'candidate_admission':False,'captures':[],'first_failure':None}
 def save():report['elapsed_s']=time.monotonic()-t;(O/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 for cap in manifest['capture_rows']:
  surfaces,identity=av.load_capture(Path(cap['pack_path']),Path(cap['receipt_path']))
  item={'step':cap['step'],'capture_identity':identity,'surfaces':[]};report['captures'].append(item)
  for sid in (313,315,317):
   try:
    projected,proof=av.retain_bound_source_faces((51010,sid),surfaces[(51010,sid)],rows[sid]['faces'],offsets[sid],retained[sid])
    proof['source_face_ancestry_ranges']=codec.index_ranges(proof.pop('source_face_ancestry'))
    p,f,n,g=projected
    prepared=av.prepare_surface((51010,sid),p,f,normals=n,source_vertex_ids=g)
    assert prepared.closed,'required intercostal closure failed'
    item['surfaces'].append({'typed_identity':[51010,sid],'status':'PASS_SHARED_STRICT_PREPARATION',
                            'source_capture_binding':proof,'diagnostics':prepared.diagnostics(),
                            'retained_faces_sha256':hashlib.sha256(f.tobytes()).hexdigest()})
   except Exception as e:
    report['first_failure']={'step':cap['step'],'typed_identity':[51010,sid],'error_type':type(e).__name__,'error':str(e)}
    report['status']='STOP_FIRST_PREPARATION_FAILURE';save();print(json.dumps(report['first_failure']));return
  save();del surfaces,projected,prepared,p,f,n,g;gc.collect()
 report['status']='PASS_EIGHT_BREATH_POSES_AND_LATE_CAPTURE_SOURCE_PREPARATION_ONLY';save()
 print(json.dumps({'status':report['status'],'capture_count':len(report['captures']),'elapsed_s':report['elapsed_s'],'report_sha256':sha(O/'report.json')}))
if __name__=='__main__':
 assert not (O/'report.json').exists()
 run()
