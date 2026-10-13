#!/usr/bin/env python3
"""Exact-coordinate quotient audit of stable489's preserved native face IDs."""
from __future__ import annotations
import gc, hashlib, importlib.util, json, os, signal, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/numi-human-fds132-m6-1350')
OUT=ROOT/'attempt-335-row489-n27-seam-all8-003/result-001'
MAIN=ROOT/'attempt-335-row489-n27-seam-all8-002/frozen-inputs/audit_branch_field_n27_v2.py'
NHA=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/liver-island-retirement-candidate-1384-002/resting-thorax.nhanatomy')
N27=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/liver-island-native-027-002')
RUN=N27/'baseline/native-run/accepted-geometry'
N27_RECEIPT=N27/'resting-anatomy-receipt.json'
N27_DECL=N27/'baseline/run-declaration.json'
N27_EXEC=N27/'baseline/execution.json'
N27_INV=N27/'baseline/native-run/invocation.json'
PUB=Path('/Users/n/numi-human-left-pq-publication-1344/src')
STEPS=(0,3807,4767,5023,5599,6207,7423,8000)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def watch():
 limit=int(2.5*1024*1024)
 while True:
  try:
   rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip())
   if rss>limit: os.kill(os.getpid(),signal.SIGTERM)
  except Exception: pass
  time.sleep(2)
def main():
 if OUT.exists(): raise RuntimeError(f'preserve existing {OUT}')
 OUT.mkdir(parents=True)
 threading.Thread(target=watch,daemon=True).start()
 sys.path.insert(0,str(PUB))
 from numilab_human import common_atlas_skin_clearance as ca
 from numilab_human import cardiac_cavity_intersections as ci
 helper=loadmod('audit335n27v2',MAIN)
 source=helper.read_source_rows()[489]
 packs={s:RUN/f'step-{s}.mrvpack' for s in STEPS}
 receipts={s:RUN/f'step-{s}.receipt.json' for s in STEPS}
 pins={str(Path(__file__)):sha(Path(__file__)),str(MAIN):sha(MAIN),str(NHA):sha(NHA),str(N27_RECEIPT):sha(N27_RECEIPT),str(N27_DECL):sha(N27_DECL),str(N27_EXEC):sha(N27_EXEC),str(N27_INV):sha(N27_INV),str(Path(ca.__file__)):sha(Path(ca.__file__)),str(Path(ci.__file__)):sha(Path(ci.__file__))}
 pins.update({str(p):sha(p) for s in STEPS for p in (packs[s],receipts[s])})
 dup=(104,2961); rows={}
 for step in STEPS:
  rec=json.loads(receipts[step].read_text())
  expected_time=float(np.float32(0.002))*step
  if int(rec.get('accepted_step',-1))!=step or float(rec.get('accepted_time_s',float('nan'))) != expected_time:
   raise RuntimeError(f'step{step}: N27 receipt step/time mismatch')
  t0=time.monotonic(); xyz,surfs,meta=ca._pack_surfaces(packs[step],{(51011,489)})
  surface=surfs[(51011,489)]; global_faces=np.asarray(surface['faces'],dtype=np.int64)
  base=int(global_faces.min()); local_faces=global_faces-base
  if len(surface['faces'])!=len(source['faces']) or not np.array_equal(local_faces,source['faces']):
   raise RuntimeError(f'step{step}: native primitive does not preserve original local face IDs/order')
  if int(global_faces.max()-base+1)!=len(source['xyz']): raise RuntimeError(f'step{step}: primitive vertex slot range mismatch')
  actual=xyz[base:base+len(source['xyz'])].copy()
  if not np.array_equal(actual[dup[0]].view('<u4'),actual[dup[1]].view('<u4')): raise RuntimeError(f'step{step}: slots104/2961 not bit-identical')
  raw_keys=[ci.float32_point_lattice_key(tuple(float(v) for v in point)) for point in actual]
  raw_records=ci._records(raw_keys,source['faces'].tolist())
  raw_owner=ci._audit_pair(raw_records,raw_records,same_surface=True)
  records,unique,qfaces,slotmap,deg=helper.exact_quotient_records(actual,source['faces'],ci)
  if records is None: raise RuntimeError(f'step{step}: quotient degenerates {deg[:20]}')
  owner=ci._audit_pair(records,records,same_surface=True)
  alias_id=int(slotmap[dup[0]])
  if int(slotmap[dup[1]])!=alias_id: raise RuntimeError('duplicate slots did not quotient together')
  alias_pairs=[]
  for fa,fb in raw_owner['triangle_pairs']:
   common=set(map(int,qfaces[int(fa)])) & set(map(int,qfaces[int(fb)]))
   if alias_id not in common: raise RuntimeError(f'step{step}: raw self pair {fa}/{fb} does not meet at duplicate-slot quotient vertex')
   points=ci.triangle_intersection_points(records[int(fa)][0],records[int(fb)][0])
   common_coords={records[int(fa)][0][list(records[int(fa)][4]).index(v)] for v in common}
   if not points or not all(ci._allowed_shared_point(p,common_coords) for p in points):
    raise RuntimeError(f'step{step}: duplicate seam faces {fa}/{fb} intersect outside shared simplex')
   alias_pairs.append({'face_ids':[int(fa),int(fb)],
    'raw_shared_source_slot_count':len(set(map(int,source['faces'][int(fa)])) & set(map(int,source['faces'][int(fb)]))),
    'quotient_shared_vertex_count':len(common),'intersection_point_count':len(points),'all_points_on_shared_simplex':True})
  if len(alias_pairs)!=int(raw_owner['count']): raise RuntimeError(f'step{step}: raw self contacts not fully classified')
  rows[str(step)]={'pack_sha256':pins[str(packs[step])], 'primitive_body':int(surface['body']),
   'original_face_ids_preserved':True,'face_count':int(len(local_faces)),
   'quotient_vertex_count':int(len(unique)),'original_slot_count':int(len(actual)),
   'quotiented_duplicate_slot_count':int(len(actual)-len(unique)),
   'duplicate_slots':[int(dup[0]),int(dup[1])],'duplicate_slots_bit_identical':True,
   'duplicate_quotient_vertex_id':alias_id,'degenerate_face_ids':deg,
   'raw_unquotiented_nonadjacent_self_intersection_count':int(raw_owner['count']),
   'nonadjacent_self_intersection_count_after_exact_coordinate_quotient':int(owner['count']),
   'allowed_adjacent_shared_simplex_pair_count':int(owner['allowed_shared_vertex_or_edge_pairs']),
   'duplicate_slot_seam_contact_face_pairs':alias_pairs,
   'duplicate_slot_seam_contact_count':len(alias_pairs),
   'aabb_candidate_face_pairs':int(owner['aabb_candidate_pairs']),
   'elapsed_seconds':time.monotonic()-t0}
  del xyz,surfs,actual,raw_records,records,unique,qfaces,slotmap;gc.collect()
  p=OUT/'report.json';tmp=OUT/'report.tmp'
  report={'schema':'numi.human.row489-exact-coordinate-seam-all8.335a.v1','status':'running','native_or_gpu_used':False,
   'source_payload_sha256':pins[str(NHA)],'pins':pins,'source_face_ids_preserved':True,'captures':rows,
   'interpretation':'Exact Float32 coordinate quotient of actual accepted N27 packed stable489; no Manifold skin build. Shared-simplex contacts are legal only when the exact intersection points lie on the quotient shared vertex/edge. This is geometric evidence, not vascular branch continuity or native admission.'}
  tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
 report['input_hashes_unchanged_at_end']=all(sha(Path(k))==v for k,v in pins.items())
 report['all_eight_complete']=set(rows)=={str(s) for s in STEPS}
 report['geometry_gates_passed'] = report['all_eight_complete'] and report['input_hashes_unchanged_at_end'] and all(r['nonadjacent_self_intersection_count_after_exact_coordinate_quotient']==0 and not r['degenerate_face_ids'] and r['duplicate_slot_seam_contact_count']==9 for r in rows.values())
 report['status']='completed_actual_native_capture_row489_self_audit_native_admission_pending' if report['geometry_gates_passed'] else 'rejected_geometric_seam_audit'
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 if not report['geometry_gates_passed']: raise RuntimeError('N27 typed seam audit gates failed')
 print(json.dumps({'status':report['status'],'report':str(OUT/'report.json'),'report_sha256':sha(OUT/'report.json'),
   'captures':{k:{'raw_unquotiented':v['raw_unquotiented_nonadjacent_self_intersection_count'],'after_quotient':v['nonadjacent_self_intersection_count_after_exact_coordinate_quotient'],'duplicate_slot_seam_contacts':len(v['duplicate_slot_seam_contact_face_pairs'])} for k,v in rows.items()}},sort_keys=True),flush=True)
if __name__=='__main__':main()
