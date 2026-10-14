#!/usr/bin/env python3
"""Exact coordinate-quotient source facet topology diagnostic; not anatomy admission."""
import hashlib,json,sys,time,signal
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path('/Users/n/numi-human-resting-anatomy-gates-publish-1398')
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
NHA=A/'liver-island-retirement-candidate-1384-002/resting-thorax.nhanatomy'
O=A/'full-anatomy-delivery-audit-1378/intercostal313-315-317-quotient-topology-003'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def orientation(t):return -1 if sum(int(t[i]>t[j]) for i in range(3) for j in range(i+1,3))%2 else 1
def inspect(p,f):
 uq,inv=np.unique(p,axis=0,return_inverse=True);qf=inv[f]
 _,gi,c=np.unique(np.sort(qf,axis=1),axis=0,return_inverse=True,return_counts=True)
 groups=[np.flatnonzero(gi==i).tolist() for i in np.flatnonzero(c>1)]
 ef=defaultdict(list);vf=defaultdict(set)
 for i,t in enumerate(qf):
  for v in t:vf[int(v)].add(i)
  for a,b in zip(t,np.roll(t,-1)):ef[tuple(sorted((int(a),int(b))))].append(i)
 parent=list(range(len(f)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for faces in ef.values():
  first=find(faces[0])
  for i in faces[1:]:parent[find(i)]=first
 comps=defaultdict(list)
 for i in range(len(f)):comps[find(i)].append(i)
 out=[]
 for g in groups:
  roots={find(i) for i in g};component=set().union(*(set(comps[r]) for r in roots))
  vs={int(v) for t in qf[g] for v in t}
  vneighbors=set().union(*(vf[v] for v in vs))-set(g)
  es={tuple(sorted((int(a),int(b)))) for t in qf[g] for a,b in zip(t,np.roll(t,-1))}
  out.append({'source_face_ids':g,'quotient_vertex_triples':qf[g].tolist(),'orientation_parities':[orientation(qf[i]) for i in g],'edge_component_face_count':len(component),'edge_component_source_face_ids':sorted(component),'edge_incidence_counts':[len(ef[e]) for e in sorted(es)],'vertex_neighbor_face_count':len(vneighbors),'vertex_neighbor_source_face_ids':sorted(vneighbors),'isolated_opposed_two_face_fragment':len(g)==2 and len(component)==2 and len(vneighbors)==0 and orientation(qf[g[0]])*orientation(qf[g[1]])==-1})
 return {'raw_vertices':len(p),'quotient_vertices':len(uq),'raw_faces':len(f),'duplicate_group_count':len(groups),'quotient_edge_component_count':len(comps),'quotient_component_size_histogram':dict(Counter(len(x) for x in comps.values())),'duplicate_groups':out,'all_duplicate_groups_isolated_opposed_two_face_fragments':all(x['isolated_opposed_two_face_fragment'] for x in out)}
def main():
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('120s source topology bound')));signal.alarm(120)
 assert sha(NHA)=='cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4'
 assert sha(ROOT/'src/numilab_human/resting_anatomy_interface_patch.py')=='e7692e4239c6a5ca4c12667b9bce94c06efa9bc3bec86cbb74e4625eeb416187'
 # Synthetic pair isolated by vertices and touching another facet distinguish classification.
 p=np.array([[0,0,0],[1,0,0],[0,1,0],[2,0,0],[2,1,0]],np.float32)
 assert inspect(p,np.array([[0,1,2],[0,2,1]]))['all_duplicate_groups_isolated_opposed_two_face_fragments']
 assert not inspect(p,np.array([[0,1,2],[0,2,1],[1,3,4]]))['all_duplicate_groups_isolated_opposed_two_face_fragments']
 sys.path.insert(0,str(ROOT/'src'))
 from numilab_human import resting_anatomy_interface_patch as codec
 t=time.monotonic();_,rows=codec.parse_payload(NHA)
 r={'status':'SOURCE_QUOTIENT_TOPOLOGY_DIAGNOSTIC_NOT_ADMISSION','script_sha256':sha(__file__),'source_sha256':sha(NHA),'surfaces':{},'candidate_admission':False}
 for sid in (313,315,317):
  row=rows[sid];assert row['body_index']==20 and row['layer']==1 and row['flags']==0
  r['surfaces'][str(sid)]=inspect(row['vertices6'][:,:3],row['faces'])
 r['elapsed_s']=time.monotonic()-t
 assert not (O/'report.json').exists();(O/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'elapsed_s':r['elapsed_s'],'report_sha256':sha(O/'report.json'),'surfaces':{sid:{k:v for k,v in x.items() if k!='duplicate_groups'} for sid,x in r['surfaces'].items()}}))
if __name__=='__main__':main()
