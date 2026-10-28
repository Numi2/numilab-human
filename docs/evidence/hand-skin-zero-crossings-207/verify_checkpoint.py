#!/usr/bin/env python3
"""Verify a restored numerical reference checkpoint; does not admit native physics."""
from pathlib import Path
import argparse,sys
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parent
for line in (R/'SHA256SUMS').read_text().splitlines():
 h,name=line.split('  ',1);p=R/name
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h,name
P=R/'candidate.npz';h=hashlib.sha256(P.read_bytes()).hexdigest();z=np.load(P,allow_pickle=False);X=z['points_world_m'];F=z['source_O_faces']
assert X.dtype==np.float32 and X.ndim==2 and X.shape[1]==3 and np.isfinite(X).all()
assert F.ndim==2 and F.shape[1]==3 and F.min()>=0 and F.max()<len(X)
assert z['envelope_source_original_id'].shape==z['envelope_source_face_row'].shape==(len(F),)
T=X.astype(float)[F];assert np.all(np.linalg.norm(np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=1)>0)
E=np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1);_,c=np.unique(E,axis=0,return_counts=True);assert np.count_nonzero(c==1)==45 and np.all(c<=2)
a=json.load(open(R/'audit/bone-crossings.json'));assert a['candidate_sha256']==h and a['complete'] and a['bank_count']==a['expected_bank_count']==185 and sum((r.get('result')or{}).get('count',0) for r in a['banks'])==0
a=json.load(open(R/'audit/self-crossings.json'));assert a['candidate_sha256']==h and a['complete'] and a['exact_triangle_pair_count']==0
a=json.load(open(R/'audit/containment.json'));assert a['candidate_sha256']==h and a['complete_bank_accounting'] and a['query_rows']==len(X) and a['inside_node_bank_pairs']==a['ambiguous_rows']==a['invalid_rows']==0
a=json.load(open(R/'qualification.json'));assert a['candidate_sha256']==h and a['reference_pose_only'] and not a['physically_admitted'] and not a['native_scene_modified']
t=json.load(open(R/'audit/topology.json'));assert t['candidate_sha256']==h and not t['strict_topology_acceptance'] and t['candidate_main_genus']==42
assert a['strict_topology_acceptance'] is False and a['anatomical_skin_accepted'] is False
parser=argparse.ArgumentParser();parser.add_argument('--require-anatomy',action='store_true');args=parser.parse_args()
if args.require_anatomy:
 print('ANATOMY_REJECTED: '+', '.join(t['strict_topology_rejection_reasons']),file=sys.stderr);sys.exit(2)
print(json.dumps({'status':'PASS_CHECKPOINT_INTEGRITY_ANATOMY_REJECTED','candidate_sha256':h,'vertices':len(X),'faces':len(F),'bone_crossings':0,'self_crossings':0,'inside_pairs':0,'native_admission':False,'anatomical_acceptance':False,'main_genus':42,'surface_components':3}))
