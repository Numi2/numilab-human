#!/usr/bin/env python3
"""Restored geometry checkpoint integrity; no physical admission."""
from pathlib import Path
import json,hashlib,argparse,sys,numpy as np
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--require-physical',action='store_true');args=ap.parse_args()
for line in (R/'SHA256SUMS').read_text().splitlines():
 h,name=line.split('  ',1);assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
p=R/'candidate.npz';h=hashlib.sha256(p.read_bytes()).hexdigest();z=np.load(p,allow_pickle=False);X=z['points_world_m'];F=z['source_O_faces'];assert X.dtype==np.float32 and np.isfinite(X).all();assert F.ndim==2 and F.shape[1]==3 and F.min()>=0 and F.max()<len(X)
sys.path.insert(0,str(R/'tools'));from topology_check import analyse
base=np.load(R/'inputs/baseline-806.npz',allow_pickle=False);b=analyse(base['points_world_m'],base['source_O_faces'],'baseline');c=analyse(X,F,'candidate')
assert c['component_count_edge_connected']==1 and c['components'][0]['genus_from_chi_and_boundary']==0
assert not c['unused_point_count'] and not c['edge_incidence_gt2_count'] and not c['exact_zero_area_face_count'] and not c['duplicate_geometric_face_groups'] and not c['exact_duplicate_coordinate_group_count']
assert c['boundary_coordinate_edge_multiset']==b['boundary_coordinate_edge_multiset'] and c['open_edge_count']==45
for n in ['bone','self','containment','topology']:
 a=json.load(open(R/'audit'/f'{n}.json'));assert a['candidate_sha256']==h
 if n=='bone':assert a['complete'] and a['bank_count']==a['expected_bank_count']==185 and sum((r.get('result')or{}).get('count',0)for r in a['banks'])==0
 if n=='self':assert a['complete'] and a['exact_triangle_pair_count']==0
 if n=='containment':assert a['complete_bank_accounting'] and a['query_rows']==len(X) and a['inside_node_bank_pairs']==a['ambiguous_rows']==a['invalid_rows']==0
 if n=='topology':assert a['strict_topology_acceptance'] and a['candidate_main_genus']==0
q=json.load(open(R/'qualification.json'));assert q['candidate_sha256']==h and q['reference_geometry_passed'] and not q['physical_admission'] and not q['native_scene_modified'] and q['accepted_physical_steps']==0
if args.require_physical:
 print('PHYSICAL_ADMISSION_PENDING: donor volume, mass ancestry, layer quality, motion and native qualification',file=sys.stderr);sys.exit(2)
print(json.dumps({'status':'PASS_RESTORED_REFERENCE_GEOMETRY_INTEGRITY','candidate_sha256':h,'bone_pairs':0,'self_pairs':0,'contained_pairs':0,'components':1,'genus':0,'physical_admission':False}))
