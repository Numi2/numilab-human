from pathlib import Path
import json,numpy as np,sys,hashlib,time,csv,argparse
R=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207');sys.path.insert(0,str(R/'authoring-deps'));import igl
P=R/'physical-transfer-209'
parser=argparse.ArgumentParser(description='Nonvisual node/bone occupancy inventory using current native Float32 physical coordinates')
parser.add_argument('--candidate',type=Path,default=P/'reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz')
parser.add_argument('--bank-manifest',type=Path,default=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json')
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();C=a.candidate;M=a.bank_manifest;D=a.output;D.mkdir(exist_ok=False)
with np.load(C) as z:q=z['points_reference_world_m_f64'].astype(np.float32).astype(np.float64);tets=z['volume_tetrahedra'];pin=set(z['pinned_volume_node_rows'].tolist())
roles=np.full(len(q),'interior',dtype='<U8');roleRows=np.full(len(q),-1,np.int64)
boundaryCSV=C.parent/'coarse-boundary-faces.csv'
byRole={r:set() for r in ['O','I','D']}
with boundaryCSV.open() as h:
 for face in csv.DictReader(h):
  r=face['role'];assert r in byRole
  for corner in range(3):
   n=int(face[f'volume_node{corner}']);assert 0<=n<len(q)
   byRole[r].add(n)
for r,nodes in byRole.items():
 assert nodes
 for row,n in enumerate(sorted(nodes)):
  assert roles[n]=='interior','boundary node belongs to two anatomical layers'
  roles[n]=r;roleRows[n]=row
assert np.isfinite(q).all() and np.all(tets>=0) and np.all(tets<len(q))
banks=json.loads(M.read_text())['operands'];rows=[];witness=[];start=time.monotonic()
for bank in banks:
 f=Path(bank['operand_path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==bank['operand_sha256']
 with np.load(f) as z:v=z['points_world_m_f64'];t=z['faces_f64'].astype(np.int64);stable=z['bank_stable_face_row']
 selected=np.flatnonzero(np.all(q>=v.min(0),axis=1)&np.all(q<=v.max(0),axis=1))
 if not len(selected):continue
 winding=igl.winding_number(v,t,q[selected]);inside=selected[np.abs(winding)>.5]
 counts={str(r):int(np.count_nonzero(roles[inside]==r)) for r in ['O','I','D']}
 clear=0
 if len(inside):
  sd,fi,near=igl.signed_distance(q[inside],v,t)[:3]
  for node,d,face,point in zip(inside,sd,fi,near):
   tri=v[t[face]];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n)
   witness.append({'volume_node_row':int(node),'role':str(roles[node]),'role_node_row':int(roleRows[node]),'bank_body':bank['body'],'bank_bone':bank['bone'],'bank_index':bank['index'],'signed_distance_m':float(d),'winding_class':'inside','clearance_more_than_5um':bool(d<-.000005),'world_point_m':q[node].tolist(),'nearest_world_point_m':point.tolist(),'outward_triangle_normal':n.tolist(),'bank_face_row':int(face),'native_stable_face_row':int(stable[face]),'preserved_anchor_or_head':bool(int(node) in pin)})
   clear+=int(d<-.000005)
 rows.append({'body':bank['body'],'bone':bank['bone'],'index':bank['index'],'inside_node_pair_count':len(inside),'inside_more_than_5um_pairs':clear,'roles':counts})
insideNodes=sorted(set(w['volume_node_row'] for w in witness));incident=np.flatnonzero(np.any(np.isin(tets,insideNodes),axis=1))
result={'status':'nonvisual_coarse_physical_boundary_penetration_inventory','candidate_sha256':hashlib.sha256(C.read_bytes()).hexdigest(),'bank_manifest_sha256':hashlib.sha256(M.read_bytes()).hexdigest(),'boundary_roles_csv_sha256':hashlib.sha256(boundaryCSV.read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start,'bank_count':len(banks),'inside_node_bank_pairs':len(witness),'unique_inside_nodes':len(insideNodes),'incident_tet_count':len(incident),'role_inside_pairs':{r:sum(w['role']==r for w in witness) for r in ['O','I','D','interior']},'bank_priorities':sorted((r for r in rows if r['inside_node_pair_count']),key=lambda r:(-r['inside_node_pair_count'],r['body'],r['bone'])),'limitations':'Node winding inventory only; not exhaustive triangle or closed-cell intersection gate. Five-micrometre count is priority classification only; every inside witness is retained.'}
(D/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');(D/'witnesses.json').write_text(json.dumps(witness,indent=2)+'\n');np.savez_compressed(D/'incident-cell-domain.npz',inside_nodes=np.array(insideNodes,np.uint32),incident_parent_tet_rows=incident.astype(np.uint32))
print(json.dumps(result,indent=2))
