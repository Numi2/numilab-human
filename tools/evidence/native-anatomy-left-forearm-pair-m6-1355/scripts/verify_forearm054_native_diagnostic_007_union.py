#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,importlib.util,json,pickle,sys
from pathlib import Path
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
C=A/'forearm054-native-diagnostic-007-candidate060';RUN=C/'baseline/native-run'
MAP=Path('/Users/n/numi-human-fds132-m6-1350/attempt-061-step0-capture-remesh-diagnostic/candidate060-adapted-source-parent-map-v2.pkl')
EXACT=Path('/Users/n/numi-human-fds132-m6-1350/attempt-060-paired-cut-sliver-contraction-001/exact-check-001/report.json')
CONSTRUCTION=Path('/Users/n/numi-human-fds132-m6-1350/attempt-060-paired-cut-sliver-contraction-001/report.json')
H=A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'
PUB=Path('/Users/n/numi-human-left-pq-publication-1344/src')
STEPS=[0,3807,4767,5023,5599,6207,7423,8000]
TARGETS=(132,134)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def json_default(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def signs_and_volumes(xyz,faces,components):
 volumes=[]
 for comp in components:
  fc=faces[np.asarray(comp,dtype=np.int64)];used=np.unique(fc)
  tri=xyz.astype(np.float64)[fc]-xyz.astype(np.float64)[used].mean(axis=0,dtype=np.float64)
  volumes.append(float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6.))
 return volumes,['positive' if v>0 else 'negative' if v<0 else 'zero' for v in volumes]
def parity(tri):
 inv=0
 for i in range(3):
  for j in range(i+1,3):inv+=int(tri[i]>tri[j])
 return -1 if inv%2 else 1

def build_union(surfaces,gids,coords):
 # Locate each source face by its canonical global candidate IDs. Shared faces are removed only when
 # exactly one face from each child uses the same three IDs and the oriented cycles are opposite.
 occurrences={};face_data=[]
 for sid in TARGETS:
  faces=np.asarray(surfaces[sid]['faces'],dtype=np.int64)
  for fi,face in enumerate(faces):
   oriented=tuple(int(gids[sid][int(v)]) for v in face)
   key=tuple(sorted(oriented))
   occurrences.setdefault(key,[]).append((sid,fi,oriented))
   face_data.append((sid,fi,oriented,key))
 shared=[];remove=set();errors=[]
 for key,items in occurrences.items():
  if len(items)==1:continue
  if len(items)==2 and {items[0][0],items[1][0]}==set(TARGETS):
   if parity(items[0][2])==-parity(items[1][2]):
    shared.append(key);remove.add((items[0][0],items[0][1]));remove.add((items[1][0],items[1][1]))
   else:errors.append({'face_global_ids':list(key),'reason':'shared_faces_not_oppositely_oriented','occurrences':[list(x) for x in items]})
  else:errors.append({'face_global_ids':list(key),'reason':'duplicate_or_nonpaired_face','occurrences':[list(x) for x in items]})
 remaining=[x for x in face_data if (x[0],x[1]) not in remove]
 used=sorted({g for _,_,trip,_ in remaining for g in trip})
 index={g:i for i,g in enumerate(used)}
 xyz=[]
 for gid in used:
  found=[]
  for sid in TARGETS:
   ix={g:i for i,g in enumerate(gids[sid])}
   if gid in ix:found.append(np.asarray(coords[sid][ix[gid]],dtype='<f4'))
  if not found:errors.append({'global_id':gid,'reason':'missing_coordinate'});xyz.append([0.,0.,0.]);continue
  if any(a.tobytes()!=found[0].tobytes() for a in found[1:]):errors.append({'global_id':gid,'reason':'shared_coordinates_not_bitwise_equal'});xyz.append(found[0]);continue
  xyz.append(found[0])
 faces=np.asarray([[index[g] for g in trip] for _,_,trip,_ in remaining],dtype=np.int64)
 return np.asarray(xyz,dtype='<f4'),faces,dict(shared_face_pairs=len(shared),removed_face_count=len(remove),remaining_face_count=len(faces),remaining_vertex_count=len(used),interface_errors=errors)

def main():
 OUT=C/'native-union-audit-002';OUT.mkdir(exist_ok=False)
 report={'schema':'numi.human.left-fds-fdp-candidate060-native-union-audit.v1','scope':'Experimental native target-union self/topology audit. Shared opposite interface faces are canceled by global candidate IDs; source cavity component winding is preserved and compared, not forced positive. No admission or all25 claim.',
         'checks':{},'captures':{},'failures':[],'complete':False,'pins':{}}
 RP=OUT/'report.json'
 def save():RP.write_text(json.dumps(report,indent=2,sort_keys=True,default=json_default)+'\n')
 def check(name,ok,**fields):
  report['checks'][name]=dict(passed=bool(ok),**fields)
  if not ok:report['failures'].append(name)
  save()
 def req(x,msg):
  if not x:raise RuntimeError(msg)
 sys.path.insert(0,str(PUB))
 from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
 from numilab_human.cardiac_cavity_geometry import analyze_topology
 h=module('forearm054_union_forward',H)
 prep=json.loads((C/'preparation.json').read_text());decl=json.loads((C/'baseline/run-declaration.json').read_text())
 audit=json.loads((C/'native-audit-003/report.json').read_text())
 exact=json.loads(EXACT.read_text());construct=json.loads(CONSTRUCTION.read_text());m=pickle.loads(MAP.read_bytes())
 check('native_local_audit_all8_passed',audit.get('status')=='experimental_native_full_grid_verified' and audit.get('failures')==[] and audit.get('available_capture_steps')==STEPS,
       status=audit.get('status'),failures=audit.get('failures'),steps=audit.get('available_capture_steps'),source_report=str(C/'native-audit-003/report.json'))
 check('exact_source_union_reference',exact.get('complete') is True and exact['checks']['union']['topology'].get('closed') is True and exact['checks']['union']['topology'].get('face_components')==12 and exact['checks']['union'].get('self_count')==0 and exact.get('shared_interface_triangles')==2146,
       complete=exact.get('complete'),union_topology=exact.get('checks',{}).get('union',{}).get('topology'),union_self_count=exact.get('checks',{}).get('union',{}).get('self_count'),shared_interface_triangles=exact.get('shared_interface_triangles'))
 check('candidate_construction_not_admitted_and_bound_pinned',construct.get('complete') is True and construct.get('admitted') is False and construct.get('candidate_sha256')==prep.get('candidate060_sha256') and construct.get('union_link_and_material_stratum_guard') is True,
       admitted=construct.get('admitted'),candidate_sha256=construct.get('candidate_sha256'),declared_bound_m=construct.get('cumulative_displacement_bound_m'),actual_displacement_m=construct.get('max_actual_displacement_m'))
 surf={int(k):v for k,v in m['surfaces'].items()};gids={sid:list(map(int,surf[sid]['global_vertex_ids'])) for sid in TARGETS}
 coords={sid:np.asarray([[np.float32(float(x)) for x in xyz] for xyz in surf[sid]['rest_vertices_exact_m']],dtype='<f4') for sid in TARGETS}
 ref_xyz,ref_faces,ref_meta=build_union(surf,gids,coords)
 ref_top=analyze_topology(ref_xyz.astype(float).tolist(),ref_faces.tolist())
 ref_vols,ref_signs=signs_and_volumes(ref_xyz,ref_faces,ref_top.get('face_components',[]))
 ref_rec,ref_dg=h.exact_rows(ref_xyz,ref_faces,ci)
 if ref_rec is None or ref_dg:ref_self=None;ref_pairs=[]
 else:
  qa=ci._audit_pair(ref_rec,ref_rec,same_surface=True);ref_self=int(qa['count']);ref_pairs=qa.get('triangle_pairs',[])
 ref_top_summary={k:ref_top.get(k) for k in ('closed_oriented_manifold_candidate','vertex_count','face_count','edge_count','face_component_count','boundary_edge_count','boundary_loop_count','euler_characteristic')}
 ref_ok=not ref_meta['interface_errors'] and ref_meta['shared_face_pairs']==2146 and ref_meta['removed_face_count']==4292 and ref_meta['remaining_face_count']==16404 and ref_meta['remaining_vertex_count']==8098 and ref_top.get('closed_oriented_manifold_candidate') is True and len(ref_top.get('face_components',[]))==12 and ref_self==0 and len(ref_dg)==0 and 'zero' not in ref_signs
 report['source_union_reference']={'construction':{k:v for k,v in ref_meta.items() if k!='interface_errors'},'topology':ref_top_summary,'component_signed_volumes_m3':ref_vols,'component_signs':ref_signs,'degenerate_count':len(ref_dg),'exact_self_pair_count':ref_self,'self_pairs':[list(map(int,x)) for x in ref_pairs[:20]],'interface_error_count':len(ref_meta['interface_errors'])}
 check('source_map_union_reconstructed_and_audited',ref_ok,**report['source_union_reference'])
 keys,*_=ca._load_target_inventory(h.INV);request_keys=set(keys)|{(51005,s) for s in (151,152,159,160,161,162,163,164)}
 native_all_keys=request_keys|{(51007,1)}
 for step in STEPS:
  pack=RUN/'accepted-geometry'/f'step-{step}.mrvpack';req(pack.is_file(),f'missing native capture {pack}')
  pos,surfaces,pmeta=ca._pack_surfaces(pack,request_keys)
  row_coords={};face_order_ok=True
  for sid in TARGETS:
   rowfaces=np.asarray(surf[sid]['faces'],dtype=np.int64);nativefaces=np.asarray(surfaces[(51005,sid)]['faces'],dtype=np.int64)
   delta=nativefaces-rowfaces
   same=nativefaces.shape==rowfaces.shape and delta.size>0 and bool(np.all(delta==delta.flat[0]))
   face_order_ok=face_order_ok and same
   if same:
    start=int(delta.flat[0]);row_coords[sid]=pos[start:start+len(gids[sid])].astype('<f4',copy=True)
  if set(row_coords)==set(TARGETS):
   xyz,faces,meta=build_union(surf,gids,row_coords)
   topo=analyze_topology(xyz.astype(float).tolist(),faces.tolist())
   rec,dg=h.exact_rows(xyz,faces,ci)
   if rec is None or dg:selfcount=None;pairs=[]
   else:
    qa=ci._audit_pair(rec,rec,same_surface=True);selfcount=int(qa['count']);pairs=qa.get('triangle_pairs',[])
   vols,component_signs=signs_and_volumes(xyz,faces,topo.get('face_components',[]))
   seam_common=set(gids[132])&set(gids[134]);seam_ok=True;seam_exact=0
   for gid in seam_common:
    a=row_coords[132][gids[132].index(gid)];b=row_coords[134][gids[134].index(gid)]
    if a.tobytes()==b.tobytes():seam_exact+=1
    else:seam_ok=False
   topo_ok=topo.get('closed_oriented_manifold_candidate') is True and len(topo.get('face_components',[]))==12 and topo.get('vertex_count')==8098 and topo.get('face_count')==16404
   winding_ok=component_signs==ref_signs and 'zero' not in component_signs
   union_ok=face_order_ok and not meta['interface_errors'] and meta['shared_face_pairs']==2146 and meta['removed_face_count']==4292 and meta['remaining_face_count']==16404 and meta['remaining_vertex_count']==8098 and seam_ok and seam_exact==1986 and topo_ok and len(dg)==0 and selfcount==0 and winding_ok
   topo_summary={k:topo.get(k) for k in ('closed_oriented_manifold_candidate','vertex_count','face_count','edge_count','face_component_count','boundary_edge_count','boundary_loop_count','euler_characteristic')}
   entry={'pack_path':str(pack),'pack_sha256':sha(pack),'native_inventory_count':len(surfaces),'native_inventory_exact_868':set(surfaces)==native_all_keys and len(surfaces)==868,
    'face_order_matches_candidate':face_order_ok,'interface':meta,'shared_seam_vertex_count':len(seam_common),'shared_seam_bitwise_equal_count':seam_exact,'topology':topo_summary,
    'component_signed_volumes_m3':vols,'component_signs':component_signs,'source_component_signs':ref_signs,'winding_matches_source_union':winding_ok,
    'degenerate_count':len(dg),'exact_self_pair_count':selfcount,'self_pairs':[list(map(int,x)) for x in pairs[:20]],'union_pass':union_ok}
  else:entry={'pack_path':str(pack),'face_order_matches_candidate':False,'union_pass':False,'reason':'native target surface index layout mismatch'};union_ok=False
  report['captures'][str(step)]=entry;check(f'union_step_{step}_exact_topology_and_self',union_ok,**entry)
 report['complete']=True
 report['status']='native_union_all8_self_topology_and_winding_pass' if not report['failures'] else 'native_union_audit_failed'
 report['pins']={str(p):sha(p) for p in (C/'preparation.json',C/'baseline/run-declaration.json',C/'resting-anatomy-receipt.json',C/'native-audit-003/report.json',MAP,EXACT,CONSTRUCTION)}
 save();print(json.dumps({'report':str(RP),'status':report['status'],'failed_checks':report['failures']},indent=2,default=json_default),flush=True)
main()
