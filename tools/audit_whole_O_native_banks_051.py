from pathlib import Path
import collections,csv,hashlib,importlib.util,json,time
import numpy as np
A=Path(__file__).resolve().parent.parent;P=A.parent.parent
OUT=A/'whole-O-native-all185-010'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 base=P/'reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz'
 nr=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz'
 ma=P/'conforming-fit-006-native-assets-545/run-002-operational-reference/registered-assets/volume/combined-node-map.csv'
 owner=A/'authoring-source-chain/build_minimal_coupled_bone32_buffer_003.py'
 z=np.load(base,allow_pickle=False);n=np.load(nr,allow_pickle=False)
 assert hashlib.sha256(base.read_bytes()).hexdigest()=='888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63'
 assert hashlib.sha256(nr.read_bytes()).hexdigest()=='e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5'
 role={}
 for r in csv.DictReader(ma.open()):
  node=int(r['joint_node_id'])-1
  if node in role:assert role[node]==r['role']
  role[node]=r['role']
 inc=collections.defaultdict(list)
 for ci,t in enumerate(z['volume_tetrahedra']):
  for k in range(4):
   f=[int(t[j]) for j in range(4) if j!=k]
   if k%2:f[1],f[2]=f[2],f[1]
   inc[tuple(sorted(f))].append((ci,f))
 outer=[];parents=[]
 for items in inc.values():
  if len(items)==1:
   ci,f=items[0]
   if all(role[x]=='O' for x in f):outer.append(f);parents.append(ci)
  elif len(items)!=2:raise ValueError('invalid source face incidence')
 F=np.array(outer,np.int64);p=z['points_current_world_m_f32']
 spec=importlib.util.spec_from_file_location('owner051',owner);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);ex=m.exact_setup()
 start=time.time();ar,_=ex.float32_surface_records(p,F);br,_=ex.float32_surface_records(n['points_world_f32'],n['faces'])
 assert len(ar)==len(F) and len(br)==len(n['faces'])
 v=ex._audit_pair(ar,br,same_surface=False);pairs=np.asarray(v.get('triangle_pairs',[]),np.int64).reshape(-1,2)
 boff=n['bank_face_offsets'];bi=np.searchsorted(boff[1:],pairs[:,1],side='right') if len(pairs) else np.array([],int)
 by=[]
 for i in sorted(set(map(int,bi))):
  pp=pairs[bi==i]
  by.append({'bank_array_row':i,'body':int(n['bank_body'][i]),'bone':int(n['bank_bone'][i]),'source_id':int(n['bank_source_id'][i]),'triangle_pairs':len(pp),'outer_faces':len(np.unique(pp[:,0]))})
 by.sort(key=lambda x:(-x['triangle_pairs'],x['body'],x['bone']))
 OUT.mkdir()
 np.savez_compressed(OUT/'witnesses.npz',triangle_pairs=pairs,outer_source_faces=F,outer_parent_current006_tet_rows=np.array(parents),bank_array_row_by_pair=bi)
 report={'status':'EXACT_F32_WHOLE_COARSE_O_NATIVE_ALL185_AUDIT','outer_faces':len(F),'native_bank_faces':len(n['faces']),'native_banks':len(boff)-1,'dropped_faces':0,'triangle_pairs':len(pairs),'banks_with_pairs':len(by),'by_bank':by,'seconds':time.time()-start,'runtime_created':0,'accepted_steps':0,'physical_admission':False,'scope':'Source current006 rest-pose raw surface intersection witnesses; intended interfaces require classification. No new geometry or motion qualification.','inputs_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [base,nr,ma,owner,Path(__file__)]}}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
if __name__=='__main__':main()
