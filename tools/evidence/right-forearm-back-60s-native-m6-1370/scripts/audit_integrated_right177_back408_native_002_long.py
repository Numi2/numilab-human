from pathlib import Path
import ast,json,pickle,hashlib,sys,time
from fractions import Fraction as F
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
D=A/'back-forearm-native-014-014-right177-60s';R=D/'baseline/native-run'
OLD=A/'back-native-014-009-60s/baseline/native-run'
CP=Path('/Users/n/numi-human-fds132-m6-1350/attempt-177-right-pentagon-option2-checkpoint-001/result-002/candidate.pkl')
H=Path('/Users/n/numi-human-fds132-m6-1350/attempt-158-right-candidate150-local-diagonal-conditioning-001/audit_candidate158_010.py')
sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,m):
 if not x:raise RuntimeError(m)
def save(p,j):p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
nodes=[n for n in ast.parse(H.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'req','exact_records','topology_volume','union_faces','exact_geometry'}]
ns={'F':F,'np':np,'ci':ci,'analyze_topology':analyze_topology};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(H),'exec'),ns)
out=D/'integrated-right-back-audit-001';out.mkdir(exist_ok=False)
decl=json.loads((D/'baseline/run-declaration.json').read_text());ex=json.loads((D/'baseline/execution.json').read_text());inv=json.loads((R/'invocation.json').read_text())
req(ex['returncode']==0 and not ex['changed_inputs'] and not ex['timed_out'],'native execution failure')
req(inv['argv']==decl['owner_cli_preview']['native_argv'] and inv['environment']==decl['owner_cli_preview']['recorded_invocation_environment'],'invocation mismatch')
req(all(sha(p)==h for p,h in decl['immutable_assets'].items()),'immutable inputs changed')
req((R/'resting-coupled.csv').read_bytes()==(OLD/'resting-coupled.csv').read_bytes(),'physiology changed')
cp=pickle.loads(CP.read_bytes());faces={sid:cp['surfaces'][sid]['faces_global'] for sid in (131,133)}
bundle=A/'right177-topology-bundle-001';bp=json.loads((bundle/'preparation.json').read_text())
data={}
for sid in (131,133):
 p=Path(bp['geometry_rows'][str(sid)]['path']);req(sha(p)==bp['geometry_rows'][str(sid)]['sha256'],'bundle changed')
 with np.load(p,allow_pickle=False) as z:data[sid]={k:z[k].copy() for k in z.files}
inventory=Path('/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv')
requests,*_=ca._load_target_inventory(inventory)
requests=set(requests)|{(51005,s) for s in range(151,165)}
pins={str(p):sha(p) for p in [CP,H,Path(__file__),D/'baseline/execution.json',D/'baseline/run-declaration.json',R/'resting-coupled.csv',OLD/'resting-coupled.csv']}
report={'complete':False,'candidate_admission':False,'scope':'Integrated back408 and right177 native verification; forearm clearance remains subject to exact capture audit and medial-back crossing unresolved.','physiology_byte_exact_009':True,'immutable_pin_count':len(decl['immutable_assets']),'captures':{},'pins':pins}
for step in decl['capture_steps']:
 p=R/'accepted-geometry'/f'step-{step}.mrvpack';rp=p.with_suffix('.receipt.json');op=OLD/'accepted-geometry'/p.name;orp=op.with_suffix('.receipt.json')
 for x in (p,rp,op,orp):pins[str(x)]=sha(x)
 receipt=json.loads(rp.read_text());oldreceipt=json.loads(orp.read_text())
 req(receipt['physical_endpoint']=='accepted' and receipt['pack_file_sha256']==sha(p),'invalid accepted capture')
 req(receipt['accepted_body_poses']==oldreceipt['accepted_body_poses'],'body poses changed')
 pos,surfs,_=ca._pack_surfaces(p,requests);oldpos,oldsurfs,_=ca._pack_surfaces(op,requests)
 req(set(surfs)==set(oldsurfs) and len(surfs)==874,'inventory differs')
 unchanged=[k for k in surfs if k not in {(51005,131),(51005,133)}]
 req(len(unchanged)==872,'unchanged count')
 req(all(np.array_equal(pos[np.asarray(surfs[k]['faces'])],oldpos[np.asarray(oldsurfs[k]['faces'])]) for k in unchanged),'non-right geometry changed')
 posed={}
 for sid in (131,133):
  d=data[sid];nf=np.asarray(surfs[(51005,sid)]['faces']);bf=np.asarray(d['faces_local'])
  delta=nf-bf;req(nf.shape==bf.shape and np.all(delta==delta.flat[0]),'face layout mismatch')
  ids=list(map(int,d['global_vertex_ids']));offset=int(delta.flat[0]);xyz=np.asarray(pos[offset:offset+len(ids)],dtype='<f4')
  posed[sid]={int(g):np.asarray(v,dtype='<f4') for g,v in zip(cp['surfaces'][sid]['global_vertex_ids'],cp['surfaces'][sid]['vertices_float32_m'])}
  posed[sid].update({g:xyz[i] for i,g in enumerate(ids)})
  np.savez_compressed(out/f'step-{step}-row-{sid}.npz',positions=xyz,faces=bf,global_vertex_ids=d['global_vertex_ids'])
 audit=ns['exact_geometry'](cp,faces,posed,f'native-step-{step}')
 req(audit['union']['shared_face_pairs']==2276 and audit['union']['referenced_shared_vertices']==2070 and not audit['union']['interface_errors'],'interface changed')
 record={'exact_geometry':audit,'original_872_surfaces_byte_exact_to_009':True,'accepted_157_body_poses_exact_to_009':True,'capture_pack_sha256':sha(p)}
 report['captures'][str(step)]=record;save(out/f'step-{step}.json',record);save(out/'report.json',report)
 print(json.dumps({'step':step,'rows':{k:v['exact_self_pair_count'] for k,v in audit['rows'].items()},'union':audit['union']['union_exact_pair_count'],'shared_pairs':audit['union']['shared_face_pairs']}),flush=True)
report['complete']=True
report['all_union_captures_clear']=all(v['exact_geometry']['union']['union_exact_pair_count']==0 for v in report['captures'].values())
report['all_rows_clear']=all(r['exact_self_pair_count']==0 for v in report['captures'].values() for r in v['exact_geometry']['rows'].values())
req(all(sha(p)==h for p,h in pins.items()),'audit inputs changed');save(out/'report.json',report)

