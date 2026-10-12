from pathlib import Path
import argparse,ast,csv,hashlib,importlib.util,json,os,signal,subprocess,sys,threading,time
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
SCRIPT=A/'forearm-shared-field-native-012-diagonal097/verify_forearm_shared_field_native_012.py'
sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
from numilab_human import cardiac_cavity_intersections as ci, common_atlas_skin_clearance as ca
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def load(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def req(v,msg):
 if not v:raise RuntimeError(msg)
def watch():
 while True:
  try:
   rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip())
   if rss>2.5*1024**2:os.kill(os.getpid(),signal.SIGTERM)
  except Exception:pass
  time.sleep(2)
def records(x,f):
 used=np.unique(f);p=np.asarray(x[used],dtype='<f4');lf=np.searchsorted(used,f)
 req(np.array_equal(used[lf],f),'global/local index adapter mismatch')
 lattice=[ci.float32_point_lattice_key(tuple(float(v) for v in row)) for row in p]
 return ci._records(lattice,lf.tolist())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--step',type=int,default=8000);args=ap.parse_args()
 C=args.scene.resolve();B=args.bundle.resolve();OUT=C/f'external-right-union-census-step-{args.step}-001'
 OUT.mkdir(exist_ok=False)
 threading.Thread(target=watch,daemon=True).start()
 v=load('external_forearm012_verify',SCRIPT);h=load('external_forearm012_forward',v.H)
 keys,*_=ca._load_target_inventory(h.INV);keys=set(keys)|{(51005,s) for s in (151,152,159,160,161,162,163,164)}
 labels={(int(r['second_semantic']),int(r['second_stable_id'])):r['source_owner_or_label'] for r in csv.DictReader(h.INV.open())}
 prep_path=B/'preparation.json';prep=json.loads(prep_path.read_text());gids={};source={};geometry_paths=[]
 for sid in (131,133):
  item=prep['geometry_rows'][str(sid)];p=Path(item['path']);req(sha(p)==item['sha256'],'bundle geometry changed');geometry_paths.append(p)
  with np.load(p,allow_pickle=False) as z:
   gids[sid]=list(map(int,z['global_vertex_ids']));source[sid]={'faces':z['faces_local'].copy()}
 tree=ast.parse(SCRIPT.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'parity','build_union'}]
 code=ast.unparse(ast.Module(body=nodes,type_ignores=[])).replace('gids[132]','gids[131]').replace('gids[134]','gids[133]')
 ns={'np':np,'TARGETS':(131,133)};exec(compile(code,str(SCRIPT),'exec'),ns)
 pack=C/'baseline/native-run/accepted-geometry'/f'step-{args.step}.mrvpack';receipt=pack.with_suffix('.receipt.json')
 pins={str(p):sha(p) for p in (Path(__file__),SCRIPT,v.H,h.INV,prep_path,*geometry_paths,pack,receipt,Path(ci.__file__),Path(ca.__file__))}
 req(json.loads(receipt.read_text())['pack_file_sha256']==sha(pack),'capture receipt mismatch')
 xyz,surfaces,_=ca._pack_surfaces(pack,keys)
 req(len(surfaces)==868,'whole-body inventory changed')
 coords={}
 for sid in (131,133):
  ff=np.asarray(surfaces[(51005,sid)]['faces'],dtype=np.int64)
  delta=ff-source[sid]['faces'];req(np.all(delta==delta.flat[0]),'target topology mismatch')
  start=int(delta.flat[0]);coords[sid]=xyz[start:start+len(gids[sid])]
 ux,uf,ancestry,meta=ns['build_union'](source,gids,coords)
 req(not meta['interface_errors'] and meta['shared_face_pairs']>0,'paired internal interface mismatch')
 first=ci._prepare_surface_aabb(records(ux,uf))
 lo=ux.min(0);hi=ux.max(0)
 report={'scope':'Exact boundary-crossing census of native013 right131+133 canceled external union against all other native surfaces at the requested accepted step. Intended shared131/133 cap is canceled by identity and opposite winding. Other contacts remain unclassified; absence of boundary intersections does not prove volumetric containment or anatomical correctness. No anatomy admission.','status':'running','pins':pins,'step':args.step,'source_union':meta,'scanned_targets':0,'targets':[]}
 def save():
  p=OUT/'report.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
 save();start=time.monotonic()
 for key,target in sorted(surfaces.items()):
  if key in ((51005,131),(51005,133)):continue
  report['scanned_targets']+=1
  ff=np.asarray(target['faces'],dtype=np.int64);tri=xyz[ff]
  select=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
  if not len(select):continue
  rec=records(xyz,ff[select]);a=ci._audit_pair_prepared_first(first,rec,same_surface=False)
  row={'semantic':key[0],'stable_id':key[1],'label':labels.get(key,'not in historical label inventory'),'body':int(target['body']),'target_faces':len(ff),'aabb_selected_faces':len(select),'intersection_count':int(a['count']),'aabb_candidate_pairs':int(a['aabb_candidate_pairs']),'classification':'unclassified' if a['count'] else 'no boundary crossings detected','pairs':[[int(ancestry[i][0]),int(ancestry[i][1]),int(select[j])] for i,j in a['triangle_pairs']]}
  report['targets'].append(row);save()
  print(json.dumps({k:row[k] for k in ('semantic','stable_id','label','intersection_count')}),flush=True)
 report.update(status='completed_census_not_anatomical_admission',elapsed_seconds=time.monotonic()-start,input_hashes_unchanged=all(sha(p)==d for p,d in pins.items()))
 save();print(json.dumps({'status':report['status'],'scanned_targets':report['scanned_targets'],'total_raw_pairs':sum(r['intersection_count'] for r in report['targets']),'report_sha256':sha(OUT/'report.json')}),flush=True)
if __name__=='__main__':main()
