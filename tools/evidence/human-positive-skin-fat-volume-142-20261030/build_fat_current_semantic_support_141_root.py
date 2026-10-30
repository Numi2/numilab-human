"""Source330 same-material Fat support for current compact positive volume.
Static compiler preparation only. Shared I geometric binding is unchanged."""
from pathlib import Path
import numpy as np,json,csv,hashlib,importlib.util,sys,time,math
from collections import Counter
H=Path(__file__).resolve().parent;ROOT=H.parent
OUT=H/'fat-current-semantic-support-141-root'
VOL=H/'fat-positive-PLC-prefix-compaction-135-root/fat-positive-compact-PLC-prefix.npz'
SHELL=H/'fat-inferred-boundary-shell-assembly-140-v2/fat-shell-refined-typed.npz'
DREC=H/'registered-current-refined-D-140-root/receipt.json'
BASE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
PINS={VOL:'7ab13a433e0cb08350c901388c0c86712269e0cf66ce70df5fe33de248357e76',
SHELL:'65a13549560123b31ecbc698680edaff48ab843e834ef127e72f59d6aaffc04a',
DREC:'f9f794b9d5db57d9b4be47b5d19db2443d48c64ad21bafe9ff074be1f568de14'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def main():
 OUT.mkdir(exist_ok=True);started=time.monotonic()
 for p,s in PINS.items():
  if sha(p)!=s:raise ValueError('pin changed '+str(p))
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'))
 reg=imp('regFat141',OWNER);parent,index,mi,pm=reg.load_parent_current(BASE)
 dr=json.loads(DREC.read_text())
 for k in ('nodeSHA256','massSHA256','sourceSHA256','tetsSHA256','labelsSHA256'):
  if pm[k]!=dr['parentCurrent'][k]:raise ValueError('parent changed '+k)
 with np.load(VOL)as z:points=z['points_world_m'];tets=z['tetrahedra']
 with np.load(SHELL)as z:
  shell=z['points_world_m'];drows=z['registered_D_shell_node_rows']
 if not np.array_equal(points[:len(shell)].view('u4'),shell.view('u4')):raise ValueError('shell prefix mismatch')
 n=len(points);chosen=np.full(n,-1,np.int32);bary=np.zeros((n,4));residual=np.zeros((n,3));dist=np.full(n,np.nan);node_support=[{}for _ in range(n)];origin=np.zeros(n,np.uint8)
 def put(i,tr,b,res,d,ties,kind):
  if tr not in ties or len(ties)!=len(set(ties)):raise ValueError('bad tied source support')
  if parent['materials'][tr]!=16:raise ValueError('wrong material query')
  if abs(np.sum(b)-1)>1e-8 or np.min(b)<-1e-8 or np.linalg.norm(b@parent['points'][parent['tets'][tr]]+res-points[i].astype(float))>1e-10:raise ValueError('query reconstruction')
  if not np.isfinite(d)or d>.02 or abs(np.linalg.norm(res)-d)>1e-10:raise ValueError('query distance')
  for pt in ties:
   g=tuple(int(parent[k][pt])for k in ('materials','donors','regions'))
   if g[0]!=16:raise ValueError('mixed-material support')
   node_support[i].setdefault(g,set()).add(int(pt))
  chosen[i]=tr;bary[i]=b;residual[i]=res;dist[i]=d;origin[i]=kind
 dfile=DREC.parent/'registered-D.csv'
 if sha(dfile)!=dr['outputSHA256'][dfile.name]:raise ValueError('D registration changed')
 count=0
 with dfile.open(newline='')as f:
  for row in csv.DictReader(f):
   v=int(row['vertex'])
   if v!=count or v>=len(drows):raise ValueError('D row domain')
   i=int(drows[v]);xyz=np.array([float(row[k])for k in ('wx','wy','wz')],np.float32)
   if not np.array_equal(xyz.view('u4'),points[i].view('u4')):raise ValueError('D node exact coordinate')
   tr=int(row['parent_tet_row']);ties=list(map(int,json.loads(row['parent_tied_tet_rows_json'])))
   if int(row['parent_registration_tie_count'])!=len(ties)-1:raise ValueError('D tie count')
   put(i,tr,np.array([float(row['parent_bary'+str(k)])for k in range(4)]),np.array([float(row['parent_residual_'+a+'_m'])for a in 'xyz']),float(row['parent_registration_distance_m']),ties,1)
   count+=1
 if count!=len(drows):raise ValueError('D coverage')
 # Chunks are exact current-query results for interruption-safe static preparation.
 todo=np.flatnonzero(chosen<0);chunk_size=2000;fallback=[]
 for chunk,start in enumerate(range(0,len(todo),chunk_size)):
  rows=todo[start:start+chunk_size];file=OUT/f'query-chunk-{chunk:04d}.npz'
  if file.exists():
   with np.load(file)as z:
    if not np.array_equal(z['node_rows'],rows)or not np.array_equal(z['points'].view('u4'),points[rows].view('u4')):raise ValueError('cached query domain mismatch')
    if str(z['volume_sha256'].item())!=PINS[VOL]or str(z['parent_nodes_sha256'].item())!=pm['nodeSHA256']or str(z['parent_labels_sha256'].item())!=pm['labelsSHA256']:raise ValueError('cached immutable inputs changed')
    for j,i in enumerate(rows):
     put(i,int(z['chosen'][j]),z['bary'][j],z['residual'][j],float(z['distance'][j]),z['ties'][z['tie_offsets'][j]:z['tie_offsets'][j+1]].astype(int).tolist(),2)
  else:
   off=[0];trlist=[]
   for i in rows:
    q=reg.closest_same_material(parent,index,mi,16,points[i].astype(float),.02,1e-8)
    put(i,int(q['tet']),q['bary'],q['residual'],float(q['distance']),list(map(int,q['ties'])),2)
    trlist.extend(q['ties']);off.append(len(trlist))
    if 'exactTieFallback'in q:
     fallback.append({'node_row':int(i),'diagnostic':q['exactTieFallback']})
   np.savez_compressed(file,node_rows=rows,points=points[rows],chosen=chosen[rows],bary=bary[rows],residual=residual[rows],distance=dist[rows],tie_offsets=np.array(off,np.int64),ties=np.array(trlist,np.int32),volume_sha256=PINS[VOL],parent_nodes_sha256=pm['nodeSHA256'],parent_labels_sha256=pm['labelsSHA256'],maximum_distance_m=.02,tie_tolerance_m=1e-8)
  progress={'status':'PREPARING_SAME_MATERIAL_FAT_SUPPORT','nodes':n,'D_reused':count,'fresh_query_count':len(todo),'complete_fresh_queries':int(start+len(rows)),'elapsed_s':time.monotonic()-started,'physical_admission':False,'native_steps':0}
  (OUT/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
  print(json.dumps(progress),flush=True)
 if np.any(chosen<0)or any(not d for d in node_support):raise ValueError('unsupported current Fat node')
 groups=sorted({tuple(int(parent[k][pt])for k in ('materials','donors','regions'))for pt in np.flatnonzero(parent['materials']==16)})
 gids={g:i for i,g in enumerate(groups)}
 ngoff=[0];ng=[];npoff=[0];npr=[]
 for supports in node_support:
  for g in sorted(supports):
   ng.append(gids[g]);npr.extend(sorted(supports[g]));npoff.append(len(npr))
  ngoff.append(len(ng))
 coff=[0];cg=[];num=[];den=[];cpoff=[0];cpr=[];cfoff=[0];cfr=[]
 for i,tet in enumerate(tets):
  supports=[node_support[int(v)]for v in tet];scale=math.lcm(*(len(s)for s in supports));scores=Counter()
  for s in supports:
   for g in s:scores[g]+=scale//len(s)
  if sum(scores.values())!=4*scale:raise ValueError('P1 partition failure')
  for g in sorted(scores):
   cg.append(gids[g]);num.append(scores[g]);den.append(4*scale)
   parents=sorted(set().union(*(s.get(g,set())for s in supports)));faces=sorted(set(int(parent['faces'][pt])for pt in parents))
   cpr.extend(parents);cpoff.append(len(cpr));cfr.extend(faces);cfoff.append(len(cfr))
  coff.append(len(cg))
  if i%100000==0:print(json.dumps({'stage':'P1_CELL_SUPPORT','cells':i,'total':len(tets)}),flush=True)
 dest=OUT/'fat-current-semantic-support.npz'
 np.savez_compressed(dest,node_group_offsets=np.array(ngoff,np.int64),node_group_ids=np.array(ng,np.int32),node_group_parent_tet_offsets=np.array(npoff,np.int64),node_group_parent_tet_rows=np.array(npr,np.int32),node_origin_kind=origin,node_parent_registration_distance_m=dist,group_tuples=np.array(groups,np.int32),cell_group_offsets=np.array(coff,np.int64),cell_group_ids=np.array(cg,np.int32),cell_group_p1_score_numerators=np.array(num,np.int64),cell_group_p1_score_denominators=np.array(den,np.int64),cell_group_parent_tet_offsets=np.array(cpoff,np.int64),cell_group_parent_tet_rows=np.array(cpr,np.int32),cell_group_stable_face_offsets=np.array(cfoff,np.int64),cell_group_stable_face_rows=np.array(cfr,np.uint32),node_parent_chosen_tet_row=chosen,node_parent_barycentric=bary,node_parent_residual_world_m=residual,volume_sha256=PINS[VOL])
 receipt={'status':'PASS_SOURCE_ONLY_COMPLETE_SAME_MATERIAL_FAT_SEMANTIC_SUPPORT','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','method':'Existing230 same-material closest point support; all tied source tets retained. P1 fractions are preferences for shared child capacity, not hard anatomical labels. I material16 mass support is separate from the unique unchanged shared-I geometry/full86 owner.','inputs':{str(p):s for p,s in PINS.items()},'source330':pm,'owner':{'path':str(OWNER),'sha256':sha(OWNER)},'counts':{'nodes':n,'cells':len(tets),'D_reused':count,'fresh_queries':len(todo),'groups':len(groups),'cell_group_slots':len(cg),'unsupported_nodes':0,'preference_group_count':len(set(cg))},'limits':{'parent_registration_distance_m':.02,'tie_tolerance_m':1e-8,'mass_transport_m':.019353,'mass_transport_NOT_evaluated_here':True},'output':{'path':str(dest),'sha256':sha(dest)},'elapsed_s':time.monotonic()-started,'maximum_query_distance_m':float(dist.max())}
 (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');(OUT/'exact-near-tie-fallbacks.json').write_text(json.dumps(fallback,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
