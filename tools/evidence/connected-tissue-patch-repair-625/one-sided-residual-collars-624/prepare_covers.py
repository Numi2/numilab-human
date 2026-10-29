from pathlib import Path
import csv,json,numpy as np,importlib.util,sys,hashlib,collections
P=Path(__file__).resolve().parent.parent
R=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024"
base=R/"closed603-merge-025/candidate-f32-linked-remesh-typed.npz"
with np.load(base)as z:F=z["faces"];N=len(z["points_world_m"])
with(R/"full-native-self-028/residual-pair-attribution-029/witness-pairs.csv").open()as h:rows=list(csv.DictReader(h))
ed=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]);ed.sort(axis=1)
keys=ed[:,0]*N+ed[:,1];owner=np.tile(np.arange(len(F)),3);order=np.argsort(keys,kind="stable");ks=keys[order];os=owner[order]
path=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/remesh_d_witness_stars_rerun022.py"
spec=importlib.util.spec_from_file_location("bd624",path);bd=importlib.util.module_from_spec(spec);sys.modules[spec.name]=bd;spec.loader.exec_module(bd)
def neighbors(sel):
 ff=F[sorted(sel)];e=np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]);e.sort(axis=1);q=np.unique(e[:,0]*N+e[:,1])
 out=set(sel)
 for a,b in zip(np.searchsorted(ks,q,"left"),np.searchsorted(ks,q,"right")):out.update(map(int,os[a:b]))
 return out
def components(sel):
 byedge=collections.defaultdict(list)
 for row in sel:
  face=F[row]
  for a,b in zip(face,np.roll(face,-1)):byedge[tuple(sorted((int(a),int(b))))].append(row)
 adj=collections.defaultdict(set)
 for values in byedge.values():
  for a in values:adj[a].update(values)
 todo=set(sel);out=[]
 while todo:
  seed=min(todo);todo.remove(seed);part={seed};stack=[seed]
  while stack:
   for b in adj[stack.pop()]:
    if b in todo:todo.remove(b);part.add(b);stack.append(b)
  out.append(part)
 return out
inventory=[]
for tid in [108,194,216,306,324]:
 pairs=[r for r in rows if str(tid)in(r["failed_transaction_a"],r["failed_transaction_b"])]
 edges=[(int(r["candidate_face_row_a"]),int(r["candidate_face_row_b"]))for r in pairs]
 covers={}
 for side in ("a","b"):covers[f"endpoint_{side}"]=set(int(r[f"candidate_face_row_{side}"])for r in pairs)
 sids=set(int(r[f"source_id_{s}"])for r in pairs for s in("a","b"))
 for sid in sids:
  if all(sid in(int(r["source_id_a"]),int(r["source_id_b"]))for r in pairs):
   covers[f"source_{sid}"]={int(r[f"candidate_face_row_{s}"])for r in pairs for s in("a","b")if int(r[f"source_id_{s}"])==sid}
 # Minimum vertex-cover alternatives for the small exact witness graph.
 best=[min(map(len,covers.values()))];solutions=[];calls=[0]
 def dfs(rem,selected):
  calls[0]+=1
  if calls[0]>8192 or len(selected)>best[0]:return
  if not rem:
   if len(selected)<best[0]:best[0]=len(selected);solutions.clear()
   if len(solutions)<64:solutions.append(set(selected))
   return
  degree=collections.Counter(v for edge in rem for v in edge)
  edge=max(rem,key=lambda e:sum(degree[v]for v in e))
  for v in sorted(edge,key=lambda v:(-degree[v],v)):
   dfs([e for e in rem if v not in e],selected|{v})
 dfs(edges,set())
 for i,sel in enumerate(solutions):covers[f"minimum_face_cover_{i}"]=sel
 seen=set();plans=[]
 for label,seeds in covers.items():
  if tuple(sorted(seeds))in seen:continue
  seen.add(tuple(sorted(seeds)))
  assert all(a in seeds or b in seeds for a,b in edges)
  sel=set(seeds)
  for ring in range(5):
   if ring:sel=neighbors(sel)
   pieces=[]
   for part in components(sel):
    try:
     loops,_=bd.boundary_cycles(F,part)
     ev={tuple(sorted((int(a),int(b))))for f in F[sorted(part)]for a,b in zip(f,np.roll(f,-1))}
     chi=len(np.unique(F[sorted(part)]))-len(ev)+len(part)
     kind="disk"if len(loops)==1 and chi==1 else"annulus"if len(loops)==2 and chi==0 else"unsupported"
     piece={"face_rows":sorted(part),"boundary_loops":[list(map(int,l))for l in loops],"kind":kind,"euler_characteristic":int(chi)}
    except Exception as ex:piece={"face_rows":sorted(part),"boundary_loops":[],"kind":"unsupported","error":repr(ex)}
    pieces.append(piece)
   plans.append({"choice":label,"ring":ring,"seeds":sorted(seeds),"covers_every_witness_pair":True,"faces":len(sel),"pieces":pieces})
 plans.sort(key=lambda r:(any(p["kind"]!="disk"for p in r["pieces"]),sum(p["kind"]=="unsupported"for p in r["pieces"]),r["faces"],sum(len(x)for p in r["pieces"]for x in p["boundary_loops"])))
 inventory.append({"transaction_id":tid,"witness_pair_count":len(pairs),"pairs":edges,"minimum_cover_search_calls":calls[0],"plans":plans})
out={"candidate_path":str(base),"candidate_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),"transactions":inventory,"physical_admission":False}
Path(__file__).with_name("collar-inventory.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps([{"transaction_id":r["transaction_id"],"pairs":r["witness_pair_count"],"best":[{"choice":v["choice"],"ring":v["ring"],"faces":v["faces"],"pieces":[{"kind":p["kind"],"f":len(p["face_rows"]),"loops":[len(l)for l in p["boundary_loops"]]}for p in v["pieces"]]}for v in r["plans"][:4]]}for r in inventory],indent=2))
