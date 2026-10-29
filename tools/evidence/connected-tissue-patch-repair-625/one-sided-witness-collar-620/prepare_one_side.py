from pathlib import Path
import csv,json,numpy as np,importlib.util,sys,hashlib
P=Path(__file__).resolve().parent.parent
R=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024"
base=R/"closed603-merge-025/candidate-f32-linked-remesh-typed.npz"
with np.load(base)as z:F=z["faces"];N=len(z["points_world_m"])
with(R/"full-native-self-028/residual-pair-attribution-029/witness-pairs.csv").open()as h:
 pairs=[d for d in csv.DictReader(h)if '227'in[d["failed_transaction_a"],d["failed_transaction_b"]]]
seeds=set(int(d["candidate_face_row_b"])for d in pairs)
ed=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]);ed.sort(axis=1)
keys=ed[:,0]*N+ed[:,1];owner=np.tile(np.arange(len(F)),3);order=np.argsort(keys,kind="stable");ks=keys[order];os=owner[order]
path=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/remesh_d_witness_stars_rerun022.py"
spec=importlib.util.spec_from_file_location("bd620",path);bd=importlib.util.module_from_spec(spec);sys.modules[spec.name]=bd;spec.loader.exec_module(bd)
def grow(sel):
 ff=F[sorted(sel)];e=np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]);e.sort(axis=1);q=np.unique(e[:,0]*N+e[:,1])
 out=set(sel)
 for a,b in zip(np.searchsorted(ks,q,"left"),np.searchsorted(ks,q,"right")):out.update(map(int,os[a:b]))
 return out
sel=seeds;plans=[]
for ring in range(0,5):
 if ring:sel=grow(sel)
 try:
  loops,_=bd.boundary_cycles(F,sel)
  edges={tuple(sorted((int(a),int(b))))for f in F[sorted(sel)]for a,b in zip(f,np.roll(f,-1))}
  chi=len(np.unique(F[sorted(sel)]))-len(edges)+len(sel)
  kind="disk"if len(loops)==1 and chi==1 else"annulus"if len(loops)==2 and chi==0 else"unsupported"
  pieces=[{"face_rows":sorted(sel),"boundary_loops":[list(map(int,l))for l in loops],"kind":kind,"euler_characteristic":chi}]
 except Exception as ex:pieces=[{"face_rows":sorted(sel),"boundary_loops":[],"kind":"unsupported","error":repr(ex)}]
 plans.append({"ring":ring,"faces":len(sel),"pieces":pieces})
out={"candidate_path":str(base),"candidate_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),"plans":plans,"seeds":sorted(seeds),"transaction_id":227,"physical_admission":False}
(Path(__file__).resolve().parent/"collar-inventory.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps([{"ring":r["ring"],"faces":r["faces"],"kind":r["pieces"][0]["kind"],"loops":[len(l)for l in r["pieces"][0]["boundary_loops"]],"chi":r["pieces"][0].get("euler_characteristic")}for r in plans]))
