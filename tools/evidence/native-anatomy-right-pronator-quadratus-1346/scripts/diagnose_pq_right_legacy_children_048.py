from pathlib import Path
import sys,json,importlib.util,numpy as np,time
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pq-right-legacy-child-diagnostic-048";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
C=A/"pq-right-legacy-binding-sensitivity-047/stable-129-unadmitted-v881-delta--5e-05.npz";N=A/"left-pq-retriangulated-native-mtp-two-002";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";prior=A/"pq-right-binding-all25-target-audit-043/report.json"
z=dict(np.load(C));p=z["vertices6"][:,:3];f=z["faces"];orig=z["face_origins"];t=h.load_tissue(T);row=h.row_data(t,129)
pins={str(x):h.sha(x) for x in [Path(__file__),H,C,T,prior,Path(ca.__file__),Path(ci.__file__),Path(cg.__file__)]}
def rec(v,ff):
 ids=np.unique(ff);r,d=h.exact_rows(v[ids],np.searchsorted(ids,ff),ci);assert not d;return r
contexts=[]
for path,hashval in json.loads(prior.read_text())["pins"].items():
 if not path.endswith("step-47519.mrvpack"):continue
 pack=Path(path);rp=pack.with_suffix(".receipt.json");assert h.sha(pack)==hashval
 rc=json.loads(rp.read_text());pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc.get("accepted_registered_body_poses"))}
 pos,s,_=ca._pack_surfaces(pack,{(51005,129),(51004,14)});sf=np.asarray(s[(51005,129)]["faces"]);off=sf-row["faces"];assert np.all(off==off.flat[0]);bv=pos[int(off.flat[0]):int(off.flat[0])+row["vc"]]
 target=rec(pos,np.asarray(s[(51004,14)]["faces"]));base=set(map(tuple,ci._audit_pair(rec(bv,row["faces"]),target,same_surface=False)["triangle_pairs"]))
 contexts.append((str(pack),pm,target,base));pins[path]=hashval;pins[str(rp)]=h.sha(rp)
assert len(contexts)==2

out={"pins":pins,"poses":[]}
for path,pm,target,base in contexts:
 v=h.forward(p,z["binding_indices"],z["weights"],row,t,pm);pairs=ci._audit_pair(rec(v,f),target,same_surface=False)["triangle_pairs"]
 added=[{"child_face":int(i),"parent_face":int(orig[i]),"target_face":int(j),"vertices":f[i].tolist(),"weights":z["weights"][f[i]].tolist(),"local_indices":z["binding_indices"][f[i]].tolist()} for i,j in pairs if (int(orig[i]),int(j)) not in base]
 out["poses"].append({"pack":path,"added":added})
out["inputs_unchanged"]=all(h.sha(x)==v for x,v in pins.items());(O/"report.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["poses"]),flush=True)
