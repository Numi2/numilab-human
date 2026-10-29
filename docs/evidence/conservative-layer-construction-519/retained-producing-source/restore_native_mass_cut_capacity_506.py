#!/usr/bin/env python3
"""Offline local geometry restoration from exact conservative-transport min cuts."""
import argparse,csv,json,time,math,re,subprocess,importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
def load(name,file):
 s=importlib.util.spec_from_file_location(name,P/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cap=load("cap506","audit_regional_mass_capacity_464.py");fit=load("fit506","fit_connected_measured_contact_501.py")
CAP=P/"native-physical-parent-mass-330/native-source-identity-control-003"
def rows(p):
 with p.open(newline="") as f:return list(csv.DictReader(f))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--candidate",type=Path,required=True);ap.add_argument("--volume",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--rounds",type=int,default=16);args=ap.parse_args()
 args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 with np.load(args.candidate) as z:a={k:z[k].copy() for k in z.files}
 x=a["points_reference_world_m_f64"].astype(np.float32).astype(float);t=a["volume_tetrahedra"];cp=a["child_parent_tet_row"].astype(int);base=x.copy()
 keys=np.stack([a[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]],axis=1)
 supply=cap.load_supply(CAP/"compiled-parent-cell-mass.csv.tets.csv",CAP/"compiled-parent-cell-mass.csv.labels.csv")
 groupkeys=sorted(supply);gm={k:i for i,k in enumerate(groupkeys)};gi=np.array([gm[tuple(map(int,k))] for k in keys])
 mass=np.array([supply[k] for k in groupkeys]);rho=np.array([cap.DENSITY_BOUNDS[k[0]] for k in groupkeys]);vlo=mass/rho[:,1];vhi=mass/rho[:,0]
 tr=rows(CAP/"compiled-parent-cell-mass.csv.tets.csv");lr=rows(CAP/"compiled-parent-cell-mass.csv.labels.csv");nr=rows(CAP/"compiled-parent-cell-mass.csv.nodes.csv")
 pm=np.array([float(r["parent_mass_kg"]) for r in tr]);pt=np.array([[int(r["parent_node"+str(k)]) for k in range(4)] for r in tr]);pp=np.array([[float(r["rest_"+k+"_m"]) for k in "xyz"] for r in nr])
 parentcent=pp[pt].mean(1)
 with np.load(P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz") as z:original=z["points_reference_world_m_f64"].astype(np.float32).astype(float)
 vmin=cap.tet_volumes_gradients(original,t)[0]*.02
 pinned=np.union1d(a["pinned_volume_node_rows"],[30,2441])
 for key in ["material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"]:
  assert np.array_equal(a[key],np.array([int(r[key]) for r in lr])[cp])
 cli=Path("/Users/numi/Work/skin-fat-native-integration-209/build-native/numi-human-physical-skin-child-mass-mapping-cli")
 cuts=[];history=[];flow_pass=False
 for outer in range(args.rounds):
  run=args.out/("mass-%03d"%outer);run.mkdir()
  node=run/"mesh.node"
  with node.open("w") as f:
   f.write(f"{len(x)} 3 0 0\n")
   for i,p in enumerate(x):f.write(str(i+1)+" "+" ".join(format(float(k),".17g") for k in p)+"\n")
  d=np.linalg.norm(x[t].mean(1)-parentcent[cp],axis=1);bound=max(.0002,math.ceil(float(d.max())/1e-6)*1e-6+1e-6)
  cmd=[str(cli),"--parent-mass-csv",str(CAP/"compiled-parent-cell-mass.csv"),"--labels-csv",str(CAP/"compiled-parent-cell-mass.csv.labels.csv"),"--child-node-file",str(node),"--child-ele-file",str(args.volume/"mesh.ele"),"--child-labels-csv",str(args.volume/"child.labels.csv"),"--coordinate-frame-sha256","0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a","--source-body-count","157","--maximum-transport-distance-m",format(bound,".17g")]
  plan={"command":cmd,"maximum_declared_parent_centroid_distance_m":float(d.max()),"predeclared_centroid_envelope_m":bound,"envelope_rule":"ceil actual declared child/reference-parent centroid distance to1um +1um; minimum200um; computed before flow","node_sha256":cap.sha(node),"physical_admission":False}
  (run/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
  r=subprocess.run(cmd,capture_output=True,text=True);(run/"stdout.txt").write_text(r.stdout);(run/"stderr.txt").write_text(r.stderr)
  item={"round":outer,"exit_code":r.returncode,"envelope_m":bound}
  history.append(item);print(json.dumps(item),flush=True)
  if r.returncode==0:flow_pass=True;break
  match=re.search(r"cut_parent_rows=(\[[0-9, ]*\]) cut_child_rows=(\[[0-9, ]*\])",r.stderr)
  if not match:item["stop"]="no supported min-cut witness";break
  pr=np.array(json.loads(match[1]),int);cr=np.array(json.loads(match[2]),int)
  if not len(pr) or not len(cr):item["stop"]="empty min-cut";break
  sk=np.unique(keys[cr],axis=0);assert len(sk)==1
  pk=np.array([[int(lr[i][k]) for k in ["material_index","geometric_donor_body","anatomical_region_id"]] for i in pr])
  assert np.all(pk==sk[0])
  lo,hi=cap.DENSITY_BOUNDS[int(sk[0,0])];cm=pm[pr].sum();cut=(pr,cr,cm/hi,cm/lo)
  if any(np.array_equal(c[0],pr) and np.array_equal(c[1],cr) for c in cuts):item["stop"]="repeated unchanged cut";break
  cuts.append(cut);item["cut"]={"parent_rows":pr.tolist(),"child_rows":cr.tolist(),"source_mass_kg":float(cm),"volume_min_m3":float(cm/hi),"volume_max_m3":float(cm/lo)}
  seed=np.unique(t[np.concatenate([c[1] for c in cuts])]);inc=np.any(np.isin(t,seed),axis=1);free=np.setdiff1d(np.unique(t[inc]),pinned);fm=np.full(len(x),-1,np.int64);fm[free]=np.arange(len(free));affected=np.flatnonzero(np.any(fm[t]>=0,axis=1))
  iterations=[]
  for inner in range(8):
   v,g=cap.tet_volumes_gradients(x,t);gv=np.bincount(gi,weights=v,minlength=len(groupkeys));cutv=np.array([v[c[1]].sum() for c in cuts]);clo=np.array([c[2] for c in cuts]);chi=np.array([c[3] for c in cuts])
   if np.all(v>0) and np.all(gv>=vlo) and np.all(gv<=vhi) and np.all(cutv>=clo) and np.all(cutv<=chi):break
   rr=np.broadcast_to(np.arange(len(affected))[:,None,None],g[affected].shape);fr=fm[t[affected]];cc=fr[:,:,None]*3+np.arange(3)[None,None,:];mask=np.broadcast_to((fr>=0)[:,:,None],g[affected].shape)
   Av=sparse.coo_matrix((g[affected][mask],(rr[mask],cc[mask])),shape=(len(affected),len(free)*3)).tocsr();vn=np.sqrt(np.asarray(Av.multiply(Av).sum(axis=1)).ravel());assert np.all(vn>0);Av=sparse.diags(1/vn)@Av
   rr=np.broadcast_to(gi[:,None,None],g.shape);fr=fm[t];cc=fr[:,:,None]*3+np.arange(3)[None,None,:];mask=np.broadcast_to((fr>=0)[:,:,None],g.shape)
   Ag=sparse.coo_matrix((g[mask],(rr[mask],cc[mask])),shape=(len(groupkeys),len(free)*3)).tocsr();gn=np.sqrt(np.asarray(Ag.multiply(Ag).sum(axis=1)).ravel());has=gn>0;assert np.all(gv[~has]>=vlo[~has]) and np.all(gv[~has]<=vhi[~has]);Ag=sparse.diags(1/gn[has])@Ag[has]
   cri=[];cci=[];cval=[]
   for j,c in enumerate(cuts):
    nodes=t[c[1]];fr=fm[nodes];gs=g[c[1]];mask=np.broadcast_to((fr>=0)[:,:,None],gs.shape);cc=fr[:,:,None]*3+np.arange(3)[None,None,:]
    cri.extend(np.full(mask.sum(),j));cci.extend(cc[mask]);cval.extend(gs[mask])
   Ac=sparse.coo_matrix((cval,(cri,cci)),shape=(len(cuts),len(free)*3)).tocsr();cn=np.sqrt(np.asarray(Ac.multiply(Ac).sum(axis=1)).ravel());assert np.all(cn>0);Ac=sparse.diags(1/cn)@Ac
   A=sparse.vstack([Av,Ag,Ac,sparse.eye(len(free)*3)],format="csc")
   lower=np.r_[(1.25*vmin[affected]-v[affected])/vn,(vlo[has]+.005*(vhi[has]-vlo[has])-gv[has])/gn[has],(clo+.005*(chi-clo)-cutv)/cn,np.full(len(free)*3,-.0001)]
   upper=np.r_[np.full(len(affected),np.inf),(vhi[has]-.005*(vhi[has]-vlo[has])-gv[has])/gn[has],(chi-.005*(chi-clo)-cutv)/cn,np.full(len(free)*3,.0001)]
   fit.CONTACT_MATRIX=None;fit.CONTACT_TARGET=None;delta,diag=fit.solve_qp(sparse.eye(len(free)*3,format="csc"),(base[free]-x[free]).ravel(),A,lower,upper)
   dx=np.zeros_like(x);dx[free]=delta.reshape(-1,3);x=(x+dx).astype(np.float32).astype(float)
   iterations.append({"inner":inner,"free_nodes":len(free),"max_correction_m":float(np.linalg.norm(dx,axis=1).max()),"qp":diag})
  item["restoration"]=iterations
  (args.out/"history.json").write_text(json.dumps(history,indent=2)+"\n")
  if not iterations:item["stop"]="no geometry correction";break
 # Update only the existing authored candidate state.
 meta=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk"
 rr={int(z["volume_node_row"]):z for z in rows(meta/"coarse-node-to-fine-surface.csv")};poses=json.loads((CAP/"registered-skin-poses-100.json").read_text())
 for n in np.flatnonzero(np.any(x!=base,axis=1)):
  pose=poses["all_bodies"][int(rr[int(n)]["frame_owner_body"])]["current"];rot=np.array(pose["local_to_world_row_major_f32"]).reshape(3,3);origin=np.array(pose["position_world_m"])
  a["points_donor_local_m_f64"][n]=np.linalg.solve(rot,x[n]-origin);a["points_current_world_m_f32"][n]=x[n];a["target_current_world_m"][n]=x[n]
 a["points_reference_world_m_f64"]=x;dest=args.out/"candidate.npz";np.savez_compressed(dest,**a)
 v,_=cap.tet_volumes_gradients(x,t);gv=np.bincount(gi,weights=v,minlength=len(groupkeys))
 rep={"script_sha256":cap.sha(__file__),"source_sha256":cap.sha(args.candidate),"candidate_sha256":cap.sha(dest),"native_conservative_mass_flow_passed":flow_pass,"history":history,"nonpositive_cells":int(np.count_nonzero(v<=0)),"failed_capacity_groups":int(np.count_nonzero((gv<vlo)|(gv>vhi))),"pinned_and_paired_targets_unchanged":bool(np.array_equal(x[pinned],base[pinned])),"max_change_m":float(np.linalg.norm(x-base,axis=1).max()),"elapsed_s":time.monotonic()-start,"physical_admission":False}
 (args.out/"receipt.json").write_text(json.dumps(rep,indent=2)+"\n");print(json.dumps(rep),flush=True)
if __name__=="__main__":main()
