from pathlib import Path
import sys,json,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");N=A/"right-edl-ehl-native-mtp-two-001";B=A/"ehl-restored-pair-native-mtp-two-001";O=N/"native-interface-rounding-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h);sha=h.sha
RP=N/"native-verification-003/report.json";r=json.loads(RP.read_text());assert r["complete"] and r["all_native_geometry_checks_pass"] and r["inputs_unchanged"]
T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";BT=B/"assets"/T.name
cr=h.row_data(h.load_tissue(T),21);br=h.row_data(h.load_tissue(BT),21)
pins={str(p):sha(p) for p in [Path(__file__),H,RP,T,BT,Path(ca.__file__),Path(ci.__file__)]}
def keys(row):
 return [row["positions"][i].tobytes()+row["local"][i].tobytes()+row["weights"][i].tobytes() for i in range(row["vc"])]
ck,bk=keys(cr),keys(br);lookup={}
for i,f in enumerate(br["faces"]):lookup.setdefault(tuple(sorted(bk[j] for j in f)),[]).append(i)
out={"scope":"Explain exact native/offline pair-table difference without altering geometry or waiving an intersection. Each differing EDL triangle must have unchanged source coordinates/maps and identical actual native triangle geometry in the prior EHL scene. Both current and parent exact predicates are evaluated against unchanged target geometry. This does not qualify existing anatomical overlaps.","pins":pins,"witnesses":[],"complete":False}
for e in r["poses"]:
 step=e["accepted_step"]
 for row in e["rows"]:
  diffs=[("added",x) for x in row["native_pairs_added_vs_offline"]]+[("removed",x) for x in row["native_pairs_removed_vs_offline"]]
  if not diffs:continue
  assert row["stable_id"]==21
  paths=[scene/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack" for scene in [N,B]]
  surfaces=[]
  for p in paths:
   pins[str(p)]=sha(p);rp=p.with_suffix(".receipt.json");pins[str(rp)]=sha(rp);assert json.loads(rp.read_text())["pack_file_sha256"]==pins[str(p)]
   surfaces.append(ca._pack_surfaces(p,{(51005,21),*[(x[0],x[1]) for _,x in diffs]})[:2])
  (cp,cs),(bp,bs)=surfaces
  for kind,(sem,sid,fi,tfi) in diffs:
   match=lookup[tuple(sorted(ck[j] for j in cr["faces"][fi]))];assert len(match)==1;bi=match[0]
   cf=np.asarray(cs[51005,21]["faces"],int)[fi];bf=np.asarray(bs[51005,21]["faces"],int)[bi]
   ct=cp[cf];bt=bp[bf]
   assert sorted(p.tobytes() for p in ct)==sorted(p.tobytes() for p in bt)
   target=cp[np.asarray(cs[sem,sid]["faces"],int)[tfi]]
   oldtarget=bp[np.asarray(bs[sem,sid]["faces"],int)[tfi]];assert np.array_equal(target,oldtarget)
   def crosses(tri):
    one,deg=h.exact_rows(tri,np.array([[0,1,2]]),ci);two,dg=h.exact_rows(target,np.array([[0,1,2]]),ci);assert not deg and not dg
    return ci._audit_pair(one,two,same_surface=False)["count"]
   now,prior=crosses(ct),crosses(bt);assert now==prior==int(kind=="added")
   out["witnesses"].append({"step":step,"native_vs_offline":kind,"target":[sem,sid,tfi],"candidate_face":fi,"parent_face":bi,"source_triangle_coordinates_and_display_maps_exact_parent":True,"actual_native_triangle_geometry_exact_parent":True,"target_native_geometry_exact_parent":True,"native_current_exact_pair_count":now,"native_parent_exact_pair_count":prior,"maximum_row_native_vs_CPU_forward_error_m":row["max_forward_error_m"]})
out.update(complete=True,inputs_unchanged=all(sha(p)==s for p,s in pins.items()),all_native_offline_differences_already_in_parent_native_geometry=bool(out["witnesses"]) and all(x["actual_native_triangle_geometry_exact_parent"] for x in out["witnesses"]))
(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({k:v for k,v in out.items() if k!="pins"}))
