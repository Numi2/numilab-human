from pathlib import Path
import json,hashlib
W=Path(__file__).resolve().parent.parent
O=Path(__file__).resolve().parent
runs={"control558":W/"accuracy-refinement-pair-558/control","candidate561":W/"domain-linear-native-561"}
out={"scope":"Four matched initial roots; same mesh, physics, dt, and budgets; domain linear stopping policy differs. Observational traces enabled. Not sustained performance or mesh fidelity qualification.","runs":{}}
for name,D in runs.items():
 rows={};traces=[];certificates=[];linear=[]
 for line in (D/"install-motion.stderr.log").read_text().splitlines():
  key,sep,payload=line.partition("=")
  if not sep or key not in {"matter_fgmres_active_work","matter_solver_domain_norms","fem_newton_iterate","fem_true_residual_cycle","fem_newton_terminal_certificate"}:continue
  try:d=json.loads(payload)
  except ValueError:continue
  root=d.get("root",d.get("control_step"))
  if root is None or root>3:continue
  if key=="matter_fgmres_active_work":rows.setdefault(root,{})["work"]=d
  elif key=="matter_solver_domain_norms":rows.setdefault(root,{})["domains"]=d
  elif key=="fem_newton_iterate":
   if d["stage"]=="final_line_search":traces.append(dict(root=root,iteration=d["iteration"],alpha=d["alpha"],fgmres=d["fgmres"]))
  elif key=="fem_newton_terminal_certificate":certificates.append(d)
  elif key=="fem_true_residual_cycle":
   linear.append({k:d[k] for k in ["root","newton_iteration","restart_cycle","environments"]})
 out["runs"][name]={"roots":rows,"newton":traces,"linear":linear,"terminal":certificates}
 out["runs"][name]["totals"]={k:sum(r["work"][k] for r in rows.values()) for k in ["used_columns_sum","cycles_with_work","budget_exhausted_solves","active_newton_begins"]}
 p=D/"install-motion.execution.json"
 if p.exists():out["runs"][name]["execution"]=json.loads(p.read_text())
(O/"native-diagnostic-comparison.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:{"totals":v["totals"],"roots":{r:{"latched":a["work"]["last_newton_latched"],"minimum_norm":a["work"]["minimum_newton_norm"],"last_norm":a["work"]["last_newton_norm"]} for r,a in v["roots"].items()}} for k,v in out["runs"].items()},indent=2))
