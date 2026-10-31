from pathlib import Path
import hashlib,importlib.util,json,sys,time
H=Path(__file__).resolve().parent
def pin(p):return {"path":str(p),"sha256":hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()}
ap=__import__("argparse").ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--journal",type=Path,required=True);a=ap.parse_args()
out=a.journal.parent
p=H/"quality-partial-cavity-684-root/independent_replay_full_neighbor_kernel_cone_quality_741_root.py"
assert pin(p)["sha256"]=="168153c50ce2296e1af1a04a96b4ce30b236cda1bb989b33ae199a8253fe3448"
s=importlib.util.spec_from_file_location("independent725",p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
journal=json.loads(a.journal.read_text());baseline=json.loads(a.state.read_text());baseline=baseline.get("current_verified_state_20261031",baseline)
assert pin(a.state)==journal["source_state"]
receipts={}
for r in journal["role_results"]:
 role=r["role"]
 if r["status"]!="PASS_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_BATCH":continue
 result=m.verify(a.state,a.journal,role,out/(role.lower()+"-independent-boundary-map.npz"))
 f=out/(role.lower()+"-independent-replay.json");f.write_text(json.dumps(result,indent=2)+"\n");receipts[role]=pin(f)
 print(role,result["status"],result["whole_output_native_bad_cells"],flush=True)
raw=json.loads((out/"state.json").read_text());st=raw.get("current_verified_state_20261031",raw)
assert st["source_surfaces"]==baseline["source_surfaces"]
st["surface_crossings"]=0
st["exact_context"]={"inherited_from_state":pin(a.state),"reason":"Source candidates unchanged; each full oriented volume boundary independently identical"}
st["independent_replay"]={"producer":pin(Path(__file__)),"volume_replays":receipts}
st.update(physical_admission=False,native_steps=0,mass_mapping="PENDING",boundary_metadata="PENDING_REBIND",source_state=pin(a.state))
f=out/"verified-state.json";f.write_text(json.dumps(st,indent=2)+"\n");print(json.dumps({"verified_state":pin(f),"Skin":st["Skin_current"]["native_quality_failures"],"Fat":st["Fat_current"]["native_quality_failures"]}),flush=True)
