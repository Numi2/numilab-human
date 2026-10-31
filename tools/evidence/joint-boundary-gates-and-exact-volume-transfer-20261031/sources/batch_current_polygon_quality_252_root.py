"""Bounded replay of existing polygon cavity improvements, preserving caches."""
from pathlib import Path
import importlib.util,sys,json,argparse,time,hashlib
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("replay251for252",H/"replay_current_polygon_quality_251_root.py");m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
def sha(f):return hashlib.file_digest(open(f,"rb"),"sha256").hexdigest()
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--input-npz",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--rounds",type=int,default=6);ap.add_argument("--seconds",type=float,default=50);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);tic=time.monotonic();current=a.input_npz;history=[]
 for i in range(a.rounds):
  if time.monotonic()-tic>a.seconds:break
  out=a.output/("pass-%03d"%i);args=argparse.Namespace(input_npz=current,output=out,seconds=min(10,a.seconds-(time.monotonic()-tic)),maximum_degree=12,maximum_operations=10000)
  m.run(args);r=json.loads((out/"report.json").read_text());current=out/"fixed-boundary-volume-proposal.npz"
  history.append({"pass":i,"report_sha256":sha(out/"report.json"),"proposal":str(current),"proposal_sha256":sha(current),"before":r["bad_cells_before"],"after":r["bad_cells_after"],"passing_operations":r["passing_operations"]})
  if not r["passing_operations"]:break
 receipt={"status":"PASS_BOUNDED_CURRENT_POLYGON_REPLAY","physical_admission":False,"native_steps":0,"input":str(a.input_npz),"input_sha256":sha(a.input_npz),"source_sha256":sha(Path(__file__)),"wrapper_sha256":sha(Path(m.__file__)),"history":history,"final_proposal":str(current),"final_sha256":sha(current),"bad_before":history[0]["before"],"bad_after":history[-1]["after"],"elapsed_s":time.monotonic()-tic}
 (a.output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt))
