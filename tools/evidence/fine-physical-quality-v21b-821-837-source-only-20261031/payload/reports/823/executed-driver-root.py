from pathlib import Path
import importlib.util,sys,json,hashlib,argparse
h=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");out=h/'all-quality-823-root';out.mkdir(exist_ok=False)
def imp(n,p,w):
 assert hashlib.sha256(p.read_bytes()).hexdigest()==w
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
b=imp('root823producer',h/'quality-partial-cavity-684-root/batch_interior_steiner_partial_quality_795_root.py','4e35df53981c2fedc0f03b268138ebf74378a09e0c55d1f7a64b8e09c813f3bf')
p=imp('root823promoter',h/'quality-partial-cavity-rebase-809-root/independent_promoter_796_807.py','5a6e2df084a097448172171cca05e1b0f6e281d64302c5592987ca9630c1a7ce')
state=h/'all-quality-821-root/round001/flips/verified-state.json'
a=argparse.Namespace(maximum_old_cells=64,maximum_points=256,maximum_star_cells=256,maximum_attempts=10000,maximum_operations=1000,construction_margin=0.)
r=b.run_state(state,out/'proposal',90.,a)
print(json.dumps({'role_summary':[{k:z.get(k) for k in ['role','status','native_bad_before','native_bad_after','log_deficit_before','log_deficit_after']} for z in r.get('role_results',[])]}),flush=True)
if any(z.get('status')=='PASS_INTERIOR_STEINER_PARTIAL_QUALITY_CONE_BATCH' for z in r.get('role_results',[])):
 print(json.dumps(p.verify_and_promote(state,out/'proposal/report.json',out/'verified-state.json',out/'independent-promoter.json')),flush=True)
