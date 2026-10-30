from pathlib import Path
import json,hashlib,subprocess,os,sys,importlib.util,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
out=H/'tetgen-explicit-no-improvement-053/skin037-run-001';out.mkdir(exist_ok=False)
r=json.loads((H/'geometry-feasibility-008-warning037/report.json').read_text())['tetgen'][0]
poly=out/'skin.poly';os.link(r['command'][-1],poly)
cmd=[str(H/'tetgen-explicit-no-improvement-053/tetgen'),'-pYq1.2/10AS512O0/0/0',str(poly)]
started=time.monotonic()
with (out/'stdout.log').open('w')as so,(out/'stderr.log').open('w')as se:
 try:p=subprocess.run(cmd,stdout=so,stderr=se,timeout=90);code=p.returncode
 except subprocess.TimeoutExpired:code='TIMEOUT'
receipt={'argv':cmd,'exit_code':code,'wall_s':time.monotonic()-started,'source_poly_sha256':hashlib.sha256(poly.read_bytes()).hexdigest(),'mesh_validity':'NOT_CHECKED','physical_admission':False,'runtime':0,'accepted_steps':0}
if code==0:
 path=H/'native-domain-217-source-004/build_native_domain_217.py';sys.path[:0]=[str(path.parent),str(P.parent/'authoring-deps')]
 sp=importlib.util.spec_from_file_location('actual_skin_053_owner',path);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);m.configure_capacity_profile('full-reference-500k-nodes-2m-cells')
 try:
  receipt['owner_result']=m._test_tetgen_result(out/'skin.1.node',out/'skin.1.ele',15,out,'skin')
  receipt['mesh_validity']='PASS_EXISTING_POSITIVE_F32_CELL_OWNER'
 except Exception as e:receipt['mesh_validity']='FAIL_EXISTING_CELL_OWNER';receipt['owner_error']=repr(e)
for ext in ('node','ele'):
 p=out/('skin.1.'+ext)
 if p.exists():
  with p.open()as f:receipt[ext+'_header']=f.readline().strip()
(out/'result.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt)[:5000],flush=True);print((out/'stdout.log').read_text()[-1800:]);print((out/'stderr.log').read_text()[-1500:])
