#!/usr/bin/env python3
"""Current-candidate envelope sensitivity; no native/admission contract change."""
import csv,hashlib,json,math,os,subprocess,time,datetime
from pathlib import Path
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706';A=H/'current-fractional-semantic-native-1047'; OUT=A/'actual-035-current-envelope-diagnostic/run-001';OUT.mkdir(parents=True,exist_ok=False)
def pin(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()}
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
oldpath=P/'conforming-fit-006-native-assets-545/run-002-operational-reference/plan.json'
old=json.loads(oldpath.read_text());assert old['mass_transport_rule']=='ceil(max child-to-declared-source-centroid distance/1um)*1um+1um, floor200um; derived before native evaluation'
basepath=A/'actual-030c-032d-diagnostic/run-001/execution.json';base=json.loads(basepath.read_text());cmd=base['command'].copy();si=cmd.index('--parent-mass-csv');s=Path(cmd[si+1]).parent
rows=list(csv.DictReader(open(s/'compiled-parent-cell-mass.csv.nodes.csv')));v=np.array([[float(r[k]) for k in ['current_x_m','current_y_m','current_z_m']] for r in rows]);rows=list(csv.DictReader(open(s/'compiled-parent-cell-mass.csv.tets.csv')));t=np.array([[int(r['parent_node'+str(k)]) for k in range(4)] for r in rows]);pc=v[t].mean(1)
zpath=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz';z=np.load(zpath);child=z['points_reference_world_m_f64'][z['volume_tetrahedra']].mean(1);parent=z['child_parent_tet_row'];d=np.linalg.norm(child-pc[parent],axis=1);r=max(.0002,(math.ceil(float(d.max())/1e-6)+1)*1e-6)
assert abs(r-.187444)<1e-12
w=int(d.argmax());cert={'schema':'numi.human.current-envelope-diagnostic.v1','purpose':'Source-only sensitivity using the old candidate-independent geometric rule on the current mesh; NOT an accepted/native radius change.','native_steps':0,'physical_admission':False,'source':pin(__file__),'historical_rule':pin(oldpath),'base_execution':pin(basepath),'current_typed_volume':pin(zpath),'source_nodes':pin(s/'compiled-parent-cell-mass.csv.nodes.csv'),'source_tets':pin(s/'compiled-parent-cell-mass.csv.tets.csv'),'old_envelope_m':.019353,'current_derived_envelope_m':r,'rule':old['mass_transport_rule'],'distance_quantiles_m':np.quantile(d,[0,.5,.9,.99,.999,1]).tolist(),'maximum_witness':{'child_row':w,'parent_row':int(parent[w]),'distance_m':float(d[w]),'child_centroid':child[w].tolist(),'source_parent_centroid':pc[parent[w]].tolist()},'unchanged':['source330 mass ledger','source parent geometry','density bounds','semantic group and one-hop checks','P1 support','carrier weights and first moments','current1020 geometry','runtime physics']}
certpath=OUT/'envelope-sensitivity-plan.json';dump(certpath,cert)
for i,a in enumerate(cmd):
 if a.startswith('NUMI_H706_FRACTIONAL_OUTPUT_DIR='):cmd[i]='NUMI_H706_FRACTIONAL_OUTPUT_DIR='+str(OUT/'verified-flow')
 if a.startswith('NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256='):cmd[i]='NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256='+pin(certpath)['sha256']
cmd[cmd.index('--maximum-transport-distance-m')+1]=format(r,'.17g')
log=OUT/'logs';log.mkdir();start=time.monotonic();timed=False
with (log/'cli.stdout.log').open('wb') as stdout,(log/'cli.stderr.log').open('wb') as stderr:
 proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr)
 try:rc=proc.wait(timeout=300)
 except subprocess.TimeoutExpired:
  timed=True;proc.terminate()
  try:rc=proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.kill();rc=proc.wait()
report={'schema':'numi.human.current-envelope-diagnostic-execution.v1','command':cmd,'source':pin(__file__),'envelope_plan':pin(certpath),'started_from_actual030c':pin(basepath),'exit_code':rc,'timed_out':timed,'elapsed_wall_seconds':time.monotonic()-start,'native_steps':0,'physical_admission':False,'status':'DIAGNOSTIC_ONLY_NO_CONTRACT_PROMOTION','stdout_log':pin(log/'cli.stdout.log'),'stderr_log':pin(log/'cli.stderr.log'),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump(OUT/'execution.json',report);print(json.dumps(report,indent=2))
