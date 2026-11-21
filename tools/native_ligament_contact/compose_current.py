"""Keep the whole accepted live-parent material package together with the recooked shoulders."""
import json,sys,hashlib
from pathlib import Path
from itertools import zip_longest
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
prior,current,output=map(Path,sys.argv[1:]);bones=json.loads(prior.read_text());full=json.loads(current.read_text())
for k,v in bones['hashes'].items():assert sha(k)==v
a,b=bones['argv'],full['argv']
# The full source scene and checked shoulder recipe use the same posture,
# mass, bed, circulation, respiratory drive, and list of ligament owners.
skip={0,3,4,a.index('--muscle-step-count')+1,a.index('--tendon-payload')+1}
assert len(a)==len(b) and all(x==y for i,(x,y) in enumerate(zip(a,b)) if i not in skip)
for i in skip:b[i]=a[i]
assert b[b.index('--muscle-step-count')+1]=='1'
full['argv']=b;full['environment']={k:v.replace(str(current.parent)+'/',str(prior.parent)+'/') if k in ('NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT','NUMI_HUMAN_COUPLED_TRACE_CSV') else v for k,v in full['environment'].items()}
full['launch_output_parent']=str(prior.parent)
full['environment'].pop('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS',None) # screenshot-free entrance
full['hashes']={**bones['hashes'],str(prior):sha(prior),str(current):sha(current)}
full['input_parent']='complete native live-parent mass, tissue and solver configuration'
output.write_text(json.dumps(full,indent=2)+'\n')
