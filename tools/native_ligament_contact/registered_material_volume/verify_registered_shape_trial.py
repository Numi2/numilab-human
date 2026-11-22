"""Compare matched native physical histories and validate the terminal role-specific cell measurements."""
import csv,json,re,sys
from pathlib import Path
from emit_skin_source_registration import sha
control,trial,release,out=map(Path,sys.argv[1:])
out.mkdir(exist_ok=True,parents=True)
ac,at=[json.loads((d/'admission.json').read_text()) for d in (control,trial)]
lc,lt=[json.loads((d/'launch.json').read_text()) for d in (control,trial)]
assert ac['exit_code']==at['exit_code']==0 and ac['accepted_steps']==at['accepted_steps']==8
assert lc['argv'][1:]==lt['argv'][1:] and lt['environment'].get('NUMI_HUMAN_REGISTERED_CELL_QUALITY')=='1'
def plain_env(j):return {k:v for k,v in j['environment'].items() if k!='NUMI_HUMAN_REGISTERED_CELL_QUALITY'}
assert plain_env(lc)==plain_env(lt)
trace='native/resting-coupled.csv'
def data(p):
    with p.open() as f:
        rd=csv.DictReader(f);return rd.fieldnames,list(rd)
cols,left=data(control/trace);current,right=data(trial/trace);assert cols==current and len(left)==len(right)==8
# Physical fields, controller history and timestamps are included. Runtime timing may differ.
ignore={'wall_s','wall_seconds','wall_elapsed_s','rtf','geometry_wall_s'}
physical=[c for c in cols if c not in ignore]
diff={k:[i+1 for i,(a,b) in enumerate(zip(left,right)) if a[k]!=b[k]] for k in physical}
diff={k:v for k,v in diff.items() if v}
receipt='native/accepted-geometry/step-8.receipt.json'
a,b=[json.loads((v/receipt).read_text()) for v in (control,trial)]
for key in ('pack_file_sha256','accepted_root_fingerprint_hex','captured_vertex_buffer_sha256','accepted_body_state_sha256','accepted_respiration_state_sha256'):
    if a[key]!=b[key]:raise ValueError('Read-only observer changed '+key)
log=(trial/'stdout.log').read_text();measured={}
for line in log.splitlines():
    if line.startswith('registered_cell_shape '):
        row=dict(term.split('=',1) for term in line.split()[1:]);role=int(row['material']);phase=row['phase']
        if phase in measured.setdefault(role,{}):raise ValueError('Duplicate phase')
        keys=('step','cells','nonfinite','inverted','volume_le_1e_18','shape_lt_1e_5','minimum_shape_local_cell')
        measured[role][phase]={k:(int(v) if k in keys else float(v)) for k,v in row.items() if k not in ('phase','material')}
expected={15:705091,16:743713}
if set(measured)!=set(expected):raise ValueError('Skin/fat subset missing')
for role,count in expected.items():
    phases=measured[role];assert set(phases)=={'accepted','reference'}
    for phase,row in phases.items():assert row['cells']==count and row['step']==(8 if phase=='accepted' else 0)
def pairs(p):
    result=set()
    for line in (p/'envelope/target-witnesses.jsonl').open():
        entry=json.loads(line);result.update((entry['semantic'],entry['stable_id'],i,j) for i,j in entry['triangle_pairs'])
    return result
before,after=pairs(release),pairs(control)
result=dict(terminal_shape=measured,reported_clinical_validation=False,accepted_roots=8,
    trace_columns=len(cols),compared_columns=physical,changed_columns=diff,excluded_columns=sorted(set(cols)-set(physical)),
    terminal_pack_sha256=a['pack_file_sha256'],accepted_root_fingerprint=a['accepted_root_fingerprint_hex'],
    native_binary_sha256=sha(Path(lt['argv'][0])),reference_trial_sha256=sha(control/'launch.json'),
    trial_sha256=sha(trial/'launch.json'),gained_skin_triangle_pairs=sorted(after-before),cleared_skin_triangle_pairs=sorted(before-after))
(out/'accepted-cell-quality.json').write_text(json.dumps(result,indent=2)+'\n')
if diff:raise ValueError('Physical history differs: '+str(list(diff)))
print(measured,flush=True)
