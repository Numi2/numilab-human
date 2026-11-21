"""Compare the full-scene candidate mass preflight and the matched native control, per physical step."""
import csv,json,re,sys
from pathlib import Path
import numpy as np
from material_context import sha

def check(control,current,masslog,out):
    r=json.loads((current/'result.json').read_text())
    if r['exit_code']!=0:raise ValueError('Native process failed')
    def rows(p):
        with (p/'resting-coupled.csv').open() as fh:return list(csv.DictReader(fh))
    old=rows(control);new=rows(current)
    if len(new)!=r['trace_rows'] or len(new)!=len(old) or list(new[0])!=list(old[0]):raise ValueError('Different native trace schema/duration')
    differing=[k for k in old[0] if any(a[k]!=b[k] for a,b in zip(old,new,strict=True))]
    if differing:raise ValueError(differing)
    log=(current/'stdout.log').read_text()
    if 'applied_to_live_world=false' not in log:raise ValueError('Ambiguous physical mass ownership')
    records=[{k:float(v) for k,v in re.findall(r'([a-z_]+)=([-+.0-9eE]+)',s)} for s in log.splitlines() if s.startswith('pelvic_material_candidate_partition ')]
    known=json.loads(masslog.read_text())['sides'];expected={}
    for side in known.values():
        for owner,kg in side['donor_kg'].items():expected[int(owner)]=expected.get(int(owner),0)+kg
    if {int(r['body']) for r in records}!=set(expected) or len(records)!=len(expected):raise ValueError('Missing donor')
    for row in records:
        if abs(row['debit_kg']-expected[int(row['body'])])>1e-12 or abs(row['remaining_kg']+row['debit_kg']-row['source_kg'])>1e-6 or row['packed_moment_error']>16*np.finfo('f4').eps:raise ValueError(row)
    receipt=json.loads((current/'step-8.receipt.json').read_text());parent=json.loads((control/'step-8.receipt.json').read_text())
    stable=[k for k in ('accepted_body_state_sha256','pack_file_sha256','base_pack_file_sha256') if receipt[k]==parent[k]]
    if len(stable)!=3:raise ValueError('Native accepted geometry changed')
    r.update(accepted_s=float(new[-1]['time_s']),end_to_end_rtf=float(new[-1]['time_s'])/r['wall_seconds'],matched_physical_columns=len(old[0]),identical_receipt_fields=stable,donors=records,trace_sha256=sha(current/'resting-coupled.csv'),candidate_ledger_sha256=sha(masslog.parent/'mass-reference.txt'))
    out.write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':check(*map(Path,sys.argv[1:]))
