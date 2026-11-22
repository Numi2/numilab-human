"""Compare the single changed Metal library at matched accepted and rejected steps."""
import json
from pathlib import Path
import sys
from material_context import sha
from summarize_release_trace import summarize
from summarize_bonded_evidence import full_run
from verify_bonded_native import matched_horizon
from compare_dense_support import observed


def extract(p,record):
    return [json.loads(s.split('=',1)[1]) for s in p.read_text().splitlines() if s.startswith(record+'=')]


def compare(root,out):
    old=root/'contact-minimum';new=root/'incremental-three';k='NUMI_MATTER_METALLIB_OVERRIDE'
    def runtime(p):
        c=json.loads((p/'launch.json').read_text());path=c['environment'].pop(k);pin=c['pins'].pop(path)
        if sha(Path(path))!=pin:raise ValueError('Metal library drift')
        return matched_horizon(c),pin
    a,lib=runtime(old);b,lib2=runtime(new)
    ids=json.loads((root/'velocity-consistent/identities.json').read_text())
    if a!=b or lib==lib2 or lib!=ids['control_metallib_sha256'] or lib2!=ids['metallib_sha256']:raise ValueError('Different physical configuration')
    result=dict(source=sha(old/'launch.json'),candidate=sha(new/'launch.json'),build=ids,run=json.loads((new/'execution.json').read_text()))
    for name,case in [('control',old),('changed',new)]:
        logs=case/'stderr.log';records=extract(logs,'fem_deformable_limit_minimum')
        status=json.loads((case/'failure.json').read_text()) if (case/'failure.json').exists() else None
        item={'minimum_contact':records,'domain_norms':extract(logs,'matter_solver_domain_norms')}
        if status:
            item.update(accepted_steps=status['segment_start'],mechanical_residual=status['matter_diagnostics'][0],material_trials=extract(logs,'fem_backtracking_trials'),
            common_rollback=status['candidate_common_coordinates']==status['accepted_common_coordinates'],respiratory_rollback=status['respiratory_candidate_status']==status['respiratory_accepted_status'])
            if records:
                detail=summarize(case);item['iterations']=detail['iterations'];item['work']=detail['work']
        else:
            receipt=json.loads((case/'terminal.receipt.json').read_text())
            item.update(accepted_steps=receipt['accepted_step'],accepted_time_s=receipt['accepted_time_s'])
        result[name]=item
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':compare(*(Path(s).resolve() for s in sys.argv[1:]))
