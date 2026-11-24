"""Read the compiled residual-witness tag and resolve the first two registered ACL node ranges."""
import json
import re
from pathlib import Path
import sys


def explain(run):
    stdout=(run/'stdout.log').read_text()
    stderr=(run/'stderr.log').read_text()
    acl=[(int(side),int(nodes)) for side,nodes in re.findall(r'^resting_inferred_acl=admitted side=(\d+).*? nodes=(\d+) ',stdout,re.M)]
    offset=re.search(r'^resting_fascia_object=compiled_reference_mode object=0 state_offset=(\d+) ',stdout,re.M)
    if acl != [(0,16),(1,16)] or not offset or int(offset[1])!=32:
        raise ValueError('Registration differs from the paired 16-node ACL cohort')
    m=re.search(r'^human_joint_failure completed_steps=(\d+) matter_status=(\d+).*?matter_object_index=(\d+) matter_diagnostics=(\[[^\n\]]+\])',stderr,re.M)
    if not m:
        execution=json.loads((run/'execution.json').read_bytes())
        return dict(completed=execution['exit']==0,execution=execution)
    norms=json.loads(m[4])
    if int(m[2])!=10 or int(norms[3])!=norms[3] or norms[3]>-(1<<20):
        return dict(completed=False,status=int(m[2]),other_diagnostics=norms)
    tag=-int(norms[3]);bank=tag>>20;row=(tag & ((1<<20)-1))-1
    if not 1<=bank<=5 or row<0:raise ValueError('Malformed residual witness')
    result=dict(accepted_steps=int(m[1]),reporting_object=int(m[3]),residual_bank=['FEM momentum','FEM pressure','transport','rigid momentum','support'][bank-1],
                peak_global_row=row,full_relative_residual=norms[0],bank_relative_residual=norms[2])
    if bank==1 and row<32:
        local=row%16
        result.update(acl_side=row//16,local_node=local,axial_ring=local//4,interior=4<=local<12)
    return result

if __name__=='__main__':
    print(json.dumps({p:explain(Path(p)) for p in sys.argv[1:]},indent=2))
