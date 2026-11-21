"""Close gravity load and moment at one accepted step from both disjoint mass owners."""
import csv,hashlib,json,re,sys
from pathlib import Path
import numpy as np
def run(directory,out):
    parity=json.loads((directory/'physical-parity.json').read_text())
    if parity['exit_code'] or parity['changed_columns']:raise ValueError('A different physical trajectory')
    log=(directory/'stdout.log').read_text();m=re.search(r'^resting_full_fem_mass_moment nodes=(\d+) mass_kg=([^ ]+) initial_kg_m=([^ ]+) accepted_kg_m=([^ ]+) accepted_step=(\d+)$',log,re.M)
    if not m:raise ValueError('Missing accepted mass moment')
    node_count,fem_mass,initial,terminal,step=m.groups();step=int(step);fem_mass=float(fem_mass)
    bodies=list(csv.DictReader((directory/'resting-body-motion-partition.csv').open()))
    rows=[r for r in bodies if int(r['accepted_step'])==step];ids=[int(r['body_index']) for r in rows]
    if sorted(ids)!=list(range(len(ids))) or len(ids)!=157:raise ValueError('Incomplete rigid partition')
    mass=np.array([float(r['mass_kg']) for r in rows])
    root=np.array([[float(r['root_'+k+'_m']) for k in 'xyz'] for r in rows])
    rel=np.array([[float(r['com_relative_root_'+k+'_m']) for k in 'xyz'] for r in rows])
    rigid_raw=(mass[:,None]*(root+rel)).sum(0)
    alt=mass@root + np.sum([[float(r['mass_weighted_relative_com_'+k+'_kg_m']) for k in 'xyz'] for r in rows],axis=0)
    if not np.allclose(rigid_raw,alt,rtol=0,atol=1e-10):raise ValueError('Native rigid moment channels disagree')
    tissue=np.fromstring(terminal,sep=',');total=mass.sum()+fem_mass
    com=(tissue+rigid_raw)/total
    def parse_line(k):
        return next(row for row in log.splitlines() if row.startswith(k))
    expected=float(re.search(r'FEM_mass_kg=([^ ]+)',parse_line('resting_tissue_shared_runtime=bound'))[1])
    if abs(expected-fem_mass)>1e-10:raise ValueError('FEM mass ownership mismatch')
    result=dict(accepted_step=step,accepted_time_s=float(rows[0]['time_s']),fem_nodes=int(node_count),bodies=len(rows),rigid_mass_kg=mass.sum(),fem_mass_kg=fem_mass,total_mass_kg=total,weight_n=total*9.81,center_xy_m=com[:2].tolist(),center_m=com.tolist(),rigid_first_kg_m=rigid_raw.tolist(),fem_first_kg_m=tissue.tolist(),fem_initial_first_kg_m=np.fromstring(initial,sep=',').tolist(),sources={p:hashlib.sha256((directory/p).read_bytes()).hexdigest() for p in ('launch.json','stdout.log','resting-body-motion-partition.csv','physical-parity.json')})
    out.write_text(json.dumps(result,indent=2)+'\n');print(total,com)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
