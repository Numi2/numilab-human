"""Compare the changed material and its matched control at their actual committed clock."""
import csv,json,re,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from check_full_motion import run as audit
from measure_supported_rest import measure

def load(p):return json.loads(p.read_bytes())
def write(p,d):p.write_text(json.dumps(d,allow_nan=False,indent=2)+'\n')
def check(w,case):
    reg=w/'native-binding';v=w/'material-import/volume';base=w.parent/'posterior-interface-082/native-pair'
    stage=case/'audit';vv=stage/'witness-material/volume';vv.mkdir(parents=True,exist_ok=True)
    spec=load(reg/'build.json');ids=[spec['new_source_node']-1]
    for s in ['mesh.ele','mesh.node','current.node','material-walls.npz','source-overlay.npz','node-mass.f64']:
        f=vv/s
        if not f.exists():f.symlink_to(v/s)
        if f.resolve()!=(v/s).resolve():raise ValueError('Invalid volume')
    write(vv/'result.json',dict(updated_sites=[dict(vertex=i)for i in ids],source=digest(v/'result.json')))
    if spec['binary']!=digest(reg/'numi-human-native') or spec['library']!=digest(reg/'libmetalrobo.dylib') or load(case/'launch.json')['material_cavity']!=digest(reg/'build.json'):raise ValueError('Unmatched native executable or cavity')
    if not any(str(reg/'libmetalrobo.dylib') in line and line.startswith('dyld[')for line in (case/'stderr.log').read_text().splitlines()):raise ValueError('Changed physical library not loaded')
    a=case/'material-check.json';audit(case,stage,a);b=case/'support-check.json';measure(case,v,b)
    def endpoint(c):
        rows=list(csv.DictReader((c/'native-traces/resting-coupled.csv').open()));work=[json.loads(s.split('=',1)[1])for s in (c/'stderr.log').read_text().splitlines()if s.startswith('matter_fgmres_active_work=')]
        out=(c/'stdout.log').read_text().splitlines();term=next((s for s in out if s.startswith('resting_integrated_body=completed')),None)
        return dict(material=load(c/'material-check.json'),support=load(c/'support-check.json'),total=load(c/'execution.json'),active_columns=[q['used_columns_sum'] for q in work],terminal=term,frames=[{k:float(r[k]) for k in ['step','time_s','pleural_pa','lung_volume_ml','PaO2_mmhg','PaCO2_mmhg','root_assistance_n']} for r in rows])
    one=endpoint(base);two=endpoint(case)
    if len(two['frames'])==0:raise ValueError('No committed new material')
    same=len(one['frames'])==len(two['frames']) and all(a['time_s']==b['time_s'] for a,b in zip(one['frames'],two['frames']))
    compare={'matched_accepted_times':same,'old':one,'new':two}
    if same:
        old=np.loadtxt(base/'physical-accepted.node',skiprows=1,usecols=(1,2,3));new=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
        if len(new)!=len(old)+1:raise ValueError('Different body')
        dx=np.linalg.norm(new[:-1]-old,axis=1);compare.update(existing_material_node_max_delta_m=float(dx.max()),existing_material_node_rms_delta_m=float(np.sqrt(np.mean(dx**2))))
    write(case/'matched.json',compare);print('accepted',two['frames'][-1]['time_s'],'shape failures',[a['below_shape']for a in two['material']['roles'].values()],'support N',two['support']['accepted_normal_force_N'],flush=True)
if __name__=='__main__':check(Path.cwd(),Path(sys.argv[1]).resolve())
