"""Cross-check the native support trace, unchanged accepted output and load path."""
import csv,hashlib,json,mmap,runpy,struct,sys
from pathlib import Path
import numpy as np
from build_bed_audit import sha

def run(control,case,vol,audit,comparison,out):
    load=lambda p:json.loads(p.read_bytes()); parse=lambda p:list(csv.DictReader(p.open()))
    if load(case/'execution.json')['exit']!=0 or load(control/'execution.json')['exit']!=0: raise ValueError('Native run did not finish')
    cfg=load(case/'launch.json'); support=Path(cfg['argv'][cfg['argv'].index('--support-contact-payload')+1])
    if cfg['pins'][str(support)]!=sha(support) or cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise ValueError('Inconsistent scene configuration')
    h,n=np.array(struct.unpack_from('<6f',support.read_bytes(),56)).reshape(2,3);n=n/np.linalg.norm(n)
    walls=np.unique(np.load(vol/'material-walls.npz')['O']);points=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3))
    initial=points(vol/'current.node')[walls];accepted=points(case/'physical-accepted.node')[walls]
    # The GPU presents Float32, even when the offline coordinate file carries doubles.
    initial=initial.astype('f4').astype(float);accepted=accepted.astype('f4').astype(float)
    reference_gaps=[float(np.min((x-h)@n)) for x in (initial,accepted,accepted)]
    trace=Path('native-traces/resting-surface-audit.csv');prev=parse(control/trace);current=parse(case/trace)
    if len(prev)!=len(current) or len(current)!=3:raise ValueError('Unexpected capture cadence')
    for a,b in zip(prev,current):
        if set(a)!=set(b) or any(a[k]!=b[k] for k in a if k not in ('min_skin_bed_gap_m','vertices_below_1mm')):raise ValueError('Unrelated surface observation changed')
    native_gaps=[float(r['min_skin_bed_gap_m'])for r in current]
    if max(abs(np.array(native_gaps)-reference_gaps))>2e-9:raise ValueError(f'Incorrect native gaps: {native_gaps}, expected {reference_gaps}')
    fixed={}
    for name in ('physical-accepted.node','physical-reference.node','native-traces/resting-coupled.csv','native-traces/resting-com-support-impulses.csv','native-traces/resting-com-momentum-diagnostic.csv'):
        fixed[name]=sha(case/name)
        if fixed[name]!=sha(control/name):raise ValueError('Observer changed physics: '+name)
    receipt=Path('native-traces/accepted-geometry/step-1.receipt.json');api=runpy.run_path(str(audit));sections=[];meta=[]
    for f in (control,case):
        r=load(f/receipt);pack=f/receipt.parent/'step-1.mrvpack'
        if sha(pack)!=r['pack_file_sha256']:raise ValueError('Stale accepted pack')
        meta.append(dict(receipt=sha(f/receipt),pack=sha(pack),launch=sha(f/'launch.json'),time_s=r['accepted_time_s'],wall_s=load(f/'execution.json')['wall_s']))
        bp={p['body_index']:p for p in r['accepted_body_poses']};digest=hashlib.sha256();cnt=0
        with pack.open('rb') as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)as memory:
            for (semantic,stable),_,xyz,tri in api['primitives'](memory,bp):
                digest.update(struct.pack('<II',semantic,stable));digest.update(np.ascontiguousarray(tri));digest.update(np.ascontiguousarray(xyz));cnt+=1
        sections.append((cnt,digest.hexdigest()))
    if sections[0]!=sections[1]:raise ValueError('Accepted anatomical surfaces changed')
    run=dict(control=meta[0],candidate=meta[1],anatomical_banks=sections[1][0],accepted_anatomical_geometry=sections[1][1],physical_hashes=fixed,matched_vertices=len(walls),support_payload=sha(support),old_m=[float(r['min_skin_bed_gap_m'])for r in prev],observer_m=native_gaps,independent_m=reference_gaps,max_gap_error_m=max(abs(np.array(native_gaps)-reference_gaps)),initial_false_gap_m=load(comparison)['trace_reported_gap_m'],observed_coupled_fields=len(parse(case/'native-traces/resting-coupled.csv')[0]))
    out.write_text(json.dumps(run,indent=2)+'\n');print(native_gaps,reference_gaps,sections[-1],flush=True)
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:]))
