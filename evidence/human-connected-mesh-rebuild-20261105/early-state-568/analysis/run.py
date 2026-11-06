from pathlib import Path
import subprocess,sys,json
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');H=Path('/Users/numi/Work/human-checkpoint-owner-20261105');O=Path(__file__).resolve().parent
candidate=W/'early-state-native-572';control=W/'domain-mesh-fidelity-pair-565/control'
commands=[('acceptance-work',[str(H/'tools/compare_human_native_mesh_trials.py'),'--candidate-dir',str(candidate),'--control-dir',str(control),'--output',str(O/'native-comparison.json')]),('matched-endpoint',[str(W/'matched-endpoint-pair-509/compare_endpoint_channels.py'),str(control/'install-motion.stdout.log.gz'),str(candidate/'install-motion.stdout.log.gz'),str(O/'matched-endpoint-channels.json')])]
for name,argv in commands:
 with (O/(name+'.log')).open('w') as f:r=subprocess.run([sys.executable,*argv],stdout=f,stderr=subprocess.STDOUT)
 print(name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
r=json.loads((O/'matched-endpoint-channels.json').read_text());strain=[dict(region=k,metric=n,value=v) for k,row in r['regional_strain'].items() for n,v in row['relative_change'].items() if v is None or v>.05]
force=[dict(region=k,channel=c,metric=n,value=row[n]) for k,x in r['regional_force_channels'].items() for c,row in x.items() for n in ['relative_resultant_difference_over_reference_nodal_l1','relative_nodal_l1_change'] if row[n] is None or row[n]>.01]
phys=[dict(channel=k,**row) for k,row in r['coupled_trace'].items() if row['relative_max_abs_change'] is None or row['relative_max_abs_change']>.01]
summary=dict(candidate=568,comparison_control=495,matched_roots=4,force_failures=force,strain_failures=strain,strict_relative_physiology_outliers=phys,body_wall_reduction_fraction=r['body_wall_reduction_fraction'],body_throughput_ratio=r['body_throughput_ratio'],physical_fidelity_qualification=False,limitations=['Short release-initialization only; no motion/breath qualification.','Canonical pointwise strain/material energy/full force owner coverage still needed.'])
(O/'gate-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(force_failures=len(force),strain_failures=len(strain),physiology_outliers=len(phys),body_throughput_ratio=r['body_throughput_ratio'])),flush=True)
