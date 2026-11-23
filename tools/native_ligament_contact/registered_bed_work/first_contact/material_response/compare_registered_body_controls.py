"""Compare supported and full-volume native runs at the same declared mass, retaining anatomy scope."""
import csv,json,re,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def run(fast,first,joined,skin,out):
    def data(p):return json.loads(p.read_bytes())
    def csvrows(p):
        with p.open() as f:return list(csv.DictReader(f))
    a=data(fast/'launch.json');b=data(joined/'launch.json')
    def declared(d):
        args=d['argv'];return float(args[args.index('--resting-reference-mass-kg')+1])
    m=declared(a)
    if abs(declared(b)-m)>1e-8 or a['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise ValueError('Unmatched mass or pressure owner')
    ca=data(fast.parent/'drive-comparison.json');state=data(fast/'execution.json')
    if state['native_exit_code'] or not all(ca['checks'].values()):raise ValueError('Unqualified mechanism control')
    lines=(fast/'stdout.log').read_text().splitlines();end,=[line for line in lines if line.startswith('resting_integrated_body=completed ')]
    if 'physiology_body_clock=matched' not in end or 'root_assistance=false' not in end:raise ValueError('Incomplete native state')
    diag=csvrows(fast/'resting-com-momentum-diagnostic.csv');late=[row for row in diag if float(row['time_s'])>=58]
    weight=9.81*m;load=float(np.mean([float(r['normal_force_last_physical_step_n']) for r in late]));sim=ca['accepted_s']
    if len(late)<2 or abs(float(late[-1]['total_body_mass_kg'])-m)>1e-7:raise ValueError('Different support mass')
    geom=data(fast/'envelope/accepted-envelope.json');rule=data(fast/'envelope/physical-banks.json')
    if geom['pack_sha256']!=rule['pack_sha256'] or geom['accepted_time_s']!=sim:raise ValueError('Unmatched anatomical evidence')
    chart=data(skin/'registration.json');report=data(joined/'execution.json');errors=(joined/'stderr.log').read_text()
    fibers=sorted(set('/'.join(z) for z in re.findall(r'registered ligament build failed name=([^ ]+) side=([01])',errors)))
    if not report['exit'] or len(fibers)!=6:raise ValueError('Native admission result changed')
    summary=data(first/'control-measurement-regression.json')
    if abs(weight-summary['world_weight_N'])>.01:raise ValueError('Different native volume mass')
    output=dict(declared_reference_mass_kg=m,source='mixed-source reference adult',
        supported_control=dict(executable=a['pins'][a['argv'][0]],launch=digest(fast/'launch.json'),accepted_s=sim,
           wall_s=state['wall_seconds'],end_to_end_factor=sim/state['wall_seconds'],bed_weight_N=weight,late_mean_bed_load_N=load,
           bed_model='registered skin/fat sections on the articulated bed-contact points',full_skin_fat_volume=False,variable_pleural_pressure=True,cycles=ca['physiology'],
           drive_response_checks=ca['checks'],inspired_volume_change=ca['inspired_volume_fractional_change'],
           endpoint_skin_self_pairs=geom['skin_self_intersections'],endpoint_physical_nonocular_pairs=rule['physical_nonocular_pairs'],endpoint=digest(fast/'envelope/accepted-envelope.json'),control_comparison=digest(fast.parent/'drive-comparison.json')),
        current_registered_material=dict(launch=digest(joined/'launch.json'),native_build=b['pins'][b['argv'][0]],shader=b['pins'][b['environment']['NUMI_MATTER_METALLIB_OVERRIDE']],material_receipt=b['material_receipt'],skin_pose_registration=digest(skin/'registration.json'),first_bed_nodes=chart['first_contact_new'],rejected_fibers=fibers,accepted_seconds=0.0),
        previous_full_volume_control=dict(accepted_s=summary['accepted_seconds'],last_bed_load_N=summary['last_bed_reaction_N'],weight_N=summary['world_weight_N'],body_factor=summary['native_body_window_factor'],evidence=digest(first/'control-measurement-regression.json')))
    out.write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
