"""Compare the late native experiment with the independent sixty-second control."""
import csv,gzip,json,sys
from pathlib import Path
from material_context import sha
def run(trial,control,training,out):
    out.mkdir(exist_ok=True)
    def table(folder,name):
        with (folder/name).open(newline='') as f:return list(csv.DictReader(f))
    src='resting-coupled.csv';aa=table(trial,src);bb=table(control,src)
    if len(aa)!=len(bb) or aa[0].keys()!=bb[0].keys():raise ValueError('Incomplete 60 second comparison')
    cols=[k for k in aa[0] if all(a[k]==b[k] for a,b in zip(aa,bb,strict=True))]
    grid=json.loads((trial/'glutes/motion.json').read_text());checked=grid['pose_samples']
    if not grid['recorded_pose_replay_qualified'] or not checked or checked[-1]['step']!=30000:raise ValueError('Incomplete native geometry')
    late=[r for r in checked if r['step']>8000];strain=json.loads((trial/'glutes/passive-strain.json').read_text());phys=json.loads((trial/'physiology.json').read_text())
    # Retain exactly the donor rows used by geometric replay with one header.
    pose_trace=trial/'resting-body-motion-partition.csv';subset=out/'pelvis-and-femur-motion.csv'
    with pose_trace.open(newline='') as f,subset.open('w',newline='') as g:
        a=csv.DictReader(f);writer=csv.DictWriter(g,fieldnames=a.fieldnames,lineterminator='\n');writer.writeheader();count=0
        for row in a:
            if int(row['body_index']) in (128,131,145):writer.writerow(row);count+=1
    if count!=3*len(checked):raise ValueError('Incomplete donor trace')
    summary=dict(accepted_time_s=grid['accepted_time_s'],accepted_roots=phys['final_accepted_step'],wall_seconds=phys['wall_seconds'],end_to_end_rtf=phys['launch_to_exit_rtf'],motion_rtf=phys['viewer_inclusive_motion_rtf'],presentation_callbacks=phys['presentation_callbacks'],wall_fps=phys['presentations_per_wall_second'],pose_samples=len(checked),frames_after_fit_window=len(late),all_sixty_second_matched_columns=len(cols),sixty_second_trace_rows=len(aa),differing_columns=sorted(set(aa[0])-set(cols)),maximum_late_gluteal_crossings=max(r['bilateral']+r['self_right']+r['self_left'] for r in late),minimum_late_normalized_determinant=min(r['minimum_normalized_determinant'] for r in late),minimum_late_relative_volume=min(r['minimum_relative_to_registered_entrance'] for r in late),passive_specimen_cell_sample_failures=sum(v['cell_sample_failures'] for v in strain['sides'].values()),source_control_launch=sha(control/'launch.json'),training_trace=sha(training/'resting-body-motion-partition.csv'),full_accepted_rigid_trace=sha(pose_trace),compact_donor_trace=sha(subset),terminal_pack=grid['accepted_pack_sha256'],terminal_receipt=grid['receipt_sha256'])
    (out/'qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
    if len(cols)!=77 or summary['maximum_late_gluteal_crossings'] or summary['minimum_late_normalized_determinant']<1e-5 or summary['passive_specimen_cell_sample_failures']:raise ValueError('Late motion outside declared gates')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
