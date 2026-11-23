"""Recover the native release, its controls, and the exact registered material inputs."""
from pathlib import Path
import json,sys
from archive_cavity_trial import archive_files,sha
root=Path(__file__).resolve().parent
files={}
def add(p,key=None):
    if not p.is_file():raise ValueError(str(p))
    files[key or str(p.relative_to(root))]=p
for f in ('fit_physical_support.py','prepare_foundation.py','build_foundation.py','admit_material_contact.py',
          'build_opening_load.py','load_recruitment.py','fit_reduced_wrench.py','remap_pose_tables.py','verify_foundation.py',
          'report_loaded_coordinates.py','audit_terminal_bed.py','run_bed_work.py','trial.py','run-material-support',
          'configure_recruit.py','compare_opening_dynamics.py','compare_registered_outer.py','compare_motion_crossings.py','check_skin_load_geometry.py','capture_mapping.py','recover_closed_frames.py','closed-packed-locations.json','closed-packed-recovery.json','closed-packed-space.json','foundation.json','support-wrench.json','opening-outer.npz','opening-muscle-load.json',
          'qualification.json','decision.json','inclination.json','tilted/foundation.json','tilted/table-check.json',
          'NumiHumanFoamFoundation.metalinc','archive_foundation.py','ranked-probe.json','render_series_boundary.py','audit_series_boundary.py','recruit-source/launch.json','certify_material_release.py'):
    add(root/f)
for name in ('endpoint-recovery.json','surface-recovery.json','fit_bed_section_curve.py'):
    if (root.parent/name).is_file():add(root.parent/name,'parent/'+name)
for n in ('owner','observe','projected','ranked','tilted/owner','boundary','coupled-surface'):
    d=root/n;ids=json.loads((d/'identities.json').read_text())
    if any(sha(d/k)!=v for k,v in ids.items()):raise ValueError('changed native build '+n)
    for f in ('identities.json','build-commands.json','shader-build-commands.json','numi-human-native',
              'metalrobo/HumanRestingSkin.metallib','metalrobo/MetalRobo.metallib','NumiHumanResolvedBed.hpp',
              'NumiHumanResolvedBed.metalinc','NumiHumanStand.metal','NumiHumanFoamFoundation.metalinc',
              'NumiHumanBedForceCurves.metalinc','NumiHumanRestingVisual.hpp','human_respiration.metal','numilab_human_myosim_visual_probe.mm','NumiHumanRegisteredRecruitment.inc'):
        if (d/f).is_file():add(d/f)
for n in ('foam-17','foam-32','admitted-32','recorded-1','motion-16','recruit-motion','projected-motion','projected-92',
          'inclined-1','tilted-motion','launcher-32','ranked-probe','geometry-88','qualified-prefix','boundary-92','coupled-92','surface-92','surface-16','release-check'):
    d=root/n
    for f in ('launch.json','execution.json','terminal.receipt.json','terminal-bed-certificate.json','failure.json',
              'resting-coupled.csv','resting-surface-audit.csv','resting-com-support-impulses.csv','stderr.log','stdout.log'):
        if (d/f).is_file():add(d/f)
    for f in ('launch.json','witness.json','witness.json.opening.csv'):
        p=root/(n+'-source')/f
        if p.is_file():add(p)
for n in ('inclined-1','admitted-32'):
    add(root/n/'registered-bed-columns.csv')
for f in ('resting-local-bed-footprint.csv','resting-local-bed-sections.csv'):add(root/'inclined-1'/f)
for n in ('geometry-88','qualified-prefix','surface-92'):
    for f in ('accepted.mrvpack.zst','resting-ocular-sclera-window.json','envelope/accepted-envelope.json','envelope/physical-banks.json','envelope/target-witnesses.jsonl'):
        add(root/n/f)
for n in ('qualified-prefix','surface-92'):add(root/n/'skin-load-geometry.json')
add(root.parent/'live-source/launch.json','parent/live-source/launch.json')
add(root/'surface-92'/'registered-bed-columns.csv')
add(root/'surface-92'/'resting-local-bed-sections.csv')
add(root/'boundary-92'/'resting-surface-audit.csv')
for f in ('opening-envelope/opening.json','opening-envelope/physical-banks.json','opening-envelope/target-witnesses.jsonl','crossing-attribution.json'):add(root/'geometry-88'/f)
for f in ('physical-outer.json','physical-banks.json','target-witnesses.jsonl'):add(root/'outer-envelope'/f)
for f in ('finite-recovery.json',):add(root.parents[1]/f,'parent/'+f)
add(root/'geometry-88/initial.mrvpack.zst')
for p in (root/'legacy-gate').iterdir():add(p)
archive_files(files,'registered-foam-'+sha(root/'decision.json')[:16]+'.tar.zst',root/'recovery.json','Calibrated support, published control, completely recovered native binaries, force observer and qualified short traces. Retained supervised experiments and failed 16-second requests. Source geometry shared with verified endpoint parent.')
