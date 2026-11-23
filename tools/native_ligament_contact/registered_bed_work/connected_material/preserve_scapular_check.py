"""Recover the exact source and matched source/assembly checks without duplicating human assets."""
from pathlib import Path
import sys
from archive_cavity_trial import archive_files,sha
root=Path(__file__).resolve().parent
source,old,a=map(lambda v:Path(v).resolve(),sys.argv[1:])
f={name:root/name for name in ('preserve_scapular_check.py','verify_scapular_frame.py','test_scapular_registration.py','screen_scapular_paths.py',
                               'material-source/scapular-frame.json','material-source/scapular-site-screen.json','material-source/scapular-paths.json','material-source/off-manifold-recovery.json','shear/recovery.json')}
for name in ('rigid.json','source-bones.npz','model-source.json','model_surfaces.py','fit_source_surfaces.py'):f['source/'+name]=source/name
f['source/'+old.name]=old;f['source/'+a.name]=a
archive_files(f,'scapular-force-frame-'+sha(root/'material-source/scapular-paths.json')[:16]+'.tar.zst',root/'material-source/scapular-recovery.json',
              'Source model and archive, retained independent cortical fit, registration screens and complete wrapped tendon measurements. The parent recovery contains the native scene and landmark checks.')
