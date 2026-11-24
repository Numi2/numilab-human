"""Retain the matched native support experiment and its exact negative anatomy result."""
import json,os,subprocess
from pathlib import Path
from archive_cavity_trial import archive_files,sha

def run(work):
    files={}; runs=['preflight-32','motion-16']
    for case in runs:
        for p in (work/case).rglob('*'):
            if p.is_file() and not p.name.endswith('.github-recovery.json'):
                files[str(p.relative_to(work))]=p
    for name in ['evaluation.md','audit_pelvic_wrench.py','build_pelvic_sections.py','NumiHumanPelvicBedSections.hpp','measure_native_support.py','native-host.json','run_pelvic_motion.sh','baseline-load.json','preflight-load.json','motion-load.json','wrench.json','motion-drift.json']:
        files[name]=work/name
    owner=work/'owner'
    for n in ('identities.json','build-commands.json','numi-human-native','NumiHumanForearmBedSections.hpp','NumiHumanPelvicBedSections.hpp','NumiHumanRestingSupportGeometry.hpp','NumiHumanRestingVisual.hpp','numilab_human_myosim_visual_probe.mm'):
        files['owner/'+n]=owner/n
    record=work/'native-support-recovery.json'
    if not record.exists():
        archive_files(files,'pelvic-native-support-'+sha(work/'motion-load.json')[:18]+'.tar.zst',record,
                      'Physical reduced support experiment in the complete coupled viewer. Includes exact samples, crossings, binary and modified sources. Parent includes the other source/build inputs.')
    rc=json.loads(record.read_bytes())
    # Retire only our already-recovered compressed output copies, excluding any
    # executable, source or input. Keep a locator at each original output path.
    for case in runs:
        for p in (work/case).glob('*.mrvpack.zst'):
            key=str(p.relative_to(work));h=sha(p);s=p.lstat()
            if rc['verified_member_sha256'][key]!=h or s.st_nlink!=1 or subprocess.run(['/usr/sbin/lsof','-t',str(p)],capture_output=True).returncode!=1:continue
            with Path(str(p)+'.github-recovery.json').open('w') as f:
                json.dump(dict(url=rc['url'],archive_sha256=rc['archive_sha256'],member=key,sha256=h),f);f.write('\n');f.flush();os.fsync(f.fileno())
            if p.lstat().st_ctime_ns!=s.st_ctime_ns:raise ValueError('Artifact edited during recovery')
            p.unlink()
if __name__=='__main__':run(Path.cwd())
