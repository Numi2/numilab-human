"""Preserve executable owners and compact diagnostics of each native globalization trial."""
from pathlib import Path
import sys
from archive_cavity_trial import archive_files,sha

def archive(root):
    case=root/'material-direction';members={}
    def add(p):members[str(p.relative_to(root))]=p
    for p in [case/n for n in ('build.py','direction.metalinc','test_kernel.mm','test_direction.py','test-kernel','test.log','compare.py','comparison.json','archive.py','owner/numi-human-native','owner/identities.json','build/receipt.json','build/libmetalrobo.dylib','build/NumiMatter.metallib',
        'quantized/compile.py','quantized/publish.inc','quantized/check.json','quantized/build/receipt.json','quantized/build/NumiMatter.metallib',
        'release/search-check.json','release/check.json','release/build.py','release/kernel.metalinc','release/prepare.inc','release/test.mm','release/test-gpu','release/verify.py','release/result.json','release/owner/identities.json','release/owner/numi-human-native','release/build/receipt.json','release/build/NumiMatter.metallib','release/build/libmetalrobo.dylib','priority-recovery.json','qualify.sh','coupled/build.py','coupled/private.json','coupled/shared.json','coupled/build/receipt.json','coupled/build/NumiMatter.metallib','dynamic/hints.metalinc','dynamic/build.py','dynamic/test_hints.mm','dynamic/test-hints','dynamic/hints.log','dynamic/check.json','dynamic/owner/identities.json','dynamic/owner/numi-human-native','dynamic/build/libmetalrobo.dylib','dynamic/build/NumiMatter.metallib','dynamic/build/receipt.json')]:add(p)
    for d in (case,case/'quantized',case/'dynamic',case/'coupled'):
        add(d/'parent/launch.json')
        for p in (d/'three').rglob('*'):
            if p.name in ('stdout.log','stderr.log','launch.json','failure.json','admission.json','execution.json','terminal.receipt.json','resting-coupled.csv'):
                add(p)
    for mode in (1,2):
        add(case/f'release/parent-{mode}/launch.json')
        for p in (case/f'release/three-{mode}').rglob('*'):
            if p.name in ('stdout.log','stderr.log','launch.json','failure.json','admission.json','execution.json','terminal.receipt.json','resting-coupled.csv'):
                add(p)
    for n in ('fold-repair/recovery.json','fold-repair/endpoint-recovery.json'):add(root/n)
    archive_files(members,'material-volume-search-'+sha(case/'comparison.json')[:16]+'.tar.zst',case/'recovery.json',
        'Recoverable native solver direction experiments with unchanged physical acceptance, source and compiled owners, matched integrated release logs and Metal tests. The parent receipts recover the registered body and physical frame.')
if __name__=='__main__':archive(Path(sys.argv[1]).resolve())
