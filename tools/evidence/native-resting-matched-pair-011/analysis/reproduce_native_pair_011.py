#!/usr/bin/env python3
"""Launch the retained integrated native scene on the SSH Mac Mini.

Reuses the exact existing owner preparation and launcher. Each invocation has
fresh outputs, input pin checks, a single-owner lock, and the recorded protocol.
No simulation state or alternate physics is implemented here.
"""
import argparse,hashlib,subprocess,tempfile,json
from pathlib import Path
SOURCE=Path('/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011')
EXPECTED={'prepare.py': '413f3b21cf162743e76304efe0a72aa4d4f31a0934f566e9105a8cf2ca9913aa', 'launch_arm.py': '0a0d9cefb55c9543440667328d0530c9accf029baffb2758e3712e03a3fdeea8'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',choices=['baseline','intervention'],default='baseline')
    ap.add_argument('--prepare-only',action='store_true',
                    help='Validate and prepare fresh declarations without running Metal.')
    args=ap.parse_args()
    for name,digest in EXPECTED.items():
        p=SOURCE/name
        if not p.is_file() or sha(p)!=digest:
            raise SystemExit('Pinned Mini owner input missing or changed: '+str(p))
    out=Path(tempfile.mkdtemp(prefix='native-coupled-pair-reproduction-',dir=SOURCE.parent))
    for name in EXPECTED:
        data=(SOURCE/name).read_bytes()
        with (out/name).open('xb') as f:f.write(data)
        if sha(out/name)!=EXPECTED[name]:raise SystemExit('Copy verification failed')
    # Preparation performs the source/runtime/asset/hash and free-space checks.
    result=subprocess.run(['/usr/bin/python3',str(out/'prepare.py')],cwd=out)
    if result.returncode:return result.returncode
    command=['/usr/bin/python3',str(out/'launch_arm.py'),args.arm]
    print(json.dumps({'prepared_directory':str(out),'launch_argv':command,
                      'launch_requested':not args.prepare_only}),flush=True)
    if args.prepare_only:return 0
    return subprocess.run(command,cwd=out).returncode
if __name__=='__main__':raise SystemExit(main())
