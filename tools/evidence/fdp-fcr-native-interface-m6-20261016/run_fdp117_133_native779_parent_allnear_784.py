from pathlib import Path
import hashlib,json,subprocess,sys
D=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378')
H=Path('/Users/n/numi-human-anatomy-increment-publish-20261016')
P=D/'fdp761-fcr117-source778-native-emission-779/native-run/accepted-geometry/step-0.mrvpack'
R=D/'fdp761-fcr117-native777-flip-source-778/resting-anatomy-receipt.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=json.loads(R.read_text());T=Path(receipt['provenance']['native_muscle_surfaces']['payload_path'])
args=[sys.executable,str(H/'tools/verify_resting_anatomy.py'),'--pack',str(P),'--pack-sha256',sha(P),'--capture-receipt',str(P.with_suffix('.receipt.json')),'--receipt-sha256',sha(P.with_suffix('.receipt.json')),'--shared-partition','51005:117','51005:133',str(T),sha(T),'--parent-bindings-receipt',str(R)+'='+sha(R)]
for sid in (117,133):
 args.extend(['--changed',f'51005:{sid}','--required-closed',f'51005:{sid}'])
for source in [Path(__file__),H/'src/numilab_human/resting_shared_partition_verify.py',H/'src/numilab_human/resting_anatomy_verify.py',D/'fdp761-fcr117-native779-owner-and-interface-proof-780/report.json',D/'fdp133-fcr117-actual770-paired-local-retessellation-777/report.json']:
 args.extend(['--source-pin',str(source)+'='+sha(source)])
if sys.argv[1:]==['--preflight']:
 args+=['--preflight-only','--output',str(D/'fdp117-133-native779-full-parent-allnear-784-preflight.json')]
else:
 assert sys.argv[1:]==['--execute']
 args+=['--output',str(D/'fdp117-133-native779-full-parent-allnear-784.json')]
raise SystemExit(subprocess.run(args).returncode)
