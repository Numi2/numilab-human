import json,os,sys,subprocess
from pathlib import Path
from evaluate_registered_bed_transfer import digest
config,closure,owner,selection,output=map(lambda v:Path(v).resolve(),sys.argv[1:])
j=json.loads(config.read_text());library=closure/'libmetalrobo.dylib';binary=owner/'numi-human-native'
bi=json.loads((owner/'identities.json').read_text());assert digest(library)==bi['lib_sha256'] and digest(binary)==bi['binary_sha256']
j['environment']['DYLD_LIBRARY_PATH']=str(closure);j['pins'][str(library)]=bi['lib_sha256'];j['pins'][str(binary)]=bi['binary_sha256']
j['environment']['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=os.environ.get('NUMI_SELECTED_TRIAL_ROOTS','1')
derived=config.parent/('qualified-input-'+j['environment']['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']+'.json')
derived.write_text(json.dumps(j,indent=2)+'\n')
subprocess.run([sys.executable,str(Path(__file__).with_name('run_seated_volume_from_disk.py')),str(derived),str(selection),str(binary),str(output)],check=True)
