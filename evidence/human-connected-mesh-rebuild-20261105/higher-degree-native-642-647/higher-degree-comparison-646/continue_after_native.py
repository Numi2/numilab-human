from pathlib import Path
import time,json,subprocess,sys
p=Path(__file__).resolve().parent;src=p.parent/'higher-degree-native-645/install-motion.execution.json';start=time.monotonic()
while not src.exists():
 if time.monotonic()-start>1200:raise RuntimeError('bounded native execution wait')
 time.sleep(2)
r=json.loads(src.read_text())
if r['exit_code'] or r['resource_stop_reason']:raise RuntimeError('Native trial failed; preserve physical/resource witness before comparison')
subprocess.run([sys.executable,str(p/'run.py')],check=True)
