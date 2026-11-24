#!/bin/bash
set -euo pipefail
work=$1
base=$2
py=$3
"$py" - "$work" <<'PY'
import sys,shutil,json,platform,subprocess
from pathlib import Path
p=Path(sys.argv[1]);free=shutil.disk_usage(p).free
if free<12*2**30:raise SystemExit('Qualification disk reserve has not been reached')
(p/'native-host.json').write_text(json.dumps(dict(free_before=free,model=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),os=platform.platform(),source_launch=str(p/'source/launch.json')),indent=2)+'\n')
PY
NUMI_RETAIN_SUPPORT_GEOMETRY=1 NUMI_CAPTURE_BED_COLUMNS=1 "$py" "$base/run_bed_work.py" "$work/source" "$work/owner" "$work/motion-16" 8000 --disabled --intermediate-steps 3999
"$py" "$work/measure_native_support.py" "$work/motion-16" "$work/motion-load.json"
"$py" "$base/measure_supported_drift.py" "$work/motion-16" > "$work/motion-drift.json"
