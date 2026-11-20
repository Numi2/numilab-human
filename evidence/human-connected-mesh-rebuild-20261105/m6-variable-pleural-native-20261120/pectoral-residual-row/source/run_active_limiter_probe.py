"""Replay the unchanged pressure/control configuration with the selected row observer."""
import runpy
from pathlib import Path

source = Path("run_residual_bank_probe.py").read_text()
source = source.replace("root / 'residual-bank-trial'", "root / 'active-limiter-trial'")
source = source.replace("for filename, expected in launch['pins'].items():", """
library = root / 'active-limiter-trace-runtime/libmetalrobo.dylib'
with library.open('rb') as stream:
    launch['pins'][str(library)] = hashlib.file_digest(stream, 'sha256').hexdigest()
for filename, expected in launch['pins'].items():""")
source = source.replace("    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),",
    """    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),
    DYLD_LIBRARY_PATH=str(library.parent),
    NM_FEM_NEWTON_TRACE_ROOT='4',""")
exec(compile(source, "run_residual_bank_probe.py", "exec"))
