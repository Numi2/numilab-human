"""Replay the unchanged pressure/control configuration with the selected row observer."""
import runpy
from pathlib import Path

source = Path("run_residual_bank_probe.py").read_text()
source = source.replace("root / 'residual-bank-trial'", "root / 'pectoral-row-trial'")
source = source.replace("for filename, expected in launch['pins'].items():", """
library = root / 'pectoral-row-trace-runtime/libmetalrobo.dylib'
with library.open('rb') as stream:
    launch['pins'][str(library)] = hashlib.file_digest(stream, 'sha256').hexdigest()
for filename, expected in launch['pins'].items():""")
source = source.replace("    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),",
    """    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),
    DYLD_LIBRARY_PATH=str(library.parent),
    NM_FEM_PECTORAL_ROW_TRACE='1',""")
exec(compile(source, "run_residual_bank_probe.py", "exec"))
