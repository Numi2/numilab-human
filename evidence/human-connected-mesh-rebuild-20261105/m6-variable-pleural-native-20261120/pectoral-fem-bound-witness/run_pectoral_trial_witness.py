"""Reuse existing bounded backtracking-witness capture for the most compressed pectoral sheet."""
from pathlib import Path

source = Path("run_residual_bank_probe.py").read_text()
source = source.replace("root / 'residual-bank-trial'", "root / 'pectoral-trial-witness'")
source = source.replace("    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),",
    """    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),
    NM_FEM_BACKTRACKING_TRACE_ROOT='4',
    NM_FEM_BACKTRACKING_TRACE_NEWTON='3',
    NM_FEM_BACKTRACKING_TRACE_OBJECT='5',""")
exec(compile(source, "run_residual_bank_probe.py", "exec"))
