"""Replay current registration and variable pressure with microstep glide refresh."""
from pathlib import Path

root = Path.cwd()
script = (root/"run_pectoral_load_probe.py").read_text()
script = script.replace("root / 'pectoral-load-trial'", "root / 'glide-refresh-trial'")
script = script.replace("root.parent / 'resting-clearance-and-atrium-012' / 'residual-bank-shader/NumiMatter.metallib'",
                        "root / 'glide-refresh-build/shader/NumiMatter.metallib'")
script = script.replace("root / 'pectoral-load-trace-runtime/libmetalrobo.dylib'",
                        "root / 'glide-refresh-build/libmetalrobo.dylib'")
script = script.replace("NM_FEM_PECTORAL_ROW_TRACE='1'", "NM_HUMAN_GLIDE_REFRESH='1'")
script = script.replace("stdout=out, stderr=err, timeout=330", "stdout=out, stderr=err, timeout=480")
exec(compile(script, "run_pectoral_load_probe.py", "exec"))
