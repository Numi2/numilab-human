from pathlib import Path

root = Path.cwd()
previous = root.parent / "resting-clearance-and-atrium-012"
builder = (previous / "build_pectoral_row_trace.py").read_text()
begin = builder.index("trace = r'''")
end = builder.index("changes = (", begin)
builder = builder[:begin] + "trace = (root / 'pectoral-energy-observer.inc').read_text()\n" + builder[end:]
builder = builder.replace("pectoral-row-trace", "pectoral-energy-trace")
builder = builder.replace("capturePectoralRow", "capturePectoralEnergy")
(root / "build_pectoral_energy_trace.py").write_text(builder)
runner = (root / "run_pectoral_load_probe.py").read_text()
runner = runner.replace("pectoral-load-trial", "pectoral-energy-trial")
runner = runner.replace("pectoral-load-trace", "pectoral-energy-trace")
(root / "run_pectoral_energy_probe.py").write_text(runner)
