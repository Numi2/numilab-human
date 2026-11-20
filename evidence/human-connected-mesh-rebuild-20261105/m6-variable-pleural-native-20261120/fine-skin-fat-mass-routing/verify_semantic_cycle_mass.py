"""Apply the native relative quota check in the independent geometry audit."""
import os
from pathlib import Path
import sys

output = Path.cwd()
previous = output.parent / 'resting-clearance-and-atrium-011'
source = (previous/'verify_mapped_nodal_mass.py').read_text()
sys.path.insert(0, str(previous))
sys.argv[1:] = ['--paired-reference']
os.chdir(previous)
changes = (
    ("p = Path(f'{mass_name}{suffix}.npz')",
     f"p = Path({str(output/'fine-current-paired-semantic-cycle-mass.npz')!r})"),
    ("target = Path(f'{verification_name}{suffix}.json')",
     f"target = Path({str(output/'fine-semantic-cycle-mass-verification.json')!r})"),
    ("np.maximum(1e-13,1e-10*expected_carrier)",
     "1e-10*expected_carrier"),
    ("'source_carrier_max_mass_error_kg':",
     "'source_carrier_max_relative_error':"
     "float((np.abs(actual_carrier-expected_carrier)/expected_carrier).max()),\n"
     "    'source_carrier_max_mass_error_kg':"),
)
for before, after in changes:
    assert source.count(before) == 1
    source = source.replace(before, after)
exec(compile(source, str(previous/'verify_mapped_nodal_mass.py'), 'exec'))
