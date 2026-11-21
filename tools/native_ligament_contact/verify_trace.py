"""Check the native coupled trace for consecutive accepted steps and matched clocks."""
import csv
import math
from pathlib import Path

def verify_trace(path: Path, expected: int, dt: float) -> None:
    if expected < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid native step request')
    seen = 0
    with path.open(newline='') as stream:
        for seen, row in enumerate(csv.DictReader(stream), 1):
            if int(row['step']) != seen or not math.isclose(float(row['time_s']), seen * dt, rel_tol=5e-6):
                raise ValueError(f'Native accepted step or clock mismatch at row {seen}')
            if any(not math.isfinite(float(value)) for value in row.values()):
                raise ValueError(f'Nonfinite coupled trace at step {seen}')
            if float(row['root_assistance_n']) or float(row['root_assistance_nm']):
                raise ValueError(f'Unexpected external assistance at step {seen}')
    if seen != expected:
        raise ValueError(f'Expected {expected} coupled steps, found {seen}')
