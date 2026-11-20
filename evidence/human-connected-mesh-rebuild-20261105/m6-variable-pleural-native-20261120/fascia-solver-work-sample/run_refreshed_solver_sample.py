"""Run the retained eight-step gravity continuation with the failure witness."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.cwd()
previous = root.parent / 'resting-clearance-and-atrium-011'
control = previous / 'first-touch-gravity-continuation8-trial'
run = root / 'glide-refresh-solver-work-sample'
run.mkdir()
launch = json.loads((control / 'launch.json').read_text())
shader = root / 'glide-refresh-build/shader/NumiMatter.metallib'
with shader.open('rb') as stream:
    launch['pins'][str(shader)] = hashlib.file_digest(stream, 'sha256').hexdigest()
library = root / 'glide-refresh-build/libmetalrobo.dylib'
with library.open('rb') as stream:
    launch['pins'][str(library)] = hashlib.file_digest(stream, 'sha256').hexdigest()
for filename, expected in launch['pins'].items():
    with open(filename, 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, filename

launch['argv'][4] = str(run / 'native')
launch['environment'].update(
    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),
    DYLD_LIBRARY_PATH=str(library.parent),
    NM_HUMAN_GLIDE_REFRESH='1',
    NM_FGMRES_SKIP_COMPLETED_CYCLE_WORK='1',
    NM_FGMRES_SKIP_COMPLETED_LINEAR_BUILDERS='1',
    NM_FGMRES_SKIP_EMPTY_FIELD_PRECONDITIONER='1',
    NM_FEM_DISTRIBUTED_GLOBAL_MEASUREMENTS='1',
    NM_FGMRES_DISTRIBUTED_CGS2='1',
    NM_FGMRES_CGS2_ICB='1',
    NM_FGMRES_OPERATOR_ICB='1',
    NM_FGMRES_PREFIX_PRECONDITIONER_ICB='1',
    NM_FEM_DEFORMABLE_COLLISION_GAP='1',
    NUMI_HUMAN_COUPLED_TRACE_CSV=str(run / 'coupled.csv'),
    NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(run / 'failure.json'),
)
(run / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
env = {key: value for key, value in os.environ.items()
       if not key.startswith(('NUMI_', 'NM_'))}
env.update(launch['environment'])
started = time.monotonic()
reason = 'native_process_exit'
with (run / 'stdout.log').open('wb') as out, (run / 'stderr.log').open('wb') as err:
    process = subprocess.Popen(launch['argv'], cwd=launch['cwd'], env=env, stdout=out, stderr=err)
    try:
        while process.poll() is None:
            log = (run / 'stdout.log').read_text()
            if 'resting_native_profile step=2 command_ms=' in log:
                reason = 'completed_two_accepted_samples'
                break
            if time.monotonic() - started > 160.0:
                reason = 'sample_time_budget_expired'
                break
            time.sleep(0.25)
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
summary = {'exit': process.returncode, 'stop_reason': reason,
           'wall_seconds': time.monotonic()-started}
(run / 'execution.json').write_text(json.dumps(summary, indent=2) + '\n')
trial_trace = run / 'native/resting-coupled.csv'
control_trace = control / 'native/resting-coupled.csv'
if (run / 'failure.json').is_file():
    failure = json.loads((run / 'failure.json').read_text())
    control_failure = json.loads((control / 'failure.json').read_text())
    candidate = failure['matter_diagnostics']
    tag = int(-candidate[3])
    bank, row = tag >> 20, (tag & ((1 << 20) - 1)) - 1
    summary.update(
        accepted_csv_byte_identity=trial_trace.read_bytes() == control_trace.read_bytes(),
        same_rejection=all(failure[key] == control_failure[key] for key in
                           ('segment_start','matter_code','matter_microsteps',
                            'matter_failing_index','matter_environment')),
        same_equilibrium_norm_and_correction=candidate[:2] == control_failure['matter_diagnostics'][:2],
        equilibrium_residual=candidate[0],
        relative_correction=candidate[1],
        accepted_time_us=failure['brain_input']['source_time_us'],
        rejected_target_time_us=failure['brain_candidate']['timestamp_us'],
    )
    if 1 <= bank <= 5 and row >= 0:
        summary.update(
            failure_bank=('fem_velocity','fem_pressure','transport','rigid','support')[bank-1],
            bank_relative_residual=candidate[2],
            fraction_squared_residual=(candidate[2] / candidate[0])**2,
            largest_bank_row=row,
        )
(run / 'residual-bank.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary), flush=True)
