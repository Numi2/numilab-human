from pathlib import Path
import os
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path: Path):
    build = tmp_path / 'build with spaces'
    probe = build / 'bin/metalrobo_numilab_human_myosim_visual_probe'
    probe.parent.mkdir(parents=True)
    args = tmp_path / 'arguments.txt'
    probe.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$NUMI_PROBE_ARGS"\n')
    probe.chmod(0o755)
    support = tmp_path / 'support.nhcnt'
    bone = tmp_path / 'not-required-in-mechanics.nhbones'
    output = tmp_path / 'output with spaces'
    def invoke(*options, magic=b'NHCNT1'):
        args.unlink(missing_ok=True)
        support.write_bytes(magic)
        result = subprocess.run([
            str(ROOT / '.numi/commands/human'), 'stand', str(tmp_path / 'input'),
            str(bone), str(tmp_path / 'tendon.nhtendon'), str(support), str(output),
            *options,
        ], env={**os.environ, 'NUMI_LAB_ROOT': str(tmp_path), 'NUMI_BUILD_DIR': str(build),
                'NUMI_PROBE_ARGS': str(args)}, text=True, capture_output=True)
        argv = args.read_text().splitlines() if args.exists() else None
        return result, argv, bone, output
    return invoke


@pytest.mark.parametrize('magic,contacts', [
    (b'NHCNT1', ['2','3','4','5','6','7']),
    (b'NHCNT2', ['5','6','8','10','12','14']),
])
def test_default_stand_is_unassisted_and_uses_live_passive_law(launch, magic, contacts):
    result, argv, _, _ = launch(magic=magic)
    assert result.returncode == 0, result.stderr
    assert '--stand-root-assistance' not in argv
    assert '--stand-remove-assistance' not in argv
    assert '--muscle-activation' not in argv
    assert '--persistent-source-passive-joint-tissue' in argv
    assert '--persistent-stand-trace' in argv
    assert '--stand-deterministic-replay' in argv
    assert argv[argv.index('--stand-contact-iterations')+1] == '64'
    assert argv[argv.index('--muscle-step-seconds')+1] == '0.0000125'
    assert argv[argv.index('--muscle-step-count')+1] == '512'
    assert [argv[i+1] for i,x in enumerate(argv) if x=='--support-stance-contact'] == contacts


def test_assistance_requires_explicit_diagnostic(launch):
    result, argv, _, _ = launch('--assisted-diagnostic')
    assert result.returncode == 0, result.stderr
    assert '--stand-root-assistance' in argv
    assert '--stand-remove-assistance' in argv


def test_mechanics_only_uses_marker_input_without_bone_payload(launch):
    result, argv, bone, output = launch('--mechanics-only', '--steps', '8')
    assert result.returncode == 0, result.stderr
    assert '--mechanics-only' in argv
    assert str(bone) not in argv
    assert argv[2] == str(output)
    assert argv[argv.index('--muscle-step-count')+1] == '8'
    assert '--stand-root-assistance' not in argv
    assert '--source-route-centrelines' not in argv


@pytest.mark.parametrize('options', [
    ('--mechanics-only', '--mechanics-only'),
    ('--assisted-diagnostic', '--assisted-diagnostic'),
    ('--mechanics-only', '--mechanics-overlay'),
])
def test_conflicting_or_duplicate_options_do_not_launch(launch, options):
    result, argv, _, _ = launch(*options)
    assert result.returncode == 2
    assert argv is None


def test_mechanics_only_does_not_bypass_payload_admission(launch):
    result, argv, _, _ = launch('--mechanics-only', magic=b'INVALID')
    assert result.returncode == 2
    assert argv is None

def test_long_horizon_release_can_skip_per_step_trace_without_assistance(launch):
    result, argv, _, _ = launch(
        '--mechanics-only', '--no-step-trace',
        '--steps', '1000', '--timestep', '0.0001',
    )
    assert result.returncode == 0, result.stderr
    assert '--persistent-stand-trace' not in argv
    assert '--stand-deterministic-replay' in argv
    assert '--stand-root-assistance' not in argv
    assert '--stand-remove-assistance' not in argv
    assert argv[argv.index('--muscle-step-count') + 1] == '1000'
    assert argv[argv.index('--muscle-step-seconds') + 1] == '0.0001'


def test_no_step_trace_is_not_silently_repeatable(launch):
    result, argv, _, _ = launch('--no-step-trace', '--no-step-trace')
    assert result.returncode == 2
    assert argv is None


def test_execute_runs_requested_horizon_once_without_assistance(launch):
    result, argv, _, _ = launch(
        '--execute', '--mechanics-only', '--steps', '10000', '--timestep', '0.001',
    )
    assert result.returncode == 0, result.stderr
    assert '--stand-deterministic-replay' not in argv
    assert '--persistent-stand-trace' not in argv
    assert '--stand-root-assistance' not in argv
    assert argv[argv.index('--muscle-step-count') + 1] == '10000'
    assert argv[argv.index('--muscle-step-seconds') + 1] == '0.001'


def test_execute_rejects_assistance(launch):
    result, argv, _, _ = launch('--execute', '--assisted-diagnostic')
    assert result.returncode == 2
    assert argv is None


def test_execute_passes_explicit_muscle_feedback_gains(launch):
    result, argv, _, _ = launch('--execute', '--muscle-feedback', '10', '1')
    assert result.returncode == 0, result.stderr
    index = argv.index('--stand-muscle-feedback')
    assert argv[index + 1:index + 3] == ['10', '1']
    assert '--stand-root-assistance' not in argv


def test_feedback_requires_explicit_execution_mode(launch):
    result, argv, _, _ = launch('--muscle-feedback', '10', '1')
    assert result.returncode == 2
    assert argv is None
