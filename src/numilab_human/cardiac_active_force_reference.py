"""Source-bound fixed-geometry ventricular active-stress residual producer.

This produces an offline FEM internal residual from candidate activation times.
It does not execute a Matter step or apply a physical external load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from . import model as human


ROOT = human.REPOSITORY_ROOT
CONFIG = ROOT / 'config/cardiac-rodero18-active-force-reference.v1.json'
SOURCE_CONFIG = ROOT / 'config/cardiac-wall-rodero18.v1.json'
SOURCE_SUPPLEMENT = ROOT / 'Docs/media/cardiac-source-activation-20260930/rodero-s4-source.pdf'
CPP = ROOT / 'tools/cardiac_active_force_reference.cpp'
SOURCE_BUFFERS = ('nodes.f64le', 'tetrahedra.u32le', 'labels.u32le', 'fibres.f64le')
ACTIVATION_BUFFERS = ('refined-arrival.f64le', 'ventricular-source-nodes.u32le',
                      'ventricular-dof-regions.u32le')


def require(value: bool, message: str) -> None:
    if not value:
        raise human.ImportError('cardiac active-force reference: ' + message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validated_inputs(asset: Path, activation: Path) -> tuple[dict, dict, dict]:
    config = human.read_json(CONFIG)
    source_config = human.read_json(SOURCE_CONFIG)
    asset_manifest_path = asset / 'manifest.json'
    activation_summary_path = activation / 'summary.json'
    asset_manifest = human.read_json(asset_manifest_path)
    activation_summary = human.read_json(activation_summary_path)
    require(config['schema'] == 'HumanPack.cardiac-rodero18-active-force-reference.v1'
            and config['anatomy_manifest_sha256'] == sha(asset_manifest_path)
            and config['candidate_activation_summary_sha256'] == sha(activation_summary_path)
            and config['source_supplement_sha256'] == sha(SOURCE_SUPPLEMENT)
            and asset_manifest['schema'] == 'HumanPack.cardiac-wall-source-asset.v1'
            and activation_summary['status'] == 'full_source_reconstruction_with_source_output_mismatch'
            and activation_summary['ventricular_tetrahedra'] == 1097534
            and activation_summary['ventricular_dofs'] == 218080
            and activation_summary['replay_bitwise'] is True
            and activation_summary['heartbeat_qualified'] is False,
            'pinned anatomy, source supplement and candidate activation identity')
    for name in SOURCE_BUFFERS:
        path = asset / name
        receipt = asset_manifest['buffers'][name]
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == receipt['bytes']
                and sha(path) == receipt['sha256'], f'source buffer {name}')
    for name in ACTIVATION_BUFFERS:
        path = activation / name
        require(path.is_file() and not path.is_symlink()
                and sha(path) == activation_summary['output_sha256'][name],
                f'candidate activation buffer {name}')
    source_params = source_config['source_mechanics']['active_ventricular']['parameters']
    mapped = (
        ('peak_isometric_tension_pa', 'peak_isometric_tension', 1000, 'kPa'),
        ('electromechanical_delay_ms', 'electromechanical_delay', 1, 'ms'),
        ('contraction_time_constant_ms', 'contraction_time_constant', 1, 'ms'),
        ('relaxation_time_constant_ms', 'relaxation_time_constant', 1, 'ms'),
        ('transient_duration_ms', 'transient_duration', 1, 'ms'),
    )
    for current, source, factor, unit in mapped:
        require(source_params[source]['unit'] == unit
                and config['source_parameters'][current] == factor * source_params[source]['value'],
                f'source active-stress parameter {source}')
    require(config['declared_variants']['length_dependence'].startswith('disabled')
            and config['declared_variants']['deformation_gradient'].startswith('identity')
            and config['force_frames_ms'] == [100, 250],
            'declared fixed-reference variant')
    return config, asset_manifest, activation_summary


def parse_native_output(line: str) -> dict[str, float | int]:
    tokens = dict(token.split('=', 1) for token in line.strip().split())
    require(set(tokens) == {'cells', 'active_cells', 'nodes', 'volume_m3',
                            'stress_volume_integral_j', 'max_tension_pa'},
            'native output fields')
    return {key: (int(value) if key in {'cells', 'active_cells', 'nodes'} else float(value))
            for key, value in tokens.items()}


def produce(asset: Path, activation: Path, output: Path) -> dict:
    asset, activation, output = (Path(p).resolve() for p in (asset, activation, output))
    config, asset_manifest, activation_summary = validated_inputs(asset, activation)
    output.mkdir(parents=True, exist_ok=True)
    binary = output / 'cardiac-active-internal-residual'
    compile_command = ['clang++', '-std=c++20', '-O3', '-march=native', str(CPP),
                       '-o', str(binary)]
    subprocess.run(compile_command, check=True, capture_output=True, text=True)
    source_parameters = config['source_parameters']
    parameter_values = [source_parameters[name] for name in (
        'peak_isometric_tension_pa', 'electromechanical_delay_ms',
        'contraction_time_constant_ms', 'relaxation_time_constant_ms',
        'transient_duration_ms')]
    frames = []
    for time_ms in config['force_frames_ms']:
        path = output / f'internal-residual-{time_ms}ms.f64le'
        replay_path = output / f'internal-residual-{time_ms}ms.replay.f64le'
        args = [str(binary), str(asset), str(activation)]
        commands = []
        for target in (path, replay_path):
            command = [*args, str(target), str(time_ms),
                       *(str(value) for value in parameter_values)]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            require(not result.stderr and len(result.stdout.strip().splitlines()) == 1,
                    f'native frame {time_ms} stdout/stderr')
            commands.append((command, result.stdout))
        require(commands[0][1] == commands[1][1] and sha(path) == sha(replay_path)
                and path.stat().st_size == 218077 * 3 * 8,
                f'native frame {time_ms} bitwise replay and dimension')
        replay_path.unlink()
        metrics = parse_native_output(commands[0][1])
        target_volume = sum(asset_manifest['topology']['regional_geometric_volume_m3'][str(label)]
                            for label in (1, 2))
        require(metrics['cells'] == 1097534 and metrics['nodes'] == 218077
                and 0 < metrics['active_cells'] <= metrics['cells']
                and abs(metrics['volume_m3'] - target_volume) < 1e-12
                and 0 < metrics['max_tension_pa'] <= parameter_values[0],
                f'native frame {time_ms} source volume/stress')
        frames.append({'time_ms': time_ms, 'file': path.name, 'sha256': sha(path),
                       'bytes': path.stat().st_size, 'native_metrics': metrics,
                       'native_stdout': commands[0][1].strip(), 'replay_bitwise': True})
    result = {
        'schema': 'numi.human.cardiac-active-force-reference.v1',
        'status': 'candidate_fixed_reference_active_internal_residual',
        'asset_manifest_sha256': sha(asset / 'manifest.json'),
        'candidate_activation_summary_sha256': sha(activation / 'summary.json'),
        'source_config_sha256': sha(SOURCE_CONFIG),
        'reference_config_sha256': sha(CONFIG),
        'source_supplement_sha256': sha(SOURCE_SUPPLEMENT),
        'source_cpp_sha256': sha(CPP), 'native_binary_sha256': sha(binary),
        'compile_command': compile_command,
        'platform': platform.platform(), 'frames': frames,
        'production_native_electromechanical_steps': 0,
        'source_model_reproduced': False,
        'physical_heart_motion': False,
        'heartbeat_qualified': False,
        'boundary': ('Offline positive FEM internal residual at F=I from candidate activation. '
                     'External nodal load sign is opposite. No stress-free reference, dynamic '
                     'deformation, native accepted state, blood-flow coupling or heartbeat.'),
    }
    path = output / 'summary.json'
    encoded = json.dumps(result, sort_keys=True, indent=2) + '\n'
    if path.exists():
        require(path.read_text() == encoded, 'existing summary changed on replay')
    else:
        pending = path.with_name(path.name + f'.{os.getpid()}.pending')
        pending.write_text(encoded)
        os.replace(pending, path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--activation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = produce(args.asset, args.activation, args.output)
    print(json.dumps({'status': result['status'], 'frames': result['frames']}))


if __name__ == '__main__':
    main()
