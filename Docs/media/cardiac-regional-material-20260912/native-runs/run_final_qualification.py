"""Capture exact native source, builds and serial physical-host qualification."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/n/MetalRobo-human-completion-20260907')
BUILD = Path('/Users/n/MetalRobo-human-completion-build-20260907')
OUT = Path('/Users/n/human-regional-material-20260912/final-attempt-001')
OUT.mkdir(exist_ok=False)
SOURCES = [
    'include/metalrobo/MatterSnapshotArchive.hpp', 'matter/CMakeLists.txt',
    'matter/include/numi/matter/matter.hpp', 'matter/include/numi/matter/shared.h',
    'matter/src/accepted_state_proof_gpu.hpp', 'matter/src/compiler_impl.cpp',
    'matter/src/metal/fem.metalinc', 'matter/src/package_impl.cpp',
    'matter/src/runtime.mm', 'matter/src/validation_impl.cpp',
    'matter/tools/fem_regional_material_check.mm',
    'matter/materials/rodero2021_ventricular_guccione_passive.nmatter',
    'matter/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter',
    'matter/tools/cardiac_material_reference.hpp',
]
TARGETS = [
    'metalrobo_matter_physics_probe', 'metalrobo_matter_snapshot_archive_probe',
    'metalrobo_numilab_human_tendon_fem_load_probe', 'numi-matter-stateful-check',
    'numi-matter-vascular-cavity-check', 'numi-matter-vascular-compiler-check',
    'numi-matter-fiber-check', 'numi-matter-cardiac-transaction-check',
    'numi-matter-cardiac-material-check', 'numi-matter-fem-material-frame-compiler-check',
    'numi-matter-fem-material-frame-check', 'numi-matter-fem-regional-material-check',
]
TESTS = [
    'matter.numi_human.tendon_fem_transaction', 'matter.runtime.snapshot_archive',
    'matter.physics.mixed_mpm_fem', 'matter.physics.monolithic_multiphysics',
    'matter.physics.topology_mutation', 'matter.physics.cohesive_mutation',
    'matter.physics.small_scale_topology_conservation', 'matter.physics.puncture_mutation',
    'matter.runtime.topology_rollback', 'matter.physics.learned_polyconvex_icnn',
    'matter.runtime.production_transaction_rollback', 'matter.physics.stateful_mpm',
    'matter.physics.stateful_fem', 'matter.compiler.stateful_roundtrip',
    'matter.compiler.source_fiber', 'matter.metal.source_fiber',
    'matter.compiler.vascular', 'matter.metal.cardiac_transaction',
    'matter.metal.vascular_cavity', 'matter.compiler.cardiac_material_source',
    'matter.compiler.fem_material_frame', 'matter.metal.fem_material_frame',
    'matter.compiler.fem_regional_material', 'matter.metal.fem_regional_material',
    'matter.runtime.inverse_identification',
]

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def info(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def read(args):
    return subprocess.check_output(args, text=True).strip()

def run(args, name):
    start = time.monotonic()
    with (OUT / name).open('wb') as log:
        result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record = {'argv': args, 'cwd': str(ROOT), 'returncode': result.returncode,
              'elapsed_seconds': time.monotonic() - start, 'log': name, **info(OUT / name)}
    print(json.dumps(record), flush=True)
    return record

evidence = {'schema': 'numilab-human.regional-material-native-execution.v1',
    'started_utc': stamp(), 'base_revision': read(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
    'host': {'model': read(['sysctl', '-n', 'hw.model']),
        'cpu': read(['sysctl', '-n', 'machdep.cpu.brand_string']),
        'memory_bytes': int(read(['sysctl', '-n', 'hw.memsize'])), 'os': read(['sw_vers'])},
    'source_before': {p: info(ROOT / p) for p in SOURCES}, 'tests': TESTS,
    'runner': info(Path(__file__))}
evidence['build'] = run(['/opt/homebrew/bin/cmake', '--build', str(BUILD), '--target', *TARGETS, '-j', '4'], 'build.log')
if evidence['build']['returncode'] == 0:
    evidence['binaries'] = {p: info(BUILD / p) for p in
        ['bin/' + t if t.startswith('metalrobo_') else 'matter/' + t for t in TARGETS]
        + ['matter/shaders/NumiMatter.metallib']}
    regex = '^(' + '|'.join(t.replace('.', r'\.') for t in TESTS) + ')$'
    evidence['ctest'] = run(['/opt/homebrew/bin/ctest', '--test-dir', str(BUILD),
        '-R', regex, '-j', '1', '-V', '--output-on-failure', '--output-junit', str(OUT / 'ctest.xml')], 'ctest.log')
evidence['source_after'] = {p: info(ROOT / p) for p in SOURCES}
evidence['source_unchanged'] = evidence['source_before'] == evidence['source_after']
evidence['finished_utc'] = stamp()
evidence['status'] = 'pass' if (evidence['build']['returncode'] == 0 and
    evidence.get('ctest', {}).get('returncode') == 0 and evidence['source_unchanged']) else 'fail'
(OUT / 'execution.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps({'status': evidence['status'], 'evidence': str(OUT / 'execution.json')}), flush=True)
raise SystemExit(0 if evidence['status'] == 'pass' else 1)
