#!/usr/bin/env python3
"""Recompute pinned cavity evidence; distinguish geometry conflicts from runtime passes."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from numilab_human import cvsim21 as cv
from numilab_human import cvsim21_anatomy as anatomy

EVIDENCE = Path('Docs/media/cardiac-cavities-20260912')
NATIVE_COMMIT = 'b91fe6832813497ab532e5dbe05ed8c8233e6b1f'
SCOPE = {'cavity_reference_association': True, 'individual_cavities_embedded': True,
         'all_cavity_domains_disjoint': False, 'native_identity_equivalence': True,
         'blood_tissue_mass_partition': False, 'physiological_calibration': False, 'standing_walking': False}
REQUIRED_OWNERS = {
    'src/numilab_human/cardiac_cavity_geometry.py', 'src/numilab_human/cardiac_cavity_intersections.py',
    'src/numilab_human/cvsim21_anatomy.py', 'src/numilab_human/cvsim21.py',
    'src/numilab_human/cvsim_parameters.py', 'src/numilab_human/physiology.py', 'src/numilab_human/model.py',
    'config/cvsim21-source.v1.json', 'config/cvsim21-cardiac-cavities.v1.json', 'sources.lock.json',
    'schemas/humanpack-cvsim21-cardiac-cavities-config.v1.schema.json', '.numi/commands/human-circulation-anatomy',
    'tools/cardiac_cavity_native_check.mm', 'tools/qualify_cardiac_cavities_native.py',
    'tools/verify_cardiac_cavities_20260912.py', 'tests/test_cardiac_cavity_geometry.py',
    'tests/test_cardiac_cavity_intersections.py', 'tests/test_cvsim21_anatomy.py',
    'tests/test_cardiac_cavities_evidence.py'}
REQUIRED_ARTIFACTS = {(EVIDENCE / (variant + '.' + suffix)).as_posix()
                      for variant in cv.VARIANTS for suffix in ('native.json', 'manifest.json', 'baseline.json')}
REQUIRED_ARTIFACTS |= {(EVIDENCE / name).as_posix() for name in (
    'native/execution.json', 'native/build.log', 'native/upstream_equation.native.log',
    'native/heldt_table_aligned.native.log', 'native/physical-mutation.json', 'native/physical-mutation.log',
    'human-tests.txt', 'independent/intersections.json', 'independent/source-decimal-audit.txt',
    'independent/exact-source-geometry-audit.py')}
STATIC_INPUTS = {
    'libnumi_matter_runtime.a': '817ec224f3df6b61372b3bef4f64d618b1554999747483992d665676f4c38d7d',
    'libnumi_matter_compiler.a': '30daf7d0126139ceddfb75ec798c8aa5228da35e1e119d0c14e1f3bb004c36d5',
    'NumiMatter.metallib': 'f84a1282fd24a9ed34fd06199596af19d49d7b0ea8561c00d8a73f50f348d86b'}
# These completed execution records predate per-run log digests. Bind the actual
# captured records here, rather than imply that a later seal was captured at run time.
CAPTURED_SHA256 = {
    'native/execution.json': '138fd9f939abfd494606834eb5bc70517bd3771557fb2803f2bb7b3887222f40',
    'native/upstream_equation.native.log': '84b941b2a579bbe2f7f12c0350e3c11f257ac4700533fa722f2d12afc80df13d',
    'native/heldt_table_aligned.native.log': 'd59b71154ef61fef126551d752fda5c4cbc2c3006bed4a37b5196380ea5f1998',
    'native/physical-mutation.log': 'd31fdc1ca7e05844dd4403447bc342aa0370de5be44d1d52a4951aa6d46015b6'}
SUMMARY_FIELDS = dict(cavity_native_check='pass', accepted_steps='64', environments_per_model='2', failed_steps='0',
    dt_seconds='0.001', physical_parameters='identical', state_pair='bitwise', environment_pair='bitwise',
    accepted_clocks='bitwise', snapshot_replay='bitwise', cross_identity_restore='rejected',
    package_fingerprints='distinct', finite_state='true', state_evolved='true', mechanical_mass_added='0',
    anatomical_disjointness='unqualified', physiological_calibration='unqualified')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    need(path.is_file() and path.stat().st_size <= 256 * 1024**2, 'missing or oversized evidence')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def checked_path(root, relative):
    need(isinstance(relative, str) and relative and '\\' not in relative and
         all(ord(c) >= 32 and ord(c) != 127 for c in relative), 'unsafe evidence path')
    p = PurePosixPath(relative)
    need(not p.is_absolute() and p.parts and '..' not in p.parts and p.as_posix() == relative, 'unsafe evidence path')
    target = root
    for component in p.parts:
        target /= component
        need(not target.is_symlink(), 'symlink evidence component')
    need(target.is_file() and target.resolve().is_relative_to(root.resolve()), 'missing or redirected evidence')
    return target


def audit_scope(receipt):
    need(set(receipt) == {'schema', 'native_commit', 'owners', 'artifacts', 'scope'}, 'receipt fields differ')
    need(receipt['schema'] == 'HumanPack.cardiac-cavities-evidence.v1' and receipt['native_commit'] == NATIVE_COMMIT,
         'receipt identity differs')
    scope = receipt['scope']
    need(isinstance(scope, dict) and set(scope) == set(SCOPE) and
         all(scope[key] is expected for key, expected in SCOPE.items()), 'unsupported scope claim')


def checked_inventory(root, inventory, required, label):
    need(isinstance(inventory, dict) and required <= inventory.keys(), 'missing required ' + label)
    paths = {}
    for relative, expected in inventory.items():
        if label == 'artifacts':
            need(isinstance(relative, str) and relative.startswith(EVIDENCE.as_posix() + '/'),
                 'artifact outside cavity evidence directory')
        need(isinstance(expected, str) and re.fullmatch(r'[0-9a-f]{64}', expected), 'invalid evidence SHA256')
        target = checked_path(root, relative)
        need(sha(target) == expected, 'hash drift: ' + relative)
        if relative.endswith('.json'):
            cv.read_json(target)
        paths[relative] = target
    return paths


def _roles(mapping):
    need(isinstance(mapping, dict) and mapping, 'missing captured input map')
    result = {}
    for path, digest in mapping.items():
        need(isinstance(path, str) and Path(path).is_absolute() and Path(path).as_posix() == path and
             '..' not in Path(path).parts and '\\' not in path and all(32 <= ord(c) != 127 for c in path), 'invalid captured path')
        name = Path(path).name
        need(name not in result and isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest), 'duplicate role or invalid captured SHA')
        result[name] = (path, digest)
    return result


def audit_native_summary(log):
    rows = [line for line in log.splitlines() if line.startswith('cavity_native_check=')]
    need(len(rows) == 1, 'missing or duplicate native summary')
    fields = rows[0].split()
    need(all(field.count('=') == 1 for field in fields), 'malformed native summary')
    pairs = [field.split('=', 1) for field in fields]
    need(len({key for key, _ in pairs}) == len(pairs) and dict(pairs) == SUMMARY_FIELDS, 'native summary fields differ')
    devices = re.findall(r'^device=Apple M4 Pro fingerprint=([0-9]+)$', log, re.MULTILINE)
    need(len(devices) == 2 and len(set(devices)) == 2, 'physical device or distinct package fingerprints missing')
    need('Metal API Validation Enabled' in log, 'Metal validation attestation missing')


def audit_execution(execution, manifests, artifact, root):
    need(execution.get('schema') == 'HumanPack.cardiac-cavity-native-execution.v1' and execution['status'] == 'pass' and
         execution['native_commit'] == execution['native_commit_after'] == NATIVE_COMMIT, 'native execution identity differs')
    need(execution['native_status_before'] == execution['native_status_after'] == '', 'native source was dirty')
    need(type(execution['build_exit_code']) is int and execution['build_exit_code'] == 0 and
         execution['build_inputs_before'] == execution['build_inputs_after'], 'native build failed or drifted')
    need(execution['runner_sha256'] == sha(root / 'tools/qualify_cardiac_cavities_native.py'), 'native runner drift')
    roles = _roles(execution['build_inputs_before'])
    expected_build = {**STATIC_INPUTS, 'cardiac_cavity_native_check.mm': sha(root / 'tools/cardiac_cavity_native_check.mm')}
    need({name: value[1] for name, value in roles.items()} == expected_build, 'native static build roles or identities differ')
    command = execution['build_command']
    need(isinstance(command, list) and len(command) == 23, 'native build command malformed')
    # Header checkout is separate from the build tree, but both include roots
    # must refer to the same captured, clean native source checkout.
    need(isinstance(command[12], str) and command[12].startswith('-I/'), 'native header root missing')
    headers = Path(command[12][2:])
    need(headers.name == 'include' and command[13] == '-I' + str(headers.parent / 'matter/include'), 'native header roots differ')
    executable = command[-1]
    expected_command = ['xcrun','clang++','-O3','-DNDEBUG','-std=c++23','-arch','arm64','-fobjc-arc','-Wall','-Wextra','-Werror',
        '-DNUMI_MATTER_METALLIB="' + roles['NumiMatter.metallib'][0] + '"', command[12], command[13],
        roles['cardiac_cavity_native_check.mm'][0], roles['libnumi_matter_runtime.a'][0], roles['libnumi_matter_compiler.a'][0],
        '-framework','Foundation','-framework','Metal','-o',executable]
    need(command == expected_command and Path(executable).is_absolute() and Path(executable).name == 'cardiac-cavity-native-check',
         'native build command differs from captured inputs')
    dependencies = execution.get('dynamic_dependencies')
    need(isinstance(dependencies, str) and dependencies.splitlines()[0] == executable + ':', 'dynamic dependency attestation missing')
    libraries = [line.strip().split(' (',1)[0] for line in dependencies.splitlines()[1:]]
    need(len(libraries) == 6 and all(path.startswith(('/System/Library/', '/usr/lib/')) for path in libraries),
         'unbound dynamic dependency')
    need(execution['environment'] == {'MTL_DEBUG_LAYER':'1'}, 'Metal validation was not enabled')
    need(len(execution['runs']) == 2 and {r['volume_coordinates'] for r in execution['runs']} == cv.VARIANTS, 'native variant coverage differs')
    def check_run(run, expected, expected_log, exit_code):
        need(type(run['exit_code']) is int and run['exit_code'] == exit_code and run['inputs_before'] == run['inputs_after'],
             'native pair failed or drifted')
        captured = _roles(run['inputs_before'])
        need({name: value[1] for name,value in captured.items()} == expected, 'runtime input identity differs')
        need(captured['cardiac-cavity-native-check'][0] == executable and captured['NumiMatter.metallib'][0] == roles['NumiMatter.metallib'][0],
             'runtime paths differ from built executable or Metal library')
        need(run['log'] == expected_log, 'wrong native log role')
        payload_names = [name for name in expected if name.endswith('.json')]
        baseline = next(name for name in payload_names if name.endswith('.baseline.json'))
        associated = next(name for name in payload_names if name != baseline)
        need(run['command'] == [executable, captured[baseline][0], captured[associated][0]], 'native command differs from captured inputs')
        return artifact('native/' + expected_log).read_text()
    common = {'cardiac-cavity-native-check': execution['executable_sha256'], 'NumiMatter.metallib': STATIC_INPUTS['NumiMatter.metallib']}
    for run in execution['runs']:
        variant = run['volume_coordinates']
        expected = {**common, variant+'.native.json': manifests[variant]['native_content_sha256'],
                    variant+'.baseline.json': manifests[variant]['source_native_sha256']}
        audit_native_summary(check_run(run, expected, variant+'.native.log', 0))
    negative = execution['negative_control']
    expected = {**common, 'upstream_equation.baseline.json': manifests['upstream_equation']['source_native_sha256'],
                'physical-mutation.json': sha(artifact('native/physical-mutation.json'))}
    log = check_run(negative, expected, 'physical-mutation.log', 1)
    need(re.findall(r'^cavity_native_check=.*$',log,re.MULTILINE) ==
         ['cavity_native_check=failed reason=anatomy association changed physical parameters or unrelated fields'],
         'negative control failed for another reason')
    mutation = cv.read_json(artifact('upstream_equation.native.json'))
    mutation['compartments'][15]['initial_volume_m3'] *= 1.01
    need(artifact('native/physical-mutation.json').read_bytes() == cv.canonical(mutation)+b'\n', 'negative control physical mutation differs')


def verify(*, root=ROOT, receipt_path=None):
    root = Path(root).resolve()
    receipt_path = Path(receipt_path) if receipt_path else root / EVIDENCE / 'receipt.json'
    need(receipt_path.is_absolute() and receipt_path.is_relative_to(root), 'receipt outside evidence root')
    receipt = cv.read_json(checked_path(root, receipt_path.relative_to(root).as_posix()))
    audit_scope(receipt)
    checked_inventory(root, receipt['owners'], REQUIRED_OWNERS, 'owners')
    paths = checked_inventory(root, receipt['artifacts'], REQUIRED_ARTIFACTS, 'artifacts')
    # The imported trusted helpers must match the source copy being audited.
    for relative in REQUIRED_OWNERS:
        need(sha(checked_path(ROOT, relative)) == receipt['owners'][relative], 'loaded owner differs from audited source')
    def artifact(name):
        return paths[(EVIDENCE / name).as_posix()]
    for name, expected in CAPTURED_SHA256.items():
        need(sha(artifact(name)) == expected, 'captured execution or log changed')
    human_tests = artifact('human-tests.txt').read_text()
    test_counts = re.findall(r'^Ran ([0-9]+) tests in [0-9.]+s$', human_tests, re.MULTILINE)
    need(len(test_counts) == 1 and int(test_counts[0]) >= 162 and human_tests.rstrip().endswith('\nOK') and
         not re.search(r'^(FAILED|ERROR:|FAIL:|OK \()', human_tests, re.MULTILINE), 'retained Human test cohort did not pass')
    manifests = {}
    for variant in sorted(cv.VARIANTS):
        config = cv.read_json(root / 'config/cvsim21-cardiac-cavities.v1.json')
        config['volume_coordinates'] = variant
        native, manifest = anatomy.compile_registration(sources=root / 'Sources', source_lock=root / 'sources.lock.json', config=config)
        for suffix, value in (('native.json', native), ('manifest.json', manifest)):
            need(artifact(variant+'.'+suffix).read_bytes() == cv.canonical(value)+b'\n', 'regenerated cavity artifact differs')
        base_config = cv.read_json(root / 'config/cvsim21-source.v1.json')
        base_config['volume_coordinates'] = variant
        baseline, _ = cv.compile_source(directory=root/'third_party/physionet/cvsim21',
            initial_path=root/'tools/cvsim21_reference/evidence/20260912/original-initial.json', config=base_config)
        need(artifact(variant+'.baseline.json').read_bytes() == cv.canonical(baseline)+b'\n', 'baseline physical payload drifted')
        audit = manifest['intersection_audit']
        need(audit['all_surfaces_embedded'] is True and audit['all_domains_disjoint'] is False, 'geometry conflict was concealed')
        pairs = [pair for pair in audit['per_pair'] if pair['count']]
        need(len(pairs)==1 and pairs[0]['first']=='right_atrium' and pairs[0]['second']=='right_ventricle' and pairs[0]['count']==42,
             'pinned interdomain conflict differs')
        need(manifest['added_mechanical_mass_kg'] == manifest['withdrawn_mechanical_mass_kg'] == 0, 'anatomical reference mutated mechanical mass')
        need(cv.read_json(artifact('independent/intersections.json')) == audit, 'independent geometric audit differs')
        manifests[variant] = manifest
    audit_execution(cv.read_json(artifact('native/execution.json')), manifests, artifact, root)
    return {'status':'pass', 'owners':len(receipt['owners']), 'artifacts':len(receipt['artifacts']),
            'individual_embedded_cavities':4, 'right_atrium_right_ventricle_intersections':42,
            'disjoint_physical_volume_admission':False, 'native_variant_pairs_passed':2,
            'blood_mass_partition':False, 'physiological_calibration':False}


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), sort_keys=True))
    except Exception as error:
        print(json.dumps({'status':'failed','error':str(error)}))
        raise SystemExit(1)
