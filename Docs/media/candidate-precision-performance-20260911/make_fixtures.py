from pathlib import Path
import subprocess, json, hashlib, sys

base = Path(__file__).parent
original = Path('/Users/n/human-compliant-equilibrium-20260910/fixture/prepared.nhinit')
binary = '/Users/n/MetalRobo-human-completion-build-20260907/bin/metalrobo_numanx_fullbody_bridge_probe'
iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 16
for dt in ([25] if iterations != 16 else [100, 50, 25]):
    folder = base / (f'fixture-exact-{dt}us' + (f'-iterations{iterations}' if iterations != 16 else ''))
    command = [binary, '--prepared-state-fixture', str(original), str(folder),
        '/Users/n/human-capsule-20260908/input/myosim-fullbody-support-primitives.nhcnt',
        '/Users/n/human-completion-20260907/input/myosim-fullbody-joint-equalities-source-compliance.nheq',
        '/Users/n/human-dynamic-limits-20260908/myosim-fullbody-joint-limits.nhlim', str(dt), str(iterations)]
    result = subprocess.run(command, capture_output=True, text=True)
    (base / (folder.name + '.log')).write_text(result.stdout + result.stderr)
    record = {'command': command, 'returncode': result.returncode,
        'source_revision': subprocess.check_output(['git', '-C', '/Users/n/MetalRobo-human-completion-20260907', 'rev-parse', 'HEAD'], text=True).strip(),
        'binary_sha256': hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(original.read_bytes()).hexdigest()}
    (base / (folder.name + '-launch.json')).write_text(json.dumps(record, indent=2))
    if result.returncode:
        raise SystemExit(result.stderr)
    payload = (folder / 'prepared.nhinit').read_bytes()
    assert payload[96:] == original.read_bytes()[96:], 'physical state changed'
    if dt == 100 and iterations == 16:
        assert payload == original.read_bytes(), '100us identity changed'
    record['physical_payload_identical'] = True
    record['output_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    (base / (folder.name + '-launch.json')).write_text(json.dumps(record, indent=2))
    print(dt, hashlib.sha256(payload).hexdigest())
