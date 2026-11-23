"""Compare inputs prepared by the ordinary launcher with both completed native runs."""
import json
import tempfile
from pathlib import Path
import sys
import run_shoulder_scene as launch


def verify(root, original):
    checks = {}
    for name, trial, inputs, count in original:
        captured = json.loads((root / name / 'launch.json').read_bytes())
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            dest = Path(tmp) / name
            invoked = []
            def capture(args, cwd, env, out):
                invoked.append((args, cwd))
                return {'exit': 0}
            launch.execute_native = capture
            launch.run(trial, inputs, dest, count)
            generated = json.loads((dest / 'launch.json').read_bytes())
            def normalize(text):
                return text.replace(str(dest),str(root / name))
            assert list(map(normalize, generated['argv'])) == captured['argv']
            assert {k:normalize(v) for k,v in generated['environment'].items()} == captured['environment']
            assert generated['pins'] == captured['pins'] and invoked == [(generated['argv'], captured['cwd'])]
            checks[name] = dict(inputs_identical=True, command_identical=True, environment_identical=True)
            altered = dict(generated, argv=list(generated['argv']))
            altered['argv'][2] = altered['argv'][3]
            try:
                launch.bind(altered, inputs)
                raise AssertionError('Accepted mismatched source')
            except ValueError:
                pass
    (root / 'launcher-parity.json').write_text(json.dumps(checks, indent=2)+'\n')


if __name__ == '__main__':
    root, compact, dense = (Path(v).resolve() for v in sys.argv[1:])
    verify(root, [('supported-8000',compact,root/'physical-input',8000), ('full-control-four',dense,root/'qualified-input',4)])
