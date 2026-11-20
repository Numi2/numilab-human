"""Run the same initial state and material limits with the alternate attachment owner."""
from pathlib import Path

source = Path('run_residual_bank_probe.py').read_text()
source = source.replace("root / 'residual-bank-trial'", "root / 'coupled-pectoral-trial'")
source = source.replace("'residual-bank-shader/NumiMatter.metallib'",
                        "'moving-attachment-shader/NumiMatter.metallib'")
source = source.replace("for filename, expected in launch['pins'].items():", """
app = root / 'coupled-pectoral-build/numi-human-native'
with app.open('rb') as stream:
    launch['pins'][str(app)] = hashlib.file_digest(stream, 'sha256').hexdigest()
launch['argv'][0] = str(app)
for filename, expected in launch['pins'].items():""")
source = source.replace("    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),",
    """    NUMI_MATTER_METALLIB_OVERRIDE=str(shader),
    NUMI_HUMAN_RESTING_COUPLED_PECTORAL_TRIAL='1',""")
exec(compile(source, 'run_residual_bank_probe.py', 'exec'))
