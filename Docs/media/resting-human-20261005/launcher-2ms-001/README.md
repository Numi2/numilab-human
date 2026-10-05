# Native resting launcher admission

On the SSH Mac mini, the nine `tests.test_resting_run` tests passed with Python 3.13 from `/Users/n/numi-human-prep-venv-20261005/bin/python`. The launcher now accepts an explicit timestep up to 2 ms, preserves the requested duration as an integer step count, and rejects drive scales above the native owner's limit of 2. The inspection-tour period is presentation-only and is retained in the invocation environment.

The actual six-second 1 ms/2 ms native comparison and forced-rejection checks are retained in Numi Lab's `matter/tools/evidence/human-resting-20261005/support-and-timestep-20261005/README.md`. The successful 2 ms run is `/Users/n/numi-human-resting-evidence-20261005/native-dt2-6s-002` (3,000 accepted steps, 6.000000285 simulated seconds, 62.993 seconds runtime wall time). This launcher test is an admission regression, not additional physical or physiological evidence. The five-minute whole-body qualification and complete anatomical intersection audit remain separate requirements.

Command (executed on the Mini, not the Air):

```sh
cd /Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project
PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest tests.test_resting_run -v
```
