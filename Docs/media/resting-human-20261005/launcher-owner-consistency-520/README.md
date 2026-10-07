# Resting launcher owner consistency

The body-scene receipt and anatomical accounting can each be valid yet refer to
different skin or rigid-body assets. The launcher now compares their declared
hashes before native execution, allowing equivalent copies with identical bytes.
Declared malformed identities fail; legacy receipts without these optional
fields retain their prior behavior.

The invocation also retains the accepted COM diagnostic flag and the common-field
failure-receipt path when supplied.

Validation on the SSH Apple M4 Pro Mac mini: 23 launcher tests passed. The corrected
skin-boundary scene passed preflight with 25 hashed assets. The older independently
valid support32 scene was rejected because its skin differs from the anatomy.
No output directory or native process was created by that preflight. The retained
fixture415 is used only for admission regression, not anatomical qualification.

Command: PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest discover -s tests -p test_resting_run.py -v.
