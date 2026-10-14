Run on SSH Numi:

    PYTHONPATH=src python -m numilab_human.resting_anatomy_verify --pack CAPTURE.mrvpack --capture-receipt CAPTURE.receipt.json --changed 51011:471 --output audit.json

All anatomical layers are targets by default. Typed semantic/stable IDs prevent aliasing; duplicate primitive chunks are retained. Exact packed-coordinate identification affects predicates only and reports original and quotient topology. Containment uses rational metres and requires a closed embedded target. Open boundaries remain open. Declared interface face pairs do not prove connected vessel lumens.

The Python API compares emitted Float32 positions with the existing rigid/material-parent owner reconstruction. It does not introduce another transform or simulator. Passing selected checks does not admit whole anatomy, mechanics, physiology or clinical validity. The current N27 capture decoded 873 typed anatomical surfaces; only receipt/section/pose input checks were run for that capture in this increment.
