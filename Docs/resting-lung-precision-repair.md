# Reproduce the retained lung source precision candidate

This recipe uses the existing Numi Human NHANAT1 ABI5 reader, the existing
source-owner pleura union builder, and the existing conforming respiratory
Kuhn field. It makes one midpoint edge collapse in stable ID 308 and derives
stable ID 310 from the five registered lobe surfaces. It does not alter the
native runtime. The retained source asset is licensed CC-BY-SA-4.0; preserve
its attribution and share-alike obligations.

The builder fails closed unless all three retained inputs match their pinned
SHA-256 values:

- NHANAT input: /Users/n/numi-human-resting-evidence-20261005/integrated-anatomy-engineering-565/resting-thorax.nhanatomy — 90caf4727bebc6889c9b23f4124c795861bed969cae2ec9bbe3127f4b1a07f32
- Receipt: /Users/n/numi-human-resting-evidence-20261005/integrated-anatomy-engineering-565/resting-anatomy-receipt.json — 5dbebdbc50b31e8660985989ab84bf7cba59b348c57b9b105ca041e64352bd39
- Stable-308 face-origin map: /Users/n/numi-human-resting-evidence-20261005/respiratory-zero-source-diagnosis-001/short-edge-batch-035/candidate/row-308-face-origin.npy — c23e2fcb357d8b8be54f0022c37ebfa7ec8606bc1c96f26724528ac9a67793e9

It also pins the existing respiratory field owner at
/Users/n/numi-human-resting-resp-source-20261006/src/numilab_human/resting_respiratory_conforming_field.py
with SHA-256 980b368d8de0cc6cd7e25a93e61e5b6f304275dbd18511207da8cc9919410bb5.

Run the focused owner tests from the Human worktree:

    PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest discover -s tests -p 'test_resting_lung_edge_repair.py'
    PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest discover -s tests -p 'test_resting_pleura_proxy.py'

Then create a new, empty output directory and run the pinned source build:

    PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python -m numilab_human.resting_lung_edge_repair --base-payload /Users/n/numi-human-resting-evidence-20261005/integrated-anatomy-engineering-565/resting-thorax.nhanatomy --base-receipt /Users/n/numi-human-resting-evidence-20261005/integrated-anatomy-engineering-565/resting-anatomy-receipt.json --face-origin /Users/n/numi-human-resting-evidence-20261005/respiratory-zero-source-diagnosis-001/short-edge-batch-035/candidate/row-308-face-origin.npy --respiratory-owner /Users/n/numi-human-resting-resp-source-20261006/src/numilab_human/resting_respiratory_conforming_field.py --output-dir /Users/n/numi-human-resting-evidence-20261005/lung-owner-reproduction-20261008

The output contains the pre-pleura intermediate, owner-derived final NHA
payload and receipt, a row-308 face-origin sidecar, exact row-310 per-face
source lineage, and a JSON build report. The code verifies that rows other
than 308 and 310 retain exact decoded geometry bytes and that the original
receipt binds the exact pinned source payload.

This is source-space candidate generation. It does not establish that native
Float32 transformed lung surfaces are intersection-free; the retained 305/308
breathing-cycle seam audit remains a separate gate.
