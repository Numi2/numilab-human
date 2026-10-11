# Correct closest-point selection at an obtuse triangle

An incorrect AC-edge condition allowed projections beyond its endpoint. For triangle [(0,0,0),(3,0,0),(1,1,0)] and point (1.4,1.6,0), the old helper returned (1.5,1.5,0), outside the triangle, instead of the nearest BC point (1.08,0.96,0). It understated distance and could corrupt anatomical source-registration diagnostics. The condition now requires d6 <= 0.

The regression checks six vertex orders at metre and millimetre scales. Before correction, two of twelve cases failed; afterward all twelve pass. Focused closest-point, Z-Anatomy, tendon and attachment checks: 38 passed, 2 skipped. A broader attempt had 120 passes, 2 skips and 9 failures caused by unavailable source/build fixtures in this checkout; its log is retained.

This corrects offline geometry preparation. Existing exact self/intersection predicates and native physiology were not changed. Frozen prior assets/runs are not rewritten. Geometry and distance claims depending on this helper require refreshed evidence; fixing the helper alone does not qualify them.

Command on SSH Mac mini:

    PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python3.13 -m pytest -q tests/test_zanatomy_closest_point.py tests/test_tendon_face_paths.py tests/test_importer.py -k 'zanatomy or tendon or attachment or closest'
