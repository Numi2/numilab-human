# Native left pronator quadratus boundary increment

The integrated native resting human now includes the corrected passive left pronator quadratus (row 130). All 155 other tissue rows, physical mass, active muscle/tendon routes, flat support contact and coupled GPU physiology remain unchanged.

The existing source-boundary reconstruction and bounded source correction preserve the reference binding maps and one-hot visual proxy. The maximum accumulated source-coordinate correction is 0.2311 mm; source area changes 0.00112% and volume 0.000626%. These are inferred reference-anatomy corrections, not measured entheses or a simulated active forearm.

The first native run exposed four self-intersection pairs at accepted step 5599 despite CPU-forward checks passing. That failed native result is retained. The existing position-preserving interior edge-flip owner repairs the local triangulation: rows 1865 and 1866 share the same raw parent face 1474, so source-parent lineage is unchanged. No coordinates or binding weights change in this precision step. Source and all 25 retained pose tests pass; all 865 current neighbors and all 859 older captured neighbors gain no strict source-parent pair identities. Older targets remain explicitly labeled historical geometry.

A fresh 16-second native run reaches 8000 accepted steps. All eight actual GPU captures have zero self intersections in row 130, a closed boundary and six positive-volume components. Actual native neighbor pair tables match the candidate verification. All 865 other displayed surfaces, all 157 body poses and the 1000-by 60 coupled-state trace match the preceding bilateral EDL/EHL scene exactly. The continuous invalid-triangle/contact checks and 252-frame native recording pass, with seven anatomical inspection layers.

Existing external overlaps remain unresolved; this increment is not completed anatomical qualification. Remaining forearm, back, temporalis and interface work precedes the final five-minute baseline/intervention run on completed anatomy. Performance work is deferred. The provisional aggregate 2 Nm/rad passive MTP configuration is unchanged.

This is a registered mixed-source reference adult, not measurements of one individual. BodyParts 3D attribution: The Database Center for Life Science, CC BY-SA 2.1 Japan. Existing Z-Anatomy material retains its CC BY-SA 4.0 attribution in the scene provenance.

On the SSH Mac mini with retained assets:

    python 3 tools/evidence/native-anatomy-left-pronator-quadratus-1344/reproduce.py --prepare-only
    python 3 tools/evidence/native-anatomy-left-pronator-quadratus-1344/reproduce.py

Exact revisions, device, binary identity, configuration, launch arguments, traces, source corrections, failure evidence and native recording are retained in this package or its hash-bound local references.
