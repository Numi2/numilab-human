# Native right pronator quadratus boundary increment

The integrated native resting human now includes the corrected passive right pronator quadratus (row 129). All 155 other tissue rows, physical mass, active muscle/tendon routes, flat support contact and coupled GPU physiology remain unchanged.

The existing source-boundary reconstruction is retained with a maximum 0.08422 mm coordinate correction. Four explicitly inferred mixed visual binding weights change at vertices 686, 881, 886 and 894; their body support and the pure attachment proxy remain fixed. The binding corrections do not change source positions or topology. Source area changes approximately 0.0006223% and volume -0.0002363%. These are reference anatomy corrections, not measured entheses or active forearm function.

The first current-pose correction failed older captured poses. Those failures and the infeasible coordinate-only repair are retained. Exact child-face localization led to two additional small mixed-binding corrections. Source and all 25 retained pose checks pass, with no new strict source-parent interfaces against all 865 current neighbors and 859 older neighbors. Historical targets remain explicitly identified as older captured geometry.

A fresh 16-second native run reaches 8,000 accepted steps. All eight actual GPU captures have zero right pronator-quadratus self intersections, a closed boundary and six positive-volume components. Native source-parent pair checks introduce no new interfaces. All 865 other displayed surfaces, all 157 body poses and the 1,000-by-60 coupled-state trace match the preceding left pronator-quadratus scene exactly. The continuous invalid-triangle/contact checks and 252-frame native recording pass, with seven inspection layers. Exact native-versus-offline pair equality is reported separately in summary.json.

Existing external overlaps remain unresolved; this increment does not complete anatomical qualification. Remaining forearm, back, temporalis and interface work precedes the final five-minute baseline/intervention run on completed anatomy. Runtime performance work remains deferred, and the provisional aggregate 2 Nm/rad passive MTP configuration is unchanged.

This is a registered mixed-source reference adult, not measurements of one individual. BodyParts3D attribution: The Database Center for Life Science, CC BY-SA 2.1 Japan. Existing Z-Anatomy material retains CC BY-SA 4.0 attribution in scene provenance.

On the SSH Mac mini with retained assets:

    python3 tools/evidence/native-anatomy-right-pronator-quadratus-1346/reproduce.py --prepare-only
    python3 tools/evidence/native-anatomy-right-pronator-quadratus-1346/reproduce.py

Exact revisions, device, binary identity, configuration, launch arguments, compact traces, corrections, failed checks and continuous native recording are retained in this package or its hash-bound local references.
