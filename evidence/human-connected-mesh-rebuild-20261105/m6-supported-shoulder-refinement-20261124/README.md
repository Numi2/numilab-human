## Posterior shoulder skin and adipose refinement on SSH Numi

In the coupled variable-pleural-pressure frame at 15.625 µs, skin node 24331 received 66.98% of the terminal bed impulse. Its eight incident edges measured 61.5–100.2 mm. Both the native skin geometry and the physical outer wall had one vertex inside 50 mm of this site.

The local material subdivision replaces 47 connected tetrahedra with 2,671 children. All 545 new coordinates participate in skin or adipose volume. The shared interface, external boundary, source attachments of the old nodes and constitutive rest geometry are inherited. In the source, loaded and retained endpoint checks, all local cells pass shape and signed volume. The largest relative error in pulled-back patch forces is under 0.002%. These endpoint positions replay the old accepted deformation.

The first Float32 mass map produced 530 density outliers. The corrected map routes 32.22 µg among descendants of face-adjacent parents sharing their material, region and mechanical owner. It conserves each parent's mass and first moment and meets the existing density bounds. The native importer accepted 314,559 nodes and 1,459,940 tetrahedra. The original integration owner regenerated P1 support. 1,082 child source-face tags changed. An independent check verified all 2,671 child tags and both native skin and fat carrier replays passed. Final source-face tags are bound by the SHA in native-import.json.

The full-scene viewer, bed-contact rows and suspended rigid-body state still require installation and a fresh motion trial with this discretization. No supported equilibrium, resolved shoulder corridor or real-time factor is established here. The reference scene stays pinned until the changed wall maps, muscle interfaces and motion pass together.

Run refine_shoulder_material.py, compile_refined_shoulder.py, bind_refined_material.py and verify_refined_ancestry.py on SSH Numi against the source recorded in registration.json. The patch contains local topology, the original 47 row identities, the full subdivision history and all three coordinate states.
