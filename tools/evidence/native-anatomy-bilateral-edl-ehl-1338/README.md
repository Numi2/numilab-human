# Bilateral EDL/EHL native anatomy increment

The existing whole-body native scene now includes the corrected left extensor digitorum longus boundary, alongside the previously integrated right EDL, both EHL surfaces and both deep masseters. This increment changes passive anatomical row22 only. All155 other tissue rows, physical mass, muscle/tendon routes, support geometry and physiology ownership are unchanged.

The existing exact boundary constructor resolves the four residual left EDL crossings at a retained pose. Lifting its result through the existing skin-binding owner preserves all2289 original union source points and adds three inferred boundary points and six net triangles. Two original mixed display maps retain the earlier bounded local correction; pure one-hot proxy maps remain unchanged. A common0.75 blend of the two actual incident ancestry maps was the lowest tested common blend that passed the source/eight-pose self checks and initial bone check;0.9 and1 also passed that limited sensitivity screen. Only0.75 received the complete865-neighbor audit.

The initial equal-map candidate failed an explicit0.5mm distance-to-each-ancestry-point guard and later created eight new initial bone pair identities. Both failures remain retained. They are not reported as passing: the selected reconstruction is explicitly inferred reference geometry, with up to1.093mm distance from individual ancestry points. Its three new points lie within0.0178mm of the original source surface; the largest sampled changed-face distance to that surface is0.116mm. These are distinct metrics, not a full Hausdorff bound. No existing source point moves. Source area changes−0.00228% and volume−0.00445%; all14 closed components remain positive.

The16-second native run completes8000 accepted steps with41 pinned inputs unchanged. All eight actual GPU captures are closed, positive in every left EDL component, and self-intersection-free. Exact native neighbor pair tables match the offline candidate at every capture. Relative to the original union mapping, there are no new strict parent interfaces among865 targets. All865 other native surfaces, all157 body poses and the1000-by60 coupled trace are byte-identical to the immediate parent. The continuous invalid-triangle/contact audit passes. The recording retains252 image frames and seven anatomical inspection layers.

Existing muscle/bone/skin/vessel overlaps remain unresolved, including2112–2630 left EDL external triangle pairs across these poses. No zero-overlap or completed-anatomy claim is made. Trapezius, temporalis, forearm and remaining interface qualification work remains, followed by the final five-minute matched run on completed anatomy. The aggregate2Nm/rad passive MTP configuration remains a declared reference sensitivity. Flat-plane contact is unchanged; performance work is deferred.

Anatomy remains a registered mixed-source reference, not measurements of one individual. BodyParts3D attribution: The Database Center for Life Science, CC BY-SA2.1 Japan. Existing Z-Anatomy assets retain CC BY-SA4.0 attribution through the scene manifest and preceding source-rights evidence.

On the SSH Mac mini with the retained exact assets:

    python3 tools/evidence/native-anatomy-bilateral-edl-ehl-1338/reproduce.py --prepare-only
    python3 tools/evidence/native-anatomy-bilateral-edl-ehl-1338/reproduce.py

Source revisions, binary identity, immutable assets, configuration, launch arguments, compact trace, actual geometry checks and continuous native recording are retained in this package or its hash-bound local references.
