# Whole-body bone registration at physical release

The native scene on SSH Numi contains 185 closed bone surfaces. The audit compares all 17,020 pairs in the original source atlas, the compiled rigid rest frame, and the rendered release frame. Each rigid transform was independently reconstructed from NHBONES1 and the native body pose. The largest difference against the emitted bone vertices is 0.411 µm. Bone source inputs and native pose/pack hashes are retained in the JSON.

At release, 78 pairs on distinct rigid owners and 84 pairs on shared owners exceed the reporting floor of 10⁻⁹ mm³. Shared-owner overlap is recorded separately because compound components can intentionally coincide. The anatomical meaning of each small pair requires its own source review. Selected cross-owner intersections give the scale and locate the error in the registration.

| Bone surfaces | Source atlas, mm³ | Compiled rest, mm³ | Supported release, mm³ |
| --- | ---: | ---: | ---: |
| Left scapula and humerus | 0 | 5,149.88 | 5,128.55 |
| Right scapula and humerus | 0 | 4,974.85 | 4,998.03 |
| Left capitate and third metacarpal | 0 | 476.00 | 475.99 |
| Right capitate and third metacarpal | 0 | 550.65 | 550.64 |

The source checks use volumetric intersection on closed surfaces after identifying coincident seam coordinates. The underlying MyoSim geometric reference and the BodyParts3D atlas are separate sources. Native force attribution requires examining the active collision and attachment owners as well. This audit measures geometry.

The retained earlier four-bone shoulder registration cleared the changed skeletal interfaces across 254 sampled poses. It also introduced external-skin intersections at the scapulae and clavicles. The later clavicle-only candidate changed banks 8 and 9. Neither result qualifies the bilateral glenohumeral joints in the present complete material scene.

The certified wrist material is kept as a candidate. The previous node-to-layer map has 23 inner-skin and 52 deep-fat vertices that disappear in its new topology. At the newly exposed cortex, 54 dermal boundary triangles and 156 fat boundary triangles meet along 54 common vertices. The existing native O/I/D contract expects exactly one skin and one fat cell at each bonded inner face. These cortical contact facets need explicit ownership if this topology is used. Labeling them as head walls or pinning the removed nodes would change the physical model.

The priority is a coupled skeletal registration with the actual joint centers, followed by matching tendon and ligament footprints, muscle paths and local tissue boundaries in the same frames. Test all nearby structures at rest and through the same retained poses before a long native run. The current full-scene control remains available to reproduce the failures.

Run the check with the published native pack reader and source archive available on Numi.

```sh
python audit_registered_skeleton.py "$native_control" "$pack_reader" "$source_atlas" skeleton.json --require-clear-separate-owners
```

The command retains the complete pair table before returning a failing status for this scene. The adjacent left-scaphoid material report links the wrist import and its independently restored recovery archives.
