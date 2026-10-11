# Native left sartorius precision repair

The integrated MTP0 resting scene now uses a topology-preserving precision repair of passive left sartorius stable50 (BodyParts3D FJ1434M). A diagnostic passive-toe-stiffness run exposed an existing 0.133 micrometre edge whose endpoints became identical in Float32 world coordinates. The native validator stopped that run at accepted physical step6591 (13.182 seconds); the failing configuration is retained and has not been adopted.

The existing edge-collapse owner merges that endpoint pair, deletes its two incident sliver triangles, and preserves all three closed source components and their topology. The endpoint binding weights differ by at most 1.68e-8; the surviving endpoint keeps its original map. Physical MyoSim attachment routes, articulated body bindings, forces and mass are unchanged. The source signed-volume change is 2.53e-18 m³. This is an explicit numerical-resolution repair of inferred visual geometry, not a new biological claim.

The repaired surface is closed and exactly self-clear across 30 retained poses (25 established poses plus five available pre-failure diagnostic poses). Exact parent-face/target-face comparisons introduce no new crossings. Existing interfaces remain unqualified by this check.

A fresh 16-second, 8,000-step native run with the delivered MTP0 joint configuration passes the continuous invalid-triangle/bed-crossing audit. At eight actual captures, the repaired surface retains three closed components, has no exact self-pairs and no new parent-face crossings against all 863 other surfaces. Every other surface, all 157 body poses and 1,000×60 coupled physiological values are exactly equal to the previous bilateral SCM/masseter scene. The recording contains 252 image frames and five timing markers, with seven inspection layers.

## Reproduce on the Mac mini

    python3 tools/evidence/native-anatomy-sartorius-precision-1330/reproduce.py --prepare-only
    python3 tools/evidence/native-anatomy-sartorius-precision-1330/reproduce.py

The existing native launcher and pinned retained assets/runtime are required. This increment builds on ../native-anatomy-bilateral-scm-masseters-1329; all source-rights and anatomical provenance remain inherited. The diagnostic MTP2 failure is retained separately from the successful MTP0 result. Neither the new posture nor a final five-minute complete-body run is qualified here. Head/back coverage, eight original defective muscle rows and unresolved tissue interfaces remain outstanding; performance work is deferred.
