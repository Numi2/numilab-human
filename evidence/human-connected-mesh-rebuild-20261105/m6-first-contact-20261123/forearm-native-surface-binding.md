# Physically registered skin/fat boundaries

The rebuilt 1,450,187-cell forearm mesh retains its skin and fat mass. Its installed registration inputs still described earlier boundary coordinates, which differed by up to 3.10 mm. The repair updates the three registered surface tables and each changed historical parent residual. It preserves donor identity, source vertex and face identities, source barycentric coordinates, physical position and the common skin/fat interface.

| Physical surface | Registered vertices | Changed coordinates | Recomputed normals | Largest position correction |
|---|---:|---:|---:|---:|
| Outer skin | 76,153 | 3,726 | 4,079 | 3.096 mm |
| Skin/fat interface | 118,196 | 2,832 | 3,258 | 3.059 mm |
| Deep fat | 103,505 | 804 | 978 | 3.040 mm |

Every emitted world coordinate matches the repaired TetGen node. All donor-local coordinates reconstruct the target within 1.12e-16 m with the recorded donor transform. These are numerical registration errors within the model. The largest revised parent distance is 14.88 mm against the existing 20 mm limit. [Coordinates and normals](forearm-registered-layers.json) · [Parent mappings](forearm-registered-parents.json)

On SSH Numi, the same C++ surface reader used in the native scene loaded the reference and regenerated files against the captured native Float32 pose. Both passed with 297,854 nodes, 596,541 faces and 86 donor frames. The check reads surface source ancestry, reciprocal layer identity, frame transforms, hash pins and parent registration. It does not integrate the material in time. The rebuilt adult still needs a joint volume map covering its 447 new interior vertices, complete conservative mass-route installation, and motion testing under gravity and bed contact. The separate 16-second compact run and four-step dense run use different material owners. [Production reader result](forearm-registered-layers-native.json) · [Exact inputs and independently verified recovery](forearm-registered-layers-recovery.json) · [Native shoulder results and remaining physics](shoulder-physical-motion.md)
