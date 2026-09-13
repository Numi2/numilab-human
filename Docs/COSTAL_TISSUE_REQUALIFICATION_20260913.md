# Costal tissue native requalification — 2026-09-13

The current isolated native branch was rebuilt and the registered costal cartilage path was rerun against the exact binding, cartilage, rigid-body, and registration inputs. The run passes the source-bound tissue mass partition and COM-frame rebase on Apple M4 Pro.

Measured result:

- 13,516 cooked FEM nodes, 46,278 tetrahedra, and 2,871 attachments.
- 0.11369939548001184 kg tissue mass removed from torso body 20.
- 64 rigid-motion mass/inertia cases and 11 negative mass admissions pass.
- Eight CPU rebase pose cases and eight Metal replay cases pass.
- Maximum Metal point error: 5.079121876416792e-7 m.
- Maximum Metal Jacobian error: 5.995823041393677e-7.
- Maximum isolated donor mass-matrix scaled error: 1.4901161193847656e-8.

The native source is branch human-blood-mass-20260913 at commit dfd3696444a8badc65e140b04b95f9fe395054df. The exact log and immutable receipt are retained under Docs/media/tissue-ownership-20260908/.

This is a registered tissue/mass and frame-preservation gate. The material remains an uncalibrated population prior; loaded thorax convergence, independent rib/sternum articulation, whole-body dynamic mass matrix, and standing/recovery/walking remain unqualified.
