# Continuous-walking engineering contract

`numi human walking-contract` is the hand-off boundary for a learned
lower-body walking task. It reads the pinned Rajagopal source and emits the
exact six `ground_pelvis` coordinate order, all 35 source coordinates, and the
complete ordered 80-muscle excitation action surface.

```sh
numi human walking-contract --sources Sources --output Build/rajagopal-walking-contract.json
```

The contract deliberately rejects a substituted floating base or an incomplete
muscle set. Core must execute the source `ground_pelvis` FunctionBased joint as
a mobile root; a policy action is a bounded excitation, not a direct joint
torque. The Core reference also provides a deterministic persistent activation
update with explicit time constants. Those constants must be provenance-locked
by the later source/contact task before training.

The flat-ground walking scenario is blocked until the following artifacts are
present and validated: per-foot BodyParts3D-to-Rajagopal registration,
conservative collision proxies, friction/compliance parameters, collision
exclusions, and a deterministic reset/replay scenario. It is likewise invalid
to animate an anatomical layer until every mesh has a validated segment
attachment; unregistered geometry stays hidden or static.

This contract is not a Human RobotPack, trained policy, calibrated gait model,
OpenSim parity result, or tissue-physics implementation.
