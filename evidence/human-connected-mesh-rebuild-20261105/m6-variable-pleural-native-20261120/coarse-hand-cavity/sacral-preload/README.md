# Fixed-bed load distribution and skin/fat preload

This increment uses the registered posterior shell and current native geometry
from [the load-transfer replay](../load-transfer/README.md) on SSH Numi,
Apple M6. Its offline authoring trials keep the native bed law, static
material energy, typed inertia, shape thresholds and unmodified force
residual. Its deep boundary has prescribed normal motion in a floating
frame. Position, pitch and roll are linearized around the reference pose.
The existing native runtime remains the sole dynamic owner.

## The initial load is concentrated

The emitted outer layer has three nodes within 1 nm of the first-contact
activation distance. At the actual initialized plane the contact law is
unloaded. The center-of-mass projection lies within the triangle formed by
the three nodes. Balancing 952.86 N of gravity and both horizontal moments
using only that triangle would require 169.32, 243.75 and 539.79 N at the
respective nodes. The largest force corresponds to about 18.9 MPa when
divided by its measured downward projected tributary area of 28.5 mm².
This is a concentrated-force estimate for the given geometry.

With an illustrative cap of 25 kPa on each projected tributary area, the
minimum force-weighted initial gap rises to 10.38 mm across 225 selected
nodes. At 50 kPa it is 5.41 mm across 172. These are geometry and moment
bounds at assumed engineering caps. The LP permits arbitrary independent
force distribution. It cannot furnish actual equilibrium or establish
pressure limits for a person.

An independent anatomical screen examined 162 deep-face nodes over the
dorsal sacrum. Cadaveric dissection documents a joined dermis, superficial
fascia and periosteum over that bone. In the registered reference mesh the
shortest geometric path passing the segment and normal tests was 6.48 mm,
and its corresponding outer point sits 19.88 mm above first touch. No
periosteal row was added. [Han et al., Clinical Anatomy](https://doi.org/10.1002/ca.23988)

## Matched preload trials

| Trial | Last converged load | First rejected load |
| --- | ---: | ---: |
| Normal-only displacement, first-touch seed | None | 1% |
| Full lateral freedom above the deep boundary, first-touch seed | None | 1% |
| Normal-only displacement, penalty-initialized | 1% | 2.5% |
| 45 mm lateral strip, penalty-initialized | 2.5% | 5% |
| Lateral strip and descending search fallback | 2.5% | 5% |
| Lateral strip and tangential deep slip, retaining all 14 attachment sites | 2.5% | 5% |

The temporary penalty provides an initial iterate. Accepted stages
re-equilibrate using the original IPC law. Adding positive diagonal shifts
when Newton's direction ascends the energy improves direction handling
without raising the attainable load. Introducing tangential motion of
selected deep nodes improves the minimum Float32 determinant ratio from
0.36481 to 0.59342 at the same 23.822 N and raises active skin sites from
85 to 101. The deep-slip state still has zero nonpositive or below-floor
cells and zero shell self- or cross-layer pairs at that load.

Bone checks on all changed faces record nine new and eleven removed pairs,
with the original skeletal frame held common. Two of the new pairs involve
the outer layer and seven the bonded interface. A lower net count does not
admit the candidate. The expansion of adipose cell 21848 approaches the
existing upper volume ratio at 5% gravity, and that stage also fails the
force residual. The initial shape and retained replay are unchanged.

The results establish a distributed contact and volume-strain target for
further authoring. They do not qualify full-body settling, loaded shoulder
behavior, zero crossings in motion, or any increase in native throughput.
The full 16-second variable-pressure test and 1× goal remain open.

## Reproduce

Use the existing authoring environment on SSH Numi. Set `T` to
`coarse-hand-cavity-024` and `W` to its parent. Supply the pinned native
capture, exact Float32 bone assets, fit and mapped mass from the preceding
replay. Expose the retained geometry tools and
`loaded-shell-seed-039` on `PYTHONPATH`. The plot-free receipt includes
every reported result, the terminal failed shape, and the saved
last-equilibrated states in the existing NPZ format.

```sh
P="$T/posterior-muscle-interface-045/paired-native-128"
D="$T/load-transfer-capacity-048"
python3 audit_vertical_support.py "$P" support-hull-reproduced.json
python3 support_pressure_sensitivity.py "$P" support-hull-01.json pressure-reproduced.json
python3 condition_preload.py "$T/loaded-shell-seed-039/predict.py" sources
patch -o sources/lateral.py "$T/loaded-shell-seed-039/lateral.py" descent-fallback/tangential-deep.patch
PYTHONPATH="$PWD/sources:$PYTHONPATH" python3 predict_distributed_preload.py \
  "$P" "$D/engagement-fit-02" support-hull-01.json "$D/native-engagement-build-02" repeated \
  --warm 1000000 --descent-fallback --lateral-strip-m 0.045 --deep-tangential
python3 audit_equilibrated_skin.py "$P" "$D/engagement-fit-02/posterior-interface-fit.npz" \
  repeated replay-audit.json --bone "$T/thickness-rebuild-034/inputs/physical-bone-initial.bin"
```
