# Split the limiting interior skull edge

The previous whole-skull fit recorded parent fat cell 26915 in 59
limiting-cell measurements. Four neighboring cavity flips passed their
orientation and volume screens but did not remove that parent. Several
flips touching it keep the unsigned boundary face set and appear to
improve shape while multiplying cavity volume. The signed boundary check
rejects those flips.

The limiting cell contains a 124.575 mm edge between original compiled
nodes 3263 and 5829. This edge belongs to six fat cells and no outer,
interface, deep or sidewall triangle. Splitting at 0.4 along the edge
preserves the complete shell boundary and each child's original parent.
Six cells become twelve. The minimum shape in the cavity improves by
18.27% in both the compiled rest and Float32 native geometry. The shell
candidate has 8,344 nodes and 32,992 cells. The verification checks both
volumes and the exact added vertex ancestry independently. The interior
vertex was held during the offline surface fit. No runtime node, force,
constraint or support was changed.

With this split, six of eight line searches accept a nonzero movement.
The final mass map balances at 13.939679838734493 kg. The longest
independently measured route to an actual receiving child is 68.21 mm,
within the existing 75 mm bound. The per-source mass closure is within
8.674e−19 kg. Both geometry frames have positive cells, and the total
initial pair inventory after the topology-only change remains identical
to all 17,878 native baseline pairs.

The resulting fit removes 3,309 pairs and introduces 2,737 different
pairs in the same bone bank, leaving 17,306. Skull outer crossings
are 702→500, interface 2,311→1,526 and deep 2,685→3,100. The fit
adds no external-bank or self intersections. Some columns exceed
original thickness limits. The candidate fails the intended deep
clearance and was not run natively. The geometric improvement and the
mass pass do not qualify shoulder loading, equilibrium, variable-pressure
motion or performance. A subsequent fit should constrain the deep
surface against actual bone faces instead of only the radial target.

## Reproduction

On SSH Numi, from the original thickness-rebuild workspace, create
`edge-cavity-035` and copy the two changed scripts plus
`fit_layers.py` and `regional_mass.py` from the parent experiment.
Apply `interior-edge.patch` inside that directory with `patch -p0`.
Use the pinned reference and prior audit workspace described by the
parent experiment.

```sh
python3 edge-cavity-035/rebuild.py edge-cavity-035/first-combined --whole-skull --all-edges --interior-edge 3263 5829
python3 edge-cavity-035/verify_rebuild.py edge-cavity-035/first-combined
```

The exact candidate and its mass routes have been recovered from the
private archive and verified against their manifest.
