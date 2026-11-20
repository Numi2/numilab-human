# Native regional FGMRES comparison

Use three paired runs at identical 2 ms root timestep, initial pack, 26 fascia regions, two ACL objects, registered ligament rows and variable pleural pressure. Compare the current solve with the existing distributed CGS2, indirect CGS2 and distributed global measurement path together. Evaluate the first four requested roots. The prediction is at least a 20% median reduction in native GPU command time over the same accepted roots, with no rejections and matched output within the retained 1% comparison tolerance. Keep the same mechanics certificate and solver budgets. Run one GPU case at a time, with one allowed initialization failure per variant. These are release-transient comparisons and cannot establish settled or full-tissue performance.

