# Grouped contact-force dispatch in the full-volume native scene

This matched Apple M6 run integrated Numi Lab commit `948c6b909088ec6e8b5c2e1b3d8437d4cd6890b4` into the current guarded Human runtime. It dispatches contact force assembly through 179,728 node groups for 15,449,893 registered rows. The exact installed mass SHA-256, bed pose, timestep, controller, pressure coupling and certificate settings match the published physical bed-placement control. Build and launch paths and binary hashes are retained.

Both runs accepted eight 15.625 microsecond roots. All coupled CSV columns match, as do the terminal ligament and bed reaction audits. The sampled Metal allocation at root 2 fell by 58,900,480 bytes. Sampled host process footprint rose at initialization; no whole-system memory improvement is established.

The control and grouped run took 238.29 and 239.71 wall seconds including initialization, with 95.90 and 96.28 seconds summed GPU spans. Grouping saved no measurable full-scene runtime in this pair despite the faster isolated kernel benchmark. Each run used 1,552 active FGMRES columns.

This headless replay covers 0.125 ms, approximately 0.00000052 times real time including startup. The bed reaction was zero at the terminal root and no resolved pleural pressure variation occurred in this interval. Supported settling, geometry through motion, variable-pressure qualification and the 16-second performance target remain open.
