# Fit the remaining posterior interfaces jointly

The sacral arm eliminated 755 sacral and all 199 bilateral triangle pairs.
The residual bone pairs are assigned exactly to one ipsilateral ilium and
femur for each muscle. The remaining 160 shell pairs are on the deep fat.
Build side-specific unions of the conservative margins of those closed
volumes with the sacrum. Keep the tested medial partition fixed. Use the
same 0.1 mm release margin and no changes to the external surfaces or
tissue constitutive laws.

Success for this geometry arm requires zero original opposing glute, skin/fat
and bone triangle pairs after Float32 emission. Use all three unchanged
bone bank inventories and the dense shell boundary. Check against other
muscles and FEM surfaces and keep any new pairs with their source locations;
a topology that introduces them cannot enter native dynamics. A runtime
replacement still requires positive tetrahedron quality, ancestry,
conservative per-muscle mass routing and a closed load-path test.
