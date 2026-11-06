# Existing native geometry certification before full scene loading

Trial 432 stopped during source import on a non-spherical cavity boundary. Running the existing native fixed-cavity checker over all 608 proposals found exactly two failures in 0.210 seconds; the other 606 pass. Its 29 negative controls pass. The prior preparation path certified oriented boundary and volume but omitted this stronger native topology certificate.

The response screen now requires a pinned native geometry checker when emitting a proposal. It retains exact native witnesses, excludes failed cavities before cumulative force/energy/strain screening, and independently checks the emitted subset again. Malformed identity, domain or exit status is a hard failure. No native certificate, mass, physical law or acceptance bound changes.

53 targeted Python tests pass. Re-screening retained 606 cavities, including the measured worst fat cell; emitted geometry has zero native geometry-certificate failures. Full scene trial 436 and matched comparison 437 are running. This does not establish mass admission, physical stepping, motion qualification or a speedup.
