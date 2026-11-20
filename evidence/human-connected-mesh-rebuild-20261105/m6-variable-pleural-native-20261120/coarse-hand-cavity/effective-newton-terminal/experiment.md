# Test with the effective reference Newton cap

Trial 043 found the requested 24-iteration native override supersedes the
cooked cap of 16. That variant actually finished at logical iteration 15.
This trial reads the declared effective cap from the pinned native launch and
uses it throughout both patched Matter owners. It reuses the same compiled
shader, physical scene, nonlinear latch, terminal certificate and rollback.
The normal runtime is unchanged.

First compare the first four accepted roots against trial 043's full run.
Require exactly matching coupled fields, zero >1% force or strain differences,
and accepted trace telemetry that reaches logical iteration 23 rather than
15. The known two-iteration rollback must remain. Only a passing startup
comparison warrants the original 90-root pair and its preregistered gates.

The four-step diagnostic also runs a fixed eight-iteration short schedule
without the retry wrapper. Keep the original effective terminal and its
latch check. Compare this run's first accepted state changes and terminal
fields against the same control. This isolates the rejected retry from the
scheduling change without changing any accepted solver tolerances.
