# Descent fallback after the invalid Newton direction

Both penalty-initialized controls rejected at a finite positive shape and
matching total reaction, due to an ascending Newton update. Test matched
controls with a diagonal modification of the Newton search metric whenever
the unmodified solve has a singular or non-descending direction. A qualifying
result must reduce the original force and moment residual below the same
threshold with the original determinant and IPC constraints. Explicitly save
the last nonlinear state meeting that criterion. The prediction is that at
least one variant will accept a load level that its corresponding control
could not reach. This offline authoring comparison does not change the
constitutive law, applied loads, or the native solver.
