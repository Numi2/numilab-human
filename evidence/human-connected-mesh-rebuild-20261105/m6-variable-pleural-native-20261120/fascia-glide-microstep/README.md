# Refresh regional fascia glide forces on the tissue clock

The matched control and opt-in trial ran on SSH Numi, Apple M6. They reused the same registered first-touch native binary and input pack, 34 ligament rows, 26 fascia regions, condensed skin/fat bed support, variable pleural pressure and 2 ms accepted body steps. The gravity continuation schedules full gravity for step eight. Pinned launch files identify the runtime and shader for each trial.

## Observed instability

A read-only GPU trace reproduced the control’s accepted CSV bit for bit and its 8 ms rollback. It samples 38 nodes and 54 tetrahedra of pectoral region 3. At root four, node 215 received 7.5913 N from the frozen external force buffer at both microticks 0 and 60. The patch’s kinetic energy rose from 1.921 mJ to 85.675 mJ. Direct force on the four corners of the dilation witness was zero. A neighboring loaded row can drive the failing cell through its connected elastic forces.

The material determinant reached its existing 2.5 limit in the next microtick. This trace provides the load history preceding the previously published element and convergence witnesses.

## Physical change

The Metal trial evaluates the same 495 unilateral plane rows at each due tissue microstep using the current accepted FEM position. The standalone transaction still holds the outer-step master pose. A row uses its existing pressure, area, closure offset and normal. It adds only its force contribution to the FEM buffer, preserving other loads. Separate per-row scratch accumulates the opposite force impulse and its moment about the master origin. The runtime reduces the accumulated impulses into the existing articulated wrench buffer in the same command transaction. The snapshot of the generalized load then uses that averaged wrench. The setup validates the standalone, single-environment, plane-only, unique-slave domain. This is a scoped experiment on the existing owner.

## Native result

The first accepted CSV row agrees exactly with the control. The trial accepted seven body steps, reaching 14 ms, compared with four steps and 8 ms in the control. Pleural pressure varied from −490.642 to −496.190 Pa. Peak normal impulse divided by the 2 ms step corresponds to 38.77 N of physical bed support. Root assistance, root torque assistance and the reported body mass residual were zero.

The attempted eighth step rejected with the existing final equilibrium test at microtick 85. The largest residual moved to FEM velocity row 357 in pectoral fascia region 5. The equilibrium residual was 0.000400030 against the unchanged 0.0001 threshold. The accepted Brain clock stayed at 14 ms. The attempted controller time was 16 ms.

The monitored process took 210.416 s including initialization and rejection, yielding 0.0000665× real time. The first four GPU step spans sum to about 90.6 s in each version. This experiment demonstrates a longer accepted prefix and approximately unchanged cost per step. The recorded surface metrics check limited predicates. Detailed transferred skin/fat admission, whole-scene motion crossings, settled rest and the 16 s at 0.1× gate remain open.

The trial keeps its private runtime-to-adapter bridge and standalone master-pose approximation in the evidence bundle. Production admission needs a tested owner contract for the refreshed force and impulse buffers, including rollback, a dedicated impulse parity measurement, and diagnostics that distinguish the initial force audit from the integrated contact load.
