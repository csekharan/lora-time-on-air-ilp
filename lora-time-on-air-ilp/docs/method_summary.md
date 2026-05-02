# Method Summary

The computational evaluation implements a regulatory-aware extension of the
baseline LoRa-class spreading-factor assignment model.

For each node \(i\) and spreading factor \(s\), the binary decision variable
\(x_{i,s}\) indicates whether node \(i\) is assigned to \(s\). The ILP minimizes
planned uplink time-on-air:

\[
\min \sum_i \sum_s a_i(W)T(B_i,s)x_{i,s}.
\]

The formulation includes one assignment per node, physical feasibility sets,
per-SF capacity/load-management constraints, dwell-time screening, per-device
uplink duty-cycle constraints, grouped downlink activation variables, and a
gateway downlink airtime budget.

The greedy baseline processes nodes in arrival order and assigns each node to
the fastest feasible spreading factor with remaining capacity. The ILP sees the
full node set and can choose globally better assignments.
