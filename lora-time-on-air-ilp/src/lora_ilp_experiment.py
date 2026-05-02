
import os, sys, math, time, json, platform, subprocess, random
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import scipy

try:
    import psutil
except Exception:
    psutil = None


# -----------------------------
# LoRa physical-layer utilities
# -----------------------------
def lora_time_on_air_seconds(
    payload_bytes,
    sf,
    bw_hz=125_000,
    coding_rate=1,     # CR = 1 means 4/5 in Semtech formula
    preamble_symbols=8,
    explicit_header=True,
    crc=True,
    low_data_rate_opt=None,
):
    """
    Semtech-style LoRa packet time-on-air formula.

    coding_rate: Semtech denominator offset, 1..4 for 4/5..4/8.
    explicit_header=True means IH=0 in Semtech formula.
    crc=True means CRC=1.
    low_data_rate_opt defaults to 1 for SF>=11 at 125 kHz, else 0.
    """
    sf = int(sf)
    payload_bytes = int(payload_bytes)

    if low_data_rate_opt is None:
        low_data_rate_opt = 1 if (sf >= 11 and bw_hz == 125_000) else 0

    ih = 0 if explicit_header else 1
    crc_val = 1 if crc else 0
    de = int(low_data_rate_opt)

    t_sym = (2 ** sf) / bw_hz
    t_preamble = (preamble_symbols + 4.25) * t_sym

    numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * crc_val - 20 * ih
    denominator = 4 * (sf - 2 * de)
    payload_symb_nb = 8 + max(math.ceil(numerator / denominator) * (coding_rate + 4), 0)
    t_payload = payload_symb_nb * t_sym

    return t_preamble + t_payload


def generate_payloads(n, rng):
    """
    Realistic small IoT packet mixture:
      70%: 8--20 bytes
      25%: 21--40 bytes
       5%: 41--51 bytes
    """
    u = rng.random(n)
    payload = np.zeros(n, dtype=int)
    mask1 = u < 0.70
    mask2 = (u >= 0.70) & (u < 0.95)
    mask3 = u >= 0.95
    payload[mask1] = rng.integers(8, 21, size=mask1.sum())
    payload[mask2] = rng.integers(21, 41, size=mask2.sum())
    payload[mask3] = rng.integers(41, 52, size=mask3.sum())
    return payload


def build_synthetic_instance(n=100, seed=7, sf_list=(7,8,9,10,11,12)):
    """
    Generates a realistic LoRa-class instance.

    Feasible SFs are built from:
      - distance-dependent path loss,
      - lognormal shadowing,
      - typical SF sensitivity thresholds at 125 kHz,
      - a link margin.
    """
    rng = np.random.default_rng(seed)
    sf_list = list(sf_list)
    S = len(sf_list)

    # Distances: nodes uniformly distributed over disk, with a small inner guard radius.
    # Radius selected to keep most generated instances feasible for 125 kHz LoRa at 14 dBm.
    r_min_km = 0.05
    r_max_km = 2.6
    u = rng.random(n)
    distances_km = np.sqrt(u * (r_max_km**2 - r_min_km**2) + r_min_km**2)

    payload = generate_payloads(n, rng)

    # Link-budget model.
    tx_power_dbm = 14.0
    freq_mhz = 915.0
    # Free-space path loss at 1 meter:
    pl_1m_db = 32.44 + 20*np.log10(freq_mhz) + 20*np.log10(0.001)
    path_loss_exp = 3.05
    shadow_sigma_db = 5.0
    shadow_db = rng.normal(0.0, shadow_sigma_db, size=n)
    d_m = distances_km * 1000.0
    path_loss_db = pl_1m_db + 10 * path_loss_exp * np.log10(d_m / 1.0) - shadow_db
    rx_power_dbm = tx_power_dbm - path_loss_db

    # Typical LoRa sensitivities at BW=125 kHz; exact values vary by radio/configuration.
    sensitivity_dbm = {
        7: -123.0,
        8: -126.0,
        9: -129.0,
        10: -132.0,
        11: -134.5,
        12: -137.0,
    }
    link_margin_db = 6.0

    # Regulatory/statutory profile for experiment.
    # Use a per-device duty-cycle budget. With W=3600 and 1%, this is 36 seconds/device.
    W_seconds = 3600.0
    U_dev_seconds = 0.01 * W_seconds

    # Dwell/channel-occupancy screen. We set it high enough not to create artificial
    # infeasibility in the main experiment while retaining the constraint in the model.
    # A tighter region-specific dwell setting can be substituted here.
    dwell_max_seconds = 4.0

    # Gateway downlink grouped-control budget.
    # Large enough to permit grouped downlink signaling in the primary experiment.
    D_gw_seconds = 36.0
    H0_bytes = 12

    # Planned and max packets in the window.
    # For an initialization-like episode, expected transmissions are small.
    a_expected = rng.choice([1, 2], size=n, p=[0.85, 0.15]).astype(float)
    a_max = a_expected + 1.0  # one retransmission allowance

    # Time-on-air matrix.
    toa = np.zeros((n, S))
    for i in range(n):
        for j, sf in enumerate(sf_list):
            toa[i, j] = lora_time_on_air_seconds(payload[i], sf)

    # Feasibility matrix.
    feasible = np.zeros((n, S), dtype=bool)
    for i in range(n):
        for j, sf in enumerate(sf_list):
            link_ok = rx_power_dbm[i] >= (sensitivity_dbm[sf] + link_margin_db)
            dwell_ok = toa[i, j] <= dwell_max_seconds
            duty_ok = a_max[i] * toa[i, j] <= U_dev_seconds
            feasible[i, j] = link_ok and dwell_ok and duty_ok

    # If any node is infeasible, soften by moving it slightly closer in repeated generation
    # is better than silently forcing feasibility. Here, signal to caller to resample.
    if not np.all(feasible.any(axis=1)):
        return None

    # Baseline capacity/load-management constraints retained for the experiment.
    # These are not statutory duty-cycle constraints; they are Section 3 occupancy/load bounds.
    # The pattern intentionally makes lower SFs scarce, creating a meaningful assignment problem.
    cap_fracs = np.array([0.22, 0.22, 0.19, 0.16, 0.12, 0.10])
    capacities = np.maximum(1, np.floor(cap_fracs * n).astype(int))

    # Ensure total slots at least n.
    while capacities.sum() < n:
        capacities[np.argmax(cap_fracs)] += 1

    # Feasibility-preserving capacity adjustment for nested LoRa feasible sets.
    # If nodes whose minimum feasible SF is at or above a threshold exceed the cumulative
    # capacity above that threshold, we add slots at that threshold. This prevents
    # synthetic capacity choices from making an otherwise physically feasible network impossible.
    min_feasible_idx = np.array([np.where(feasible[i])[0].min() for i in range(n)])
    for k in range(S-1, -1, -1):
        demand_suffix = int(np.sum(min_feasible_idx >= k))
        cap_suffix = int(np.sum(capacities[k:]))
        if cap_suffix < demand_suffix:
            capacities[k] += (demand_suffix - cap_suffix)

    return {
        "n": n,
        "seed": seed,
        "sf_list": sf_list,
        "S": S,
        "distances_km": distances_km,
        "payload_bytes": payload,
        "rx_power_dbm": rx_power_dbm,
        "toa": toa,
        "feasible": feasible,
        "a_expected": a_expected,
        "a_max": a_max,
        "W_seconds": W_seconds,
        "U_dev_seconds": U_dev_seconds,
        "dwell_max_seconds": dwell_max_seconds,
        "D_gw_seconds": D_gw_seconds,
        "H0_bytes": H0_bytes,
        "capacities": capacities,
    }


def make_instance_with_retry(n, seed_start=7, max_tries=200):
    for offset in range(max_tries):
        inst = build_synthetic_instance(n=n, seed=seed_start + offset)
        if inst is not None:
            return inst
    raise RuntimeError(f"Could not generate feasible instance for n={n} after {max_tries} tries")


# -----------------------------
# Greedy baseline
# -----------------------------
def greedy_assignment(inst, arrival_seed=100):
    rng = np.random.default_rng(arrival_seed)
    n, S = inst["n"], inst["S"]
    sf_list = inst["sf_list"]
    feasible = inst["feasible"]
    capacities = inst["capacities"].copy()
    toa = inst["toa"]
    a_expected = inst["a_expected"]

    remaining = capacities.copy()
    assigned = np.full(n, -1, dtype=int)
    order = rng.permutation(n)

    for i in order:
        # Fastest feasible SF with remaining capacity.
        for j, sf in enumerate(sf_list):
            if feasible[i, j] and remaining[j] > 0:
                assigned[i] = j
                remaining[j] -= 1
                break

    feasible_all = np.all(assigned >= 0)
    total_toa = float(np.sum([a_expected[i] * toa[i, assigned[i]] for i in range(n) if assigned[i] >= 0]))
    return {
        "assigned_idx": assigned,
        "feasible_all": feasible_all,
        "total_toa": total_toa,
        "remaining_capacity": remaining,
        "order": order,
    }


# -----------------------------
# MILP solver
# -----------------------------
def solve_ilp(inst, time_limit_seconds=60.0, mip_rel_gap=0.0):
    n, S = inst["n"], inst["S"]
    toa = inst["toa"]
    feasible = inst["feasible"]
    a_expected = inst["a_expected"]
    a_max = inst["a_max"]
    U_dev = inst["U_dev_seconds"]
    D_gw = inst["D_gw_seconds"]
    capacities = inst["capacities"]
    H0 = inst["H0_bytes"]
    sf_list = inst["sf_list"]

    num_x = n * S
    num_y = S
    num_vars = num_x + num_y

    def x_index(i, j):
        return i*S + j
    def y_index(j):
        return num_x + j

    c = np.zeros(num_vars)
    for i in range(n):
        for j in range(S):
            c[x_index(i,j)] = a_expected[i] * toa[i,j]

    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)

    # Feasibility, dwell, and duty screens via upper bound 0 for infeasible x.
    for i in range(n):
        for j in range(S):
            if not feasible[i, j]:
                ub[x_index(i,j)] = 0.0

    constraints = []
    lower = []
    upper = []

    # Number of constraints:
    # assignment n; capacity S; per-device duty n; x<=y n*S; y<=sum S; gateway downlink 1
    m = n + S + n + n*S + S + 1
    A = lil_matrix((m, num_vars), dtype=float)
    row = 0

    # Assignment: sum_s x_i,s = 1
    for i in range(n):
        for j in range(S):
            A[row, x_index(i,j)] = 1.0
        lower.append(1.0); upper.append(1.0)
        row += 1

    # Baseline capacity/load-management constraints: sum_i x_i,s <= C_s
    for j in range(S):
        for i in range(n):
            A[row, x_index(i,j)] = 1.0
        lower.append(-np.inf); upper.append(float(capacities[j]))
        row += 1

    # Per-device duty-cycle constraints: sum_s amax_i * ToA_i,s * x_i,s <= U_dev
    # With one abstract band, this is per node.
    for i in range(n):
        for j in range(S):
            A[row, x_index(i,j)] = a_max[i] * toa[i,j]
        lower.append(-np.inf); upper.append(float(U_dev))
        row += 1

    # x_i,s <= y_s
    for i in range(n):
        for j in range(S):
            A[row, x_index(i,j)] = 1.0
            A[row, y_index(j)] = -1.0
            lower.append(-np.inf); upper.append(0.0)
            row += 1

    # y_s <= sum_i x_i,s  => y_s - sum_i x_i,s <= 0
    for j in range(S):
        A[row, y_index(j)] = 1.0
        for i in range(n):
            A[row, x_index(i,j)] = -1.0
        lower.append(-np.inf); upper.append(0.0)
        row += 1

    # Gateway grouped downlink budget: sum_s T(H0,s)y_s <= D_gw
    for j, sf in enumerate(sf_list):
        A[row, y_index(j)] = lora_time_on_air_seconds(H0, sf)
    lower.append(-np.inf); upper.append(float(D_gw))
    row += 1

    assert row == m

    lc = LinearConstraint(A.tocsr(), np.array(lower), np.array(upper))
    bounds = Bounds(lb, ub)
    integrality = np.ones(num_vars)

    start = time.perf_counter()
    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=lc,
        options={"time_limit": time_limit_seconds, "mip_rel_gap": mip_rel_gap, "disp": False},
    )
    elapsed = time.perf_counter() - start

    assigned = np.full(n, -1, dtype=int)
    y = np.zeros(S, dtype=int)

    if res.success and res.x is not None:
        x = res.x[:num_x].reshape((n,S))
        for i in range(n):
            assigned[i] = int(np.argmax(x[i]))
        y = (res.x[num_x:] > 0.5).astype(int)

    total_toa = float(res.fun) if res.fun is not None else np.nan
    return {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "objective": total_toa,
        "assigned_idx": assigned,
        "active_y": y,
        "solve_time_seconds": elapsed,
        "num_variables": num_vars,
        "num_constraints": m,
    }


def assignment_dataframe(inst, greedy, ilp):
    n = inst["n"]
    sf_list = inst["sf_list"]
    rows = []
    for i in range(n):
        gi = greedy["assigned_idx"][i]
        ii = ilp["assigned_idx"][i]
        rows.append({
            "node": i+1,
            "distance_km": inst["distances_km"][i],
            "payload_bytes": int(inst["payload_bytes"][i]),
            "rx_power_dbm": inst["rx_power_dbm"][i],
            "a_expected": inst["a_expected"][i],
            "a_max": inst["a_max"][i],
            "greedy_sf": int(sf_list[gi]) if gi >= 0 else None,
            "ilp_sf": int(sf_list[ii]) if ii >= 0 else None,
            "greedy_toa_s": inst["toa"][i, gi] if gi >= 0 else np.nan,
            "ilp_toa_s": inst["toa"][i, ii] if ii >= 0 else np.nan,
        })
    return pd.DataFrame(rows)


# -----------------------------
# System information
# -----------------------------
def get_cpu_model():
    # Linux-friendly CPU model extraction.
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"

def get_system_info():
    info = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "os": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "processor": get_cpu_model(),
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "memory_total_gb": None,
        "scipy_version": scipy.__version__,
        "milp_solver": "scipy.optimize.milp using HiGHS backend",
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    if psutil is not None:
        try:
            info["physical_cores"] = psutil.cpu_count(logical=False)
            info["logical_cores"] = psutil.cpu_count(logical=True)
            info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        except Exception:
            pass
    return info


# -----------------------------
# Plotting
# -----------------------------
def plot_total_toa(outdir, greedy_total, ilp_total):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["Greedy", "ILP"]
    values = [greedy_total, ilp_total]
    ax.bar(labels, values)
    ax.set_ylabel("Total planned uplink ToA (s)")
    ax.set_title("Total Uplink ToA: Greedy vs ILP")
    improvement = 100.0 * (greedy_total - ilp_total) / greedy_total if greedy_total > 0 else 0
    ax.text(0.5, max(values)*0.94, f"ILP improvement: {improvement:.1f}%", ha="center")
    fig.tight_layout()
    path = outdir / "fig_total_toa_greedy_vs_ilp.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_sf_distribution(outdir, inst, greedy, ilp):
    sf_list = inst["sf_list"]
    S = len(sf_list)
    greedy_counts = np.array([(greedy["assigned_idx"] == j).sum() for j in range(S)])
    ilp_counts = np.array([(ilp["assigned_idx"] == j).sum() for j in range(S)])
    x = np.arange(S)
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - width/2, greedy_counts, width, label="Greedy")
    ax.bar(x + width/2, ilp_counts, width, label="ILP")
    ax.set_xticks(x)
    ax.set_xticklabels([f"SF{sf}" for sf in sf_list])
    ax.set_ylabel("Number of nodes")
    ax.set_title("Spreading-Factor Assignment Distribution")
    ax.legend()
    fig.tight_layout()
    path = outdir / "fig_sf_distribution.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_distance_vs_sf(outdir, inst, ilp):
    sf_list = inst["sf_list"]
    assigned_sf = np.array([sf_list[j] for j in ilp["assigned_idx"]])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.scatter(inst["distances_km"], assigned_sf, s=24, alpha=0.8)
    ax.set_xlabel("Distance from gateway (km)")
    ax.set_ylabel("Assigned spreading factor")
    ax.set_yticks(sf_list)
    ax.set_title("Distance vs Assigned SF under ILP")
    fig.tight_layout()
    path = outdir / "fig_distance_vs_assigned_sf.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def plot_solver_scaling(outdir, scaling_df):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(scaling_df["N"], scaling_df["solve_time_seconds"], marker="o")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("ILP solve time (s)")
    ax.set_title("Solver Runtime Scaling")
    fig.tight_layout()
    path = outdir / "fig_solver_scaling.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# -----------------------------
# Main experiment runner
# -----------------------------
def run_experiment(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    system_info = get_system_info()
    with open(outdir / "system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)

    # Primary instance for figures 1--3.
    inst = make_instance_with_retry(n=100, seed_start=42)
    greedy = greedy_assignment(inst, arrival_seed=2026)

    # Make sure the greedy baseline gives a complete assignment for a fair comparison.
    # If not, resample a few times.
    if not greedy["feasible_all"]:
        for s in range(43, 120):
            inst = make_instance_with_retry(n=100, seed_start=s)
            greedy = greedy_assignment(inst, arrival_seed=2026)
            if greedy["feasible_all"]:
                break

    ilp = solve_ilp(inst, time_limit_seconds=60.0)
    if not ilp["success"]:
        raise RuntimeError(f"Primary ILP failed: {ilp['message']}")

    assign_df = assignment_dataframe(inst, greedy, ilp)
    assign_df.to_csv(outdir / "results_assignments_primary.csv", index=False)

    greedy_total = greedy["total_toa"]
    ilp_total = ilp["objective"]
    improvement_pct = 100.0 * (greedy_total - ilp_total) / greedy_total

    # Scaling experiment.
    scaling_rows = []
    for idx, n in enumerate([25, 50, 100, 200, 500]):
        inst_n = make_instance_with_retry(n=n, seed_start=500 + idx*100)
        ilp_n = solve_ilp(inst_n, time_limit_seconds=90.0)
        scaling_rows.append({
            "N": n,
            "success": ilp_n["success"],
            "status": ilp_n["status"],
            "message": ilp_n["message"],
            "objective_s": ilp_n["objective"],
            "solve_time_seconds": ilp_n["solve_time_seconds"],
            "num_variables": ilp_n["num_variables"],
            "num_constraints": ilp_n["num_constraints"],
        })
    scaling_df = pd.DataFrame(scaling_rows)
    scaling_df.to_csv(outdir / "summary_solver_scaling.csv", index=False)

    # Summary metrics.
    sf_list = inst["sf_list"]
    summary = {
        "primary_N": inst["n"],
        "primary_seed": inst["seed"],
        "sf_list": sf_list,
        "bandwidth_hz": 125000,
        "window_seconds": inst["W_seconds"],
        "per_device_duty_budget_seconds": inst["U_dev_seconds"],
        "dwell_screen_seconds": inst["dwell_max_seconds"],
        "gateway_downlink_budget_seconds": inst["D_gw_seconds"],
        "grouped_control_payload_bytes": inst["H0_bytes"],
        "capacity_by_sf": {f"SF{sf_list[j]}": int(inst["capacities"][j]) for j in range(len(sf_list))},
        "greedy_complete_assignment": bool(greedy["feasible_all"]),
        "greedy_total_toa_seconds": greedy_total,
        "ilp_total_toa_seconds": ilp_total,
        "ilp_improvement_percent": improvement_pct,
        "ilp_solve_time_seconds": ilp["solve_time_seconds"],
        "ilp_num_variables": ilp["num_variables"],
        "ilp_num_constraints": ilp["num_constraints"],
        "ilp_solver_message": ilp["message"],
        "greedy_sf_counts": {f"SF{sf_list[j]}": int((greedy["assigned_idx"] == j).sum()) for j in range(len(sf_list))},
        "ilp_sf_counts": {f"SF{sf_list[j]}": int((ilp["assigned_idx"] == j).sum()) for j in range(len(sf_list))},
        "system_info": system_info,
    }
    with open(outdir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable reproducibility report.
    report = []
    report.append("LoRa ILP Computational Experiment - Reproducibility Report")
    report.append("="*64)
    report.append("")
    report.append("System / execution environment")
    for k, v in system_info.items():
        report.append(f"- {k}: {v}")
    report.append("")
    report.append("Primary experiment settings")
    report.append(f"- N: {inst['n']}")
    report.append(f"- SF set: {sf_list}")
    report.append("- Bandwidth: 125 kHz")
    report.append(f"- Payload distribution: 70% 8-20 bytes, 25% 21-40 bytes, 5% 41-51 bytes")
    report.append(f"- Window W: {inst['W_seconds']} s")
    report.append(f"- Per-device duty-cycle budget: {inst['U_dev_seconds']} s")
    report.append(f"- Dwell/channel-occupancy screen: {inst['dwell_max_seconds']} s")
    report.append(f"- Gateway grouped downlink budget: {inst['D_gw_seconds']} s")
    report.append(f"- Grouped control payload H0: {inst['H0_bytes']} bytes")
    report.append(f"- Capacity/load-management slots by SF: {summary['capacity_by_sf']}")
    report.append("")
    report.append("Primary results")
    report.append(f"- Greedy complete assignment: {greedy['feasible_all']}")
    report.append(f"- Greedy total planned uplink ToA: {greedy_total:.6f} s")
    report.append(f"- ILP total planned uplink ToA: {ilp_total:.6f} s")
    report.append(f"- ILP improvement over greedy: {improvement_pct:.2f}%")
    report.append(f"- ILP solve time: {ilp['solve_time_seconds']:.6f} s")
    report.append(f"- ILP variables: {ilp['num_variables']}")
    report.append(f"- ILP constraints: {ilp['num_constraints']}")
    report.append(f"- ILP solver message: {ilp['message']}")
    report.append("")
    report.append("Scaling summary")
    report.append(scaling_df.to_string(index=False))
    (outdir / "reproducibility_report.txt").write_text("\n".join(report))

    # Plots.
    plot_paths = []
    plot_paths.append(plot_total_toa(outdir, greedy_total, ilp_total))
    plot_paths.append(plot_sf_distribution(outdir, inst, greedy, ilp))
    plot_paths.append(plot_distance_vs_sf(outdir, inst, ilp))
    plot_paths.append(plot_solver_scaling(outdir, scaling_df))

    return summary, scaling_df, plot_paths


if __name__ == "__main__":
    outdir = Path("/mnt/data/lora_ilp_experiment_outputs")
    summary, scaling_df, plot_paths = run_experiment(outdir)
    print(json.dumps({
        "output_directory": str(outdir),
        "summary": {
            "greedy_total_toa_seconds": summary["greedy_total_toa_seconds"],
            "ilp_total_toa_seconds": summary["ilp_total_toa_seconds"],
            "ilp_improvement_percent": summary["ilp_improvement_percent"],
            "ilp_solve_time_seconds": summary["ilp_solve_time_seconds"],
            "ilp_num_variables": summary["ilp_num_variables"],
            "ilp_num_constraints": summary["ilp_num_constraints"],
            "system_processor": summary["system_info"]["processor"],
            "system_memory_total_gb": summary["system_info"]["memory_total_gb"],
        },
        "plots": [str(p) for p in plot_paths],
    }, indent=2))
