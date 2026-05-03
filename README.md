# LoRa Time-on-Air ILP Optimization Experiments

This repository contains Python code for simulating and solving a LoRa-class gateway initialization problem using integer linear programming (ILP). The experiments compare a simple online greedy initializer against a globally optimized ILP assignment that minimizes planned uplink time-on-air subject to physical feasibility, load-management, and regulatory-aware constraints.

The code supports the paper draft:

> **Modeling and Optimizing Time-On-Air and Energy Usage in LoRa-Class Networks**  
> Chandra N. Sekharan, Ruben Dominguez, Jose Baca

## Overview

Low-power wide-area networks such as LoRa use spreading factors to trade off range, airtime, and energy consumption. Higher spreading factors improve link budget but substantially increase packet time-on-air. During gateway-driven initialization, the gateway must assign each node a feasible spreading factor while respecting link-budget feasibility, practical load limits, and regulatory constraints.

This repository implements a computational evaluation of that assignment problem. It generates synthetic LoRa-class network instances, computes LoRa time-on-air values, constructs feasible spreading-factor sets, solves an ILP optimization model, compares it against a greedy baseline, and produces summary statistics and figures.

## Main features

The code includes:

- Semtech-style LoRa time-on-air calculation.
- Synthetic node generation using distance, path loss, shadowing, payload size, and receiver sensitivity.
- Feasible spreading-factor set construction.
- Greedy online initializer.
- Regulatory-aware ILP solver using `scipy.optimize.milp` with the HiGHS backend.
- Monte Carlo evaluation over multiple randomly generated feasible instances.
- Runtime scaling experiments.
- Publication-oriented output tables and figures.

## Optimization model

For each node \(i\) and spreading factor \(s\), the binary variable \(x_{i,s}\) indicates whether node \(i\) is assigned to spreading factor \(s\). The ILP minimizes planned aggregate uplink time-on-air:

\[
\min \sum_i \sum_s a_i(W)T(B_i,s)x_{i,s}.
\]

The formulation includes:

- one spreading-factor assignment per node,
- physical feasibility constraints \(s \in \mathcal{F}_i\),
- per-spreading-factor capacity/load-management constraints,
- conservative dwell-time screening,
- per-device uplink duty-cycle compliance,
- grouped gateway downlink airtime constraints,
- binary activation variables for active spreading factors.

The per-spreading-factor capacity constraints are **load-management constraints**, not statutory duty-cycle rules. Statutory uplink duty-cycle compliance is modeled per transmitting end device.

## Repository structure

```text
.
├── src/
│   ├── lora_ilp_experiment.py      # Main experiment script
│   └── run_lora_mc.py              # Monte Carlo experiment driver, if included
├── data/
│   └── sample_outputs/             # Sample CSV/JSON/report outputs
├── figures/                        # Representative generated figures
├── docs/                           # Method notes
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

Depending on how the experiment is run, generated outputs may be written to an `outputs/`, `results/`, or configured experiment-output directory.

## Requirements

Recommended Python version:

```text
Python 3.10 or newer
```

Required Python packages:

```text
numpy
pandas
scipy
matplotlib
psutil
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

The ILP is solved using SciPy's `milp` interface, which uses the HiGHS optimization backend distributed with SciPy.

## Installation

Clone the repository:

```bash
git clone https://github.com/csekharan/lora-time-on-air-ilp.git
cd lora-time-on-air-ilp
```

Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the main experiment

From the repository root:

```bash
python src/lora_ilp_experiment.py
```

The script generates a synthetic LoRa-class network instance, solves the greedy and ILP assignments, and writes outputs such as:

```text
results_assignments_primary.csv
summary_metrics.json
summary_solver_scaling.csv
reproducibility_report.txt
fig_total_toa_greedy_vs_ilp.png
fig_greedy_to_ilp_transition_heatmap.png
fig_distance_vs_assigned_sf.png
fig_solver_scaling.png
```

The exact output directory depends on the path configured inside the script.

## Running the Monte Carlo experiment

If `src/run_lora_mc.py` is present, run:

```bash
python src/run_lora_mc.py
```

The Monte Carlo experiment estimates the ILP improvement over a domain of randomly generated feasible instances rather than reporting only one instance. Typical reported statistics include:

- mean percentage time-on-air reduction,
- standard deviation,
- median reduction,
- approximate confidence interval,
- mean fraction of nodes reassigned by the ILP,
- solver runtime as a function of network size.

This is the preferred way to support claims about expected improvement.

## Experimental settings

The default synthetic generator uses a LoRa-class setting with:

- spreading factors \(SF7,\ldots,SF12\),
- 125 kHz bandwidth,
- small IoT-style payload sizes,
- distance-dependent path loss,
- lognormal shadowing,
- SF-specific receiver sensitivity thresholds,
- a link-margin requirement,
- a per-device duty-cycle budget over an observation window,
- a conservative dwell/channel-occupancy screen,
- grouped gateway downlink control signaling.

These settings are intended to be realistic enough for computational evaluation, but they are not tied to a single deployment or regional channel plan. Users should adjust parameters for a specific regulatory region, hardware platform, or deployment scenario.

## Important modeling notes

### Greedy versus ILP

The greedy initializer processes nodes in arrival order and assigns each node to the fastest feasible spreading factor with remaining capacity. This is simple and online, but arrival-order dependent.

The ILP sees the full node set and can choose a globally better assignment. Its advantage is most visible when per-SF capacity/load-management constraints create competition for low-airtime spreading-factor slots.

### Regulatory constraints

The code distinguishes between:

- **per-device statutory or regional duty-cycle constraints**, and
- **per-SF capacity/load-management constraints** used for network performance and collision control.

The capacity constraints should not be interpreted as network-wide statutory duty-cycle rules.

### Grouped downlink control

The complete ILP uses a grouped-control downlink model in which a compact gateway control message is sent per active spreading factor. If a deployment requires per-node unicast downlink messages instead, the downlink-budget constraint can be replaced by the corresponding unicast constraint.

## Representative outputs

The experiments can generate figures such as:

1. **Greedy-to-ILP transition heatmap**  
   Shows how nodes move from greedy-assigned SFs to ILP-assigned SFs.

2. **Distance versus assigned SF**  
   Checks whether the ILP assignment is physically plausible.

3. **Solver runtime scaling**  
   Shows how ILP solve time changes as the number of nodes increases.

4. **Distribution of ToA reductions**  
   Summarizes Monte Carlo improvement across randomly generated instances.

## Reproducibility

The scripts use random seeds for synthetic instance generation. For paper-quality results, report:

- the number of generated feasible instances,
- the network-size range,
- seed schedule,
- solver and SciPy version,
- CPU and memory information,
- mean, standard deviation, median, and confidence interval of improvement.

The generated `reproducibility_report.txt` captures key system and experiment settings.

## Customizing the experiment

Common parameters to modify include:

- number of nodes \(N\),
- spreading-factor set,
- bandwidth,
- payload-size distribution,
- path-loss exponent,
- shadowing variance,
- transmit power,
- receiver sensitivity thresholds,
- per-SF capacity profile,
- duty-cycle window and budget,
- dwell-time threshold,
- grouped downlink payload size.

These parameters are defined inside the experiment scripts and can be edited directly for deployment-specific studies.

## Troubleshooting

### `scipy.optimize.milp` is unavailable

Install a newer SciPy version:

```bash
pip install --upgrade scipy
```

### Solver reports infeasibility

The generated instance may be physically or capacity infeasible. Try:

- increasing per-SF capacities,
- reducing maximum node distance,
- reducing shadowing variance,
- relaxing dwell-time thresholds,
- increasing the duty-cycle budget,
- regenerating with a different seed.

### Git line-ending warnings on Windows

Warnings such as `LF will be replaced by CRLF` are usually harmless. For consistent line endings, add a `.gitattributes` file with:

```text
* text=auto eol=lf
*.png binary
*.jpg binary
*.pdf binary
*.zip binary
```

Then run:

```bash
git add --renormalize .
git commit -m "Normalize line endings"
```

## Citation

If you use this code, please cite the associated paper draft or this repository. A `CITATION.cff` file is included for GitHub citation support.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Authors

Chandra N. Sekharan  
Ruben Dominguez  
Department of Computer Science  
Texas A&M University--Corpus Christi
