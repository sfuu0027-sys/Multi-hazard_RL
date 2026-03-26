# Multi-Hazard Lifecycle Management (RL/CEM)


This repository provides an open and reproducible implementation of multi-hazard (taking long-term deterioration + earthquake + fire as an example) life-cycle management: By constructing an environment through long-term deterioration-abrupt disturbance-restoration, it generates structural performance curves and constructs a unified objective based on discounted resilience loss, threshold-triggered risk, and cost to search for interpretable intervention strategies (pre-reinforcement/maintenance/post-disaster repair). 

The Cross-Entropy Method (CEM) is adopted for multi-objective optimization to select the optimal combination of intervention strategies. It also presents training procedures and ablation studies.

---

## 1. Key Concept

- **State**

```math
s_t=[F(t),\, t/T]
```

- **Actions**

```math
P\in\{0,1,2\},\quad M_t\in\{0,1,2\},\quad R_t\in\{0,1,2\}
```

Here, 'P' denotes pre-reinforcement at t=0, 'M_t' denotes periodic maintenance, and 'R_t' denotes post-disaster repair.

- **Resilience loss**

```math
L_R=\int_0^T e^{-\rho t}\,(1-F(t))\,dt
```

- **Risk**

```math
L_{\text{risk}}=\int_0^T e^{-\rho t}\,\mathbf{1}[F(t)<F_{\text{crit}}]\cdot C_f(F(t))\,dt
```

- **Cost / NPV**

```math
\mathrm{NPV}=C_P(P)+\int_0^T e^{-\rho t}\,\bigl(C_M(M_t)+C_R(R_t)\bigr)\,dt
```

- **Repository Structure**

```math
\max_{\pi}\ J(\pi)=\mathbb{E}_{\pi}\left[\sum_{t} r_t\right]
```

---

## 2. Repository Structure

- [RL_Code/](RL_Code/)
  - [train_lifecycle_rl.py](RL_Code/train_lifecycle_rl.py)：main entry point (training / re-plotting / batch cases)
  - `case*.json`：configuration files for each case (hazard rates, weights, CEM settings, output paths)
  - `run_config.json`：an example default configuration for a single case
  - `cem_results_*.json`、`iter_trajectories_*.json`：training outputs and trajectory logs
  - `fig/`：output figure directory (summary plots + per-case subfolders)

---

## 3. Dependencies

### 3.1 Python Requirements

Main dependencies：

- Python 3.10+（recommended）
- numpy
- matplotlib
- torch（optional: used for accelerating batch evaluation when available; the code falls back automatically if absent）

This project assumes a conda environment named `cudadev` by default：

```bash
conda activate cudadev
```

## 4. Quick Start (Training and Plotting)

All commands below should be executed from the repository root.

### 4.1 Train All Cases (200 iterations)

```bash
conda activate cudadev
python RL_Code/train_lifecycle_rl.py --all
```

Outputs：

- `RL_Code/cem_results_case*.json`
- `RL_Code/iter_trajectories_case*.json`
- Summary figures and per-case figures under `RL_Code/fig/` 

### 4.2 Re-plot Only (No Retraining)

If `cem_results_case*.json` already exist, you can regenerate comparison trajectories and figures without retraining：

```bash
python RL_Code/train_lifecycle_rl.py --all --replot
```

### 4.3 Run a Single Case

```bash
python RL_Code/train_lifecycle_rl.py RL_Code/case1_baseline.json
```

Re-plot a single case only：

```bash
python RL_Code/train_lifecycle_rl.py RL_Code/case1_baseline.json --replot
```

### 4.4 Toggle Figure Titles

By default, the script**does not display titles**at the top of plots. To enable titles：

```bash
python RL_Code/train_lifecycle_rl.py --all --replot --titles
```

---

## 5. Configuration (JSON)

All case configurations are stored in `RL_Code/case*.json`，with the following structure：

- `io`：output paths（fig、json、log）
- `lifecycle`：lifecycle model parameters（$T$、$\Delta t$、thresholds、 costs、 deterioration、 hazards、 discounting, etc.）
- `cem`：CEM settings (iterations, population size, elite fraction, number of evaluation episodes, etc.)

---

## 6. Randomness and Reproducibility

- Training randomness is controlled by `seed` ；Minor differences may appear across machines or torch versions.
- `--replot` loads the optimal parameters from saved result files and re-simulates under a fixed `compare_seed` to ensure consistent comparison plots.

---

## 7. Citation

If you use the code, figures, or experimental settings from this repository, please cite the repository URL in your work and add a BibTeX entry following your publication requirements.

---

## 8. License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
