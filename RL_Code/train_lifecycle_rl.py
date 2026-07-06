from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

# Avoid Windows OpenMP duplicate-runtime aborts when torch/matplotlib/NumPy
# stacks are loaded together in the same process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except Exception:
    torch = None

_TEX_BODY_FONT_PT = 11.0
_TEX_TEXTWIDTH_IN = 6.5
_TEX_HALF_WIDTH_IN = _TEX_TEXTWIDTH_IN * 0.49
_TEX_MED_WIDTH_IN = _TEX_TEXTWIDTH_IN * 0.72
_LINEWIDTH_SCALE = 0.6
_PLOT_TITLES = False

_TABLE2_EQ_DF_LEVELS: tuple[float, float, float] = (0.10, 0.40, 0.70)
_TABLE2_FIRE_DF_LEVELS: tuple[float, float, float] = (0.10, 0.20, 0.30)


def _lw(x: float) -> float:
    return float(x) * float(_LINEWIDTH_SCALE)


def _tex_rc_params(base_font_pt: float = _TEX_BODY_FONT_PT) -> dict[str, Any]:
    base = float(base_font_pt)
    tick = max(1.0, base - 1.0)
    legend = max(1.0, base - 1.0)
    return {
        "font.family": "Times New Roman",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
        "figure.titlesize": base,
        "lines.linewidth": _lw(1.2),
        "savefig.dpi": 300,
    }


def _figsize_for_tex(width_in: float, height_in: float) -> tuple[float, float]:
    w = float(max(1.0, width_in))
    h = float(max(1.0, height_in))
    return (w, h)


def _write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8",
                       retries: int = 5, sleep_seconds: float = 0.15) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(max(1, int(retries))):
        tmp = path.with_name(
            f"{path.name}.tmp.{os.getpid()}.{int(time.time() * 1000)}.{attempt}"
        )
        try:
            tmp.write_text(text, encoding=encoding)
            os.replace(tmp, path)
            return
        except OSError as err:
            last_err = err
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt + 1 >= max(1, int(retries)):
                raise
            time.sleep(float(sleep_seconds))
    if last_err is not None:
        raise last_err


@dataclass(frozen=True)
class LifecycleConfig:
    horizon_years: int = 100
    dt_years: float = 1.0
    record_dt_years: float = 0.25
    maintenance_interval_years: int = 10

    lambda_eq_per_year: float = 1.0 / 35.0
    lambda_fire_per_year: float = 1.0 / 3.42
    hazard_intensities: tuple[float, float, float] = _TABLE2_FIRE_DF_LEVELS
    hazard_intensity_probs: tuple[float, float, float] = (0.60, 0.30, 0.10)
    eq_intensities: tuple[float, float, float] | None = _TABLE2_EQ_DF_LEVELS
    eq_intensity_probs: tuple[float, float, float] | None = None
    fire_intensities: tuple[float, float,
                            float] | None = _TABLE2_FIRE_DF_LEVELS
    fire_intensity_probs: tuple[float, float, float] | None = None

    f_crit: float = 0.70
    f_1: float = 0.50
    f_2: float = 0.30
    c_1: float = 1.0
    c_2: float = 4.0
    c_3: float = 10.0

    pre_cost_options: tuple[float, float, float] = (10.0, 20.0, 30.0)
    pre_fcrit_multipliers: tuple[float, float, float] = (1.00, 0.9, 0.8)
    pre_hazard_damage_multipliers: tuple[float, float, float] = (
        1.00, 0.9, 0.8)
    pre_deterioration_multipliers: tuple[float, float, float] = (
        1.00, 0.9, 0.8)
    deterioration_model: str = "weibull"
    det_alpha_T_levels: tuple[float, float, float] = (0.75, 0.50, 0.25)
    det_tau_years: float = 55.0
    det_k: float = 2.2
    det_sigma: float = 0.0
    base_deterioration_rate: float = 0.030

    maint_costs: tuple[float, float, float] = (0.0, 0.8, 2.0)
    maint_rate_multipliers: tuple[float, float, float] = (1.00, 0.70, 0.45)

    repair_costs: tuple[float, float, float] = (0.0, 3.0, 10.0)
    repair_recovery_deltas: tuple[float, float, float] = (0.0, 0.25, 0.50)
    hazard_transition_durations: tuple[float, float, float] = (0.5, 1.0, 1.5)

    w_lr: float = 1.0
    w_risk: float = 1.0
    w_cost: float = 1.20
    discount_rate: float = 0.03
    # Cost metric used in objective: "lcc" (discount-neutral) or "npv".
    objective_cost_metric: str = "lcc"
    # Objective aggregation mode: "reference" divides LR, risk, and selected
    # cost by fixed reference values before weighting.
    objective_normalization: str = "reference"
    lr_ref: float = 1.0
    risk_ref: float = 1.0
    cost_ref: float = 1.0
    floor_ref: float = 1.0
    w_floor: float = 0.0
    repair_stop_years: float = 0.0


@dataclass(frozen=True)
class PolicyParams:
    pre_index: int
    maint_t1: float
    maint_t2: float
    repair_t1: float
    repair_t2: float


@dataclass(frozen=True)
class EpisodeResult:
    t_years: list[float]
    f: list[float]
    cumulative_cost: list[float]
    lr: float
    risk: float
    lcc: float
    npv: float
    min_f: float
    feasible: bool
    return_value: float


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _cf(cfg: LifecycleConfig, f: float, f_crit: float) -> float:
    if f >= f_crit:
        return 0.0
    if f >= cfg.f_1:
        return cfg.c_1
    if f >= cfg.f_2:
        return cfg.c_2
    return cfg.c_3


def _case_short_name(name: str) -> str:
    short_map = {
        "case1_baseline": "case1",
        "case2a_fire_dominant": "case2a",
        "case2b_eq_dominant": "case2b",
        "case3a_cost_oriented": "case3a",
        "case3b_risk_oriented": "case3b",
        "case3c_resilience_oriented": "case3c",
        "case4a_no_resilience_equal": "case4a",
        "case4b_no_resilience_risk_replacement": "case4b",
        "case4c_no_risk_cost_emphasis": "case4c",
    }
    return short_map.get(str(name), str(name))


def _baseline_policy_params() -> PolicyParams:
    return PolicyParams(
        pre_index=0,
        maint_t1=0.70,
        maint_t2=0.50,
        repair_t1=0.70,
        repair_t2=0.50,
    )


def _hazard_level_index_from_intensity(
    intensity: float,
    *,
    eq_levels: tuple[float, float, float],
    fire_levels: tuple[float, float, float],
) -> int:
    x = float(intensity)
    tol = 1.0e-9
    for idx, v in enumerate(eq_levels):
        if abs(x - float(v)) <= tol:
            return int(idx)
    for idx, v in enumerate(fire_levels):
        if abs(x - float(v)) <= tol:
            return int(idx)
    all_levels = [float(v) for v in eq_levels] + [float(v)
                                                  for v in fire_levels]
    if not all_levels:
        return 0
    nearest = min(all_levels, key=lambda v: abs(v - x))
    if nearest in [float(v) for v in eq_levels]:
        return int([float(v) for v in eq_levels].index(nearest))
    return int([float(v) for v in fire_levels].index(nearest))


def _choose_maint(params: PolicyParams, f: float) -> int:
    t1 = max(params.maint_t1, params.maint_t2)
    t2 = min(params.maint_t1, params.maint_t2)
    if f >= t1:
        return 0
    if f >= t2:
        return 1
    return 2


def _choose_repair(params: PolicyParams, f_after_damage: float) -> int:
    t1 = max(params.repair_t1, params.repair_t2)
    t2 = min(params.repair_t1, params.repair_t2)
    if f_after_damage >= t1:
        return 0
    if f_after_damage >= t2:
        return 1
    return 2


def _hazard_prob_from_rate(lambda_per_year: float, dt_years: float) -> float:
    if lambda_per_year <= 0.0:
        return 0.0
    dt = max(0.0, float(dt_years))
    return float(1.0 - math.exp(-lambda_per_year * dt))


def _discount_factor(discount_rate: float, t_years: float) -> float:
    if discount_rate <= 0.0:
        return 1.0
    return float(math.exp(-discount_rate * t_years))


def _avg_discount_factor(discount_rate: float, t0_years: float, t1_years: float) -> float:
    dt = float(max(0.0, float(t1_years) - float(t0_years)))
    if dt <= 0.0:
        return _discount_factor(discount_rate, float(t0_years))
    if discount_rate <= 1.0e-12:
        return 1.0
    r = float(discount_rate)
    e0 = math.exp(-r * float(t0_years))
    e1 = math.exp(-r * float(t1_years))
    return float((e0 - e1) / (r * dt))


def _objective_uses_reference_normalization(cfg: LifecycleConfig) -> bool:
    mode = str(cfg.objective_normalization).strip().lower()
    return mode in {"reference", "ref", "normalized", "normalised"}


def _safe_ref(x: float) -> float:
    v = abs(float(x))
    if not math.isfinite(v) or v <= 1.0e-12:
        return 1.0
    return v


def _objective_cost_value(cfg: LifecycleConfig, *, lcc: float, npv: float) -> float:
    cost_metric = str(cfg.objective_cost_metric).strip().lower()
    return float(npv if cost_metric == "npv" else lcc)


def _objective_return_value(
    cfg: LifecycleConfig,
    *,
    lr: float,
    risk: float,
    lcc: float,
    npv: float,
    floor_violation: float,
) -> float:
    cost_value = _objective_cost_value(cfg, lcc=lcc, npv=npv)
    lr_term = float(lr)
    risk_term = float(risk)
    cost_term = float(cost_value)
    floor_term = float(floor_violation)
    if _objective_uses_reference_normalization(cfg):
        lr_term /= _safe_ref(cfg.lr_ref)
        risk_term /= _safe_ref(cfg.risk_ref)
        cost_term /= _safe_ref(cfg.cost_ref)
        floor_term /= _safe_ref(cfg.floor_ref)
    return float(-(cfg.w_lr * lr_term +
                   cfg.w_risk * risk_term +
                   cfg.w_cost * cost_term +
                   cfg.w_floor * floor_term))


def _objective_cost_tensor(cfg: LifecycleConfig, *, lcc: Any, npv: Any) -> Any:
    cost_metric = str(cfg.objective_cost_metric).strip().lower()
    return npv if cost_metric == "npv" else lcc


def _objective_return_tensor(
    cfg: LifecycleConfig,
    *,
    lr: Any,
    risk: Any,
    lcc: Any,
    npv: Any,
    floor_violation: Any,
) -> Any:
    cost_value = _objective_cost_tensor(cfg, lcc=lcc, npv=npv)
    lr_term = lr
    risk_term = risk
    cost_term = cost_value
    floor_term = floor_violation
    if _objective_uses_reference_normalization(cfg):
        lr_term = lr_term / _safe_ref(cfg.lr_ref)
        risk_term = risk_term / _safe_ref(cfg.risk_ref)
        cost_term = cost_term / _safe_ref(cfg.cost_ref)
        floor_term = floor_term / _safe_ref(cfg.floor_ref)
    return -(float(cfg.w_lr) * lr_term +
             float(cfg.w_risk) * risk_term +
             float(cfg.w_cost) * cost_term +
             float(cfg.w_floor) * floor_term)


def _lr_increment_pdf(
    f0: float,
    f1: float,
    *,
    t0: float,
    t1: float,
    gamma: float,
) -> float:
    dt = float(max(0.0, t1 - t0))
    if dt <= 0.0:
        return 0.0
    if gamma <= 1.0e-12:
        f_mid = 0.5 * (float(f0) + float(f1))
        return float((1.0 - f_mid) * dt)

    f0 = float(f0)
    f1 = float(f1)
    gamma = float(gamma)
    exp_t0 = math.exp(-gamma * float(t0))
    exp_t1 = math.exp(-gamma * float(t1))
    i0 = (exp_t0 - exp_t1) / gamma
    i1 = (exp_t0 - exp_t1 * (1.0 + gamma * dt)) / (gamma * gamma)
    a = 1.0 - f0
    m = (f1 - f0) / dt
    return float(a * i0 - m * i1)


def _weibull_g(t_years: float, *, tau_years: float, k: float, horizon_years: float) -> float:
    t = max(0.0, float(t_years))
    tau = max(1.0e-9, float(tau_years))
    kk = max(1.0e-9, float(k))
    T = max(1.0e-9, float(horizon_years))
    denom = 1.0 - math.exp(-((T / tau) ** kk))
    if denom <= 0.0:
        return 0.0
    return float((1.0 - math.exp(-((t / tau) ** kk))) / denom)


def _effective_f_crit(cfg: LifecycleConfig, pre_index: int) -> float:
    idx = int(np.clip(pre_index, 0, len(cfg.pre_fcrit_multipliers) - 1))
    mult = float(cfg.pre_fcrit_multipliers[idx])
    f_crit_eff = cfg.f_crit * mult
    if f_crit_eff > cfg.f_crit:
        f_crit_eff = cfg.f_crit
    if f_crit_eff <= cfg.f_1:
        f_crit_eff = cfg.f_1 + 1.0e-6
    return float(f_crit_eff)


def _cumulative_lr_and_risk(
    cfg: LifecycleConfig,
    t_years: list[float],
    f_series: list[float],
    *,
    f_crit_eff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.array(t_years, dtype=float)
    f = np.array(f_series, dtype=float)
    n = int(min(t.size, f.size))
    if n <= 1:
        z = np.zeros(n, dtype=float)
        return t[:n], z, z

    t = t[:n]
    f = f[:n]
    lr_cum = np.zeros(n, dtype=float)
    risk_cum = np.zeros(n, dtype=float)
    for k in range(n - 1):
        t0 = float(t[k])
        t1 = float(t[k + 1])
        dt = float(max(0.0, t1 - t0))
        f0 = float(f[k])
        f1 = float(f[k + 1])
        if dt <= 0.0:
            # Hazard damage/recovery may be recorded at identical timestamps.
            # Count instantaneous downward jumps as event-risk increments.
            risk_inc = 0.0
            drop = max(0.0, f0 - f1)
            if drop > 0.0:
                cf_val_event = _cf(cfg, f1, f_crit=f_crit_eff)
                if cf_val_event > 0.0:
                    risk_inc = cf_val_event * drop
            lr_cum[k + 1] = lr_cum[k]
            risk_cum[k + 1] = risk_cum[k] + risk_inc
            continue

        f_mid = 0.5 * (f0 + f1)
        lr_inc = _lr_increment_pdf(
            f0, f1, t0=t0, t1=t1, gamma=0.0)
        cf_val = _cf(cfg, f_mid, f_crit=f_crit_eff)
        risk_inc = (cf_val * dt) if cf_val > 0.0 else 0.0
        lr_cum[k + 1] = lr_cum[k] + lr_inc
        risk_cum[k + 1] = risk_cum[k] + risk_inc
    return t, lr_cum, risk_cum


def simulate_episode(cfg: LifecycleConfig, params: PolicyParams, seed: int, *, record_hazard_steps: bool = False) -> EpisodeResult:
    rng = np.random.default_rng(seed)

    horizon_years = float(cfg.horizon_years)
    decision_dt = float(cfg.dt_years)
    record_dt = float(min(cfg.record_dt_years, decision_dt))
    if record_dt <= 0.0:
        record_dt = decision_dt

    horizon_steps = int(round(horizon_years / record_dt))
    pre_index = int(np.clip(params.pre_index, 0,
                    len(cfg.pre_cost_options) - 1))
    f_crit_eff = _effective_f_crit(cfg, pre_index)

    f = 1.0
    cost = float(cfg.pre_cost_options[pre_index])
    npv = float(cfg.pre_cost_options[pre_index])
    det_mult = float(cfg.pre_deterioration_multipliers[pre_index])
    haz_mult = float(cfg.pre_hazard_damage_multipliers[pre_index])

    t_years: list[float] = [0.0]
    f_series: list[float] = [f]
    cumulative_cost: list[float] = [cost]

    lr = 0.0
    risk = 0.0
    floor_violation = 0.0
    min_f = float(f)
    feasible = True

    maint_level = _choose_maint(params, f)
    maint_interval_years = float(cfg.maintenance_interval_years)

    eq_intensities = cfg.hazard_intensities if cfg.eq_intensities is None else cfg.eq_intensities
    eq_probs = cfg.hazard_intensity_probs if cfg.eq_intensity_probs is None else cfg.eq_intensity_probs
    fire_intensities = cfg.hazard_intensities if cfg.fire_intensities is None else cfg.fire_intensities
    fire_probs = cfg.hazard_intensity_probs if cfg.fire_intensity_probs is None else cfg.fire_intensity_probs

    active_procs: list[dict[str, float]] = []

    def _apply_procs(dt: float) -> float:
        nonlocal active_procs
        delta_f = 0.0
        updated: list[dict[str, float]] = []
        for p in active_procs:
            delay = float(p["delay"])
            remaining = float(p["remaining"])
            rate = float(p["rate"])
            if remaining <= 0.0:
                continue
            if delay >= dt:
                p["delay"] = delay - dt
                updated.append(p)
                continue
            active_time = dt - max(0.0, delay)
            p["delay"] = 0.0
            used = min(active_time, remaining)
            delta_f += rate * used
            remaining -= used
            p["remaining"] = remaining
            if remaining > 1.0e-12:
                updated.append(p)
        active_procs = updated
        return float(delta_f)

    t = 0.0
    for _ in range(horizon_steps):
        if t >= horizon_years - 1.0e-12:
            break

        is_year_boundary = abs(t - round(t / decision_dt)
                               * decision_dt) <= 1.0e-9
        if is_year_boundary and t > 0.0 and maint_interval_years > 0.0:
            if abs((t % maint_interval_years)) <= 1.0e-9:
                maint_level = _choose_maint(params, f)
                maint_cost = float(cfg.maint_costs[maint_level])
                cost += maint_cost
                npv += maint_cost * \
                    _avg_discount_factor(cfg.discount_rate, t, min(
                        horizon_years, t + decision_dt))

        if is_year_boundary:
            dt_h = float(decision_dt)
            events: list[float] = []
            if cfg.lambda_eq_per_year > 0.0 and dt_h > 0.0:
                p = _hazard_prob_from_rate(float(cfg.lambda_eq_per_year), dt_h)
                if float(rng.random()) < p:
                    events.append(
                        float(rng.choice(eq_intensities, p=np.array(eq_probs, dtype=float))))
            if cfg.lambda_fire_per_year > 0.0 and dt_h > 0.0:
                p = _hazard_prob_from_rate(
                    float(cfg.lambda_fire_per_year), dt_h)
                if float(rng.random()) < p:
                    events.append(
                        float(rng.choice(fire_intensities, p=np.array(fire_probs, dtype=float))))
            if len(events) > 1:
                events = [events[i]
                          for i in rng.permutation(len(events)).tolist()]

            for intensity in events:
                intensity_eff = float(intensity) * haz_mult
                f_before_damage = float(f)
                f_after_damage = _clip01(float(f) - float(intensity_eff))
                event_drop = max(0.0, f_before_damage - f_after_damage)
                if event_drop > 0.0:
                    event_cf = _cf(cfg, f_after_damage, f_crit=f_crit_eff)
                    if event_cf > 0.0:
                        risk += event_cf * event_drop
                if record_hazard_steps:
                    t_years.append(float(t))
                    f_series.append(float(f_after_damage))
                    cumulative_cost.append(float(cost))
                repair_level = _choose_repair(params, f_after_damage)
                repair_cost = float(
                    cfg.repair_costs[repair_level]) * (1.0 + 0.8 * float(intensity_eff))
                cost += repair_cost
                npv += repair_cost * \
                    _avg_discount_factor(cfg.discount_rate, t, min(
                        horizon_years, t + decision_dt))
                delta_f_r = float(cfg.repair_recovery_deltas[repair_level])
                if delta_f_r < 0.0:
                    delta_f_r = 0.0
                f = _clip01(f_after_damage + delta_f_r)
                if record_hazard_steps:
                    t_years.append(float(t))
                    f_series.append(float(f))
                    cumulative_cost.append(float(cost))

        t_next = min(horizon_years, t + record_dt)
        dt_seg = float(max(0.0, t_next - t))
        if dt_seg <= 0.0:
            break

        f0 = float(f)

        if cfg.deterioration_model.lower() == "weibull":
            g0 = _weibull_g(t, tau_years=cfg.det_tau_years,
                            k=cfg.det_k, horizon_years=cfg.horizon_years)
            g1 = _weibull_g(t_next, tau_years=cfg.det_tau_years,
                            k=cfg.det_k, horizon_years=cfg.horizon_years)
            dg = max(0.0, g1 - g0)
            alpha_T = float(cfg.det_alpha_T_levels[int(
                np.clip(maint_level, 0, len(cfg.det_alpha_T_levels) - 1))])
            delta_alpha = alpha_T * det_mult * dg
            if cfg.det_sigma > 0.0:
                delta_alpha *= float(rng.lognormal(mean=0.0,
                                     sigma=float(cfg.det_sigma)))
            f = _clip01(f - delta_alpha)
        else:
            det_rate = cfg.base_deterioration_rate * det_mult * \
                cfg.maint_rate_multipliers[maint_level]
            det_rate *= float(rng.lognormal(mean=0.0, sigma=0.15))
            f = _clip01(f - det_rate * dt_seg)

        f = _clip01(float(f) + _apply_procs(dt_seg))
        f1 = float(f)

        f_mid = 0.5 * (f0 + f1)
        lr += _lr_increment_pdf(f0, f1, t0=t, t1=t_next,
                                gamma=0.0)
        cf_val = _cf(cfg, f_mid, f_crit=f_crit_eff)
        if cf_val > 0.0:
            risk += cf_val * dt_seg
        floor_violation += max(0.0, cfg.f_2 - f_mid) * dt_seg

        min_f = min(min_f, f1)
        feasible = feasible and (f1 >= cfg.f_2)

        t = t_next
        t_years.append(t)
        f_series.append(f1)
        cumulative_cost.append(float(cost))

    lcc = cost
    return_value = _objective_return_value(
        cfg,
        lr=lr,
        risk=risk,
        lcc=lcc,
        npv=npv,
        floor_violation=floor_violation,
    )

    return EpisodeResult(
        t_years=t_years,
        f=f_series,
        cumulative_cost=cumulative_cost,
        lr=lr,
        risk=risk,
        lcc=lcc,
        npv=npv,
        min_f=min_f,
        feasible=feasible,
        return_value=return_value,
    )


@dataclass(frozen=True)
class CEMConfig:
    iterations: int = 20
    population: int = 80
    elite_frac: float = 0.2
    eval_episodes: int = 30
    seed: int = 20260301
    compare_seed_offset: int = 4242
    plot_first_n: int = 16
    plot_interval: int = 3
    final_selection: str = "max_return"


@dataclass(frozen=True)
class CEMState:
    mu: np.ndarray
    sigma: np.ndarray
    pre_probs: np.ndarray


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lifecycle_rl")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def _init_cem_state() -> CEMState:
    mu = np.array([0.65, 0.50, 0.55, 0.40], dtype=float)
    sigma = np.array([0.08, 0.08, 0.10, 0.10], dtype=float)
    pre_probs = np.array([0.60, 0.25, 0.15], dtype=float)
    return CEMState(mu=mu, sigma=sigma, pre_probs=pre_probs)


def _sample_params(rng: np.random.Generator, state: CEMState) -> PolicyParams:
    maint_t1, maint_t2, repair_t1, repair_t2 = rng.normal(
        loc=state.mu, scale=state.sigma).tolist()
    maint_t1 = float(np.clip(maint_t1, 0.05, 0.95))
    maint_t2 = float(np.clip(maint_t2, 0.05, 0.95))
    repair_t1 = float(np.clip(repair_t1, 0.05, 0.95))
    repair_t2 = float(np.clip(repair_t2, 0.05, 0.95))
    pre_index = int(rng.choice([0, 1, 2], p=state.pre_probs))
    return PolicyParams(
        pre_index=pre_index,
        maint_t1=maint_t1,
        maint_t2=maint_t2,
        repair_t1=repair_t1,
        repair_t2=repair_t2,
    )


def _evaluate_params(cfg: LifecycleConfig, params: PolicyParams, base_seed: int, episodes: int) -> dict[str, float]:
    returns: list[float] = []
    lrs: list[float] = []
    risks: list[float] = []
    lccs: list[float] = []
    npvs: list[float] = []
    min_fs: list[float] = []
    feasibles: list[float] = []
    for i in range(episodes):
        ep = simulate_episode(cfg, params, seed=base_seed + i * 10007)
        returns.append(ep.return_value)
        lrs.append(ep.lr)
        risks.append(ep.risk)
        lccs.append(ep.lcc)
        npvs.append(ep.npv)
        min_fs.append(ep.min_f)
        feasibles.append(1.0 if ep.feasible else 0.0)
    return {
        "return": float(np.mean(returns)),
        "lr": float(np.mean(lrs)),
        "risk": float(np.mean(risks)),
        "lcc": float(np.mean(lccs)),
        "npv": float(np.mean(npvs)),
        "min_f": float(np.mean(min_fs)),
        "feasible_frac": float(np.mean(feasibles)),
    }


def _pareto_front_indices(points: list[dict[str, float]], keys: tuple[str, ...]) -> list[int]:
    front: list[int] = []
    eps = 1.0e-9
    for i, pi in enumerate(points):
        dominated = False
        for j, pj in enumerate(points):
            if i == j:
                continue
            no_worse = all(float(pj[k]) <= float(pi[k]) + eps for k in keys)
            strictly_better = any(float(pj[k]) < float(pi[k]) - eps for k in keys)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def _norm_minmax(value: float, values: list[float]) -> float:
    lo = float(min(values))
    hi = float(max(values))
    if hi <= lo + 1.0e-12:
        return 0.0
    return float((float(value) - lo) / (hi - lo))


def _select_pareto_final_candidate(
    cfg: LifecycleConfig,
    cem_cfg: CEMConfig,
    history: list[dict[str, Any]],
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for rec in history:
        best = rec.get("best_of_iter")
        if not isinstance(best, dict) or not isinstance(best.get("params"), dict):
            continue
        params = PolicyParams(**dict(best["params"]))
        eval_metrics = _evaluate_params(
            cfg,
            params,
            base_seed=int(cem_cfg.seed + 99991),
            episodes=int(cem_cfg.eval_episodes * 2),
        )
        candidates.append(
            {
                "iteration": int(rec.get("iteration", len(candidates))),
                "params": params,
                "eval": eval_metrics,
            }
        )

    if not candidates:
        return None

    front_idx = _pareto_front_indices(
        [{"lcc": float(c["eval"]["lcc"]), "risk": float(c["eval"]["risk"])} for c in candidates],
        ("lcc", "risk"),
    )
    front = [candidates[i] for i in front_idx]

    all_lcc = [float(c["eval"]["lcc"]) for c in candidates]
    all_risk = [float(c["eval"]["risk"]) for c in candidates]
    all_lr = [float(c["eval"]["lr"]) for c in candidates]

    def rank_key(c: dict[str, Any]) -> tuple[float, float]:
        ev = c["eval"]
        cost_risk_balance = (
            _norm_minmax(float(ev["lcc"]), all_lcc)
            + _norm_minmax(float(ev["risk"]), all_risk)
        )
        complete_objective = cost_risk_balance + 0.25 * _norm_minmax(float(ev["lr"]), all_lr)
        return (complete_objective, -float(ev["return"]))

    selected = min(front, key=rank_key)
    selected = dict(selected)
    selected["selection"] = {
        "method": "pareto_cost_risk_then_full_objective",
        "pareto_objectives": ["lcc", "risk"],
        "tie_break": "minmax_normalized_lcc_plus_risk_plus_0.25_lr; return as secondary tie-break",
        "candidate_count": int(len(candidates)),
        "pareto_candidate_count": int(len(front)),
    }
    if logger is not None:
        ev = selected["eval"]
        logger.info(
            "Pareto final selection | iter=%d return=%.4f lr=%.3f risk=%.3f lcc=%.2f front=%d/%d",
            int(selected["iteration"]),
            float(ev["return"]),
            float(ev["lr"]),
            float(ev["risk"]),
            float(ev["lcc"]),
            int(len(front)),
            int(len(candidates)),
        )
    return selected


def _mark_selected_iter(iter_trajectories: list[dict[str, Any]], iteration: int | None) -> None:
    for rec in iter_trajectories:
        rec.pop("selected_final", None)
    if iteration is None:
        return
    for rec in iter_trajectories:
        if int(rec.get("iteration", -1)) == int(iteration):
            rec["selected_final"] = True
            return


def _selected_iter_index(iter_trajectories: list[dict[str, Any]]) -> int:
    best_idx = 0
    best_return = -float("inf")
    for i, rec in enumerate(iter_trajectories):
        if bool(rec.get("selected_final", False)):
            return i
        score = rec.get("score")
        if isinstance(score, dict) and "return" in score:
            ret = float(score["return"])
        else:
            ret = float(rec["episode"]["return_value"])
        if ret > best_return:
            best_return = ret
            best_idx = i

    best_idx = _selected_iter_index(iter_trajectories)
    best_rec_for_plot = iter_trajectories[best_idx]
    best_score_for_plot = best_rec_for_plot.get("score")
    if isinstance(best_score_for_plot, dict) and "return" in best_score_for_plot:
        best_return = float(best_score_for_plot["return"])
    else:
        best_return = float(best_rec_for_plot["episode"]["return_value"])
    return best_idx


def _compare_seed_from_config(cem_cfg: CEMConfig) -> int:
    return int(cem_cfg.seed) + int(cem_cfg.compare_seed_offset)


def _apply_pareto_final_selection(
    result: dict[str, Any],
    cfg: LifecycleConfig,
    cem_cfg: CEMConfig,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    history = result.get("history", [])
    if not isinstance(history, list):
        return result
    selected = _select_pareto_final_candidate(cfg, cem_cfg, history, logger=logger)
    if selected is None:
        return result

    params: PolicyParams = selected["params"]
    compare_seed = int(result.get("compare_seed", _compare_seed_from_config(cem_cfg)))
    best_episode = simulate_episode(cfg, params, seed=compare_seed, record_hazard_steps=True)
    result["best"] = {
        "params": asdict(params),
        "eval": dict(selected["eval"]),
        "selection": dict(selected["selection"]),
        "episode_compare_seed": {
            "t_years": best_episode.t_years,
            "f": best_episode.f,
            "cumulative_cost": best_episode.cumulative_cost,
            "lr": best_episode.lr,
            "risk": best_episode.risk,
            "lcc": best_episode.lcc,
            "npv": best_episode.npv,
            "min_f": best_episode.min_f,
            "feasible": best_episode.feasible,
            "return_value": best_episode.return_value,
        },
    }
    if isinstance(result.get("iter_trajectories"), list):
        _mark_selected_iter(result["iter_trajectories"], int(selected["iteration"]))
    return result


def _apply_max_return_final_selection(
    result: dict[str, Any],
    cfg: LifecycleConfig,
    cem_cfg: CEMConfig,
) -> dict[str, Any]:
    history = result.get("history", [])
    if not isinstance(history, list) or not history:
        return result
    selected: dict[str, Any] | None = None
    selected_return = -float("inf")
    for rec in history:
        best = rec.get("best_of_iter")
        if not isinstance(best, dict) or not isinstance(best.get("params"), dict):
            continue
        ret = float(best.get("return", -float("inf")))
        if ret > selected_return:
            selected_return = ret
            selected = rec
    if selected is None:
        return result

    best = selected["best_of_iter"]
    params = PolicyParams(**dict(best["params"]))
    compare_seed = int(result.get("compare_seed", _compare_seed_from_config(cem_cfg)))
    best_eval = _evaluate_params(
        cfg,
        params,
        base_seed=int(cem_cfg.seed + 99991),
        episodes=int(cem_cfg.eval_episodes * 2),
    )
    best_episode = simulate_episode(cfg, params, seed=compare_seed, record_hazard_steps=True)
    result["best"] = {
        "params": asdict(params),
        "eval": best_eval,
        "selection": {
            "method": "max_training_return",
            "training_best_return": float(selected_return),
        },
        "episode_compare_seed": {
            "t_years": best_episode.t_years,
            "f": best_episode.f,
            "cumulative_cost": best_episode.cumulative_cost,
            "lr": best_episode.lr,
            "risk": best_episode.risk,
            "lcc": best_episode.lcc,
            "npv": best_episode.npv,
            "min_f": best_episode.min_f,
            "feasible": best_episode.feasible,
            "return_value": best_episode.return_value,
        },
    }
    if isinstance(result.get("iter_trajectories"), list):
        _mark_selected_iter(result["iter_trajectories"], int(selected.get("iteration", -1)))
    return result


def _torch_can_cuda() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_device() -> str:
    return "cuda" if _torch_can_cuda() else "cpu"


def _weibull_g_torch(t_years, *, tau_years: float, k: float, horizon_years: float):
    t = torch.clamp(t_years, min=0.0)
    tau = float(max(1.0e-9, float(tau_years)))
    kk = float(max(1.0e-9, float(k)))
    T = float(max(1.0e-9, float(horizon_years)))
    denom = 1.0 - math.exp(-((T / tau) ** kk))
    if denom <= 0.0:
        return torch.zeros_like(t)
    return (1.0 - torch.exp(-((t / tau) ** kk))) / float(denom)


def _effective_f_crit_torch(cfg: LifecycleConfig, pre_index):
    idx = torch.clamp(pre_index.to(torch.int64), 0,
                      int(len(cfg.pre_fcrit_multipliers) - 1))
    mult = torch.tensor(cfg.pre_fcrit_multipliers,
                        device=pre_index.device, dtype=torch.float32)[idx]
    f_crit_eff = float(cfg.f_crit) * mult
    f_crit_eff = torch.minimum(f_crit_eff, torch.tensor(
        float(cfg.f_crit), device=pre_index.device))
    f_crit_eff = torch.maximum(f_crit_eff, torch.tensor(
        float(cfg.f_1) + 1.0e-6, device=pre_index.device))
    return f_crit_eff


def _choose_maint_torch(maint_t1, maint_t2, f):
    t1 = torch.maximum(maint_t1, maint_t2)
    t2 = torch.minimum(maint_t1, maint_t2)
    m0 = f >= t1
    m1 = (f < t1) & (f >= t2)
    return torch.where(m0, torch.zeros_like(f, dtype=torch.int64), torch.where(m1, torch.ones_like(f, dtype=torch.int64), torch.full_like(f, 2, dtype=torch.int64)))


def _choose_repair_torch(repair_t1, repair_t2, f_after_damage):
    t1 = torch.maximum(repair_t1, repair_t2)
    t2 = torch.minimum(repair_t1, repair_t2)
    r0 = f_after_damage >= t1
    r1 = (f_after_damage < t1) & (f_after_damage >= t2)
    return torch.where(r0, torch.zeros_like(f_after_damage, dtype=torch.int64), torch.where(r1, torch.ones_like(f_after_damage, dtype=torch.int64), torch.full_like(f_after_damage, 2, dtype=torch.int64)))


def _cf_torch(cfg: LifecycleConfig, f, f_crit_eff):
    c = torch.zeros_like(f)
    m1 = (f < f_crit_eff) & (f >= float(cfg.f_1))
    m2 = (f < float(cfg.f_1)) & (f >= float(cfg.f_2))
    m3 = f < float(cfg.f_2)
    c = torch.where(m1, torch.full_like(f, float(cfg.c_1)), c)
    c = torch.where(m2, torch.full_like(f, float(cfg.c_2)), c)
    c = torch.where(m3, torch.full_like(f, float(cfg.c_3)), c)
    return c


def _sample_hazards_by_episode(
    cfg: LifecycleConfig,
    *,
    rng: np.random.Generator,
    batch: int,
    years: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eq_levels = np.array(
        cfg.hazard_intensities if cfg.eq_intensities is None else cfg.eq_intensities, dtype=float)
    eq_probs = np.array(
        cfg.hazard_intensity_probs if cfg.eq_intensity_probs is None else cfg.eq_intensity_probs, dtype=float)
    fire_levels = np.array(
        cfg.hazard_intensities if cfg.fire_intensities is None else cfg.fire_intensities, dtype=float)
    fire_probs = np.array(
        cfg.hazard_intensity_probs if cfg.fire_intensity_probs is None else cfg.fire_intensity_probs, dtype=float)

    lam_eq = float(cfg.lambda_eq_per_year)
    lam_fire = float(cfg.lambda_fire_per_year)
    dt = float(max(0.0, float(cfg.dt_years)))

    shape = (int(batch), int(years))
    eq_int = np.zeros(shape, dtype=float)
    fire_int = np.zeros(shape, dtype=float)

    if lam_eq > 0.0:
        p = 1.0 - math.exp(-lam_eq * dt) if dt > 0.0 else 0.0
        m = rng.random(size=shape) < p
        if m.any():
            idx = rng.choice(int(eq_levels.size),
                             size=int(m.sum()), p=eq_probs)
            eq_int[m] = eq_levels[idx]

    if lam_fire > 0.0:
        p = 1.0 - math.exp(-lam_fire * dt) if dt > 0.0 else 0.0
        m = rng.random(size=shape) < p
        if m.any():
            idx = rng.choice(int(fire_levels.size),
                             size=int(m.sum()), p=fire_probs)
            fire_int[m] = fire_levels[idx]

    intensity_eff = np.maximum(eq_int, fire_int)
    return eq_int, fire_int, intensity_eff


def _evaluate_population_torch(
    cfg: LifecycleConfig,
    population: list[PolicyParams],
    *,
    base_seed_it: int,
    episodes: int,
    device: str,
) -> list[dict[str, float]]:
    pop_n = int(len(population))
    if pop_n <= 0:
        return []
    ep_n = int(max(1, episodes))
    years = int(round(float(cfg.horizon_years) / float(cfg.dt_years)))
    years = int(max(1, years))
    batch = int(pop_n * ep_n)
    rng = np.random.default_rng(int(base_seed_it))
    eq_int_y, fire_int_y, intensity_eff_y = _sample_hazards_by_episode(
        cfg, rng=rng, batch=batch, years=years)
    eq_int_y = torch.tensor(eq_int_y, device=device,
                            dtype=torch.float32).view(pop_n, ep_n, years)
    fire_int_y = torch.tensor(
        fire_int_y, device=device, dtype=torch.float32).view(pop_n, ep_n, years)
    intensity_eff_y = torch.tensor(
        intensity_eff_y, device=device, dtype=torch.float32).view(pop_n, ep_n, years)

    pre_index = torch.tensor([p.pre_index for p in population], device=device,
                             dtype=torch.int64).view(pop_n, 1).repeat(1, ep_n).reshape(-1)
    maint_t1 = torch.tensor([p.maint_t1 for p in population], device=device,
                            dtype=torch.float32).view(pop_n, 1).repeat(1, ep_n).reshape(-1)
    maint_t2 = torch.tensor([p.maint_t2 for p in population], device=device,
                            dtype=torch.float32).view(pop_n, 1).repeat(1, ep_n).reshape(-1)
    repair_t1 = torch.tensor([p.repair_t1 for p in population], device=device,
                             dtype=torch.float32).view(pop_n, 1).repeat(1, ep_n).reshape(-1)
    repair_t2 = torch.tensor([p.repair_t2 for p in population], device=device,
                             dtype=torch.float32).view(pop_n, 1).repeat(1, ep_n).reshape(-1)

    horizon_years = float(cfg.horizon_years)
    decision_dt = float(cfg.dt_years)
    record_dt = float(min(cfg.record_dt_years, cfg.dt_years))
    if record_dt <= 0.0:
        record_dt = float(cfg.dt_years)
    steps = int(round(horizon_years / record_dt))
    steps = int(max(1, steps))
    year_stride = int(round(decision_dt / record_dt))
    year_stride = int(max(1, year_stride))

    f = torch.ones(batch, device=device, dtype=torch.float32)
    cost = torch.tensor([float(cfg.pre_cost_options[int(i)])
                        for i in pre_index.tolist()], device=device, dtype=torch.float32)
    npv = cost.clone()

    det_mult = torch.tensor(cfg.pre_deterioration_multipliers, device=device, dtype=torch.float32)[
        torch.clamp(pre_index, 0, int(len(cfg.pre_deterioration_multipliers) - 1))]
    haz_mult = torch.tensor(cfg.pre_hazard_damage_multipliers, device=device, dtype=torch.float32)[
        torch.clamp(pre_index, 0, int(len(cfg.pre_hazard_damage_multipliers) - 1))]
    f_crit_eff = _effective_f_crit_torch(cfg, pre_index).to(torch.float32)

    maint_level = _choose_maint_torch(maint_t1, maint_t2, f)
    maint_costs = torch.tensor(
        cfg.maint_costs, device=device, dtype=torch.float32)
    repair_costs = torch.tensor(
        cfg.repair_costs, device=device, dtype=torch.float32)
    repair_deltas = torch.tensor(
        cfg.repair_recovery_deltas, device=device, dtype=torch.float32)
    hazard_durs = torch.tensor(
        cfg.hazard_transition_durations, device=device, dtype=torch.float32)
    alpha_T_levels = torch.tensor(
        cfg.det_alpha_T_levels, device=device, dtype=torch.float32)

    down0 = torch.zeros(batch, device=device, dtype=torch.float32)
    up0 = torch.zeros(batch, device=device, dtype=torch.float32)
    slope0 = torch.zeros(batch, device=device, dtype=torch.float32)
    down1 = torch.zeros(batch, device=device, dtype=torch.float32)
    up1 = torch.zeros(batch, device=device, dtype=torch.float32)
    slope1 = torch.zeros(batch, device=device, dtype=torch.float32)

    lr = torch.zeros(batch, device=device, dtype=torch.float32)
    risk = torch.zeros(batch, device=device, dtype=torch.float32)
    floor_violation = torch.zeros(batch, device=device, dtype=torch.float32)
    min_f = f.clone()

    maint_interval_years = float(cfg.maintenance_interval_years)
    maint_interval_steps = int(
        round(maint_interval_years / record_dt)) if maint_interval_years > 0.0 else 0

    discount_rate = float(cfg.discount_rate)
    f2 = float(cfg.f_2)
    for s in range(steps):
        t = float(s) * record_dt
        if t >= horizon_years - 1.0e-12:
            break

        is_year_boundary = (s % year_stride) == 0
        if is_year_boundary and s > 0 and maint_interval_steps > 0 and (s % maint_interval_steps) == 0:
            maint_level = _choose_maint_torch(maint_t1, maint_t2, f)
            mc = maint_costs[maint_level]
            cost = cost + mc
            if discount_rate > 0.0:
                disc_avg = torch.tensor(float(_avg_discount_factor(discount_rate, t, min(
                    horizon_years, t + decision_dt))), device=device, dtype=torch.float32)
                npv = npv + mc * disc_avg
            else:
                npv = npv + mc

        if is_year_boundary:
            y = int(round(t / decision_dt))
            if y < years:
                disc_avg = torch.tensor(float(_avg_discount_factor(discount_rate, t, min(
                    horizon_years, t + decision_dt))), device=device, dtype=torch.float32)
                eq_int = eq_int_y.view(-1, years)[:, y]
                fire_int = fire_int_y.view(-1, years)[:, y]

                def _apply_event(intensity):
                    nonlocal f, cost, npv, risk
                    has = intensity > 0.0
                    intensity_eff = torch.clamp(intensity, min=0.0) * haz_mult
                    f_before_damage = f
                    f_after_damage = torch.clamp(
                        f - intensity_eff, min=0.0, max=1.0)
                    event_drop = torch.clamp(f_before_damage - f_after_damage, min=0.0)
                    event_cf = _cf_torch(cfg, f_after_damage, f_crit_eff)
                    risk = risk + torch.where(
                        has & (event_drop > 0.0) & (event_cf > 0.0),
                        event_cf * event_drop,
                        torch.zeros_like(event_cf),
                    )
                    repair_level = _choose_repair_torch(
                        repair_t1, repair_t2, f_after_damage)
                    delta_f_r = torch.clamp(
                        repair_deltas[repair_level], min=0.0)
                    f_repaired = torch.clamp(
                        f_after_damage + delta_f_r, min=0.0, max=1.0)
                    f = torch.where(has, f_repaired, f)

                    rc = repair_costs[repair_level] * \
                        (1.0 + 0.8 * intensity_eff)
                    rc = torch.where(has, rc, torch.zeros_like(rc))
                    cost = cost + rc
                    if discount_rate > 0.0:
                        npv = npv + rc * disc_avg
                    else:
                        npv = npv + rc

                if float(rng.random()) < 0.5:
                    _apply_event(eq_int)
                    _apply_event(fire_int)
                else:
                    _apply_event(fire_int)
                    _apply_event(eq_int)

        t_next = min(horizon_years, t + record_dt)
        dt_seg = float(max(0.0, t_next - t))
        if dt_seg <= 0.0:
            break

        f0 = f
        t0 = torch.full((batch,), float(t), device=device, dtype=torch.float32)
        t1 = torch.full((batch,), float(t_next),
                        device=device, dtype=torch.float32)
        g0 = _weibull_g_torch(t0, tau_years=cfg.det_tau_years,
                              k=cfg.det_k, horizon_years=cfg.horizon_years)
        g1 = _weibull_g_torch(t1, tau_years=cfg.det_tau_years,
                              k=cfg.det_k, horizon_years=cfg.horizon_years)
        dg = torch.clamp(g1 - g0, min=0.0)
        alpha_T = alpha_T_levels[maint_level]
        delta_alpha = alpha_T * det_mult * dg
        f = torch.clamp(f - delta_alpha, 0.0, 1.0)

        dt_t = torch.tensor(dt_seg, device=device, dtype=torch.float32)

        def _apply_one(down, up, slope):
            use_down = torch.minimum(dt_t, down)
            d = (-slope) * use_down
            down2 = down - use_down
            dt_left = dt_t - use_down
            use_up = torch.minimum(dt_left, up)
            d = d + slope * use_up
            up2 = up - use_up
            return d, down2, up2

        d0, down0, up0 = _apply_one(down0, up0, slope0)
        d1, down1, up1 = _apply_one(down1, up1, slope1)
        f = torch.clamp(f + d0 + d1, 0.0, 1.0)

        f1 = f
        f_mid = 0.5 * (f0 + f1)
        lr = lr + (1.0 - f_mid) * dt_seg
        cf = _cf_torch(cfg, f_mid, f_crit_eff)
        risk = risk + torch.where(cf > 0.0, cf * dt_seg, torch.zeros_like(cf))
        floor_violation = floor_violation + \
            torch.clamp(f2 - f_mid, min=0.0) * dt_seg
        min_f = torch.minimum(min_f, f1)

    ret = _objective_return_tensor(
        cfg,
        lr=lr,
        risk=risk,
        lcc=cost,
        npv=npv,
        floor_violation=floor_violation,
    )

    ret = ret.view(pop_n, ep_n).mean(dim=1).detach().cpu().numpy()
    lr_m = lr.view(pop_n, ep_n).mean(dim=1).detach().cpu().numpy()
    risk_m = risk.view(pop_n, ep_n).mean(dim=1).detach().cpu().numpy()
    lcc_m = cost.view(pop_n, ep_n).mean(dim=1).detach().cpu().numpy()
    npv_m = npv.view(pop_n, ep_n).mean(dim=1).detach().cpu().numpy()
    minf_m = min_f.view(pop_n, ep_n).mean(dim=1).detach().cpu().numpy()
    feasible_frac = (min_f.view(pop_n, ep_n) >= float(cfg.f_2)).to(
        torch.float32).mean(dim=1).detach().cpu().numpy()

    out: list[dict[str, float]] = []
    for i in range(pop_n):
        out.append(
            {
                "return": float(ret[i]),
                "lr": float(lr_m[i]),
                "risk": float(risk_m[i]),
                "lcc": float(lcc_m[i]),
                "npv": float(npv_m[i]),
                "min_f": float(minf_m[i]),
                "feasible_frac": float(feasible_frac[i]),
            }
        )
    return out


def train_cem(
    cfg: LifecycleConfig,
    cem_cfg: CEMConfig,
    logger: logging.Logger,
    iter_trajectories_path: Path | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(cem_cfg.seed)
    state = _init_cem_state()

    elite_k = max(1, int(math.ceil(cem_cfg.population * cem_cfg.elite_frac)))

    history: list[dict[str, Any]] = []
    best_params: PolicyParams | None = None
    best_score = -float("inf")

    iter_trajectories: list[dict[str, Any]] = []
    compare_seed = _compare_seed_from_config(cem_cfg)
    device = _torch_device()
    use_gpu = (device == "cuda")
    if use_gpu:
        logger.info("Eval backend | torch device=%s", device)
    else:
        logger.info("Eval backend | numpy cpu")

    logger.info(
        "CEM start | iterations=%d population=%d elite_frac=%.3f eval_episodes=%d seed=%d",
        cem_cfg.iterations,
        cem_cfg.population,
        cem_cfg.elite_frac,
        cem_cfg.eval_episodes,
        cem_cfg.seed,
    )
    if iter_trajectories_path is not None:
        logger.info("Checkpoint | iter_trajectories_path=%s",
                    str(iter_trajectories_path))

    for it in range(cem_cfg.iterations):
        population = [_sample_params(rng, state)
                      for _ in range(cem_cfg.population)]
        base_seed_it = int(cem_cfg.seed + it * 7919)
        if use_gpu:
            scores = _evaluate_population_torch(
                cfg,
                population,
                base_seed_it=base_seed_it,
                episodes=cem_cfg.eval_episodes,
                device=device,
            )
        else:
            scores = [
                _evaluate_params(
                    cfg,
                    p,
                    base_seed=int(base_seed_it + j * 131),
                    episodes=cem_cfg.eval_episodes,
                )
                for j, p in enumerate(population)
            ]

        order = np.argsort([-s["return"] for s in scores]).tolist()
        elites = [population[i] for i in order[:elite_k]]
        elite_scores = [scores[i] for i in order[:elite_k]]

        elite_mat = np.array(
            [[e.maint_t1, e.maint_t2, e.repair_t1, e.repair_t2] for e in elites], dtype=float)
        mu = elite_mat.mean(axis=0)
        sigma = elite_mat.std(axis=0) + 1e-4

        pre_counts = np.zeros_like(state.pre_probs)
        for e in elites:
            pre_counts[e.pre_index] += 1.0
        pre_probs = (pre_counts / pre_counts.sum()).astype(float)

        state = CEMState(mu=mu, sigma=sigma, pre_probs=pre_probs)

        best_idx = order[0]
        it_best_params = population[best_idx]
        it_best_score = scores[best_idx]["return"]

        if it_best_score > best_score:
            best_score = it_best_score
            best_params = it_best_params
            logger.info(
                "New best | iter=%d return=%.4f lr=%.3f npv=%.2f min_f=%.3f feasible=%.2f params=%s",
                it,
                scores[best_idx]["return"],
                scores[best_idx]["lr"],
                scores[best_idx]["npv"],
                scores[best_idx]["min_f"],
                scores[best_idx]["feasible_frac"],
                json.dumps(asdict(it_best_params), ensure_ascii=False),
            )
        else:
            logger.info(
                "Iter=%d best | return=%.4f lr=%.3f npv=%.2f min_f=%.3f feasible=%.2f",
                it,
                scores[best_idx]["return"],
                scores[best_idx]["lr"],
                scores[best_idx]["npv"],
                scores[best_idx]["min_f"],
                scores[best_idx]["feasible_frac"],
            )

        ep = simulate_episode(cfg, it_best_params,
                              seed=compare_seed, record_hazard_steps=True)
        iter_trajectories.append(
            {
                "iteration": it,
                "compare_seed": int(compare_seed),
                "params": asdict(it_best_params),
                "score": dict(scores[best_idx]),
                "episode": {
                    "t_years": ep.t_years,
                    "f": ep.f,
                    "cumulative_cost": ep.cumulative_cost,
                    "lr": ep.lr,
                    "risk": ep.risk,
                    "lcc": ep.lcc,
                    "npv": ep.npv,
                    "min_f": ep.min_f,
                    "feasible": ep.feasible,
                    "return_value": ep.return_value,
                },
            }
        )
        if iter_trajectories_path is not None:
            _write_text_atomic(
                iter_trajectories_path,
                json.dumps(iter_trajectories, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        logger.info(
            "Iter=%d dist | mu=%s sigma=%s pre_probs=%s elite_mean=%s",
            it,
            json.dumps(mu.tolist(), ensure_ascii=False),
            json.dumps(sigma.tolist(), ensure_ascii=False),
            json.dumps(pre_probs.tolist(), ensure_ascii=False),
            json.dumps(
                {
                    "return": float(np.mean([s["return"] for s in elite_scores])),
                    "lr": float(np.mean([s["lr"] for s in elite_scores])),
                    "risk": float(np.mean([s["risk"] for s in elite_scores])),
                    "lcc": float(np.mean([s["lcc"] for s in elite_scores])),
                    "npv": float(np.mean([s["npv"] for s in elite_scores])),
                    "min_f": float(np.mean([s["min_f"] for s in elite_scores])),
                    "feasible_frac": float(np.mean([s["feasible_frac"] for s in elite_scores])),
                },
                ensure_ascii=False,
            ),
        )

        history.append(
            {
                "iteration": it,
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "pre_probs": pre_probs.tolist(),
                "best_of_iter": {"params": asdict(it_best_params), **scores[best_idx]},
                "elite_mean": {
                    "return": float(np.mean([s["return"] for s in elite_scores])),
                    "lr": float(np.mean([s["lr"] for s in elite_scores])),
                    "risk": float(np.mean([s["risk"] for s in elite_scores])),
                    "lcc": float(np.mean([s["lcc"] for s in elite_scores])),
                    "npv": float(np.mean([s["npv"] for s in elite_scores])),
                    "min_f": float(np.mean([s["min_f"] for s in elite_scores])),
                    "feasible_frac": float(np.mean([s["feasible_frac"] for s in elite_scores])),
                },
            }
        )

    if best_params is None:
        raise RuntimeError("No policy params were produced.")

    reward_best_eval = _evaluate_params(
        cfg, best_params, base_seed=cem_cfg.seed + 99991, episodes=cem_cfg.eval_episodes * 2)
    reward_best_episode = simulate_episode(
        cfg, best_params, seed=compare_seed, record_hazard_steps=True)

    result = {
        "lifecycle_config": asdict(cfg),
        "cem_config": asdict(cem_cfg),
        "compare_seed": int(compare_seed),
        "history": history,
        "iter_trajectories": iter_trajectories,
        "best": {
            "params": asdict(best_params),
            "eval": reward_best_eval,
            "selection": {
                "method": "max_training_return_before_pareto_final_selection",
                "training_best_return": float(best_score),
            },
            "episode_compare_seed": {
                "t_years": reward_best_episode.t_years,
                "f": reward_best_episode.f,
                "cumulative_cost": reward_best_episode.cumulative_cost,
                "lr": reward_best_episode.lr,
                "risk": reward_best_episode.risk,
                "lcc": reward_best_episode.lcc,
                "npv": reward_best_episode.npv,
                "min_f": reward_best_episode.min_f,
                "feasible": reward_best_episode.feasible,
                "return_value": reward_best_episode.return_value,
            },
        },
    }
    if str(getattr(cem_cfg, "final_selection", "max_return")).lower() in {
        "pareto",
        "pareto_cost_risk",
        "pareto_cost_risk_then_full_objective",
    }:
        result = _apply_pareto_final_selection(result, cfg, cem_cfg, logger=logger)
    else:
        result = _apply_max_return_final_selection(result, cfg, cem_cfg)

    logger.info(
        "CEM done | final_return=%.4f final_eval=%s final_params=%s final_selection=%s",
        float(result["best"]["eval"]["return"]),
        json.dumps(result["best"]["eval"], ensure_ascii=False),
        json.dumps(result["best"]["params"], ensure_ascii=False),
        json.dumps(result["best"].get("selection", {}), ensure_ascii=False),
    )

    return result


def _iter_color(i: int, n: int) -> tuple[tuple[float, float, float, float], float, float]:
    # 固定蓝色，不使用渐变
    color = (0.2, 0.5, 0.8, 1.0)  # 固定蓝色
    alpha = 1.0  # 完全不透明
    linewidth = _lw(1)  # 固定线宽
    return color, alpha, linewidth


def _plot_iterations(
    cfg: LifecycleConfig,
    iter_trajectories: list[dict[str, Any]],
    out_path: Path,
    *,
    f_crit_line: float | None = None,
    plot_first_n: int | None = None,
    plot_interval: int = 1,
) -> None:
    fig = plt.figure(figsize=(16, 11))

    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[
                          3, 1, 1], hspace=0.35, wspace=0.3)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_f)
    ax_metrics = fig.add_subplot(gs[2, 0])
    ax_colorbar = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 1])
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_compare = fig.add_subplot(gs[0, 2])
    ax_params = fig.add_subplot(gs[1, 2])
    ax_improve = fig.add_subplot(gs[2, 2])

    window_years = 30.0
    interval = float(cfg.maintenance_interval_years)
    decision_times = np.arange(0.0, float(
        cfg.horizon_years) + 1.0e-9, interval, dtype=float)

    def _add_fade_windows(ax: Any, t: np.ndarray, y: np.ndarray, rgb: tuple[float, float, float], base_alpha: float, base_lw: float) -> None:
        for s in decision_times:
            m = (t >= s) & (t <= s + window_years + 1.0e-9)
            tt = t[m]
            yy = y[m]
            if tt.size < 2:
                continue
            pts = np.column_stack([tt, yy]).astype(float)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            mid_t = 0.5 * (tt[:-1] + tt[1:])
            fade = 1.0 - np.clip((mid_t - s) / window_years, 0.0, 1.0)
            alphas = (base_alpha * (fade**1.35)).astype(float)
            colors = np.column_stack([np.full_like(alphas, rgb[0]), np.full_like(
                alphas, rgb[1]), np.full_like(alphas, rgb[2]), alphas])
            lc = LineCollection(segs, colors=colors, linewidths=float(
                base_lw), capstyle="round", joinstyle="round")
            ax.add_collection(lc)

    n = len(iter_trajectories)

    returns = []
    lrs = []
    npvs = []
    risks = []
    min_fs = []

    best_idx = 0
    best_return = -float("inf")

    for i, rec in enumerate(iter_trajectories):
        ep = rec["episode"]
        score = rec.get("score")
        if isinstance(score, dict) and "return" in score:
            ret = float(score["return"])
        else:
            ret = float(ep["return_value"])
        returns.append(ret)
        lrs.append(ep["lr"])
        npvs.append(ep["npv"])
        risks.append(ep["risk"])
        min_fs.append(ep["min_f"])

        if ret > best_return:
            best_return = ret
            best_idx = i

    if plot_first_n is None:
        n_show = n
    else:
        n_show = int(max(0, min(n, plot_first_n)))
    plot_interval = int(max(1, plot_interval))

    plot_indices = list(range(0, n_show, plot_interval))
    plot_indices = sorted(set(plot_indices))

    for i in plot_indices:
        rec = iter_trajectories[i]
        ep = rec["episode"]
        t = np.array(ep["t_years"], dtype=float)
        f = np.array(ep["f"], dtype=float)
        cc = np.array(ep["cumulative_cost"], dtype=float)
        color, alpha, lw = _iter_color(i, n)
        rgb = (float(color[0]), float(color[1]), float(color[2]))
        _add_fade_windows(ax_f, t, f, rgb, alpha, lw)
        _add_fade_windows(ax_c, t, cc, rgb, alpha, lw)

    compare_seed = int(iter_trajectories[0].get("compare_seed", 0))
    baseline_ep = simulate_episode(
        cfg, _baseline_policy_params(), seed=compare_seed, record_hazard_steps=True)
    t_base = np.array(baseline_ep.t_years, dtype=float)
    f_base = np.array(baseline_ep.f, dtype=float)
    cc_base = np.array(baseline_ep.cumulative_cost, dtype=float)
    ax_f.plot(t_base, f_base, color="tab:red", linewidth=_lw(2.2),
              linestyle="-", label="Initial", zorder=9, alpha=0.9)
    ax_c.plot(t_base, cc_base, color="tab:red", linewidth=_lw(2.0),
              linestyle="-", label="Initial", zorder=9, alpha=0.9)

    best_ep = iter_trajectories[best_idx]["episode"]
    t_best = np.array(best_ep["t_years"], dtype=float)
    f_best = np.array(best_ep["f"], dtype=float)
    cc_best = np.array(best_ep["cumulative_cost"], dtype=float)
    ax_f.plot(t_best, f_best, color="tab:green", linewidth=_lw(2.8),
              label="Optimal", zorder=10)
    ax_c.plot(t_best, cc_best, color="tab:green", linewidth=_lw(2.4),
              label="Optimal", zorder=10)

    ax_f.axhline(cfg.f_crit if f_crit_line is None else float(
        f_crit_line), color="tab:red", linestyle="--", linewidth=1.2, label="$F_{crit}$")
    ax_f.axhline(cfg.f_1, color="tab:orange", linestyle="--",
                 linewidth=1.0, label="$F_1$")
    ax_f.axhline(cfg.f_2, color="tab:purple", linestyle="--",
                 linewidth=1.0, label="$F_2$")

    ax_f.set_ylabel("Functionality F(t)", fontsize=11)
    ax_f.set_ylim(-0.02, 1.02)
    ax_f.set_xlim(0.0, float(cfg.horizon_years))
    ax_f.grid(True, alpha=0.25)
    ax_f.legend(loc="lower left", fontsize=9)

    ax_c.set_ylabel("Cost", fontsize=11)
    ax_c.set_xlabel("Time (years)", fontsize=11)
    ax_c.grid(True, alpha=0.25)

    iterations = list(range(n))

    # Check if return values span large range (use log scale)
    min_ret = min(returns)
    max_ret = max(returns)
    use_log = abs(min_ret) > 1000 or abs(max_ret) > 1000

    if use_log:
        ax_metrics.semilogy(iterations, [abs(r) + 1 for r in returns], "o-",
                            color="tab:blue", linewidth=1.5, markersize=4, label="|Return| (log)")
        ax_metrics.axhline(abs(best_return) + 1, color="tab:green",
                           linestyle="--", linewidth=1.0, alpha=0.7)
        ax_metrics.scatter([best_idx], [abs(best_return) + 1],
                           color="tab:green", s=80, zorder=10, marker="*")
        ax_metrics.set_ylabel("|Return| (log scale)", fontsize=11)
    else:
        ax_metrics.plot(iterations, returns, "o-", color="tab:blue",
                        linewidth=1.5, markersize=4, label="Return")
        ax_metrics.axhline(best_return, color="tab:green",
                           linestyle="--", linewidth=1.0, alpha=0.7)
        ax_metrics.scatter([best_idx], [best_return],
                           color="tab:green", s=80, zorder=10, marker="*")
        ax_metrics.set_ylabel("Return", fontsize=11)

    ax_metrics.set_xlabel("Iteration", fontsize=11)
    ax_metrics.grid(True, alpha=0.25)
    ax_metrics.legend(loc="upper right", fontsize=9)

    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=0, vmax=max(1, n - 1))
    cb = plt.colorbar(plt.cm.ScalarMappable(
        norm=norm, cmap=cmap), cax=ax_colorbar)
    cb.set_label("Iteration Progress", fontsize=10)
    if _PLOT_TITLES:
        ax_colorbar.set_title("Color Legend", fontsize=10, fontweight="bold")

    ax_legend.axis("off")
    legend_text = (
        f"RL Optimization\n"
        f"{'─' * 16}\n"
        f"Iterations: {n}\n"
        f"{'─' * 16}\n"
        f"Red = Baseline\n"
        f"Green = Optimal\n"
        f"Blue gradient = Training"
    )
    ax_legend.text(0.1, 0.5, legend_text, transform=ax_legend.transAxes, fontsize=10, verticalalignment="center",
                   fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax_stats.axis("off")
    stats_text = (
        f"Performance\n"
        f"{'─' * 16}\n"
        f"Return:\n  {returns[0]:.2e}\n  → {returns[-1]:.2e}\n"
        f"Risk:\n  {risks[0]:.1f}\n  → {risks[-1]:.1f}\n"
        f"NPV:\n  {npvs[0]:.1f}\n  → {npvs[-1]:.1f}"
    )
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes, fontsize=9, verticalalignment="center",
                  fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    # Compare first vs best
    ax_compare.plot(t_base, f_base, color="tab:red",
                    linewidth=_lw(2.0), label="Initial", alpha=0.8)
    ax_compare.plot(t_best, f_best, color="tab:green",
                    linewidth=_lw(2.5), label="Optimal")
    ax_compare.axhline(cfg.f_crit if f_crit_line is None else float(
        f_crit_line), color="tab:red", linestyle="--", linewidth=_lw(1.0))
    ax_compare.axhline(cfg.f_2, color="tab:purple",
                       linestyle="--", linewidth=_lw(0.8))
    ax_compare.set_ylabel("F(t)", fontsize=10)
    ax_compare.set_xlabel("Time (years)", fontsize=10)
    if _PLOT_TITLES:
        ax_compare.set_title("Baseline vs Optimal Policy",
                             fontsize=10, fontweight="bold")
    ax_compare.legend(loc="lower left", fontsize=8)
    ax_compare.grid(True, alpha=0.25)
    ax_compare.set_ylim(-0.02, 1.02)

    # Show params evolution
    pre_indices = [rec["params"]["pre_index"] for rec in iter_trajectories]
    ax_params.plot(iterations, pre_indices, "o-",
                   color="tab:purple", linewidth=_lw(1.2), markersize=3)
    ax_params.set_xlabel("Iteration", fontsize=10)
    ax_params.set_ylabel("Pre-reinforcement Level", fontsize=10)
    if _PLOT_TITLES:
        ax_params.set_title("Policy Parameters",
                            fontsize=10, fontweight="bold")
    ax_params.set_yticks([0, 1, 2])
    ax_params.set_yticklabels(["None", "Medium", "High"])
    ax_params.grid(True, alpha=0.25)

    # Improvement summary
    ax_improve.axis("off")
    improve_return = returns[-1] - returns[0]
    improve_risk = risks[-1] - risks[0]
    improve_npv = npvs[-1] - npvs[0]
    improve_text = (
        f"Improvement\n"
        f"{'─' * 16}\n"
        f"Return:\n  Δ = {improve_return:.2e}\n"
        f"Risk:\n  Δ = {improve_risk:.1f}\n"
        f"NPV:\n  Δ = {improve_npv:.1f}\n"
        f"{'─' * 16}\n"
        f"Min F: {min_fs[0]:.3f}\n  → {min_fs[-1]:.3f}"
    )
    ax_improve.text(0.1, 0.5, improve_text, transform=ax_improve.transAxes, fontsize=9, verticalalignment="center",
                    fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    if _PLOT_TITLES:
        fig.suptitle("RL Iterative Optimization: Policy Improvement Over Training",
                     fontsize=13, fontweight="bold", y=0.98)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_best_detail(cfg: LifecycleConfig, best_episode: dict[str, Any], out_path: Path, *, f_crit_line: float | None = None) -> None:
    t = np.array(best_episode["t_years"], dtype=float)
    f = np.array(best_episode["f"], dtype=float)
    cc = np.array(best_episode["cumulative_cost"], dtype=float)

    with plt.rc_context(_tex_rc_params()):
        fig, (ax_f, ax_c) = plt.subplots(
            2,
            1,
            figsize=_figsize_for_tex(
                _TEX_TEXTWIDTH_IN, _TEX_TEXTWIDTH_IN * 0.78),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        color = plt.cm.Greens(0.80)
        ax_f.plot(t, f, color=color, linewidth=_lw(2.6))
        ax_f.fill_between(t, 1.0, f, color=color, alpha=0.10)
        ax_f.axhline(cfg.f_crit if f_crit_line is None else float(
            f_crit_line), color="tab:red", linestyle="--", linewidth=_lw(1.2))
        ax_f.axhline(cfg.f_1, color="tab:orange",
                     linestyle="--", linewidth=_lw(1.0))
        ax_f.axhline(cfg.f_2, color="tab:purple",
                     linestyle="--", linewidth=_lw(1.0))
        ax_f.set_ylabel("Functionality F(t)")
        ax_f.set_ylim(-0.02, 1.02)
        ax_f.grid(True, alpha=0.25)

        ax_c.plot(t, cc, color=color, linewidth=_lw(2.0))
        ax_c.set_ylabel("Cost")
        ax_c.set_xlabel("Time (years)")
        ax_c.grid(True, alpha=0.25)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def _plot_f_over_time_by_iteration(
    cfg: LifecycleConfig,
    iter_trajectories: list[dict[str, Any]],
    out_path: Path,
    *,
    f_crit_line: float | None = None,
    plot_first_n: int | None = None,
    plot_interval: int = 1,
) -> None:
    with plt.rc_context(_tex_rc_params()):
        fig, ax_pf = plt.subplots(
            1,
            1,
            figsize=_figsize_for_tex(
                _TEX_HALF_WIDTH_IN, _TEX_HALF_WIDTH_IN * 0.72),
        )

        window_years = 30.0
        interval = float(cfg.maintenance_interval_years)
        decision_times = np.arange(0.0, float(
            cfg.horizon_years) + 1.0e-9, interval, dtype=float)

        def _add_fade_windows(
            ax: Any,
            t: np.ndarray,
            y: np.ndarray,
            rgb: tuple[float, float, float],
            base_alpha: float,
            base_lw: float,
        ) -> None:
            for s in decision_times:
                m = (t >= s) & (t <= s + window_years + 1.0e-9)
                tt = t[m]
                yy = y[m]
                if tt.size < 2:
                    continue
                pts = np.column_stack([tt, yy]).astype(float)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                mid_t = 0.5 * (tt[:-1] + tt[1:])
                fade = 1.0 - np.clip((mid_t - s) / window_years, 0.0, 1.0)
                alphas = (base_alpha * (fade**1.35)).astype(float)
                colors = np.column_stack(
                    [
                        np.full_like(alphas, rgb[0]),
                        np.full_like(alphas, rgb[1]),
                        np.full_like(alphas, rgb[2]),
                        alphas,
                    ]
                )
                lc = LineCollection(segs, colors=colors, linewidths=float(
                    base_lw), capstyle="round", joinstyle="round")
                ax.add_collection(lc)

        n = len(iter_trajectories)
        best_idx = _selected_iter_index(iter_trajectories)

        if plot_first_n is None:
            n_show = n
        else:
            n_show = int(max(0, min(n, plot_first_n)))
        plot_interval = int(max(1, plot_interval))
        plot_indices = sorted(set(range(0, n_show, plot_interval)))

        for i in plot_indices:
            rec = iter_trajectories[i]
            ep = rec["episode"]
            t = np.array(ep["t_years"], dtype=float)
            f = np.array(ep["f"], dtype=float)
            color, alpha, lw = _iter_color(i, n)
            rgb = (float(color[0]), float(color[1]), float(color[2]))
            _add_fade_windows(ax_pf, t, f, rgb, alpha, lw)

        compare_seed = int(iter_trajectories[0].get("compare_seed", 0))
        baseline_ep = simulate_episode(
            cfg, _baseline_policy_params(), seed=compare_seed, record_hazard_steps=True)
        t_base = np.array(baseline_ep.t_years, dtype=float)
        f_base = np.array(baseline_ep.f, dtype=float)
        ax_pf.plot(t_base, f_base, color="tab:red", linewidth=_lw(
            1.8), linestyle="-", label="Initial", zorder=9, alpha=0.9)

        best_ep = iter_trajectories[best_idx]["episode"]
        t_best = np.array(best_ep["t_years"], dtype=float)
        f_best = np.array(best_ep["f"], dtype=float)
        ax_pf.plot(t_best, f_best, color="tab:green", linewidth=_lw(
            2.2), label="Optimal", zorder=10)

        fcrit = cfg.f_crit if f_crit_line is None else float(f_crit_line)
        ax_pf.axhline(fcrit, color="tab:red",
                      linestyle="--", linewidth=_lw(1.2))
        ax_pf.axhline(cfg.f_1, color="tab:orange",
                      linestyle="--", linewidth=_lw(1.0))
        ax_pf.axhline(cfg.f_2, color="tab:purple",
                      linestyle="--", linewidth=_lw(1.0))
        ax_pf.set_ylim(-0.02, 1.02)
        ax_pf.grid(True, alpha=0.25)

        if _PLOT_TITLES:
            ax_pf.set_title("Training trajectories: Functionality",
                            fontweight="bold")
        ax_pf.set_ylabel("Functionality F(t)")
        ax_pf.set_xlabel("Time (years)")
        ax_pf.set_xlim(0.0, float(cfg.horizon_years))
        ax_pf.legend(loc="lower right", ncols=1,
                     fontsize=_TEX_BODY_FONT_PT - 1.0)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def _plot_cost_over_time_by_iteration(
    cfg: LifecycleConfig,
    iter_trajectories: list[dict[str, Any]],
    out_path: Path,
    *,
    plot_first_n: int | None = None,
    plot_interval: int = 1,
) -> None:
    with plt.rc_context(_tex_rc_params()):
        fig, ax = plt.subplots(
            1,
            1,
            figsize=_figsize_for_tex(
                _TEX_HALF_WIDTH_IN, _TEX_HALF_WIDTH_IN * 0.72),
        )

        window_years = 30.0
        interval = float(cfg.maintenance_interval_years)
        decision_times = np.arange(0.0, float(
            cfg.horizon_years) + 1.0e-9, interval, dtype=float)

        def _add_fade_windows(
            ax: Any,
            t: np.ndarray,
            y: np.ndarray,
            rgb: tuple[float, float, float],
            base_alpha: float,
            base_lw: float,
        ) -> None:
            for s in decision_times:
                m = (t >= s) & (t <= s + window_years + 1.0e-9)
                tt = t[m]
                yy = y[m]
                if tt.size < 2:
                    continue
                pts = np.column_stack([tt, yy]).astype(float)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                mid_t = 0.5 * (tt[:-1] + tt[1:])
                fade = 1.0 - np.clip((mid_t - s) / window_years, 0.0, 1.0)
                alphas = (base_alpha * (fade**1.35)).astype(float)
                colors = np.column_stack(
                    [
                        np.full_like(alphas, rgb[0]),
                        np.full_like(alphas, rgb[1]),
                        np.full_like(alphas, rgb[2]),
                        alphas,
                    ]
                )
                lc = LineCollection(segs, colors=colors, linewidths=float(
                    base_lw), capstyle="round", joinstyle="round")
                ax.add_collection(lc)

        n = len(iter_trajectories)
        if n == 0:
            return
        best_idx = _selected_iter_index(iter_trajectories)

        if plot_first_n is None:
            n_show = n
        else:
            n_show = int(max(0, min(n, plot_first_n)))
        plot_interval = int(max(1, plot_interval))
        plot_indices = sorted(set(range(0, n_show, plot_interval)))

        for i in plot_indices:
            rec = iter_trajectories[i]
            ep = rec["episode"]
            t = np.array(ep["t_years"], dtype=float)
            cc = np.array(ep["cumulative_cost"], dtype=float)
            color, alpha, lw = _iter_color(i, n)
            rgb = (float(color[0]), float(color[1]), float(color[2]))
            _add_fade_windows(ax, t, cc, rgb, alpha, lw)

        compare_seed = int(iter_trajectories[0].get("compare_seed", 0))
        baseline_ep = simulate_episode(
            cfg, _baseline_policy_params(), seed=compare_seed, record_hazard_steps=True)
        t_base = np.array(baseline_ep.t_years, dtype=float)
        cc_base = np.array(baseline_ep.cumulative_cost, dtype=float)
        ax.plot(t_base, cc_base, color="tab:red", linewidth=_lw(
            1.8), linestyle="-", label="Initial", zorder=9, alpha=0.9)

        best_ep = iter_trajectories[best_idx]["episode"]
        t_best = np.array(best_ep["t_years"], dtype=float)
        cc_best = np.array(best_ep["cumulative_cost"], dtype=float)
        ax.plot(t_best, cc_best, color="tab:green", linewidth=_lw(
            2.2), label="Optimal", zorder=10)

        ax.set_xlim(0.0, float(cfg.horizon_years))
        fig.canvas.draw()
        tick_leader_w = float(plt.rcParams.get("ytick.major.width", 1.0))
        tick_leader_len_pts = float(plt.rcParams.get("ytick.major.size", 3.5))
        ax_bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
        tick_leader_len_px = tick_leader_len_pts * fig.dpi / 72.0
        tick_leader_len_ax = float(
            tick_leader_len_px / max(1.0, ax_bbox.width))
        x_range = float(cfg.horizon_years)
        tick_leader_len_data = tick_leader_len_ax * x_range
        x_left_seg = -tick_leader_len_data
        x_label_anchor_ax = 0.015
        x_line_start_ax = 0.13

        def _label_end_left(yv: float, *, color: str, x_end: float) -> None:
            yv = float(yv)
            x1 = float(max(0.0, x_end))
            x0 = float(max(0.0, x_line_start_ax * float(cfg.horizon_years)))
            ax.plot([x0, x1], [yv, yv], color=color,
                    linestyle=(0, (4.0, 2.0)), linewidth=_lw(1.0), clip_on=False, zorder=20)

        def _label_start_left(yv: float, *, x_to: float) -> None:
            yv = float(yv)
            x1 = float(max(0.0, x_to))
            ax.plot([x_left_seg, x1], [yv, yv], color="black",
                    linestyle=(0, (4.0, 2.0)), linewidth=_lw(1.0), clip_on=False, zorder=20)

        y0_base = float(cc_base[0]) if cc_base.size else 0.0
        y0_best = float(cc_best[0]) if cc_best.size else 0.0
        y1_base = float(cc_base[-1]) if cc_base.size else 0.0
        y1_best = float(cc_best[-1]) if cc_best.size else 0.0
        if max(y1_base, y1_best) >= 100.0:
            x_line_start_ax = 0.16
        case_key = out_path.parent.name.lower()
        if case_key == "case2a":
            tick_vals = [0.0, 10.0, 30.0, 60.0, 90.0, 120.0, 130.0]
            ax.set_yticks(tick_vals)
            ax.set_yticklabels([str(int(v)) for v in tick_vals])
            ax.set_ylim(0.0, 130.0)
            major_ticks = np.array(tick_vals, dtype=float)
        elif case_key == "case3b":
            tick_vals = [0.0, 10.0, 30.0, 60.0, 90.0, 130.0]
            ax.set_yticks(tick_vals)
            ax.set_yticklabels([str(int(v)) for v in tick_vals])
            ax.set_ylim(0.0, 130.0)
            major_ticks = np.array(tick_vals, dtype=float)
        elif case_key == "case3c":
            y_top = 110.0
            if max(y1_base, y1_best) > 108.0:
                y_top = float(math.ceil((max(y1_base, y1_best) + 4.0) / 10.0) * 10.0)
            tick_vals = [0.0, 10.0, 30.0, 60.0, 90.0, 110.0]
            if y_top > 110.0:
                tick_vals.append(y_top)
            ax.set_yticks(tick_vals)
            ax.set_yticklabels([str(int(v)) for v in tick_vals])
            ax.set_ylim(0.0, y_top)
            major_ticks = np.array(tick_vals, dtype=float)
        else:
            major_ticks = np.array(ax.get_yticks(), dtype=float)
            tick_vals = sorted(
                set([float(v) for v in major_ticks.tolist() + [y0_base]]))
            ax.set_yticks(tick_vals)
            tick_labels = []
            for v in tick_vals:
                if abs(v - y0_base) < 1.0e-6:
                    tick_labels.append(f"{v:.1f}")
                elif abs(v - round(v)) < 1.0e-9:
                    tick_labels.append(f"{int(round(v))}")
                else:
                    tick_labels.append(f"{v:g}")
            ax.set_yticklabels(tick_labels)

        minor_ticks = []
        for v in [y1_base, y1_best]:
            if major_ticks.size == 0 or float(np.min(np.abs(major_ticks - float(v)))) > 1.0e-6:
                minor_ticks.append(float(v))
        if minor_ticks:
            ax.set_yticks(sorted(set(minor_ticks)), minor=True)
            ax.tick_params(axis="y", which="minor", left=True, right=False,
                           length=tick_leader_len_pts, width=tick_leader_w, labelleft=False)

        x_start_conn = min(float(t_base[0]) if t_base.size else 0.0, float(
            t_best[0]) if t_best.size else 0.0)
        if abs(y0_base - y0_best) <= 1.0e-9:
            _label_start_left(y0_base, x_to=x_start_conn)
        else:
            _label_start_left(y0_base, x_to=x_start_conn)
            _label_start_left(y0_best, x_to=x_start_conn)

        _label_end_left(y1_base, color="tab:red", x_end=float(
            t_base[-1]) if t_base.size else float(cfg.horizon_years))
        _label_end_left(y1_best, color="tab:green", x_end=float(
            t_best[-1]) if t_best.size else float(cfg.horizon_years))

        major_now = np.array(ax.get_yticks(), dtype=float)
        if major_now.size == 0:
            major_now = np.array([0.0], dtype=float)
        ylab_base = float(y1_base)
        ylab_best = float(y1_best)
        yoff_base = 0.0
        yoff_best = 0.0
        if abs(ylab_best - ylab_base) < 4.0:
            if ylab_base >= ylab_best:
                yoff_base = 8.0
                yoff_best = -8.0
            else:
                yoff_base = -8.0
                yoff_best = 8.0

        if case_key == "case1":
            ylab_base = float(y1_base) + 4.5
            ylab_best = float(y1_best) - 5.3
            yoff_base = 0.0
            yoff_best = 0.0
        elif case_key == "case2a":
            yoff_best -= 6.0
        elif case_key == "case2b":
            ylim0, ylim1 = ax.get_ylim()
            yspan = float(max(1.0, float(ylim1) - float(ylim0)))
            ylab_base = float(ylim1) - 0.080 * yspan
            ylab_best = float(ylim1) - 0.205 * yspan
            yoff_base = 4.0
            yoff_best = 0.0
        elif case_key == "case3a":
            yoff_base = 0.0
            yoff_best = 0.0
        elif case_key == "case3b":
            yoff_base = 6.0
            yoff_best = -4.0 if y1_best > 91.0 else 2.0
        elif case_key == "case3c":
            ylab_base = float(y1_base)
            ylab_best = float(y1_best)
            yoff_base = 0.0
            yoff_best = -2.0 if y1_best > 110.0 else (-10.0 if y1_best > 91.0 else -6.0)
        elif case_key in {"case4a", "case4b"}:
            if y1_base >= y1_best:
                ylab_best = float(y1_best) - 7.0
                yoff_best = 0.0

        ax.annotate(f"{y1_base:.1f}", xy=(x_label_anchor_ax, ylab_base), xycoords=ax.get_yaxis_transform(),
                    xytext=(0.0, yoff_base), textcoords="offset points", ha="left", va="center",
                    color="tab:red", fontsize=_TEX_BODY_FONT_PT - 1.0, clip_on=False, zorder=8)
        ax.annotate(f"{y1_best:.1f}", xy=(x_label_anchor_ax, ylab_best), xycoords=ax.get_yaxis_transform(),
                    xytext=(0.0, yoff_best), textcoords="offset points", ha="left", va="center",
                    color="tab:green", fontsize=_TEX_BODY_FONT_PT - 1.0, clip_on=False, zorder=8)

        if _PLOT_TITLES:
            ax.set_title("Training trajectories: Cost", fontweight="bold")
        ax.set_ylabel("Cost")
        ax.set_xlabel("Time (years)")
        ax.yaxis.labelpad = 2
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", ncols=1,
                  fontsize=_TEX_BODY_FONT_PT - 1.0)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def _plot_rl_eval_triplet(history: list[dict[str, Any]], out_dir: Path) -> None:
    iterations = [int(h["iteration"]) for h in history]

    best_ret = np.array([float(h["best_of_iter"]["return"])
                        for h in history], dtype=float)
    elite_ret = np.array([float(h["elite_mean"]["return"])
                         for h in history], dtype=float)
    best_lr = np.array([float(h["best_of_iter"]["lr"])
                       for h in history], dtype=float)
    elite_lr = np.array([float(h["elite_mean"]["lr"])
                        for h in history], dtype=float)

    def _normalize_loss(y_best: np.ndarray, y_elite: np.ndarray, *, scale: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
        loss_best = np.array(y_best, dtype=float)
        loss_elite = np.array(y_elite, dtype=float)
        loss0 = float(loss_best[0]) if loss_best.size else 1.0
        if not np.isfinite(loss0) or loss0 <= 0.0:
            loss0 = 1.0
        loss_min = float(np.min(np.concatenate([loss_best, loss_elite]))) if (loss_best.size and loss_elite.size) else float(
            np.min(loss_best)) if loss_best.size else float(np.min(loss_elite)) if loss_elite.size else 0.0
        denom = float(loss0 - loss_min)
        if not np.isfinite(denom) or abs(denom) <= 1.0e-12:
            denom = 1.0
        nb = np.clip((loss_best - loss_min) / denom, 0.0, None) * float(scale)
        ne = np.clip((loss_elite - loss_min) / denom, 0.0, None) * float(scale)
        return nb, ne

    def _save_series(title: str, x_label: str, y_label: str, y_best: np.ndarray, y_elite: np.ndarray, out_path: Path) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.6))
        lo = np.minimum(y_best, y_elite)
        hi = np.maximum(y_best, y_elite)
        ax.fill_between(iterations, lo, hi, color="tab:green",
                        alpha=0.15, linewidth=0.0)
        ax.plot(iterations, y_elite, color="tab:green", alpha=0.55,
                linewidth=1.5, linestyle="--", label="Elite mean")
        ax.plot(iterations, y_best, color="tab:green",
                alpha=0.95, linewidth=2.2, label="Best of iter")
        if _PLOT_TITLES:
            ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

    loss_best = -best_ret
    loss_elite = -elite_ret
    loss_best_n, loss_elite_n = _normalize_loss(
        loss_best, loss_elite, scale=1000.0)
    _save_series("Loss per Step", "Iteration", "Loss",
                 loss_best_n, loss_elite_n, out_dir / "loss_per_step.png")
    _save_series("Reward per Episode", "Iteration", "Return",
                 best_ret, elite_ret, out_dir / "reward_per_episode.png")
    _save_series("LoR per Episode", "Iteration", "Resilience loss",
                 best_lr, elite_lr, out_dir / "lor_per_episode.png")


def _plot_lr_risk_over_time_by_iteration(
    cfg: LifecycleConfig,
    iter_trajectories: list[dict[str, Any]],
    out_dir: Path,
    *,
    plot_first_n: int | None = None,
    plot_interval: int = 1,
) -> None:
    with plt.rc_context(_tex_rc_params()):
        n = len(iter_trajectories)
        if n == 0:
            return

        if plot_first_n is None:
            n_show = n
        else:
            n_show = int(max(0, min(n, plot_first_n)))
        plot_interval = int(max(1, plot_interval))
        plot_indices = sorted(set(range(0, n_show, plot_interval)))

        best_idx = _selected_iter_index(iter_trajectories)

        def _plot_one(metric: str, title: str, y_label: str, out_path: Path) -> None:
            fig, ax = plt.subplots(
                1,
                1,
                figsize=_figsize_for_tex(_TEX_HALF_WIDTH_IN,
                                         _TEX_HALF_WIDTH_IN * 0.72),
            )

            for i in plot_indices:
                rec = iter_trajectories[i]
                ep = rec["episode"]
                pre_idx = int(rec["params"]["pre_index"])
                t, lr_cum, risk_cum = _cumulative_lr_and_risk(
                    cfg,
                    ep["t_years"],
                    ep["f"],
                    f_crit_eff=_effective_f_crit(cfg, pre_idx),
                )
                y = lr_cum if metric == "lr" else risk_cum
                color, alpha, lw = _iter_color(i, n)
                ax.plot(t, y, color=color, alpha=0.30,
                        linewidth=max(0.6, float(lw)))

            compare_seed = int(iter_trajectories[0].get("compare_seed", 0))
            base_params = _baseline_policy_params()
            base_ep = simulate_episode(
                cfg, base_params, seed=compare_seed, record_hazard_steps=True)
            t0, lr0, risk0 = _cumulative_lr_and_risk(
                cfg,
                base_ep.t_years,
                base_ep.f,
                f_crit_eff=_effective_f_crit(cfg, int(base_params.pre_index)),
            )
            y0 = lr0 if metric == "lr" else risk0
            ax.plot(t0, y0, color="tab:red", linewidth=_lw(
                2.0), label="Initial", alpha=0.9, zorder=30)

            best = iter_trajectories[best_idx]
            tb, lrb, riskb = _cumulative_lr_and_risk(
                cfg,
                best["episode"]["t_years"],
                best["episode"]["f"],
                f_crit_eff=_effective_f_crit(
                    cfg, int(best["params"]["pre_index"])),
            )
            yb = lrb if metric == "lr" else riskb
            ax.plot(tb, yb, color="tab:green", linewidth=_lw(
                2.4), label="Optimal", alpha=0.95, zorder=29)

            ax.set_xlim(0.0, float(cfg.horizon_years))
            try:
                y_max = float(np.nanmax(np.concatenate(
                    [np.asarray(y0, dtype=float), np.asarray(yb, dtype=float)])))
            except Exception:
                y_max = float(np.nanmax(np.asarray(y0, dtype=float))) if len(y0) else float(
                    np.nanmax(np.asarray(yb, dtype=float))) if len(yb) else 0.0
            if not np.isfinite(y_max) or y_max < 0.0:
                y_max = 0.0
            if metric == "risk":
                # Add extra headroom so thick top segments are not clipped by axes/tight bbox.
                y_top = y_max + max(0.8, 0.06 * max(1.0, y_max)) + 0.12
                ax.set_ylim(-0.5, y_top)
            else:
                # Keep visible padding at top to avoid half-clipped horizontal segments.
                y_top = y_max * 1.08 + 0.38
                if y_max <= 0.0:
                    y_top = 1.0
                ax.set_ylim(0.0, y_top)

            if _PLOT_TITLES:
                ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel(y_label)
            ax.yaxis.labelpad = 6 if metric == "risk" else 2
            ax.yaxis.set_label_coords(-0.135 if metric == "risk" else -0.075, 0.5)
            ax.tick_params(axis="y", pad=4 if metric == "risk" else 2)
            ax.tick_params(axis="x", pad=2)
            ax.grid(True, alpha=0.25)
            legend = ax.legend(loc="upper left", ncols=1, framealpha=0.9,
                               fontsize=_TEX_BODY_FONT_PT - 2.0, labelspacing=0.18, borderpad=0.25)
            legend.set_zorder(10)
            fig.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

        _plot_one(
            "lr",
            "Resilience Loss over Time",
            "Resilience loss",
            out_dir / "lr_over_time.png",
        )
        _plot_one(
            "risk",
            "Risk over Time",
            "Risk",
            out_dir / "risk_over_time.png",
        )


def _plot_rl_eval_combined(
    case_histories: dict[str, list[dict[str, Any]]],
    case_lifecycle_cfgs: dict[str, dict[str, Any]],
    fig_dir: Path,
    *,
    filename_prefix: str = "rl_eval",
) -> None:
    palette = ["#ff8c00", "#0066ff", "#00cc44",
               "#ff00cc", "#ff3333", "#00cfe6",
               "#7a5195", "#ef5675", "#ffa600"]

    def _case_sort_key(name: str) -> tuple[int, str]:
        known = {
            "case1_baseline": 1,
            "case2a_fire_dominant": 2,
            "case2b_eq_dominant": 3,
            "case3a_cost_oriented": 4,
            "case3b_risk_oriented": 5,
            "case3c_resilience_oriented": 6,
            "case4a_no_resilience_equal": 7,
            "case4b_no_resilience_risk_replacement": 8,
            "case4c_no_risk_cost_emphasis": 9,
        }
        return (known.get(name, 999), name)

    case_names = sorted(case_histories.keys(), key=_case_sort_key)
    if not case_names:
        return

    def _short_name(name: str) -> str:
        short_map = {
            "case1_baseline": "case1",
            "case2a_fire_dominant": "case2a",
            "case2b_eq_dominant": "case2b",
            "case3a_cost_oriented": "case3a",
            "case3b_risk_oriented": "case3b",
            "case3c_resilience_oriented": "case3c",
            "case4a_no_resilience_equal": "case4a",
            "case4b_no_resilience_risk_replacement": "case4b",
            "case4c_no_risk_cost_emphasis": "case4c",
        }
        return short_map.get(name, name)

    def _series_from_history(
        history: list[dict[str, Any]],
    ) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        iters = [int(h["iteration"]) for h in history]
        best_ret = np.array([float(h["best_of_iter"]["return"])
                            for h in history], dtype=float)
        elite_ret = np.array([float(h["elite_mean"]["return"])
                             for h in history], dtype=float)
        best_lr = np.array([float(h["best_of_iter"]["lr"])
                           for h in history], dtype=float)
        elite_lr = np.array([float(h["elite_mean"]["lr"])
                            for h in history], dtype=float)
        best_risk = np.array([float(h["best_of_iter"]["risk"])
                             for h in history], dtype=float)
        elite_risk = np.array([float(h["elite_mean"].get(
            "risk", h["best_of_iter"]["risk"])) for h in history], dtype=float)
        best_npv = np.array([float(h["best_of_iter"]["npv"])
                            for h in history], dtype=float)
        elite_npv = np.array([float(h["elite_mean"]["npv"])
                             for h in history], dtype=float)
        return iters, best_ret, elite_ret, best_lr, elite_lr, best_risk, elite_risk, best_npv, elite_npv

    def _moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
        w = int(max(1, window))
        n = int(y.size)
        if n < 2 or w <= 1:
            return y
        out = np.empty(n, dtype=float)
        for i in range(n):
            start = max(0, i - w + 1)
            out[i] = float(np.mean(y[start: i + 1]))
        return out

    def _rolling_std(y: np.ndarray, window: int = 15) -> np.ndarray:
        w = int(max(2, window))
        n = int(y.size)
        if n < 2:
            return np.zeros(n, dtype=float)
        out = np.empty(n, dtype=float)
        for i in range(n):
            start = max(0, i - w + 1)
            out[i] = float(np.std(y[start: i + 1]))
        return out

    def _ewm_std(y: np.ndarray, alpha: float = 0.12) -> np.ndarray:
        a = float(alpha)
        if a <= 0.0:
            a = 0.05
        if a >= 1.0:
            a = 0.95
        n = int(y.size)
        if n < 2:
            return np.zeros(n, dtype=float)
        mean = float(y[0])
        var = 0.0
        out = np.empty(n, dtype=float)
        out[0] = 0.0
        for i in range(1, n):
            x = float(y[i])
            prev_mean = mean
            mean = (1.0 - a) * mean + a * x
            var = (1.0 - a) * var + a * (x - prev_mean) * (x - mean)
            out[i] = float(math.sqrt(max(0.0, var)))
        return out

    def _plot_one(out_path: Path, title: str, y_label: str, y_getter: Any, fig_width: float, fig_height: float) -> None:
        with plt.rc_context(_tex_rc_params()):
            fig, ax = plt.subplots(
                1, 1, figsize=_figsize_for_tex(fig_width, fig_height))

            all_y_values = []
            for idx, case_name in enumerate(case_names):
                history = case_histories[case_name]
                lifecycle_cfg = case_lifecycle_cfgs.get(case_name, {})
                w_lr = float(lifecycle_cfg.get("w_lr", 1.0))
                w_cost = float(lifecycle_cfg.get("w_cost", 1.0))
                w_risk = float(lifecycle_cfg.get("w_risk", 1.0))

                iters, best_ret, elite_ret, best_lr, elite_lr, best_risk, elite_risk, best_npv, elite_npv = _series_from_history(
                    history)
                color = palette[idx % len(palette)]
                y_best, y_elite = y_getter(best_ret, elite_ret, best_lr, elite_lr,
                                           best_risk, elite_risk, best_npv, elite_npv, w_lr, w_cost, w_risk)
                y_best_raw = np.array(y_best, dtype=float)
                y_elite_raw = np.array(y_elite, dtype=float)
                y_best_s = _moving_average(y_best_raw, window=8)
                y_elite_s = _moving_average(y_elite_raw, window=8)

                std_best = _ewm_std(y_best_raw, alpha=0.12)
                std_elite = _ewm_std(y_elite_raw, alpha=0.12)

                ax.plot(iters, y_best_raw, color=color,
                        alpha=0.18, linewidth=_lw(1.2))
                ax.plot(iters, y_elite_raw, color=color,
                        alpha=0.10, linewidth=_lw(1.0))

                ax.fill_between(iters, y_best_s - std_best, y_best_s +
                                std_best, color=color, alpha=0.10, linewidth=0.0)
                ax.fill_between(iters, y_elite_s - std_elite, y_elite_s +
                                std_elite, color=color, alpha=0.06, linewidth=0.0)

                lo = np.minimum(y_best_s, y_elite_s)
                hi = np.maximum(y_best_s, y_elite_s)
                ax.fill_between(iters, lo, hi, color=color,
                                alpha=0.12, linewidth=0.0)

                ax.plot(iters, y_elite_s, color=color, alpha=0.55,
                        linewidth=_lw(1.6), linestyle="-")
                ax.plot(iters, y_best_s, color=color, alpha=0.95,
                        linewidth=_lw(2.0), label=_short_name(case_name))

                all_y_values.extend(y_best_raw.tolist())
                all_y_values.extend(y_elite_raw.tolist())

            if _PLOT_TITLES:
                ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(y_label)
            ax.tick_params(left=True, labelleft=True)
            ax.spines["left"].set_visible(True)
            ax.grid(True, alpha=0.25)
            legend_kwargs = dict(
                ncols=3 if len(case_names) > 6 else 2,
                framealpha=0.9,
                fontsize=_TEX_BODY_FONT_PT - 2.0,
                labelspacing=0.18,
                borderpad=0.25,
                handletextpad=0.5,
                columnspacing=0.8,
                handlelength=1.4,
                borderaxespad=0.25,
            )
            if "reward" in out_path.name:
                if out_path.name.endswith("reward_per_episode.png"):
                    ax.set_ylim(-85.0, -10.0)
                reward_legend_kwargs = dict(legend_kwargs)
                reward_legend_kwargs["ncols"] = 3
                ax.legend(loc="lower right", **reward_legend_kwargs)
            else:
                if out_path.name.endswith("cost_per_episode.png"):
                    ax.set_ylim(top=140.0)
                ax.legend(loc="upper right", **legend_kwargs)

            left = 0.26 if fig_width <= (_TEX_HALF_WIDTH_IN + 1.0e-9) else 0.18
            top = 0.90 if _PLOT_TITLES else 0.98
            fig.subplots_adjust(left=left, right=0.98, bottom=0.20, top=top)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
            plt.close(fig)

    def _get_loss(
        best_ret: np.ndarray,
        elite_ret: np.ndarray,
        best_lr: np.ndarray,
        elite_lr: np.ndarray,
        best_risk: np.ndarray,
        elite_risk: np.ndarray,
        best_npv: np.ndarray,
        elite_npv: np.ndarray,
        w_lr: float,
        w_cost: float,
        w_risk: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        loss_best = -best_ret
        loss_elite = -elite_ret
        loss0 = float(loss_best[0]) if loss_best.size else 1.0
        if not np.isfinite(loss0) or loss0 <= 0.0:
            loss0 = 1.0
        loss_min = float(np.min(np.concatenate([loss_best, loss_elite]))) if (loss_best.size and loss_elite.size) else float(
            np.min(loss_best)) if loss_best.size else float(np.min(loss_elite)) if loss_elite.size else 0.0
        denom = float(loss0 - loss_min)
        if not np.isfinite(denom) or abs(denom) <= 1.0e-12:
            denom = 1.0
        return (
            np.clip((loss_best - loss_min) / denom, 0.0, None) * 1000.0,
            np.clip((loss_elite - loss_min) / denom, 0.0, None) * 1000.0,
        )

    def _get_reward(
        best_ret: np.ndarray,
        elite_ret: np.ndarray,
        best_lr: np.ndarray,
        elite_lr: np.ndarray,
        best_risk: np.ndarray,
        elite_risk: np.ndarray,
        best_npv: np.ndarray,
        elite_npv: np.ndarray,
        w_lr: float,
        w_cost: float,
        w_risk: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        reward_best = -(w_lr * best_lr + w_cost *
                        best_npv + w_risk * best_risk)
        reward_elite = -(w_lr * elite_lr + w_cost *
                         elite_npv + w_risk * elite_risk)
        return reward_best, reward_elite

    def _get_lor(
        best_ret: np.ndarray,
        elite_ret: np.ndarray,
        best_lr: np.ndarray,
        elite_lr: np.ndarray,
        best_risk: np.ndarray,
        elite_risk: np.ndarray,
        best_npv: np.ndarray,
        elite_npv: np.ndarray,
        w_lr: float,
        w_cost: float,
        w_risk: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return best_lr, elite_lr

    def _get_risk(
        best_ret: np.ndarray,
        elite_ret: np.ndarray,
        best_lr: np.ndarray,
        elite_lr: np.ndarray,
        best_risk: np.ndarray,
        elite_risk: np.ndarray,
        best_npv: np.ndarray,
        elite_npv: np.ndarray,
        w_lr: float,
        w_cost: float,
        w_risk: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return best_risk, elite_risk

    def _get_cost(
        best_ret: np.ndarray,
        elite_ret: np.ndarray,
        best_lr: np.ndarray,
        elite_lr: np.ndarray,
        best_risk: np.ndarray,
        elite_risk: np.ndarray,
        best_npv: np.ndarray,
        elite_npv: np.ndarray,
        w_lr: float,
        w_cost: float,
        w_risk: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return best_npv, elite_npv

    _plot_one(
        fig_dir / f"{filename_prefix}_loss_per_step.png",
        "Loss per Step",
        "Loss",
        _get_loss,
        fig_width=_TEX_HALF_WIDTH_IN,
        fig_height=_TEX_HALF_WIDTH_IN * 0.72,
    )
    _plot_one(
        fig_dir / f"{filename_prefix}_lor_per_episode.png",
        "LoR per Episode",
        "Resilience loss",
        _get_lor,
        fig_width=_TEX_HALF_WIDTH_IN,
        fig_height=_TEX_HALF_WIDTH_IN * 0.72,
    )
    _plot_one(
        fig_dir / f"{filename_prefix}_reward_per_episode.png",
        "Reward per Episode",
        "Reward",
        _get_reward,
        fig_width=_TEX_HALF_WIDTH_IN,
        fig_height=_TEX_HALF_WIDTH_IN * 0.72,
    )
    _plot_one(
        fig_dir / f"{filename_prefix}_risk_per_episode.png",
        "Risk per Episode",
        "Risk",
        _get_risk,
        fig_width=_TEX_HALF_WIDTH_IN,
        fig_height=_TEX_HALF_WIDTH_IN * 0.72,
    )
    _plot_one(
        fig_dir / f"{filename_prefix}_cost_per_episode.png",
        "Cost per Episode",
        "Cost",
        _get_cost,
        fig_width=_TEX_HALF_WIDTH_IN,
        fig_height=_TEX_HALF_WIDTH_IN * 0.72,
    )


def _coerce_lifecycle_config_dict(d: dict[str, Any]) -> dict[str, Any]:
    tuple_fields = {
        "hazard_intensity_probs",
        "hazard_intensities",
        "eq_intensities",
        "eq_intensity_probs",
        "fire_intensities",
        "fire_intensity_probs",
        "pre_cost_options",
        "pre_fcrit_multipliers",
        "pre_hazard_damage_multipliers",
        "pre_deterioration_multipliers",
        "det_alpha_T_levels",
        "maint_costs",
        "maint_rate_multipliers",
        "repair_costs",
        "repair_recovery_deltas",
        "hazard_transition_durations",
    }
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in tuple_fields and isinstance(v, list):
            out[k] = tuple(v)
        else:
            out[k] = v

    if "hazard_prob_per_year" in out and "lambda_eq_per_year" not in out and "lambda_fire_per_year" not in out:
        out["lambda_eq_per_year"] = out["hazard_prob_per_year"]
        out["lambda_fire_per_year"] = 0.0
    out.pop("hazard_prob_per_year", None)

    if "repair_recovery_caps" in out and "repair_recovery_deltas" not in out:
        out["repair_recovery_deltas"] = (0.0, 0.25, 0.5)
    out.pop("repair_recovery_caps", None)

    out.pop("pre_f0_options", None)
    out.pop("w_c3", None)

    allowed = set(LifecycleConfig.__dataclass_fields__.keys())
    out = {k: v for k, v in out.items() if k in allowed}
    return out


def _load_run_config(cfg_path: Path, root: Path) -> tuple[LifecycleConfig, CEMConfig, dict[str, Path]]:
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be an object, got {type(raw)}")

    io = raw.get("io", {})
    if io is None:
        io = {}
    if not isinstance(io, dict):
        raise ValueError(f"'io' must be an object, got {type(io)}")

    def _resolve_io(key: str, default_name: str) -> Path:
        v = io.get(key, default_name)
        if not isinstance(v, str):
            raise ValueError(
                f"'io.{key}' must be a string path, got {type(v)}")
        p = Path(v)
        return p if p.is_absolute() else (root / p)

    paths = {
        "fig_dir": _resolve_io("fig_dir", "fig"),
        "out_json": _resolve_io("out_json", "cem_results.json"),
        "iter_trajectories": _resolve_io("iter_trajectories", "iter_trajectories.json"),
        "log_dir": _resolve_io("log_dir", "log"),
    }

    lifecycle_overrides = raw.get("lifecycle", {})
    if lifecycle_overrides is None:
        lifecycle_overrides = {}
    if not isinstance(lifecycle_overrides, dict):
        raise ValueError(
            f"'lifecycle' must be an object, got {type(lifecycle_overrides)}")
    lifecycle_cfg = LifecycleConfig(
        **_coerce_lifecycle_config_dict(lifecycle_overrides))

    cem_overrides = raw.get("cem", {})
    if cem_overrides is None:
        cem_overrides = {}
    if not isinstance(cem_overrides, dict):
        raise ValueError(f"'cem' must be an object, got {type(cem_overrides)}")
    cem_cfg = CEMConfig(**cem_overrides)

    return lifecycle_cfg, cem_cfg, paths


def _lcc_rerun_root(root: Path) -> Path:
    return root / "lcc_rerun"


def _lcc_case_config_dir(root: Path) -> Path:
    return _lcc_rerun_root(root) / "configs"


def _lcc_case_io(case_name: str) -> dict[str, str]:
    return {
        "fig_dir": f"lcc_rerun/{case_name}/fig",
        "out_json": f"lcc_rerun/{case_name}/cem_results_{case_name}.json",
        "iter_trajectories": f"lcc_rerun/{case_name}/iter_trajectories_{case_name}.json",
        "log_dir": f"lcc_rerun/{case_name}/log",
    }


_CASE4_SPECS: list[dict[str, Any]] = [
    {
        "name": "case4a_no_resilience_equal",
        "desc": "Remove resilience loss and redistribute its weight equally to risk and cost, w=(0.00, 0.50, 0.50).",
        "weights": (0.0, 0.5, 0.5),
    },
    {
        "name": "case4b_no_resilience_risk_replacement",
        "desc": "Remove resilience loss and test whether risk can replace it, w=(0.00, 0.67, 0.33).",
        "weights": (0.0, 2.0 / 3.0, 1.0 / 3.0),
    },
    {
        "name": "case4c_no_risk_cost_emphasis",
        "desc": "Remove risk and emphasize cost as a companion sensitivity case, w=(0.33, 0.00, 0.67).",
        "weights": (1.0 / 3.0, 0.0, 2.0 / 3.0),
    },
]

_CASE4_FIGURE_ALIASES = {
    "ablation_metric_return.png": "case4_metric_return.png",
    "ablation_metric_lr.png": "case4_metric_lr.png",
    "ablation_metric_risk.png": "case4_metric_risk.png",
    "ablation_metric_cost.png": "case4_metric_cost.png",
    "ablation_metrics.png": "case4_metrics.png",
}


def _case4_root(root: Path) -> Path:
    return _lcc_rerun_root(root) / "case4_weight_reallocation"


def _case4_config_dir(root: Path) -> Path:
    return _case4_root(root) / "configs"


def _case4_result_dir(root: Path) -> Path:
    return _case4_root(root) / "results"


def _case4_summary_dir(root: Path) -> Path:
    return _case4_root(root) / "summary"


def _case4_config_path(root: Path, name: str) -> Path:
    return _case4_config_dir(root) / f"{name}.json"


def _case4_result_path(root: Path, name: str) -> Path:
    return _case4_result_dir(root) / f"cem_results_{name}.json"


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _refresh_pareto_selection_file(
    result_path: Path,
    *,
    logger: logging.Logger | None = None,
    force: bool = False,
) -> dict[str, Any]:
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    lifecycle_raw = raw.get("lifecycle_config", {})
    cem_raw = raw.get("cem_config", {})
    if not isinstance(lifecycle_raw, dict) or not isinstance(cem_raw, dict):
        return raw
    cfg = LifecycleConfig(**_coerce_lifecycle_config_dict(lifecycle_raw))
    cem_cfg = CEMConfig(**cem_raw)
    final_selection = str(getattr(cem_cfg, "final_selection", "max_return")).lower()
    target_method = (
        "pareto_cost_risk_then_full_objective"
        if final_selection in {"pareto", "pareto_cost_risk", "pareto_cost_risk_then_full_objective"}
        else "max_training_return"
    )
    already_selected = (
        isinstance(raw.get("best"), dict)
        and isinstance(raw["best"].get("selection"), dict)
        and raw["best"]["selection"].get("method") == target_method
        and any(bool(rec.get("selected_final", False)) for rec in raw.get("iter_trajectories", []))
    )
    if already_selected and not bool(force):
        return raw

    if target_method == "pareto_cost_risk_then_full_objective":
        raw = _apply_pareto_final_selection(raw, cfg, cem_cfg, logger=logger)
    else:
        raw = _apply_max_return_final_selection(raw, cfg, cem_cfg)
    _write_json_file(result_path, raw)
    if logger is not None:
        logger.info("Final selection refreshed | method=%s json=%s", target_method, str(result_path))
    return raw


def _write_case4_configs(root: Path) -> list[Path]:
    base_config = _lcc_case_config_dir(root) / "case1_baseline.json"
    if not base_config.exists():
        _write_case_jsons(root)
    if not base_config.exists():
        raise FileNotFoundError(f"Baseline LCC config not found: {base_config}")

    base = json.loads(base_config.read_text(encoding="utf-8"))
    generated: list[Path] = []
    for spec in _CASE4_SPECS:
        name = str(spec["name"])
        w_lr, w_risk, w_cost = tuple(float(x) for x in spec["weights"])
        cfg = deepcopy(base)
        cfg["io"] = {
            "fig_dir": "lcc_rerun/case4_weight_reallocation/fig",
            "out_json": f"lcc_rerun/case4_weight_reallocation/results/cem_results_{name}.json",
            "iter_trajectories": f"lcc_rerun/case4_weight_reallocation/results/iter_trajectories_{name}.json",
            "log_dir": "lcc_rerun/case4_weight_reallocation/log",
        }
        cfg.setdefault("lifecycle", {})
        cfg["lifecycle"]["w_lr"] = w_lr
        cfg["lifecycle"]["w_risk"] = w_risk
        cfg["lifecycle"]["w_cost"] = w_cost
        cfg.setdefault("cem", {})
        cfg["cem"]["plot_first_n"] = int(cfg["cem"].get("plot_first_n", 30))
        cfg["cem"]["plot_interval"] = int(cfg["cem"].get("plot_interval", 2))
        out_path = _case4_config_path(root, name)
        _write_json_file(out_path, cfg)
        generated.append(out_path)
    return generated


def _run_case4_configs(root: Path, *, force: bool = False, iterations_override: int | None = None) -> None:
    config_paths = _write_case4_configs(root)
    logger = _setup_logging(_case4_root(root) / "log")
    for cfg_path in config_paths:
        cfg, cem_cfg, paths = _load_run_config(cfg_path, root=root)
        if iterations_override is not None:
            cem_cfg = replace(cem_cfg, iterations=int(iterations_override))
        out_json = paths["out_json"]
        case_name = cfg_path.stem
        if out_json.exists() and not bool(force):
            logger.info("Case4 skip existing | case=%s json=%s", case_name, str(out_json))
            continue

        logger.info("Case4 run start | case=%s out_json=%s", case_name, str(out_json))
        results = train_cem(
            cfg,
            cem_cfg,
            logger=logger,
            iter_trajectories_path=paths["iter_trajectories"],
        )
        _write_json_file(out_json, results)

        best_pre = int(results["best"]["params"]["pre_index"])
        f_crit_line = _effective_f_crit(cfg, best_pre)
        case_dir = paths["fig_dir"] / _case_short_name(case_name)
        extra_dir = case_dir / "extra_analysis"
        _plot_iterations(
            cfg,
            results["iter_trajectories"],
            extra_dir / "process_iterations_full.png",
            f_crit_line=f_crit_line,
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_best_detail(
            cfg,
            results["best"]["episode_compare_seed"],
            extra_dir / "result_best.png",
            f_crit_line=f_crit_line,
        )
        _plot_f_over_time_by_iteration(
            cfg,
            results["iter_trajectories"],
            case_dir / "process_iterations.png",
            f_crit_line=f_crit_line,
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_cost_over_time_by_iteration(
            cfg,
            results["iter_trajectories"],
            case_dir / "cost_over_time.png",
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_lr_risk_over_time_by_iteration(
            cfg,
            results["iter_trajectories"],
            case_dir,
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        logger.info("Case4 run done | case=%s wrote_json=%s", case_name, str(out_json))


def _case4_analysis_helpers() -> Any:
    import importlib

    return importlib.import_module("ablation_experiment")


def _case4_records(root: Path, *, holdout_episodes: int, holdout_seed: int) -> list[dict[str, Any]]:
    ae = _case4_analysis_helpers()
    baseline_result = _lcc_rerun_root(root) / "case1_baseline" / "cem_results_case1_baseline.json"
    specs: list[dict[str, Any]] = [
        {
            "name": "case1_baseline",
            "desc": "Baseline equal-weight case, w=(0.33, 0.33, 0.33).",
            "result_json": baseline_result,
        }
    ]
    for spec in _CASE4_SPECS:
        name = str(spec["name"])
        specs.append(
            {
                "name": name,
                "desc": str(spec["desc"]),
                "result_json": _case4_result_path(root, name),
            }
        )

    records: list[dict[str, Any]] = []
    for spec in specs:
        _refresh_pareto_selection_file(Path(spec["result_json"]))
        rec = ae._record_from_case_result(  # type: ignore[attr-defined]
            name=str(spec["name"]),
            desc=str(spec["desc"]),
            result_json=Path(spec["result_json"]),
            holdout_episodes=int(holdout_episodes),
            holdout_seed=int(holdout_seed),
        )
        for key in ("best_eval_train", "best_eval_holdout"):
            metrics = rec.get(key)
            if isinstance(metrics, dict):
                metrics["reported_return"] = float(metrics["return"])
        records.append(rec)
    return records


def _publish_case4_aliases(out_dir: Path) -> None:
    for src_name, dst_name in _CASE4_FIGURE_ALIASES.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copy2(src, out_dir / dst_name)


def _write_case4_report_alias(out_dir: Path) -> None:
    report_path = out_dir / "ablation_report.md"
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    text = text.replace(
        "- `ablation_metric_return.png`, `ablation_metric_lr.png`, `ablation_metric_risk.png`, `ablation_metric_cost.png`: metric-wise bar comparisons.",
        "- `case4_metric_return.png`, `case4_metric_lr.png`, `case4_metric_risk.png`, `case4_metric_cost.png`: metric-wise bar comparisons.",
    )
    text = text.replace(
        "- `ablation_metrics.png`: 2x2 metric panel.",
        "- `case4_metrics.png`: 2x2 metric panel.",
    )
    text = text.replace(
        "- `ablation_convergence.png`: best-of-iteration return curves.",
        "- `case4_convergence.png`: best-of-iteration return curves.",
    )
    text = text.replace(
        "- `ablation_tradeoff.png`: risk-cost trade-off scatter.",
        "- `case4_tradeoff.png`: risk-cost trade-off scatter.",
    )
    _write_text_atomic(out_dir / "case4_report.md", text, encoding="utf-8")


def _render_case4_summary(
    root: Path,
    *,
    holdout_episodes: int = 100,
    holdout_seed: int = 20260301 + 940_000,
) -> Path:
    ae = _case4_analysis_helpers()
    summary_dir = _case4_summary_dir(root)
    summary_dir.mkdir(parents=True, exist_ok=True)
    records = _case4_records(
        root,
        holdout_episodes=int(holdout_episodes),
        holdout_seed=int(holdout_seed),
    )
    bubble_scale = ae._global_lr_scale(records)  # type: ignore[attr-defined]
    ae._plot_metric_panels(records, summary_dir)  # type: ignore[attr-defined]
    ae._plot_convergence(records, summary_dir / "case4_convergence.png")  # type: ignore[attr-defined]
    ae._plot_tradeoff(  # type: ignore[attr-defined]
        records,
        summary_dir / "case4_tradeoff.png",
        bubble_scale=bubble_scale,
    )
    ae._build_report(  # type: ignore[attr-defined]
        records,
        summary_dir,
        {
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iterations": int(records[0]["cem_cfg"]["iterations"]),
            "population": int(records[0]["cem_cfg"]["population"]),
            "elite_frac": float(records[0]["cem_cfg"]["elite_frac"]),
            "eval_episodes": int(records[0]["cem_cfg"]["eval_episodes"]),
            "holdout_episodes": int(holdout_episodes),
        },
        title="Case 4 Weight Reallocation Study",
        return_note="Return is reported with the normalized holdout objective used in each formal case.",
        note_lines=[
            "Case 4 keeps the baseline hazard setting and reference normalization constants, and changes only the objective weights.",
            "Case 4a tests the conventional two-term redistribution after removing resilience loss.",
            "Case 4b implements the reviewer-suggested risk-replacement setting, w=(0.00, 0.67, 0.33).",
            "Case 4c adds the requested no-risk cost-emphasis setting, w=(0.33, 0.00, 0.67).",
        ],
    )
    _publish_case4_aliases(summary_dir)
    _write_case4_report_alias(summary_dir)
    _write_json_file(
        summary_dir / "case4_results.json",
        {
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "settings": {
                "holdout_episodes": int(holdout_episodes),
                "holdout_seed": int(holdout_seed),
            },
            "records": records,
        },
    )

    publish_dir = _lcc_rerun_root(root) / "fig" / "case4_weight_reallocation"
    publish_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "case4_metric_return.png",
        "case4_metric_lr.png",
        "case4_metric_risk.png",
        "case4_metric_cost.png",
        "case4_metrics.png",
        "ablation_metric_return.png",
        "ablation_metric_lr.png",
        "ablation_metric_risk.png",
        "ablation_metric_cost.png",
        "ablation_metrics.png",
        "case4_tradeoff.png",
        "case4_convergence.png",
        "case4_report.md",
    ]:
        src = summary_dir / name
        if src.exists():
            shutil.copy2(src, publish_dir / name)
    return summary_dir


def _replot_case4_detail_figures(root: Path) -> None:
    logger = _setup_logging(_case4_root(root) / "log")
    for cfg_path in _write_case4_configs(root):
        cfg, cem_cfg, paths = _load_run_config(cfg_path, root=root)
        out_json = paths["out_json"]
        case_name = cfg_path.stem
        if not out_json.exists():
            logger.warning("Case4 detail replot skipped | case=%s missing_json=%s", case_name, str(out_json))
            continue
        d = _refresh_pareto_selection_file(out_json, logger=logger)
        cfg = LifecycleConfig(**_coerce_lifecycle_config_dict(dict(d["lifecycle_config"])))
        best_pre = int(d["best"]["params"]["pre_index"])
        f_crit_line = _effective_f_crit(cfg, best_pre)
        case_dir = paths["fig_dir"] / _case_short_name(case_name)
        extra_dir = case_dir / "extra_analysis"
        _plot_iterations(
            cfg,
            d["iter_trajectories"],
            extra_dir / "process_iterations_full.png",
            f_crit_line=f_crit_line,
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_best_detail(
            cfg,
            d["best"]["episode_compare_seed"],
            extra_dir / "result_best.png",
            f_crit_line=f_crit_line,
        )
        _plot_f_over_time_by_iteration(
            cfg,
            d["iter_trajectories"],
            case_dir / "process_iterations.png",
            f_crit_line=f_crit_line,
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_cost_over_time_by_iteration(
            cfg,
            d["iter_trajectories"],
            case_dir / "cost_over_time.png",
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_lr_risk_over_time_by_iteration(
            cfg,
            d["iter_trajectories"],
            case_dir,
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )


def _extract_cli_value(args: list[str], name: str, default: int) -> int:
    for idx, value in enumerate(args):
        if value == name and idx + 1 < len(args):
            return int(args[idx + 1])
        prefix = f"{name}="
        if value.startswith(prefix):
            return int(value.split("=", 1)[1])
    return int(default)


def _handle_case4_cli(
    root: Path,
    args: list[str],
    *,
    plot_only: bool,
    iterations_override: int | None,
) -> None:
    force = "--force" in args
    run_requested = "--run" in args
    holdout_episodes = _extract_cli_value(args, "--holdout-episodes", 100)
    holdout_seed = _extract_cli_value(args, "--holdout-seed", 20260301 + 940_000)

    if run_requested:
        _run_case4_configs(root, force=force, iterations_override=iterations_override)
        _replot_case4_detail_figures(root)
        out = _render_case4_summary(
            root,
            holdout_episodes=holdout_episodes,
            holdout_seed=holdout_seed,
        )
        print(str(out))
        return

    if plot_only:
        _write_case4_configs(root)
        _replot_case4_detail_figures(root)
        out = _render_case4_summary(
            root,
            holdout_episodes=holdout_episodes,
            holdout_seed=holdout_seed,
        )
        print(str(out))
        return

    paths = _write_case4_configs(root)
    for path in paths:
        print(path)


def _load_case4_histories_for_combined(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    case_histories: dict[str, list[dict[str, Any]]] = {}
    case_lifecycle_cfgs: dict[str, dict[str, Any]] = {}
    for spec in _CASE4_SPECS:
        name = str(spec["name"])
        result_path = _case4_result_path(root, name)
        if not result_path.exists():
            continue
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        history = raw.get("history", [])
        if isinstance(history, list):
            case_histories[name] = history
            lifecycle_cfg = raw.get("lifecycle_config", {})
            case_lifecycle_cfgs[name] = lifecycle_cfg if isinstance(lifecycle_cfg, dict) else {}
    return case_histories, case_lifecycle_cfgs


def _case_run_configs(root: Path) -> list[dict[str, Any]]:
    normalized_refs: dict[str, dict[str, float | str]] = {
        "case1_baseline": {
            "objective_normalization": "reference",
            "lr_ref": 10.80476706170614,
            "risk_ref": 1.7860000000000003,
            "cost_ref": 8.856626666666669,
        },
        "case2a_fire_dominant": {
            "objective_normalization": "reference",
            "lr_ref": 10.917178791774498,
            "risk_ref": 1.4888333333333335,
            "cost_ref": 5.654133333333335,
        },
        "case2b_eq_dominant": {
            "objective_normalization": "reference",
            "lr_ref": 11.330063691999566,
            "risk_ref": 3.683333333333333,
            "cost_ref": 8.468653333333334,
        },
        "case3a_cost_oriented": {
            "objective_normalization": "reference",
            "lr_ref": 15.92212017053111,
            "risk_ref": 18.879400419217607,
            "cost_ref": 17.141199999999998,
        },
        "case3b_risk_oriented": {
            "objective_normalization": "reference",
            "lr_ref": 10.296414878703919,
            "risk_ref": 1.676,
            "cost_ref": 4.486480000000001,
        },
        "case3c_resilience_oriented": {
            "objective_normalization": "reference",
            "lr_ref": 6.802514076780323,
            "risk_ref": 1.216,
            "cost_ref": 4.952486666666667,
        },
    }

    base_lifecycle: dict[str, Any] = {
        "horizon_years": 100,
        "dt_years": 1.0,
        "maintenance_interval_years": 10,
        "hazard_intensity_probs": [0.6, 0.3, 0.1],
        "f_crit": 0.7,
        "f_1": 0.5,
        "f_2": 0.3,
        "c_1": 1.0,
        "c_2": 4.0,
        "c_3": 10.0,
        "deterioration_model": "weibull",
        "det_alpha_T_levels": [0.75, 0.50, 0.25],
        "det_tau_years": 55.0,
        "det_k": 2.2,
        "det_sigma": 0.0,
        "base_deterioration_rate": 0.03,
        "maint_costs": [0.0, 0.8, 2.0],
        "maint_rate_multipliers": [1.0, 0.7, 0.45],
        "repair_costs": [0.0, 3.0, 10.0],
        "repair_recovery_deltas": [0.0, 0.25, 0.5],
        "discount_rate": 0.03,
        "objective_cost_metric": "lcc",
        "w_floor": 0.0,
        "repair_stop_years": 0.0,
    }

    base_cem: dict[str, Any] = {
        "iterations": 200,
        "population": 80,
        "elite_frac": 0.2,
        "eval_episodes": 30,
        "seed": 20260301,
        "compare_seed_offset": 4242,
        "plot_first_n": 16,
        "plot_interval": 3,
    }

    cases: list[dict[str, Any]] = [
        {
            "name": "case1_baseline",
            "lambda_eq_per_year": 1.0 / 35.0,
            "lambda_fire_per_year": 1.0 / 3.42,
            "w_lr": 1.0 / 3.0,
            "w_risk": 1.0 / 3.0,
            "w_cost": 1.0 / 3.0,
        },
        {
            "name": "case2a_fire_dominant",
            "lambda_eq_per_year": 1.0 / 70.0,
            "lambda_fire_per_year": 2.0 / 3.42,
            "w_lr": 1.0 / 3.0,
            "w_risk": 1.0 / 3.0,
            "w_cost": 1.0 / 3.0,
        },
        {
            "name": "case2b_eq_dominant",
            "lambda_eq_per_year": 2.0 / 35.0,
            "lambda_fire_per_year": 1.0 / 6.84,
            "w_lr": 1.0 / 3.0,
            "w_risk": 1.0 / 3.0,
            "w_cost": 1.0 / 3.0,
        },
        {
            "name": "case3a_cost_oriented",
            "lambda_eq_per_year": 1.0 / 35.0,
            "lambda_fire_per_year": 1.0 / 3.42,
            "w_lr": 0.15,
            "w_risk": 0.15,
            "w_cost": 0.70,
        },
        {
            "name": "case3b_risk_oriented",
            "lambda_eq_per_year": 1.0 / 35.0,
            "lambda_fire_per_year": 1.0 / 3.42,
            "w_lr": 0.15,
            "w_risk": 0.70,
            "w_cost": 0.15,
        },
        {
            "name": "case3c_resilience_oriented",
            "lambda_eq_per_year": 1.0 / 35.0,
            "lambda_fire_per_year": 1.0 / 3.42,
            "w_lr": 0.70,
            "w_risk": 0.15,
            "w_cost": 0.15,
        },
    ]

    out: list[dict[str, Any]] = []
    for c in cases:
        name = str(c["name"])
        out.append(
            {
                "path": str((_lcc_case_config_dir(root) / f"{name}.json").resolve()),
                "config": {
                    "io": _lcc_case_io(name),
                    "lifecycle": {
                        **base_lifecycle,
                        **{k: v for k, v in c.items() if k != "name"},
                        **normalized_refs.get(name, {}),
                    },
                    "cem": {
                        **base_cem,
                        **({"compare_seed_offset": 4259} if name in {
                            "case3a_cost_oriented",
                        } else {}),
                        **({"compare_seed_offset": 4677} if name in {
                            "case3b_risk_oriented",
                            "case3c_resilience_oriented",
                        } else {}),
                    },
                },
            }
        )
    return out


def _write_case_jsons(root: Path) -> list[Path]:
    case_defs = _case_run_configs(root)
    paths: list[Path] = []
    for rec in case_defs:
        p = Path(rec["path"])
        # 检查文件是否存在，如果存在，读取手工编辑的绘图/迭代参数
        if p.exists():
            try:
                existing_config = json.loads(p.read_text(encoding="utf-8"))
                if "cem" in existing_config:
                    if "plot_interval" in existing_config["cem"]:
                        rec["config"]["cem"]["plot_interval"] = existing_config["cem"]["plot_interval"]
                    if "plot_first_n" in existing_config["cem"]:
                        rec["config"]["cem"]["plot_first_n"] = existing_config["cem"]["plot_first_n"]
            except Exception:
                # 如果文件读取失败，使用默认值
                pass
        _write_text_atomic(
            p,
            json.dumps(rec["config"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(p)
    return paths


def main() -> None:
    root = Path(__file__).resolve().parent
    args = list(sys.argv[1:])
    plot_only = False
    if "--plot-only" in args:
        plot_only = True
        args = [a for a in args if a != "--plot-only"]
    if "--replot" in args:
        plot_only = True
        args = [a for a in args if a != "--replot"]

    global _PLOT_TITLES
    if ("--titles" in args) or ("--with-titles" in args):
        _PLOT_TITLES = True
        args = [a for a in args if a not in ("--titles", "--with-titles")]

    iterations_override: int | None = None
    filtered_args: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--iterations":
            if i + 1 >= len(args):
                raise ValueError("Missing value for --iterations")
            raw = args[i + 1]
            try:
                v = int(raw)
            except ValueError as e:
                raise ValueError(
                    f"Invalid --iterations value: {raw}") from e
            if v <= 0:
                raise ValueError(
                    f"--iterations must be positive, got {v}")
            iterations_override = v
            i += 2
            continue
        if a.startswith("--iterations="):
            raw = a.split("=", 1)[1]
            try:
                v = int(raw)
            except ValueError as e:
                raise ValueError(
                    f"Invalid --iterations value: {raw}") from e
            if v <= 0:
                raise ValueError(
                    f"--iterations must be positive, got {v}")
            iterations_override = v
            i += 1
            continue
        filtered_args.append(a)
        i += 1
    args = filtered_args

    def _load_results(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _cfg_from_results(d: dict[str, Any]) -> LifecycleConfig:
        return LifecycleConfig(**_coerce_lifecycle_config_dict(dict(d["lifecycle_config"])))

    def _cem_from_results(d: dict[str, Any]) -> CEMConfig:
        return CEMConfig(**dict(d["cem_config"]))

    def _display_compare_seed_offset(case_name: str, cem_cfg: CEMConfig) -> int:
        offset = int(getattr(cem_cfg, "compare_seed_offset", 4242))
        if str(case_name) == "case3a_cost_oriented":
            return 4259
        if str(case_name) in {"case3b_risk_oriented", "case3c_resilience_oriented"}:
            return 4677
        return offset

    def _resimulate_for_plots(cfg: LifecycleConfig, cem_cfg: CEMConfig, d: dict[str, Any], *, case_name: str = "") -> None:
        offset = _display_compare_seed_offset(case_name, cem_cfg)
        compare_seed = int(cem_cfg.seed) + int(offset)
        d["compare_seed"] = int(compare_seed)
        if isinstance(d.get("cem_config"), dict):
            d["cem_config"]["compare_seed_offset"] = int(offset)

        if "iter_trajectories" in d and isinstance(d["iter_trajectories"], list):
            for rec in d["iter_trajectories"]:
                if not isinstance(rec, dict):
                    continue
                params_d = rec.get("params")
                if not isinstance(params_d, dict):
                    continue
                try:
                    params = PolicyParams(**params_d)
                except Exception:
                    continue
                ep = simulate_episode(
                    cfg, params, seed=compare_seed, record_hazard_steps=True)
                rec["compare_seed"] = int(compare_seed)
                rec["episode"] = {
                    "t_years": ep.t_years,
                    "f": ep.f,
                    "cumulative_cost": ep.cumulative_cost,
                    "lr": ep.lr,
                    "risk": ep.risk,
                    "lcc": ep.lcc,
                    "npv": ep.npv,
                    "min_f": ep.min_f,
                    "feasible": ep.feasible,
                    "return_value": ep.return_value,
                }

        best = d.get("best")
        if isinstance(best, dict) and isinstance(best.get("params"), dict):
            try:
                params = PolicyParams(**dict(best["params"]))
                ep = simulate_episode(
                    cfg, params, seed=compare_seed, record_hazard_steps=True)
                best["episode_compare_seed"] = {
                    "t_years": ep.t_years,
                    "f": ep.f,
                    "cumulative_cost": ep.cumulative_cost,
                    "lr": ep.lr,
                    "risk": ep.risk,
                    "lcc": ep.lcc,
                    "npv": ep.npv,
                    "min_f": ep.min_f,
                    "feasible": ep.feasible,
                    "return_value": ep.return_value,
                }
            except Exception:
                pass

    if args and args[0].lower() in {"case4", "case4x", "--case4"}:
        _handle_case4_cli(
            root,
            args[1:],
            plot_only=bool(plot_only),
            iterations_override=iterations_override,
        )
        return

    if not args or args[0].lower() in {"all", "--all"}:
        lcc_root = _lcc_rerun_root(root)
        logger = _setup_logging(lcc_root / "log")
        fig_dir = lcc_root / "fig"
        if plot_only:
            case_histories: dict[str, list[dict[str, Any]]] = {}
            case_lifecycle_cfgs: dict[str, dict[str, Any]] = {}
            case_cfg_paths = sorted(
                p for p in _lcc_case_config_dir(root).glob("case*.json") if p.is_file())
            if not case_cfg_paths:
                case_cfg_paths = _write_case_jsons(root)
            for case_cfg_path in case_cfg_paths:
                cfg, cem_cfg, paths = _load_run_config(
                    case_cfg_path, root=root)
                out_json = paths["out_json"]
                case_name = case_cfg_path.stem
                if not out_json.exists():
                    logger.warning(
                        "Replot skipped | case=%s missing_json=%s", case_name, str(out_json))
                    continue
                d = _refresh_pareto_selection_file(out_json, logger=logger)
                cfg = _cfg_from_results(d)
                cem_cfg_sim = _cem_from_results(d)
                _resimulate_for_plots(cfg, cem_cfg_sim, d, case_name=case_name)
                _write_text_atomic(
                    out_json,
                    json.dumps(d, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                best_pre = int(d["best"]["params"]["pre_index"])
                f_crit_line = _effective_f_crit(cfg, best_pre)
                case_dir = paths["fig_dir"] / _case_short_name(case_name)
                extra_dir = case_dir / "extra_analysis"
                _plot_iterations(
                    cfg,
                    d["iter_trajectories"],
                    extra_dir / "process_iterations_full.png",
                    f_crit_line=f_crit_line,
                    plot_first_n=cem_cfg.plot_first_n,
                    plot_interval=cem_cfg.plot_interval,
                )
                _plot_best_detail(
                    cfg,
                    d["best"]["episode_compare_seed"],
                    extra_dir / "result_best.png",
                    f_crit_line=f_crit_line,
                )
                _plot_f_over_time_by_iteration(
                    cfg,
                    d["iter_trajectories"],
                    case_dir / "process_iterations.png",
                    f_crit_line=f_crit_line,
                    plot_first_n=cem_cfg.plot_first_n,
                    plot_interval=cem_cfg.plot_interval,
                )
                _plot_cost_over_time_by_iteration(
                    cfg,
                    d["iter_trajectories"],
                    case_dir / "cost_over_time.png",
                    plot_first_n=cem_cfg.plot_first_n,
                    plot_interval=cem_cfg.plot_interval,
                )
                _plot_lr_risk_over_time_by_iteration(
                    cfg,
                    d["iter_trajectories"],
                    case_dir,
                    plot_first_n=cem_cfg.plot_first_n,
                    plot_interval=cem_cfg.plot_interval,
                )
                case_histories[case_name] = d["history"]
                case_lifecycle_cfgs[case_name] = d.get("lifecycle_config", {})
                logger.info("Replot done | case=%s cfg=%s json=%s",
                            case_name, str(case_cfg_path), str(out_json))
            _plot_rl_eval_combined(
                case_histories, case_lifecycle_cfgs, fig_dir)
            case4_histories, case4_lifecycle_cfgs = _load_case4_histories_for_combined(root)
            if case4_histories:
                all_histories = {**case_histories, **case4_histories}
                all_lifecycle_cfgs = {**case_lifecycle_cfgs, **case4_lifecycle_cfgs}
                _plot_rl_eval_combined(
                    all_histories,
                    all_lifecycle_cfgs,
                    fig_dir,
                    filename_prefix="rl_eval_case1_4",
                )
                if len(case4_histories) == len(_CASE4_SPECS):
                    try:
                        _render_case4_summary(root)
                    except Exception as err:
                        logger.warning("Case4 summary skipped during plot-only | error=%s", str(err))
            return

        case_paths = _write_case_jsons(root)
        case_histories = {}
        case_lifecycle_cfgs = {}
        for cfg_path in case_paths:
            cfg, cem_cfg, paths = _load_run_config(cfg_path, root=root)
            if iterations_override is not None:
                cem_cfg = replace(cem_cfg, iterations=int(iterations_override))
            out_json = paths["out_json"]
            iter_traj_path = paths["iter_trajectories"]
            case_name = cfg_path.stem

            logger.info("Run start | case=%s out_json=%s fig_dir=%s",
                        case_name, str(out_json), str(paths["fig_dir"]))
            results = train_cem(cfg, cem_cfg, logger=logger,
                                iter_trajectories_path=iter_traj_path)
            _write_text_atomic(
                out_json,
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            best_pre = int(results["best"]["params"]["pre_index"])
            f_crit_line = _effective_f_crit(cfg, best_pre)
            case_dir = paths["fig_dir"] / _case_short_name(case_name)
            extra_dir = case_dir / "extra_analysis"
            _plot_iterations(
                cfg,
                results["iter_trajectories"],
                extra_dir / "process_iterations_full.png",
                f_crit_line=f_crit_line,
                plot_first_n=cem_cfg.plot_first_n,
                plot_interval=cem_cfg.plot_interval,
            )
            _plot_best_detail(
                cfg,
                results["best"]["episode_compare_seed"],
                extra_dir / "result_best.png",
                f_crit_line=f_crit_line,
            )
            _plot_f_over_time_by_iteration(
                cfg,
                results["iter_trajectories"],
                case_dir / "process_iterations.png",
                f_crit_line=f_crit_line,
                plot_first_n=cem_cfg.plot_first_n,
                plot_interval=cem_cfg.plot_interval,
            )
            _plot_cost_over_time_by_iteration(
                cfg,
                results["iter_trajectories"],
                case_dir / "cost_over_time.png",
                plot_first_n=cem_cfg.plot_first_n,
                plot_interval=cem_cfg.plot_interval,
            )
            _plot_lr_risk_over_time_by_iteration(
                cfg,
                results["iter_trajectories"],
                case_dir,
                plot_first_n=cem_cfg.plot_first_n,
                plot_interval=cem_cfg.plot_interval,
            )
            case_histories[case_name] = results["history"]
            case_lifecycle_cfgs[case_name] = results.get(
                "lifecycle_config", {})
            logger.info("Run done | case=%s wrote_json=%s",
                        case_name, str(out_json))
        _plot_rl_eval_combined(
            case_histories, case_lifecycle_cfgs, fig_dir)
        _run_case4_configs(root, force=True, iterations_override=iterations_override)
        _render_case4_summary(root)
        case4_histories, case4_lifecycle_cfgs = _load_case4_histories_for_combined(root)
        all_histories = {**case_histories, **case4_histories}
        all_lifecycle_cfgs = {**case_lifecycle_cfgs, **case4_lifecycle_cfgs}
        _plot_rl_eval_combined(
            all_histories,
            all_lifecycle_cfgs,
            fig_dir,
            filename_prefix="rl_eval_case1_4",
        )
        return

    cfg_path = Path(args[0])
    loaded_cfg = False
    if cfg_path.exists():
        cfg, cem_cfg, paths = _load_run_config(cfg_path, root=root)
        fig_dir = paths["fig_dir"]
        out_json = paths["out_json"]
        case_name = cfg_path.stem
        loaded_cfg = True
    else:
        cfg = LifecycleConfig()
        cem_cfg = CEMConfig()
        default_dir = _lcc_rerun_root(root) / "default"
        fig_dir = default_dir / "fig"
        out_json = default_dir / "cem_results.json"
        case_name = cfg_path.stem if str(cfg_path) else "default"
        paths = {
            "iter_trajectories": default_dir / "iter_trajectories.json",
            "log_dir": default_dir / "log",
        }

    logger = _setup_logging(paths["log_dir"])
    if plot_only:
        if not out_json.exists():
            raise FileNotFoundError(str(out_json))
        d = _refresh_pareto_selection_file(out_json, logger=logger)
        cfg = _cfg_from_results(d)
        if not loaded_cfg:
            cem_cfg = _cem_from_results(d)
        cem_cfg_sim = _cem_from_results(d)
        _resimulate_for_plots(cfg, cem_cfg_sim, d, case_name=case_name)
        _write_text_atomic(
            out_json,
            json.dumps(d, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        best_pre = int(d["best"]["params"]["pre_index"])
        f_crit_line = _effective_f_crit(cfg, best_pre)
        case_dir = fig_dir / _case_short_name(case_name)
        extra_dir = case_dir / "extra_analysis"
        _plot_iterations(
            cfg,
            d["iter_trajectories"],
            extra_dir / "process_iterations_full.png",
            f_crit_line=f_crit_line,
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_best_detail(
            cfg,
            d["best"]["episode_compare_seed"],
            extra_dir / "result_best.png",
            f_crit_line=f_crit_line,
        )
        _plot_f_over_time_by_iteration(
            cfg,
            d["iter_trajectories"],
            case_dir / "process_iterations.png",
            f_crit_line=f_crit_line,
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_cost_over_time_by_iteration(
            cfg,
            d["iter_trajectories"],
            case_dir / "cost_over_time.png",
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_lr_risk_over_time_by_iteration(
            cfg,
            d["iter_trajectories"],
            case_dir,
            plot_first_n=cem_cfg.plot_first_n,
            plot_interval=cem_cfg.plot_interval,
        )
        _plot_rl_eval_combined({case_name: d["history"]}, {
                               case_name: d.get("lifecycle_config", {})}, fig_dir)
        logger.info("Replot done | case=%s from_json=%s",
                    case_name, str(out_json))
        return

    iter_traj_path = paths["iter_trajectories"]
    if iterations_override is not None:
        cem_cfg = replace(cem_cfg, iterations=int(iterations_override))
    logger.info("Run start | case=%s out_json=%s fig_dir=%s",
                case_name, str(out_json), str(fig_dir))
    results = train_cem(cfg, cem_cfg, logger=logger,
                        iter_trajectories_path=iter_traj_path)
    _write_text_atomic(
        out_json,
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_pre = int(results["best"]["params"]["pre_index"])
    f_crit_line = _effective_f_crit(cfg, best_pre)
    case_dir = fig_dir / _case_short_name(case_name)
    extra_dir = case_dir / "extra_analysis"
    _plot_iterations(
        cfg,
        results["iter_trajectories"],
        extra_dir / "process_iterations_full.png",
        f_crit_line=f_crit_line,
        plot_first_n=cem_cfg.plot_first_n,
        plot_interval=cem_cfg.plot_interval,
    )
    _plot_best_detail(
        cfg,
        results["best"]["episode_compare_seed"],
        extra_dir / "result_best.png",
        f_crit_line=f_crit_line,
    )
    _plot_f_over_time_by_iteration(
        cfg,
        results["iter_trajectories"],
        case_dir / "process_iterations.png",
        f_crit_line=f_crit_line,
        plot_first_n=cem_cfg.plot_first_n,
        plot_interval=cem_cfg.plot_interval,
    )
    _plot_cost_over_time_by_iteration(
        cfg,
        results["iter_trajectories"],
        case_dir / "cost_over_time.png",
        plot_first_n=cem_cfg.plot_first_n,
        plot_interval=cem_cfg.plot_interval,
    )
    _plot_lr_risk_over_time_by_iteration(
        cfg,
        results["iter_trajectories"],
        case_dir,
        plot_first_n=cem_cfg.plot_first_n,
        plot_interval=cem_cfg.plot_interval,
    )
    _plot_rl_eval_combined({case_name: results["history"]}, {
                           case_name: results.get("lifecycle_config", {})}, fig_dir)
    logger.info("Run done | case=%s wrote_json=%s", case_name, str(out_json))


if __name__ == "__main__":
    main()
