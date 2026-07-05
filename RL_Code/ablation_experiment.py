from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import train_lifecycle_rl as tl

# ---------------------------
# Centralized plot font config
# Tune all figure text sizes here.
# ---------------------------
_PLOT_FONT: dict[str, float] = {
    # Metric panel fonts
    "metric_title": 40.0,
    "metric_axis": 40.0,
    "metric_ticks": 40.0,
    "metric_value": 40.0,
    # Convergence fonts
    "conv_title": 20.0,
    "conv_axis": 16.0,
    "conv_ticks": 14.0,
    "conv_legend": 14.0,
    # Tradeoff fonts
    "trade_title": 20.0,
    "trade_axis": 20.0,
    "trade_ticks": 20.0,
    "trade_label": 16.0,
    "trade_legend": 20.0,
    "trade_legend_title": 20.0,
    "trade_bubble_legend": 16.0,
    "trade_bubble_legend_title": 16.0,
    "trade_cbar_label": 20.0,
    "trade_cbar_ticks": 20.0,
    "trade_footer": 20.0,
}

# Muted fills with restrained academic hatch patterns.
_BAR_COLORS: list[str] = [
    "#6F8FBF",
    "#79AFC4",
    "#98C7A3",
    "#E5D48F",
    "#D78895",
    "#A996D2",
    "#C79E74",
]
_BAR_HATCHES: list[str] = [
    "////",
    "\\\\\\\\",
    "----",
    "||||",
    "xxxx",
    "....",
    "++",
]
_BAR_EDGE = "#222222"
_BAR_STEP = 1.18
_BAR_WIDTH = 0.58

plt.rcParams["hatch.linewidth"] = 0.7


def _font(name: str) -> float:
    return float(_PLOT_FONT[name])


def _apply_full_frame(ax: Any, *, linewidth: float = 0.9, color: str = "#333333") -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(float(linewidth))
        ax.spines[side].set_color(color)


def _display_variant_name(name: str) -> str:
    name_map = {
        "baseline_full": "Default",
        "no_risk_term": "No risk",
        "no_resilience_term": "No resilience",
        "no_cost_term": "No cost",
        "no_pre_reinforcement": "No pre-reinforcement",
        "no_maintenance": "No maintenance",
        "no_repair": "No repair",
    }
    return name_map.get(str(name), str(name))


def _make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"ablation_{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _default_base_config_path() -> Path:
    return Path(__file__).resolve().parent / "normalized_cases" / "case1_baseline_normalized.json"


def _base_cfg(config_path: Path | None = None) -> tuple[tl.LifecycleConfig, tl.CEMConfig, Path]:
    path = _default_base_config_path() if config_path is None else Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalized baseline config not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    lifecycle_raw = raw.get("lifecycle", {})
    cem_raw = raw.get("cem", {})
    if not isinstance(lifecycle_raw, dict):
        raise ValueError(f"'lifecycle' must be an object in {path}")
    if not isinstance(cem_raw, dict):
        raise ValueError(f"'cem' must be an object in {path}")

    lifecycle_cfg = tl.LifecycleConfig(
        **tl._coerce_lifecycle_config_dict(lifecycle_raw)  # type: ignore[attr-defined]
    )
    cem_cfg = tl.CEMConfig(**cem_raw)
    return lifecycle_cfg, cem_cfg, path.resolve()


def _variants(base: tl.LifecycleConfig) -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_full",
            "desc": "Full objective and full intervention modules.",
            "cfg": base,
        },
        {
            "name": "no_risk_term",
            "desc": "Remove risk term from reward.",
            "cfg": replace(base, w_lr=0.5, w_risk=0.0, w_cost=0.5),
        },
        {
            "name": "no_resilience_term",
            "desc": "Remove resilience-loss term from reward.",
            "cfg": replace(base, w_lr=0.0, w_risk=0.5, w_cost=0.5),
        },
        {
            "name": "no_cost_term",
            "desc": "Remove cost term from reward.",
            "cfg": replace(base, w_lr=0.5, w_risk=0.5, w_cost=0.0),
        },
        {
            "name": "no_pre_reinforcement",
            "desc": "Disable pre-reinforcement effect and pre-cost differentiation.",
            "cfg": replace(
                base,
                pre_cost_options=(10.0, 10.0, 10.0),
                pre_fcrit_multipliers=(1.0, 1.0, 1.0),
                pre_hazard_damage_multipliers=(1.0, 1.0, 1.0),
                pre_deterioration_multipliers=(1.0, 1.0, 1.0),
            ),
        },
        {
            "name": "no_maintenance",
            "desc": "Disable periodic maintenance actions throughout the lifecycle.",
            "cfg": replace(
                base,
                maintenance_interval_years=0,
            ),
        },
        {
            "name": "no_repair",
            "desc": "Disable post-disaster repair recovery and repair cost actions.",
            "cfg": replace(
                base,
                repair_costs=(0.0, 0.0, 0.0),
                repair_recovery_deltas=(0.0, 0.0, 0.0),
            ),
        },
    ]


def _best_params_from_result(result: dict[str, Any]) -> tl.PolicyParams:
    p = result["best"]["params"]
    return tl.PolicyParams(
        pre_index=int(p["pre_index"]),
        maint_t1=float(p["maint_t1"]),
        maint_t2=float(p["maint_t2"]),
        repair_t1=float(p["repair_t1"]),
        repair_t2=float(p["repair_t2"]),
    )


def _reported_return_from_terms(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Raw scalarisation used for paper-comparable ablation return plots."""
    w_lr = float(cfg.get("w_lr", 1.0 / 3.0))
    w_risk = float(cfg.get("w_risk", 1.0 / 3.0))
    w_cost = float(cfg.get("w_cost", 1.0 / 3.0))
    lr = float(metrics["lr"])
    risk = float(metrics["risk"])
    cost = float(metrics.get("lcc", metrics.get("npv", 0.0)))
    return float(-(w_lr * lr + w_risk * risk + w_cost * cost))


def _ensure_reported_return(record: dict[str, Any]) -> None:
    cfg = record.get("cfg", {})
    if not isinstance(cfg, dict):
        cfg = {}
    for key in ("best_eval_train", "best_eval_holdout"):
        metrics = record.get(key)
        if isinstance(metrics, dict):
            metrics["reported_return"] = _reported_return_from_terms(metrics, cfg)


def _metric_value(record: dict[str, Any], metric_key: str) -> float:
    metrics = record["best_eval_holdout"]
    if metric_key == "reported_return":
        if "reported_return" not in metrics:
            _ensure_reported_return(record)
        return float(record["best_eval_holdout"]["reported_return"])
    return float(metrics[metric_key])


def _metric_axis_spec(metric_key: str, vals: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(vals, dtype=float)
    if metric_key == "reported_return":
        vmin = float(np.min(vals))
        y_bottom = min(vmin * 2.45, -1.0)
        return {
            "scale": "symlog",
            "scale_kwargs": {"linthresh": 1.0, "linscale": 1.0, "base": 10},
            "ymin": y_bottom,
            "ymax": 0.0,
            "label_suffix": " (symlog)",
        }

    positive = vals[vals > 0.0]
    if positive.size == 0:
        return {
            "scale": "linear",
            "scale_kwargs": {},
            "ymin": 0.0,
            "ymax": 1.0,
            "label_suffix": "",
        }

    vmin = float(np.min(positive))
    vmax = float(np.max(vals))
    top_scale = 1.8
    if metric_key == "lr":
        top_scale = 2.6
    if metric_key == "risk":
        top_scale = 4.8
    return {
        "scale": "log",
        "scale_kwargs": {"base": 10},
        "ymin": max(vmin / 1.8, 1.0e-3),
        "ymax": max(vmax * top_scale, vmin * 2.0),
        "label_suffix": " (log)",
    }


def _annotation_style_for_metric(metric_key: str, value: float) -> tuple[int, str]:
    if metric_key == "reported_return":
        return (-8, "top") if value <= 0.0 else (8, "bottom")
    return (8, "bottom")


def _metric_annotation_override(
    metric_key: str,
    variant_name: str,
    *,
    panel_kind: str,
) -> dict[str, Any] | None:
    return None


def _run_one_variant(
    *,
    variant: dict[str, Any],
    cem_cfg: tl.CEMConfig,
    seed: int,
    holdout_episodes: int,
    out_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    name = str(variant["name"])
    cfg = variant["cfg"]
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    cem_cfg_v = replace(cem_cfg, seed=int(seed))
    logger.info("Ablation run start | variant=%s seed=%d", name, seed)
    result = tl.train_cem(cfg, cem_cfg_v, logger=logger)

    (run_dir / "cem_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_params = _best_params_from_result(result)
    holdout = tl._evaluate_params(  # type: ignore[attr-defined]
        cfg,
        best_params,
        base_seed=int(seed + 700_000),
        episodes=int(holdout_episodes),
    )
    holdout = dict(holdout)
    compare_ep = result["best"]["episode_compare_seed"]
    payload = {
        "name": name,
        "desc": str(variant["desc"]),
        "seed": int(seed),
        "cfg": {
            "w_lr": float(cfg.w_lr),
            "w_risk": float(cfg.w_risk),
            "w_cost": float(cfg.w_cost),
            "objective_cost_metric": str(cfg.objective_cost_metric),
            "objective_normalization": str(cfg.objective_normalization),
            "lr_ref": float(cfg.lr_ref),
            "risk_ref": float(cfg.risk_ref),
            "cost_ref": float(cfg.cost_ref),
            "floor_ref": float(cfg.floor_ref),
            "w_floor": float(cfg.w_floor),
            "pre_cost_options": list(map(float, cfg.pre_cost_options)),
            "pre_fcrit_multipliers": list(map(float, cfg.pre_fcrit_multipliers)),
            "pre_hazard_damage_multipliers": list(
                map(float, cfg.pre_hazard_damage_multipliers)
            ),
            "pre_deterioration_multipliers": list(
                map(float, cfg.pre_deterioration_multipliers)
            ),
        },
        "cem_cfg": {
            "iterations": int(cem_cfg_v.iterations),
            "population": int(cem_cfg_v.population),
            "elite_frac": float(cem_cfg_v.elite_frac),
            "eval_episodes": int(cem_cfg_v.eval_episodes),
        },
        "best_eval_train": result["best"]["eval"],
        "best_eval_holdout": holdout,
        "best_params": result["best"]["params"],
        "compare_seed_episode": {
            "lr": float(compare_ep["lr"]),
            "risk": float(compare_ep["risk"]),
            "npv": float(compare_ep["npv"]),
            "return_value": float(compare_ep["return_value"]),
        },
        "history": result["history"],
    }
    _ensure_reported_return(payload)
    (run_dir / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Ablation run done | variant=%s holdout_return=%.4f holdout_risk=%.3f holdout_cost=%.2f",
        name,
        float(holdout["return"]),
        float(holdout["risk"]),
        float(holdout["lcc"]),
    )
    return payload


def _plot_one_metric_bar(
    records: list[dict[str, Any]],
    *,
    metric_key: str,
    metric_ylabel: str,
    higher_better: bool,
    out_path: Path,
) -> None:
    names = [r["name"] for r in records]
    labels = []
    for n in names:
        txt = _display_variant_name(n)
        txt = txt.replace(" No ", "\nNo ")
        labels.append(txt)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(16.0, 9.8))
    vals = np.array([_metric_value(r, metric_key) for r in records], dtype=float)
    bars = ax.bar(
        x,
        vals,
        width=0.72,
        color=[_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(len(vals))],
        edgecolor=_BAR_EDGE,
        linewidth=0.95,
        alpha=0.96,
        zorder=3,
    )
    for i, b in enumerate(bars):
        b.set_hatch(_BAR_HATCHES[i % len(_BAR_HATCHES)])
    ax.set_facecolor("#F7F7F7")
    x_tick_shift = 0.12
    ax.set_xticks(x + x_tick_shift)
    ax.set_xticklabels(labels, rotation=14, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="both", labelsize=_font("metric_ticks"))
    ax.tick_params(axis="x", pad=8)
    axis_spec = _metric_axis_spec(metric_key, vals)
    ax.set_yscale(axis_spec["scale"], **axis_spec["scale_kwargs"])
    ax.set_ylim(float(axis_spec["ymin"]), float(axis_spec["ymax"]))
    ax.set_ylabel(metric_ylabel + str(axis_spec["label_suffix"]), fontsize=_font("metric_axis"))
    ax.grid(True, axis="y", which="both", linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    _apply_full_frame(ax, linewidth=0.95)
    ax.margins(x=0.12 if metric_key == "reported_return" else 0.06)
    baseline = float(vals[0])
    ax.axhline(
        baseline,
        color="#2F4F4F",
        linestyle=":",
        linewidth=1.0,
        alpha=0.8,
        zorder=2,
    )
    for i, b in enumerate(bars):
        v = float(vals[i])
        delta = v - baseline
        sign = "+" if delta >= 0 else ""
        marker = ""
        if i != 0:
            marker = " \u2191" if higher_better and delta >= 0 else ""
            marker = marker or (" \u2193" if (not higher_better and delta <= 0) else "")
        label = f"{v:.2f}\n({sign}{delta:.2f}){marker}"
        y_shift, va = _annotation_style_for_metric(metric_key, v)
        label_font = 26.0 if metric_key == "reported_return" else _font("metric_value")
        override = _metric_annotation_override(metric_key, str(names[i]), panel_kind="single")
        xytext = (0, y_shift)
        ha = "center"
        if override is not None:
            xytext = tuple(override["xytext"])
            ha = str(override["ha"])
            va = str(override["va"])
        ax.annotate(
            label,
            xy=(b.get_x() + b.get_width() / 2.0, v),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=label_font,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.72, linewidth=0.0),
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _metric_specs() -> list[tuple[str, str, bool, str, str]]:
    return [
        ("reported_return", "Return", True, "ablation_metric_return.png", "(a) Return"),
        ("lr", "Resilience loss", False, "ablation_metric_lr.png", "(b) Resilience loss"),
        ("risk", "Risk", False, "ablation_metric_risk.png", "(c) Risk"),
        ("lcc", "Cost", False, "ablation_metric_cost.png", "(d) Cost"),
    ]


def _plot_metric_grid(records: list[dict[str, Any]], out_path: Path) -> None:
    labels = []
    for rec in records:
        txt = _display_variant_name(str(rec["name"]))
        labels.append(txt.replace(" No ", "\nNo "))
    x = np.arange(len(records))

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    for ax, (metric_key, metric_ylabel, higher_better, _filename, caption) in zip(
        axes.flat, _metric_specs()
    ):
        vals = np.array(
            [_metric_value(r, metric_key) for r in records],
            dtype=float,
        )
        bars = ax.bar(
            x,
            vals,
            width=0.72,
            color=[_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(len(vals))],
            edgecolor=_BAR_EDGE,
            linewidth=0.6,
            alpha=0.96,
            zorder=3,
        )
        for i, b in enumerate(bars):
            b.set_hatch(_BAR_HATCHES[i % len(_BAR_HATCHES)])

        axis_spec = _metric_axis_spec(metric_key, vals)
        ax.set_yscale(axis_spec["scale"], **axis_spec["scale_kwargs"])
        ax.set_ylim(float(axis_spec["ymin"]), float(axis_spec["ymax"]))

        baseline = float(vals[0])
        ax.axhline(
            baseline,
            color="#2F4F4F",
            linestyle=":",
            linewidth=0.6,
            alpha=0.8,
            zorder=2,
        )
        for i, b in enumerate(bars):
            v = float(vals[i])
            delta = v - baseline
            sign = "+" if delta >= 0 else ""
            marker = ""
            if i != 0:
                marker = " \u2191" if higher_better and delta >= 0 else ""
                marker = marker or (" \u2193" if (not higher_better and delta <= 0) else "")
            label = f"{v:.2f}\n({sign}{delta:.2f}){marker}"
            y_shift, va = _annotation_style_for_metric(metric_key, v)
            override = _metric_annotation_override(metric_key, str(records[i]["name"]), panel_kind="grid")
            xytext = (0, 4 if y_shift > 0 else -4)
            ha = "center"
            if override is not None:
                raw_xytext = tuple(override["xytext"])
                xytext = (int(raw_xytext[0]), 4 if int(raw_xytext[1]) > 0 else -4)
                ha = str(override["ha"])
                va = str(override["va"])
            ax.annotate(
                label,
                xy=(b.get_x() + b.get_width() / 2.0, v),
                xytext=xytext,
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=8.8,
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", alpha=0.74, linewidth=0.0),
            )

        ax.set_facecolor("#F7F7F7")
        ax.set_xticks(x + 0.10)
        ax.set_xticklabels(labels, rotation=14, ha="right", rotation_mode="anchor", fontsize=9.2)
        ax.tick_params(axis="y", labelsize=9.4)
        ax.tick_params(axis="x", pad=6)
        ax.set_ylabel(metric_ylabel + str(axis_spec["label_suffix"]), fontsize=10.5)
        ax.grid(True, axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.40, zorder=0)
        _apply_full_frame(ax, linewidth=0.8)
        ax.margins(x=0.12 if metric_key == "reported_return" else 0.06)
        ax.text(0.5, -0.34, caption, transform=ax.transAxes, ha="center", va="top", fontsize=11.5)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.12, wspace=0.30, hspace=0.58)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_panels(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, ylabel, higher_better, filename, _caption in _metric_specs():
        _plot_one_metric_bar(
            records,
            metric_key=key,
            metric_ylabel=ylabel,
            higher_better=higher_better,
            out_path=out_dir / filename,
        )
    shutil.copy2(out_dir / "ablation_metric_cost.png", out_dir / "ablation_metric_npv.png")
    _plot_metric_grid(records, out_dir / "ablation_metrics.png")


def _copy_key_ablation_figures(out_dir: Path) -> None:
    src_dir = out_dir
    dst_dir = Path(__file__).resolve().parent / "fig" / "ablation"
    dst_dir.mkdir(parents=True, exist_ok=True)
    fig_names = [
        "ablation_metric_return.png",
        "ablation_metric_lr.png",
        "ablation_metric_risk.png",
        "ablation_metric_cost.png",
        "ablation_metric_npv.png",
        "ablation_metrics.png",
        "ablation_tradeoff.png",
    ]
    for name in fig_names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def _plot_convergence(records: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for rec in records:
        hist = rec["history"]
        xs = [int(h["iteration"]) for h in hist]
        ys = [float(h["best_of_iter"]["return"]) for h in hist]
        ax.plot(xs, ys, linewidth=1.8, label=rec["name"])
    ax.set_xlabel("Iteration", fontsize=_font("conv_axis"))
    ax.set_ylabel("Best return of iteration", fontsize=_font("conv_axis"))
    ax.set_title("Ablation Convergence", fontsize=_font("conv_title"))
    ax.tick_params(axis="both", labelsize=_font("conv_ticks"))
    ax.grid(True, alpha=0.25)
    _apply_full_frame(ax, linewidth=0.9)
    ax.legend(loc="best", fontsize=_font("conv_legend"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_tradeoff(records: list[dict[str, Any]], out_path: Path) -> None:
    names = [str(r["name"]) for r in records]
    cost = np.array([float(r["best_eval_holdout"]["lcc"]) for r in records], dtype=float)
    risk = np.array([float(r["best_eval_holdout"]["risk"]) for r in records], dtype=float)
    lr = np.array([float(r["best_eval_holdout"]["lr"]) for r in records], dtype=float)
    ret = np.array([_metric_value(r, "reported_return") for r in records], dtype=float)

    lr_min = float(np.min(lr))
    lr_max = float(np.max(lr))
    if lr_max - lr_min <= 1.0e-12:
        bubble_sizes = np.full_like(lr, 420.0)
    else:
        # Lower LR is better -> larger bubbles.
        bubble_sizes = 220.0 + 760.0 * (lr_max - lr) / (lr_max - lr_min)

    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    scatter_ref = ax.scatter(
        cost,
        risk,
        s=bubble_sizes,
        c=ret,
        cmap="viridis",
        alpha=0.88,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    _apply_full_frame(ax, linewidth=0.9)
    ax.tick_params(axis="both", labelsize=_font("trade_ticks"))

    risk_positive = risk[risk > 0.0]
    if risk_positive.size > 0:
        ax.set_yscale("log", base=10)
        ax.set_ylim(max(float(np.min(risk_positive)) / 1.8, 1.0e-3), float(np.max(risk)) * 1.8)

    label_offsets = {
        "baseline_full": (16, -8),
        "no_risk_term": (12, 8),
        "no_resilience_term": (-14, -16),
        "no_cost_term": (-28, 12),
        "no_pre_reinforcement": (16, 8),
        "no_maintenance": (12, 18),
        "no_repair": (20, -10),
    }
    label_align = {
        "baseline_full": ("left", "center"),
        "no_risk_term": ("left", "bottom"),
        "no_resilience_term": ("right", "top"),
        "no_cost_term": ("right", "bottom"),
        "no_pre_reinforcement": ("left", "bottom"),
        "no_maintenance": ("left", "bottom"),
        "no_repair": ("left", "top"),
    }
    for i, name in enumerate(names):
        dx, dy = label_offsets.get(name, (8, 8))
        ha, va = label_align.get(name, ("left", "bottom"))
        ax.annotate(
            _display_variant_name(name),
            xy=(cost[i], risk[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=_font("trade_label"),
            ha=ha,
            va=va,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7, alpha=0.55),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.72, linewidth=0.0),
        )

    cbar = fig.colorbar(scatter_ref, ax=ax, pad=0.03, fraction=0.05)
    cbar.set_label("Holdout return", rotation=90, fontsize=_font("trade_cbar_label"))
    cbar.ax.tick_params(labelsize=_font("trade_cbar_ticks"))

    # Bubble-size legend for LR values.
    lr_refs = np.array([np.min(lr), np.median(lr), np.max(lr)], dtype=float)
    if lr_max - lr_min <= 1.0e-12:
        size_refs = np.full(3, float(bubble_sizes[0]))
    else:
        size_refs = 220.0 + 760.0 * (lr_max - lr_refs) / (lr_max - lr_min)
    size_handles = [
        ax.scatter([], [], s=float(size_refs[i]), color="#6A6A6A", alpha=0.78, edgecolors="none")
        for i in range(3)
    ]
    ax.legend(
        size_handles,
        [f"LR={float(v):.2f}" for v in lr_refs],
        title="Bubble size (LR)",
        loc="upper right",
        bbox_to_anchor=(0.96, 0.98),
        ncols=1,
        frameon=True,
        facecolor="white",
        framealpha=0.82,
        fontsize=_font("trade_bubble_legend"),
        title_fontsize=_font("trade_bubble_legend_title"),
    )

    x_pad = max(2.4, float((np.max(cost) - np.min(cost)) * 0.10))
    ax.set_xlim(float(np.min(cost) - x_pad), float(np.max(cost) + x_pad))
    ax.set_xlabel("Cost", fontsize=_font("trade_axis"))
    ax.set_ylabel("Risk (log)", fontsize=_font("trade_axis"))
    ax.set_title("Risk-Cost Trade-off on Holdout Episodes", fontsize=_font("trade_title"), pad=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _build_report(records: list[dict[str, Any]], out_dir: Path, settings: dict[str, Any]) -> None:
    baseline = records[0]["best_eval_holdout"]
    baseline_return = _metric_value(records[0], "reported_return")
    rank = sorted(
        records,
        key=lambda r: _metric_value(r, "reported_return"),
        reverse=True,
    )
    lines: list[str] = []
    lines.append("# Ablation Study Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- run_time: `{settings['run_time']}`")
    lines.append(f"- iterations: `{settings['iterations']}`")
    lines.append(f"- population: `{settings['population']}`")
    lines.append(f"- elite_frac: `{settings['elite_frac']}`")
    lines.append(f"- eval_episodes: `{settings['eval_episodes']}`")
    lines.append(f"- holdout_episodes: `{settings['holdout_episodes']}`")
    lines.append("- Return is reported with the paper-comparable raw scalarisation of LR, risk, and cost under each ablation weight setting.")
    lines.append("")
    lines.append("## Holdout Metrics")
    lines.append("")
    lines.append("| Variant | Return | LR | Risk | Cost | Min F | Feasible |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for rec in records:
        m = rec["best_eval_holdout"]
        lines.append(
            f"| {rec['name']} | {_metric_value(rec, 'reported_return'):.4f} | {float(m['lr']):.3f} | {float(m['risk']):.3f} | {float(m['lcc']):.2f} | {float(m['min_f']):.3f} | {float(m['feasible_frac']):.3f} |"
        )
    lines.append("")
    lines.append("## Relative To Baseline")
    lines.append("")
    lines.append("| Variant | dReturn | dLR | dRisk | dCost |")
    lines.append("|---|---:|---:|---:|---:|")
    for rec in records[1:]:
        m = rec["best_eval_holdout"]
        lines.append(
            f"| {rec['name']} | {_metric_value(rec, 'reported_return') - baseline_return:+.4f} | {float(m['lr']) - float(baseline['lr']):+.3f} | {float(m['risk']) - float(baseline['risk']):+.3f} | {float(m['lcc']) - float(baseline['lcc']):+.2f} |"
        )
    lines.append("")
    lines.append("## Quick Findings")
    lines.append("")
    lines.append(
        f"- Best holdout return: `{rank[0]['name']}` ({_metric_value(rank[0], 'reported_return'):.4f})."
    )
    worst_risk = max(records, key=lambda r: float(r["best_eval_holdout"]["risk"]))
    best_risk = min(records, key=lambda r: float(r["best_eval_holdout"]["risk"]))
    lines.append(
        f"- Lowest risk: `{best_risk['name']}` ({float(best_risk['best_eval_holdout']['risk']):.3f}); highest risk: `{worst_risk['name']}` ({float(worst_risk['best_eval_holdout']['risk']):.3f})."
    )
    best_cost = min(records, key=lambda r: float(r["best_eval_holdout"]["lcc"]))
    worst_cost = max(records, key=lambda r: float(r["best_eval_holdout"]["lcc"]))
    lines.append(
        f"- Lowest cost: `{best_cost['name']}` ({float(best_cost['best_eval_holdout']['lcc']):.2f}); highest cost: `{worst_cost['name']}` ({float(worst_cost['best_eval_holdout']['lcc']):.2f})."
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `ablation_metric_return.png`, `ablation_metric_lr.png`, `ablation_metric_risk.png`, `ablation_metric_cost.png`: metric-wise bar comparisons.")
    lines.append("- `ablation_metrics.png`: 2x2 metric panel.")
    lines.append("- `ablation_convergence.png`: best-of-iteration return curves.")
    lines.append("- `ablation_tradeoff.png`: risk-cost trade-off scatter.")
    lines.append("")
    (out_dir / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_ablation(
    *,
    out_root: Path,
    iterations: int | None,
    population: int | None,
    eval_episodes: int | None,
    holdout_episodes: int,
    elite_frac: float | None,
    base_seed: int | None,
    base_config_path: Path | None = None,
) -> Path:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dir = out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(out_dir / "ablation.log")

    base_cfg, source_cem_cfg, source_config_path = _base_cfg(base_config_path)
    iterations_v = int(source_cem_cfg.iterations if iterations is None else iterations)
    population_v = int(source_cem_cfg.population if population is None else population)
    eval_episodes_v = int(source_cem_cfg.eval_episodes if eval_episodes is None else eval_episodes)
    elite_frac_v = float(source_cem_cfg.elite_frac if elite_frac is None else elite_frac)
    base_seed_v = int(source_cem_cfg.seed if base_seed is None else base_seed)

    variants = _variants(base_cfg)
    cem_cfg = tl.CEMConfig(
        iterations=iterations_v,
        population=population_v,
        elite_frac=elite_frac_v,
        eval_episodes=eval_episodes_v,
        seed=base_seed_v,
        plot_first_n=0,
        plot_interval=9999,
    )

    logger.info("Ablation setup | out_dir=%s base_config=%s", str(out_dir), str(source_config_path))
    records: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        seed = int(base_seed_v + idx * 1000)
        rec = _run_one_variant(
            variant=variant,
            cem_cfg=cem_cfg,
            seed=seed,
            holdout_episodes=holdout_episodes,
            out_dir=out_dir,
            logger=logger,
        )
        records.append(rec)
    for rec in records:
        _ensure_reported_return(rec)

    _plot_metric_panels(records, out_dir)
    _plot_convergence(records, out_dir / "ablation_convergence.png")
    _plot_tradeoff(records, out_dir / "ablation_tradeoff.png")
    _copy_key_ablation_figures(out_dir)

    summary_payload = {
        "run_time": run_time,
        "settings": {
            "base_config": str(source_config_path),
            "iterations": iterations_v,
            "population": population_v,
            "elite_frac": elite_frac_v,
            "eval_episodes": eval_episodes_v,
            "holdout_episodes": int(holdout_episodes),
            "base_seed": base_seed_v,
        },
        "records": records,
    }
    (out_dir / "ablation_all_results.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _build_report(
        records,
        out_dir,
        {
            "run_time": run_time,
            "iterations": iterations_v,
            "population": population_v,
            "elite_frac": elite_frac_v,
            "eval_episodes": eval_episodes_v,
            "holdout_episodes": int(holdout_episodes),
        },
    )
    logger.info("Ablation completed | results=%s", str(out_dir))
    return out_dir


def redraw_only_from_results(*, out_root: Path, results_json: Path | None = None) -> Path:
    out_dir = out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    src = results_json if results_json is not None else (out_dir / "ablation_all_results.json")
    if not src.exists():
        raise FileNotFoundError(f"Results JSON not found: {src}")

    payload = json.loads(src.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"No records found in results JSON: {src}")
    for rec in records:
        if isinstance(rec, dict):
            _ensure_reported_return(rec)
    payload["records"] = records

    settings = payload.get("settings", {})
    run_time = str(payload.get("run_time", "plot_only"))
    _plot_metric_panels(records, out_dir)
    _plot_convergence(records, out_dir / "ablation_convergence.png")
    _plot_tradeoff(records, out_dir / "ablation_tradeoff.png")
    _copy_key_ablation_figures(out_dir)
    _build_report(
        records,
        out_dir,
        {
            "run_time": run_time,
            "iterations": int(settings.get("iterations", 0)),
            "population": int(settings.get("population", 0)),
            "elite_frac": float(settings.get("elite_frac", 0.0)),
            "eval_episodes": int(settings.get("eval_episodes", 0)),
            "holdout_episodes": int(settings.get("holdout_episodes", 0)),
        },
    )
    (out_dir / "ablation_all_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments for lifecycle CEM policy search.")
    parser.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Output directory for ablation results. Default: <script_dir>/ablation_results.",
    )
    parser.add_argument("--plot-only", action="store_true", help="Redraw figures/report from existing results JSON without training.")
    parser.add_argument(
        "--results-json",
        type=str,
        default="",
        help="Path to existing ablation_all_results.json (used with --plot-only).",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="",
        help="Normalized baseline case config. Default: normalized_cases/case1_baseline_normalized.json.",
    )
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--population", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--holdout-episodes", type=int, default=100)
    parser.add_argument("--elite-frac", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_root = (
        (script_dir / "ablation_results").resolve()
        if not str(args.out_root).strip()
        else Path(args.out_root).resolve()
    )
    if bool(args.plot_only):
        results_json = Path(args.results_json).resolve() if str(args.results_json).strip() else None
        out_dir = redraw_only_from_results(out_root=out_root, results_json=results_json)
        print(str(out_dir))
        return

    run_dir = run_ablation(
        out_root=out_root,
        iterations=None if int(args.iterations) <= 0 else int(args.iterations),
        population=None if int(args.population) <= 0 else int(args.population),
        eval_episodes=None if int(args.eval_episodes) <= 0 else int(args.eval_episodes),
        holdout_episodes=args.holdout_episodes,
        elite_frac=None if float(args.elite_frac) < 0.0 else float(args.elite_frac),
        base_seed=None if int(args.seed) < 0 else int(args.seed),
        base_config_path=Path(args.base_config).resolve() if str(args.base_config).strip() else None,
    )
    print(str(run_dir))


if __name__ == "__main__":
    main()
