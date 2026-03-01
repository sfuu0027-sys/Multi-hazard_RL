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
    "trade_label": 20.0,
    "trade_legend": 20.0,
    "trade_legend_title": 20.0,
    "trade_bubble_legend": 16.0,
    "trade_bubble_legend_title": 16.0,
    "trade_cbar_label": 20.0,
    "trade_cbar_ticks": 20.0,
    "trade_footer": 20.0,
}


def _font(name: str) -> float:
    return float(_PLOT_FONT[name])


def _display_variant_name(name: str) -> str:
    name_map = {
        "baseline_full": "Default",
        "no_risk_term": "No risk",
        "no_resilience_term": "No resilience",
        "no_cost_term": "No cost",
        "no_pre_reinforcement": "No pre-reinforcement",
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


def _base_cfg() -> tl.LifecycleConfig:
    # Same hazard setting as case1 baseline.
    return tl.LifecycleConfig(
        lambda_eq_per_year=1.0 / 35.0,
        lambda_fire_per_year=1.0 / 3.42,
        w_lr=1.0 / 3.0,
        w_risk=1.0 / 3.0,
        w_cost=1.0 / 3.0,
        discount_rate=0.03,
        w_floor=0.0,
    )


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
    compare_ep = result["best"]["episode_compare_seed"]
    payload = {
        "name": name,
        "desc": str(variant["desc"]),
        "seed": int(seed),
        "cfg": {
            "w_lr": float(cfg.w_lr),
            "w_risk": float(cfg.w_risk),
            "w_cost": float(cfg.w_cost),
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
    (run_dir / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Ablation run done | variant=%s holdout_return=%.4f holdout_risk=%.3f holdout_npv=%.2f",
        name,
        float(holdout["return"]),
        float(holdout["risk"]),
        float(holdout["npv"]),
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
    colors = ["#4C72B0", "#4E9FB5", "#86C5A3", "#E6D58C", "#CC6677"]
    hatches = ["x", "/", "-", "\\", "o"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(16.0, 9.8))
    vals = np.array([float(r["best_eval_holdout"][metric_key]) for r in records], dtype=float)
    bars = ax.bar(
        x,
        vals,
        width=0.72,
        color=[colors[i % len(colors)] for i in range(len(vals))],
        edgecolor="black",
        linewidth=1.0,
        alpha=0.95,
        zorder=3,
    )
    for i, b in enumerate(bars):
        b.set_hatch(hatches[i % len(hatches)])
    ax.set_facecolor("#F7F7F7")
    x_tick_shift = 0.12
    ax.set_xticks(x + x_tick_shift)
    ax.set_xticklabels(labels, rotation=14, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="both", labelsize=_font("metric_ticks"))
    ax.set_ylabel(metric_ylabel, fontsize=_font("metric_axis"))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    span = max(1.0, abs(vmax - vmin))
    y_bottom = vmin - 0.30 * span
    y_top = vmax + 0.40 * span
    if vmax <= 0.0:
        # For all-negative panels, lift the zero line by giving more room below.
        y_bottom = vmin - 0.42 * span
        y_top = max(2.0, vmax + 0.25 * span)
    if vmin >= 0.0:
        y_bottom = 0.0
    ax.set_ylim(y_bottom, y_top)
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
        # Flip label direction near plot bounds to avoid clipping.
        if v >= 0:
            place_above = (y_top - v) >= (0.12 * span)
        else:
            place_above = (v - y_bottom) < (0.12 * span)
        y_shift = 8 if place_above else -8
        ax.annotate(
            label,
            xy=(b.get_x() + b.get_width() / 2.0, v),
            xytext=(0, y_shift),
            textcoords="offset points",
            ha="center",
            va="bottom" if place_above else "top",
            fontsize=_font("metric_value"),
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.72, linewidth=0.0),
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_panels(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        ("return", "Return", True, "ablation_metric_return.png"),
        ("lr", "Resilience loss", False, "ablation_metric_lr.png"),
        ("risk", "Risk", False, "ablation_metric_risk.png"),
        ("npv", "Cost", False, "ablation_metric_npv.png"),
    ]
    for key, ylabel, higher_better, filename in metric_specs:
        _plot_one_metric_bar(
            records,
            metric_key=key,
            metric_ylabel=ylabel,
            higher_better=higher_better,
            out_path=out_dir / filename,
        )


def _copy_key_ablation_figures(out_dir: Path) -> None:
    src_dir = out_dir
    dst_dir = Path(__file__).resolve().parent / "fig" / "ablation"
    dst_dir.mkdir(parents=True, exist_ok=True)
    fig_names = [
        "ablation_metric_return.png",
        "ablation_metric_lr.png",
        "ablation_metric_risk.png",
        "ablation_metric_npv.png",
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
    ax.legend(loc="best", fontsize=_font("conv_legend"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_tradeoff(records: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    names = [str(r["name"]) for r in records]
    npv = np.array([float(r["best_eval_holdout"]["npv"]) for r in records], dtype=float)
    risk = np.array([float(r["best_eval_holdout"]["risk"]) for r in records], dtype=float)
    lr = np.array([float(r["best_eval_holdout"]["lr"]) for r in records], dtype=float)
    ret = np.array([float(r["best_eval_holdout"]["return"]) for r in records], dtype=float)

    lr_min = float(np.min(lr))
    lr_max = float(np.max(lr))
    if lr_max - lr_min <= 1.0e-12:
        bubble_sizes = np.full_like(lr, 420.0)
    else:
        # Lower LR is better -> larger bubbles.
        bubble_sizes = 220.0 + 760.0 * (lr_max - lr) / (lr_max - lr_min)

    sc = ax.scatter(
        npv,
        risk,
        s=bubble_sizes,
        c=ret,
        cmap="viridis",
        alpha=0.88,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    label_offsets = {
        "baseline_full": (16, 8),
        "no_risk_term": (18, -14),
        "no_resilience_term": (16, 8),
        "no_cost_term": (-98, 24),
        "no_pre_reinforcement": (16, 8),
    }
    for i, name in enumerate(names):
        dx, dy = label_offsets.get(name, (8, 8))
        ax.annotate(
            _display_variant_name(name),
            xy=(npv[i], risk[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=_font("trade_label"),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.72, linewidth=0.0),
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
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
        ncols=1,
        frameon=True,
        facecolor="white",
        framealpha=0.82,
        fontsize=_font("trade_bubble_legend"),
        title_fontsize=_font("trade_bubble_legend_title"),
    )

    ax.set_xlabel("Cost", fontsize=_font("trade_axis"))
    ax.set_ylabel("Risk", fontsize=_font("trade_axis"))
    ax.tick_params(axis="both", labelsize=_font("trade_ticks"))
    ax.set_title("Risk-Cost Trade-off on Holdout Episodes", fontsize=_font("trade_title"), pad=7)
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_pad = max(2.4, float((np.max(npv) - np.min(npv)) * 0.10))
    y_pad = max(0.2, float((np.max(risk) - np.min(risk)) * 0.10))
    ax.set_xlim(float(np.min(npv) - x_pad), float(np.max(npv) + x_pad))
    ax.set_ylim(max(0.0, float(np.min(risk) - y_pad)), float(np.max(risk) + y_pad))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _build_report(records: list[dict[str, Any]], out_dir: Path, settings: dict[str, Any]) -> None:
    baseline = records[0]["best_eval_holdout"]
    rank = sorted(
        records,
        key=lambda r: float(r["best_eval_holdout"]["return"]),
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
    lines.append("")
    lines.append("## Holdout Metrics")
    lines.append("")
    lines.append("| Variant | Return | LR | Risk | NPV | Min F | Feasible |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for rec in records:
        m = rec["best_eval_holdout"]
        lines.append(
            f"| {rec['name']} | {float(m['return']):.4f} | {float(m['lr']):.3f} | {float(m['risk']):.3f} | {float(m['npv']):.2f} | {float(m['min_f']):.3f} | {float(m['feasible_frac']):.3f} |"
        )
    lines.append("")
    lines.append("## Relative To Baseline")
    lines.append("")
    lines.append("| Variant | dReturn | dLR | dRisk | dNPV |")
    lines.append("|---|---:|---:|---:|---:|")
    for rec in records[1:]:
        m = rec["best_eval_holdout"]
        lines.append(
            f"| {rec['name']} | {float(m['return']) - float(baseline['return']):+.4f} | {float(m['lr']) - float(baseline['lr']):+.3f} | {float(m['risk']) - float(baseline['risk']):+.3f} | {float(m['npv']) - float(baseline['npv']):+.2f} |"
        )
    lines.append("")
    lines.append("## Quick Findings")
    lines.append("")
    lines.append(
        f"- Best holdout return: `{rank[0]['name']}` ({float(rank[0]['best_eval_holdout']['return']):.4f})."
    )
    worst_risk = max(records, key=lambda r: float(r["best_eval_holdout"]["risk"]))
    best_risk = min(records, key=lambda r: float(r["best_eval_holdout"]["risk"]))
    lines.append(
        f"- Lowest risk: `{best_risk['name']}` ({float(best_risk['best_eval_holdout']['risk']):.3f}); highest risk: `{worst_risk['name']}` ({float(worst_risk['best_eval_holdout']['risk']):.3f})."
    )
    best_cost = min(records, key=lambda r: float(r["best_eval_holdout"]["npv"]))
    worst_cost = max(records, key=lambda r: float(r["best_eval_holdout"]["npv"]))
    lines.append(
        f"- Lowest NPV: `{best_cost['name']}` ({float(best_cost['best_eval_holdout']['npv']):.2f}); highest NPV: `{worst_cost['name']}` ({float(worst_cost['best_eval_holdout']['npv']):.2f})."
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `ablation_metric_return.png`, `ablation_metric_lr.png`, `ablation_metric_risk.png`, `ablation_metric_npv.png`: metric-wise bar comparisons.")
    lines.append("- `ablation_convergence.png`: best-of-iteration return curves.")
    lines.append("- `ablation_tradeoff.png`: risk-cost trade-off scatter.")
    lines.append("")
    (out_dir / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_ablation(
    *,
    out_root: Path,
    iterations: int,
    population: int,
    eval_episodes: int,
    holdout_episodes: int,
    elite_frac: float,
    base_seed: int,
) -> Path:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dir = out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(out_dir / "ablation.log")

    base_cfg = _base_cfg()
    variants = _variants(base_cfg)
    cem_cfg = tl.CEMConfig(
        iterations=int(iterations),
        population=int(population),
        elite_frac=float(elite_frac),
        eval_episodes=int(eval_episodes),
        seed=int(base_seed),
        plot_first_n=0,
        plot_interval=9999,
    )

    logger.info("Ablation setup | out_dir=%s", str(out_dir))
    records: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        seed = int(base_seed + idx * 1000)
        rec = _run_one_variant(
            variant=variant,
            cem_cfg=cem_cfg,
            seed=seed,
            holdout_episodes=holdout_episodes,
            out_dir=out_dir,
            logger=logger,
        )
        records.append(rec)

    _plot_metric_panels(records, out_dir)
    _plot_convergence(records, out_dir / "ablation_convergence.png")
    _plot_tradeoff(records, out_dir / "ablation_tradeoff.png")
    _copy_key_ablation_figures(out_dir)

    summary_payload = {
        "run_time": run_time,
        "settings": {
            "iterations": int(iterations),
            "population": int(population),
            "elite_frac": float(elite_frac),
            "eval_episodes": int(eval_episodes),
            "holdout_episodes": int(holdout_episodes),
            "base_seed": int(base_seed),
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
            "iterations": int(iterations),
            "population": int(population),
            "elite_frac": float(elite_frac),
            "eval_episodes": int(eval_episodes),
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
    parser.add_argument("--iterations", type=int, default=28)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--holdout-episodes", type=int, default=50)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260315)
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
        iterations=args.iterations,
        population=args.population,
        eval_episodes=args.eval_episodes,
        holdout_episodes=args.holdout_episodes,
        elite_frac=args.elite_frac,
        base_seed=args.seed,
    )
    print(str(run_dir))


if __name__ == "__main__":
    main()
