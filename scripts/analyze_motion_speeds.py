"""Report root-frame expert velocities and clip-uniform raw-frame references."""

# ruff: noqa: E402 -- ensure this repository wins over other editable `src` packages.

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

AXES = ("vx", "vy", "vz", "wx", "wy", "wz")
QUANTILES = {"min": 0.0, "p01": 0.01, "p05": 0.05, "p50": 0.5,
             "p95": 0.95, "p99": 0.99, "max": 1.0}


@dataclass(frozen=True)
class AnalysisConfig:
    motion_root: Path = _REPO_ROOT / "src/assets/motions"
    selected_dir: Path = _REPO_ROOT / "src/assets/motions/go2"
    exclude_patterns: tuple[str, ...] = ("go2_jump_*.npz",)
    output_dir: Path = _REPO_ROOT / "docs/motion_analysis/go2"


def load_velocities(path: Path) -> tuple[np.ndarray, float]:
    """Read body index zero (base_link); NPZ quaternions are wxyz."""
    with np.load(path, allow_pickle=False) as data:
        fps = np.asarray(data["fps"]).reshape(-1)
        if fps.size != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
            raise ValueError(f"{path}: invalid FPS")
        quat = np.asarray(data["body_quat_w"][:, 0], dtype=np.float64)
        linear = np.asarray(data["body_lin_vel_w"][:, 0], dtype=np.float64)
        angular = np.asarray(data["body_ang_vel_w"][:, 0], dtype=np.float64)
    n = len(quat)
    if n < 2 or quat.shape != (n, 4) or linear.shape != (n, 3) or angular.shape != (n, 3):
        raise ValueError(f"{path}: invalid root-array shapes")
    if not all(np.isfinite(array).all() for array in (quat, linear, angular)):
        raise ValueError(f"{path}: nonfinite root states")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if (norm < 1e-6).any():
        raise ValueError(f"{path}: zero quaternion")
    quat /= norm

    def inverse_rotate(vector):
        cross = 2.0 * np.cross(quat[:, 1:], vector)
        return vector - quat[:, :1] * cross + np.cross(quat[:, 1:], cross)

    return np.concatenate((inverse_rotate(linear), inverse_rotate(angular)), axis=-1), float(fps[0])


def summarize(values: np.ndarray, weights: np.ndarray | None = None) -> dict:
    result = {}
    for index, axis in enumerate(AXES):
        column = values[:, index]
        if weights is None:
            quantiles = np.quantile(column, list(QUANTILES.values()))
        else:
            order = np.argsort(column)
            cdf = np.cumsum(weights[order]) / weights.sum()
            indices = np.searchsorted(cdf, list(QUANTILES.values()), side="left")
            quantiles = column[order[np.clip(indices, 0, len(column) - 1)]]
        result[axis] = {name: float(value) for name, value in zip(QUANTILES, quantiles)}
        result[axis]["mean"] = float(np.average(column, weights=weights))
    return result


def render_plot(path, selected, values, transition_values, transition_weights):
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), layout="constrained",
                             gridspec_kw={"height_ratios": (2, 1)})
    components = ((0, "vx", "m/s"), (1, "vy", "m/s"), (5, "wz", "rad/s"))
    y = np.arange(len(selected))
    for column, (index, name, unit) in enumerate(components):
        ax = axes[0, column]
        stats = [record["stats"][name] for record in selected]
        ax.hlines(y, [s["min"] for s in stats], [s["max"] for s in stats], color="0.8", label="Min–max")
        ax.hlines(y, [s["p01"] for s in stats], [s["p99"] for s in stats], color="#2563eb", lw=2, label="P01–P99")
        ax.plot([s["mean"] for s in stats], y, "o", color="#dc2626", ms=4, label="Mean")
        ax.set_yticks(y, [Path(r["file"]).stem.removeprefix("go2_") for r in selected] if column == 0 else [])
        ax.invert_yaxis()
        ax.set_xlabel(f"{name} [{unit}]")
        ax.grid(axis="x", alpha=0.2)
        if column == 0:
            ax.legend(loc="best", fontsize=8)
        ax = axes[1, column]
        edges = np.linspace(values[:, index].min(), values[:, index].max(), 61)
        ax.hist(values[:, index], bins=edges, density=True, histtype="step", color="0.4", label="Stored frames")
        ax.hist(transition_values[:, index], bins=edges, weights=transition_weights,
                density=True, histtype="step", color="#2563eb", label="Clip-uniform raw-frame reference")
        ax.set_xlabel(f"{name} [{unit}]")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Go2 selected experts: {len(selected)} clips / {len(values)} frames; base_link coordinates")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict):
    active = summary["selected"]
    rows = ["# Go2 Expert Motion Speed Analysis", "",
            "Velocities come from NPZ `body_*_vel_w[:, 0]`, inverse-rotated by the normalized "
            "`body_quat_w[:, 0]` (wxyz) into the full `base_link` frame, matching the AMP tracking rewards. "
            "These are measured trajectory velocities; the files do not provide target velocity commands.", "",
            f"The complete asset directory contains **{summary['all']['clips']} clips / {summary['all']['frames']} frames**. "
            f"The current AMP selection contains **{active['clips']} clips / {active['frames']} frames**, "
            f"with {active['duration_s']:.2f} seconds of stored frames at {active['fps']} Hz. "
            "Selection uses the immediate Go2 directory and excludes `go2_jump_*.npz`.", "",
            "## Selected dataset: frame-weighted velocities", "",
            "| Axis | Unit | Min | P01 | P05 | Median | P95 | P99 | Max |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for index, axis in enumerate(AXES):
        stats = active["stats"][axis]
        rows.append(f"| {axis} | {'m/s' if index < 3 else 'rad/s'} | " +
                    " | ".join(f"{stats[name]:.3f}" for name in QUANTILES) + " |")
    rows += ["", "## Clip-uniform raw-frame reference", "",
             "Each selected clip has equal probability. These weighted statistics characterize the original "
             "discrete frames, excluding each clip's last frame, with empirical CDF weight "
             "`1 / (num_clips * (clip_frames - 1))` per eligible frame. Training samples continuous valid times, "
             "interpolates world poses/velocities, and preloads 2,000,000 transitions; the table is a raw-frame "
             "reference rather than a measurement of the interpolated cache.", "",
             "| Axis | P01 | P05 | Median | P95 | P99 |",
             "|---|---:|---:|---:|---:|---:|"]
    for axis in ("vx", "vy", "wz"):
        stats = summary["amp_sampling"]["stats"][axis]
        rows.append(f"| {axis} | " + " | ".join(f"{stats[name]:.3f}" for name in ("p01", "p05", "p50", "p95", "p99")) + " |")
    canter = [r for r in summary["clips"] if r["selected"] and "canter" in r["file"]]
    canter_frames = sum(r["frames"] for r in canter)
    rows += ["", f"Canter has {len(canter)} clips / {canter_frames} frames "
             f"({100 * canter_frames / active['frames']:.1f}% of stored frames), but "
             f"{100 * len(canter) / active['clips']:.1f}% of clip-uniform sample probability. "
             "Frame pooling therefore understates the high-speed prior used by AMP.", "",
             "## Per-clip statistics", "",
             "Linear units are m/s; angular units are rad/s. Full component quantiles are in `speeds.csv` and `speeds.json`.", "",
             "| Clip | Selected | Frames | Mean vx | vx min–max | Mean vy | vy min–max | Mean wz | wz min–max |",
             "|---|---|---:|---:|---|---:|---|---:|---|"]
    for record in summary["clips"]:
        cells = [Path(record["file"]).name, "yes" if record["selected"] else "no", str(record["frames"])]
        for axis in ("vx", "vy", "wz"):
            stats = record["stats"][axis]
            cells.extend((f"{stats['mean']:.3f}", f"[{stats['min']:.3f}, {stats['max']:.3f}]"))
        rows.append("| " + " | ".join(cells) + " |")
    rows += ["", "## Interpretation", "",
             "Clip means describe sustained speed better than instantaneous extrema, which include starts, "
             "stops, and gait oscillations. The canter clips are short; their extrema should not be treated "
             "as demonstrated sustained target speeds. Most other clips move along one axis or turn in place. "
             "The marginal ranges do not establish coverage of every combined forward/lateral/yaw command.", "",
             "Canter per-clip means `(vx, vy)` are " + "; ".join(
                 f"`{Path(r['file']).stem}`: ({r['stats']['vx']['mean']:.3f}, {r['stats']['vy']['mean']:.3f}) m/s"
                 for r in canter) + ". Lateral drift in the canter clips makes the high-speed expert prior asymmetric.", "",
             "The staged command limits and transition steps are defined in "
             "`src/tasks/amp_loco/config/go2/env_cfgs.py`.", "",
             "![Go2 expert speed ranges and sampling distributions](speeds.png)", "",
             "Reproduce with `python scripts/analyze_motion_speeds.py`.", ""]
    path.write_text("\n".join(rows))


def main(cfg: AnalysisConfig):
    records, selected_values, all_values = [], [], []
    selected_dir = cfg.selected_dir.resolve()
    for path in sorted(cfg.motion_root.rglob("*.npz")):
        values, fps = load_velocities(path)
        selected = path.parent.resolve() == selected_dir and not any(path.match(pattern) for pattern in cfg.exclude_patterns)
        record = {"file": str(path.relative_to(cfg.motion_root)), "selected": selected,
                  "frames": len(values), "fps": fps, "duration_s": len(values) / fps,
                  "stats": summarize(values)}
        records.append(record)
        all_values.append(values)
        if selected:
            selected_values.append(values)
    selected = [r for r in records if r["selected"]]
    if not selected:
        raise ValueError(f"No selected clips found in {selected_dir}")
    if len({r["fps"] for r in selected}) != 1:
        raise ValueError("Selected expert FPS values differ")
    values = np.concatenate(selected_values)
    transitions = np.concatenate([v[:-1] for v in selected_values])
    weights = np.concatenate([np.full(len(v) - 1, 1 / (len(selected) * (len(v) - 1))) for v in selected_values])
    summary = {
        "frame": "base_link; full inverse wxyz quaternion rotation; NPZ body index 0",
        "selection": {"selected_dir": str(cfg.selected_dir), "exclude_patterns": cfg.exclude_patterns},
        "all": {"clips": len(records), "frames": sum(len(v) for v in all_values),
                "stats": summarize(np.concatenate(all_values))},
        "selected": {"clips": len(selected), "frames": len(values), "fps": selected[0]["fps"],
                     "duration_s": sum(r["duration_s"] for r in selected), "stats": summarize(values)},
        "amp_sampling": {"sampling_model": "clip-uniform raw-frame reference before interpolation",
                         "eligible_current_frames": len(transitions), "stats": summarize(transitions, weights)},
        "clips": records,
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "speeds.json").write_text(json.dumps(summary, indent=2) + "\n")
    flat_rows = [{**{key: record[key] for key in ("file", "selected", "frames", "fps", "duration_s")},
                  **{f"{axis}_{stat}": value for axis, stats in record["stats"].items() for stat, value in stats.items()}}
                 for record in records]
    with (cfg.output_dir / "speeds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    render_plot(cfg.output_dir / "speeds.png", selected, values, transitions, weights)
    write_report(cfg.output_dir / "README.md", summary)
    print(json.dumps({"report": str(cfg.output_dir / "README.md"), "selected": summary["selected"],
                      "amp_sampling": summary["amp_sampling"]}, indent=2))


if __name__ == "__main__":
    main(tyro.cli(AnalysisConfig))
