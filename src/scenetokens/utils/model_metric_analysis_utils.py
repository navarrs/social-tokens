import math
from itertools import product
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from omegaconf import DictConfig

from scenetokens.utils.constants import SMALL_EPSILON


plt.style.use("seaborn-v0_8-whitegrid")

MODEL_NAME_MAP = {
    "wayformer": "Wayformer",
    "scenetransformer": "SceneTransformer",
    "scenetokens-student": "ST",
    "scenetokens-teacher-unmasked": "Causal-ST",
    "safe-scenetokens": "Safe-ST",
}


def _plot_sample_selection_sweep_lineplot(  # noqa: PLR0915
    config: DictConfig, log: Logger, output_path: Path, metrics_df: pd.DataFrame, suffix: str = ""
) -> None:
    """Wrapper to call plot_sample_selection_sweep_lineplot for a specific metrics file.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_df (pd.DataFrame): DataFrame containing the metrics data.
        suffix (str): Suffix to append to output filenames to distinguish different metrics files.
    """
    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    log.info("Plotting sample selection sweep lineplots for metrics: %s", metrics)
    retention_pcts = list(map(float, config.sample_retention_percentages))

    # Create a colormap for the strategies
    cmap = plt.cm.get_cmap(config.get("lineplot_colormap", "tab10"))
    colors = [cmap(i) for i in range(cmap.N)]
    colormap = {s: colors[i % len(colors)] for i, s in enumerate(config.sample_selection_strategies_to_compare)}

    for model, split in product(config.models_to_compare, config.sample_selection_splits_to_compare):
        subsplit = split.split("/")[-1]
        log.info("Creating sweep plot for model=%s, split=%s", model, split)

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
        for i, metric in enumerate(metrics):
            ax = axes[0, i]
            column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric

            # Plot each sample selection strategy
            best_value, best_strategy, best_x, best_y, all_y_values = np.inf, None, None, None, []
            for strategy in config.sample_selection_strategies_to_compare:
                sweep_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}"
                sweep_df = metrics_df[metrics_df["Name"].str.startswith(sweep_prefix)]

                # Collect y values across retention percentages
                y = []
                for pct in retention_pcts:
                    row = sweep_df[sweep_df["Name"] == f"{sweep_prefix}_{pct}"]
                    if not row.empty and column in row.columns:
                        val = row.iloc[0][column]
                        y.append(val)
                        all_y_values.append(val)
                    else:
                        y.append(np.nan)
                y = np.array(y, dtype=float)
                if metric == "Runtime":
                    y = y / 3600.0  # convert to hours
                    all_y_values = [v / 3600.0 for v in all_y_values]

                # Track best value
                if np.nanmin(y) < best_value:
                    best_value = np.nanmin(y)
                    best_strategy = strategy
                    idx = int(np.nanargmin(y))
                    best_x = retention_pcts[idx]
                    best_y = y[idx]

                # Plot metrics across retention percentages
                ax.plot(retention_pcts, y, marker="o", ms=6, lw=2.5, c=colormap[strategy], alpha=0.9, label=strategy)

            # Highlight best strategy point
            if best_strategy is not None:
                ax.plot(best_x, best_y, marker="*", ms=16, c=colormap[best_strategy], mec="k", zorder=10, label="Best")

            # Add base model (no sample selection) reference line
            base_experiment = f"{config.sample_selection_benchmark}_{model}"
            base_experiment_df = metrics_df[metrics_df["Name"] == base_experiment]
            if not base_experiment_df.empty and column in base_experiment_df.columns:
                base_value = base_experiment_df[column].min()
                if metric == "Runtime":
                    base_value = base_value / 3600.0  # convert to hours
                ax.axhline(base_value, ls="--", lw=2, c="black", alpha=0.7, label="Base model")
                all_y_values.append(base_value)

            # Auto-scale y-axis with padding
            if all_y_values:
                ymin, ymax = np.nanmin(all_y_values), np.nanmax(all_y_values)
                padding = 0.5 * (ymax - ymin) if ymax > ymin else 0.05
                ax.set_ylim(ymin - padding, ymax + padding)

            ax.set_title(metric.replace("_", " "), pad=10)
            ax.set_xlabel("Data Retention (%)")
            ax.set_ylabel("Metric Value")
            ax.set_xticks(retention_pcts)
            ax.set_xticklabels([f"{int(p * 100)}%" for p in retention_pcts])
            ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend().remove()

        # Shared legend at the bottom
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)
        plt.tight_layout(rect=(0, 0.1, 1, 1))

        output_filepath = output_path / f"{model}_{subsplit}{suffix}.png"
        fig.savefig(output_filepath, dpi=200)
        plt.close(fig)
        log.info("Saved sweep plot to %s", output_filepath)


def _plot_joint_sample_selection_sweep_lineplot(
    config: DictConfig, log: Logger, output_path: Path, metrics_dataframes: dict[str, pd.DataFrame]
) -> None:
    """Creates subplots for each strategy, showing different dataframe versions (which are metrics file versions) as
    separate lines.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_dataframes (dict[str, pd.DataFrame]): Dictionary of DataFrames containing the metrics for each file.
    """
    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    log.info("Plotting sample selection sweep lineplots for metrics: %s", metrics)
    retention_pcts = list(map(float, config.sample_retention_percentages))

    # Create a colormap for the dataframe versions
    cmap = plt.cm.get_cmap(config.get("lineplot_colormap", "tab10"))
    colors = [cmap(i) for i in range(cmap.N)]
    colormap = {version: colors[i % len(colors)] for i, version in enumerate(metrics_dataframes.keys())}

    for model, split in product(config.models_to_compare, config.sample_selection_splits_to_compare):
        subsplit = split.split("/")[-1]

        for strategy in config.sample_selection_strategies_to_compare:
            log.info("Creating sweep plot for model=%s, split=%s, strategy=%s", model, split, strategy)

            fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
            for i, metric in enumerate(metrics):
                ax = axes[0, i]
                column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric

                # Plot each dataframe version for this strategy
                all_y_values = []
                for version_key, sweep_df in metrics_dataframes.items():
                    sweep_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}"

                    # Collect y values across retention percentages
                    y = []
                    for pct in retention_pcts:
                        row = sweep_df[sweep_df["Name"] == f"{sweep_prefix}_{pct}"]
                        if not row.empty and column in row.columns:
                            val = row.iloc[0][column]
                            y.append(val)
                            all_y_values.append(val)
                        else:
                            y.append(np.nan)
                    y = np.array(y, dtype=float)
                    if metric == "Runtime":
                        y = y / 3600.0  # convert to hours
                        all_y_values = [v / 3600.0 for v in all_y_values]

                    # Plot metrics across retention percentages
                    ax.plot(
                        retention_pcts,
                        y,
                        marker="o",
                        ms=6,
                        lw=2.5,
                        c=colormap[version_key],
                        alpha=0.9,
                        label=version_key,
                    )

                # Auto-scale y-axis with padding
                if all_y_values:
                    ymin, ymax = np.nanmin(all_y_values), np.nanmax(all_y_values)
                    padding = 0.5 * (ymax - ymin) if ymax > ymin else 0.05
                    ax.set_ylim(ymin - padding, ymax + padding)

                ax.set_title(metric.replace("_", " "), pad=10)
                ax.set_xlabel("Data Retention (%)")
                ax.set_ylabel("Metric Value")
                ax.set_xticks(retention_pcts)
                ax.set_xticklabels([f"{int(p * 100)}%" for p in retention_pcts])
                ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.4)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend().remove()

            # Shared legend at the bottom
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)
            plt.tight_layout(rect=(0, 0.1, 1, 1))

            output_filepath = output_path / f"{model}_{subsplit}_{strategy}.png"
            fig.savefig(output_filepath, dpi=200)
            plt.close(fig)
            log.info("Saved sweep plot to %s", output_filepath)


def plot_sample_selection_sweep_lineplot(config: DictConfig, log: Logger, output_path: Path) -> None:
    """For each (model, subsplit), creates a figure with one subplot per metric. Each subplot shows metric values across
    retention percentages for all sample selection strategies, plus a horizontal base-model reference line. Highlights
    best strategy, auto-scales y-axis, and adds confidence bands when available.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
    """
    output_path = output_path / "sample_selection_lineplots"
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metrics CSV
    metrics_dataframes = {}
    for file in config.sample_selection_files:
        log.info("Processing sample selection sweep lineplots for file: %s", file)
        metrics_filepath = Path(file)
        if not metrics_filepath.exists():
            log.error("Sample selection CSV not found at %s", metrics_filepath)
            return
        metrics_df = pd.read_csv(metrics_filepath)
        suffix = Path(file).stem.split("_")[-1]
        metrics_dataframes[suffix] = metrics_df
        _plot_sample_selection_sweep_lineplot(config, log, output_path, metrics_df, f"_{suffix}")

    if len(metrics_dataframes) > 1:
        _plot_joint_sample_selection_sweep_lineplot(config, log, output_path, metrics_dataframes)


def plot_sample_selection_sweep_heatmap(config: DictConfig, log: Logger, output_path: Path) -> None:  # noqa: PLR0915, PLR0912
    """For each split and metric, creates a figure with P heatmaps (one per retention percentage). Each heatmap shows:
    - rows: models
    - columns: strategies
    - color: metric value (lower is better)
    """
    output_path = output_path / "sample_selection_heatmaps"
    output_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 14,
        }
    )

    # Load metrics CSV
    base_path = Path(config.base_path)
    metrics_filepath = base_path / config.metrics_file
    if not metrics_filepath.exists():
        log.error("Metrics file not found at %s", metrics_filepath)
        return
    metrics_df = pd.read_csv(metrics_filepath)

    cmap = sns.color_palette(config.get("heatmap_colormap", "mako_r"), as_cmap=True)
    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    log.info("Plotting sample selection sweep heatmaps for metrics: %s", metrics)
    retention_pcts = list(map(float, config.sample_retention_percentages))
    models = config.models_to_compare
    strategies = config.sample_selection_strategies_to_compare
    highlight_color = config.get("highlight_color", "dodgerblue")

    for split in config.sample_selection_splits_to_compare:
        subsplit = split.split("/")[-1]
        log.info("Creating heatmap sweep plots for split=%s", split)

        for metric in metrics:
            column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric

            heatmap_data = {}
            all_values = []

            # Create all strategy heatmaps
            for pct in retention_pcts:
                data = np.full((len(models), len(strategies)), np.nan)
                for i, model in enumerate(models):
                    for j, strategy in enumerate(strategies):
                        run_name = f"{config.sample_selection_benchmark}_{model}_{strategy}_{pct}"
                        row = metrics_df[metrics_df["Name"] == run_name]
                        if not row.empty and column in row.columns:
                            val = row.iloc[0][column]
                            data[i, j] = val
                            all_values.append(val)
                heatmap_data[pct] = data

            # Base-only heatmap
            base_data = np.full((len(models), 1), np.nan)
            base_values = {}
            for i, model in enumerate(models):
                base_name = f"{config.sample_selection_benchmark}_{model}"
                row = metrics_df[metrics_df["Name"] == base_name]
                if not row.empty and column in row.columns:
                    val = row[column].min()
                    base_data[i, 0] = val
                    base_values[i] = val
                    all_values.append(val)
                else:
                    base_values[i] = None

            if not all_values:
                log.warning("No data found for metric=%s, split=%s", metric, split)
                continue

            vmin = np.nanmin(all_values)
            vmax = np.nanmax(all_values)

            # Figure layout
            num_pcts = len(retention_pcts)
            fig, axes = plt.subplots(
                1,
                num_pcts + 1,
                figsize=(3.8 * num_pcts + 2.2, 0.65 * len(models) + 2.2),
                squeeze=False,
                gridspec_kw={"width_ratios": [1] * num_pcts + [0.45]},
            )
            axes = axes[0]

            # Plot strategy heatmaps
            for k, (ax, pct) in enumerate(zip(axes[:num_pcts], retention_pcts, strict=False)):
                data = heatmap_data[pct]
                masked_data = np.ma.masked_invalid(data)
                im = ax.imshow(masked_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                im.cmap.set_bad(color="#eeeeee")

                ax.set_title(f"{int(pct * 100)}%", pad=6)
                ax.set_xticks(range(len(strategies)))
                ax.set_xticklabels(strategies, rotation=35, ha="right", rotation_mode="anchor")

                if k == 0:
                    ax.set_yticks(range(len(models)))
                    ax.set_yticklabels(models)
                    ax.tick_params(axis="y", pad=6)
                else:
                    ax.set_yticks([])

                # Highlight best per row
                for i in range(data.shape[0]):
                    row = data[i]
                    if np.all(np.isnan(row)):
                        continue

                    j = np.nanargmin(row)
                    best_val = row[j]
                    base_val = base_values.get(i)
                    if base_val is not None and best_val < base_val:
                        edge_color = highlight_color
                        marker_color = highlight_color
                    else:
                        edge_color = "black"
                        marker_color = "black"

                    if config.add_rectangle_annotation:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=edge_color, linewidth=3))
                    ax.plot(j, i, marker="*", ms=18, mec=marker_color, mew=1, c=marker_color, zorder=10)

                # Subtle grid
                ax.set_xticks(np.arange(-0.5, len(strategies), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
                ax.grid(which="minor", color="black", alpha=1, linewidth=1)
                ax.grid(which="major", color="black", alpha=0.01, linewidth=1)
                ax.tick_params(which="minor", bottom=False, left=False)

            # Baseline heatmap
            ax_base = axes[-1]
            masked_base = np.ma.masked_invalid(base_data)
            im = ax_base.imshow(masked_base, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            im.cmap.set_bad(color="#eeeeee")

            ax_base.set_title("Base", pad=6)
            ax_base.set_xticks([])
            ax_base.set_yticks([])
            ax_base.grid(which="minor", color="black", alpha=1, linewidth=1)
            ax_base.grid(which="major", color="black", alpha=0.01, linewidth=1)
            ax_base.set_xticks(np.arange(-0.5, 1, 1), minor=True)
            ax_base.set_yticks(np.arange(-0.5, len(models), 1), minor=True)

            # Colorbar
            cax = fig.add_axes(rect=(0.92, 0.25, 0.03, 0.61))
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=12)

            # Legend handles to show best strategies
            legend = [
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color=highlight_color,
                    linestyle="None",
                    markersize=10,
                    label="Best strategy improves over base",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="black",
                    linestyle="None",
                    markersize=10,
                    label="Best strategy ≥ base",
                ),
            ]

            # Save figure
            output_file = output_path / f"{metric}_{subsplit}.png"
            fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=9)
            fig.suptitle(f"{metric.replace('_', ' ')} — {split}", y=1.02)
            plt.tight_layout(rect=(0, 0, 0.9, 1))
            fig.savefig(output_file, dpi=200, bbox_inches="tight")
            plt.close(fig)

            log.info("Saved heatmaps to %s", output_file)


def _plot_distribution_shift_comparison(
    summary_df: pd.DataFrame, output_path: Path, colormap: str, id_metric: str, ood_metric: str
) -> None:
    """Plots a comparison of In-Distribution (ID) vs Out-of-Distribution (OOD) performance for different models,
    highlighting the performance gaps.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        output_path (Path): Directory to save the generated plot.
        colormap (str): Name of the matplotlib colormap to use for consistent coloring.
        id_metric (str): Name of the ID metric.
        ood_metric (str): Name of the OOD metric.
    """
    assert id_metric in summary_df.columns, f"ID metric '{id_metric}' not found in summary_df columns"
    assert ood_metric in summary_df.columns, f"OOD metric '{ood_metric}' not found in summary_df columns"

    palette = sns.color_palette(colormap, len(summary_df))
    models = summary_df["Model"].to_numpy()

    def _plot_bars(ax: Axes, metric: str, title: str) -> None:
        values = summary_df[metric].to_numpy()
        bars = ax.bar(models, values, color=palette, alpha=0.8, edgecolor="black", linewidth=1.5)

        ax.set_ylabel(metric, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9, rotation=30)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                x = bar.get_x() + bar.get_width() / 2.0
                ax.text(x, height, f"{height:.3f}", ha="center", va="bottom", fontsize=10)
        ax.yaxis.grid(visible=True, alpha=0.3)

        # Tight y-limits for better use of space
        ymin, ymax = np.nanmin(values), np.nanmax(values)  # pyright: ignore[reportCallIssue, reportArgumentType]
        padding = (ymax - ymin) * 0.15 if ymax > ymin else 0.1
        ax.set_ylim(ymin - padding * 0.4, ymax + padding)

    #  Create a second figure focusing on ID vs OOD comparison with performance gaps
    n_models = models.shape[0]
    n_rows = 1
    n_cols = 3
    fig, (ax1, ax2, ax3) = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_models * n_cols, 6))
    fig.suptitle("Distribution Shift Analysis", fontsize=14, fontweight="bold")

    # ID Performance (brierFDE)
    _plot_bars(ax1, id_metric, "In-Distribution (ID) Performance")

    # OOD Performance (brierFDE)
    _plot_bars(ax2, ood_metric, "Out-of-Distribution (OOD) Performance")

    # Performance Gap (OOD - ID, absolute difference)
    id_values = summary_df[id_metric].to_numpy()
    ood_values = summary_df[ood_metric].to_numpy()
    gap_values = ((ood_values - id_values) / (id_values + SMALL_EPSILON)) * 100  # pyright: ignore[reportOperatorIssue]

    # Color bars based on gap direction: red for performance drop, green for improvement
    gap_colors = ["#f07569" if gap > 0 else "#7cbf7c" for gap in gap_values]
    bars = ax3.bar(models, gap_values, color=gap_colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
    ax3.set_ylabel("Performance Gap (OOD - ID)", fontsize=11, fontweight="bold")
    ax3.set_title("Generalization Gap", fontsize=12, fontweight="bold")
    ax3.tick_params(axis="x", labelsize=12, rotation=30)

    for bar, gap in zip(bars, gap_values, strict=False):
        height = bar.get_height()
        if not np.isnan(height):
            va = "bottom" if height > 0 else "top"
            x = bar.get_x() + bar.get_width() / 2.0
            ax3.text(x, height, f"{gap:.3f}", ha="center", va=va, fontsize=8, fontweight="bold")
    ax3.yaxis.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    output_file = output_path / "distribution_shift_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved as '{output_file}'")


def _plot_benchmark_comparison(
    summary_df: pd.DataFrame, metrics: dict[str, str], output_path: Path, colormap: str
) -> None:
    """Plots a benchmark comparison across different models for specified metrics.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        metrics (dict[str, str]): Dictionary mapping metric column names to display names.
        output_path (Path): Directory to save the generated plot.
        colormap (str): Name of the matplotlib colormap to use for consistent coloring.
    """
    # Layout
    num_metrics = len(metrics)
    n_models = summary_df["Model"].shape[0]
    n_cols = min(2, num_metrics)
    n_rows = math.ceil(num_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_models * n_cols, 4.0 * n_rows), constrained_layout=True)
    fig.suptitle("Model Performance Comparison", fontsize=20, fontweight="bold")

    axes = np.atleast_1d(axes).flatten()

    # Consistent color palette
    palette = sns.color_palette(colormap, len(summary_df))
    model_order = summary_df["Model"].to_numpy()

    # Plot each metric
    for idx, metric_name in enumerate(metrics.values()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        values = summary_df[metric_name].to_numpy()
        bars = ax.bar(model_order, values, color=palette, edgecolor="black", linewidth=1.0, alpha=0.8)

        # Titles & labels
        ax.set_title(metric_name, pad=12)
        ax.set_ylabel("Metric Value", fontsize=12)

        # Ticks
        ax.tick_params(axis="x", labelsize=10)
        ax.set_axisbelow(True)

        # Tight y-limits for better use of space
        ymin, ymax = np.nanmin(values), np.nanmax(values)
        padding = (ymax - ymin) * 0.15 if ymax > ymin else 0.1
        ax.set_ylim(ymin - padding * 0.4, ymax + padding)

        # Value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="medium",
                )

        # Highlight best model
        best_idx = np.nanargmin(values) if "↓" in metric_name else np.nanargmax(values)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(4)

    # Hide unused subplots
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    # Save
    output_file = output_path / "benchmark_comparison.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"\n✓ Plot saved as '{output_file}'")


def _plot_performance_gaps(
    summary_df: pd.DataFrame, output_path: Path, metric_pairs: list[tuple[str, str, str]]
) -> None:
    """Plots comprehensive performance gaps (absolute and percentage) between OOD and ID metrics for multiple metrics.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        output_path (Path): Directory to save the generated plot.
        metric_pairs (list[tuple[str, str, str]]): List of tuples containing (ID metric column name, OOD metric column
            name, metric display name).
    """
    gap_data = {}
    for id_col, ood_col, metric_name in metric_pairs:
        if id_col in summary_df.columns and ood_col in summary_df.columns:
            id_vals = summary_df[id_col].to_numpy()
            ood_vals = summary_df[ood_col].to_numpy()
            ood_id_diff = ood_vals - id_vals
            gap_data[metric_name] = {
                "absolute": ood_id_diff,
                "percent": (ood_id_diff / (id_vals + SMALL_EPSILON)) * 100,
            }

    if gap_data:
        num_metrics = len(gap_data)
        num_models = summary_df["Model"].shape[0]
        horizontal_size = num_models * num_metrics * 1.5
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(horizontal_size, 6))
        fig.suptitle("Performance Gaps (OOD - ID)", fontsize=14, fontweight="bold")
        x = np.arange(len(summary_df))
        width = 0.25
        for i, (metric_name, gaps) in enumerate(gap_data.items()):
            offset = width * i
            # Absolute gaps
            ax1.bar(x + offset, gaps["absolute"], width, label=metric_name, alpha=0.8, edgecolor="black", linewidth=1)
            # Percentage gaps
            ax2.bar(x + offset, gaps["percent"], width, label=metric_name, alpha=0.8, edgecolor="black", linewidth=1)

        ax1.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
        ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Absolute Gap (OOD - ID)", fontsize=12, fontweight="bold")
        ax1.set_title("Absolute Performance Gaps\n(Positive = OOD performs worse)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x + (num_metrics - 1) * width)
        ax1.set_xticklabels(summary_df["Model"].values, ha="right", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.yaxis.grid(visible=True, alpha=0.3)
        ax1.set_axisbelow(True)

        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1.5)
        ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Percentage Gap (%)", fontsize=12, fontweight="bold")
        ax2.set_title(
            "Percentage Performance Gaps\n(Positive = OOD worse, % relative to ID)", fontsize=12, fontweight="bold"
        )
        ax2.set_xticks(x + (num_metrics - 1) * width)
        ax2.set_xticklabels(summary_df["Model"].values, ha="right", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.yaxis.grid(visible=True, alpha=0.3)
        ax2.set_axisbelow(True)

        plt.tight_layout()
        output_file = output_path / "performance_gaps.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved as '{output_file}'")

        # Print gap statistics
        print("\n" + "=" * 80)
        print("Performance Gap Analysis (OOD - ID):")
        print("=" * 80)
        for metric_name, gaps in gap_data.items():
            abs_gaps = gaps["absolute"]
            pct_gaps = gaps["percent"]
            print(f"\n{metric_name}:")
            for i, model in enumerate(summary_df["Model"].values):
                print(f"  {model:30s}: {abs_gaps[i]:+.4f} (Absolute) | {pct_gaps[i]:+.2f}% (Relative)")
            print(f"  Average Gap: {np.mean(abs_gaps):+.4f} | {np.mean(pct_gaps):+.2f}%")
            print(f"  Max Gap:     {np.max(abs_gaps):+.4f} | {np.max(pct_gaps):+.2f}%")


def _plot_grouped_bar_chart(
    summary_df: pd.DataFrame, metrics: dict[str, str], output_path: Path, key_metrics_display: list[str]
) -> None:
    """Plots a grouped bar chart comparing multiple key metrics across different models.

    Args:
        summary_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        metrics (dict[str, str]): Dictionary mapping metric column names to display names.
        output_path (Path): Directory to save the generated plot.
        key_metrics_display (list[str]): List of key metric column names to include in the grouped bar chart.
    """
    fig3, ax = plt.subplots(figsize=(14, 7))

    # Select key metrics for grouped comparison
    available_metrics = [m for m in key_metrics_display if m in summary_df.columns]

    y_min, y_max = float("inf"), float("-inf")
    if available_metrics:
        x = np.arange(len(summary_df))
        width = 0.2
        for i, metric in enumerate(available_metrics):
            offset = width * i
            values = summary_df[metric].to_numpy()
            y_min = min(y_min, np.nanmin(values))
            y_max = max(y_max, np.nanmax(values))
            ax.bar(x + offset, values, width, label=metric, alpha=0.8, edgecolor="black", linewidth=1)

        # Tight y-limits for better use of space
        padding = (y_max - y_min) * 0.15 if y_max > y_min else 0.1
        ax.set_ylim(y_min - padding * 0.4, y_max + padding)

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
        ax.set_title("Multi-Metric Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(summary_df["Model"].values, rotation=35, ha="right")
        ax.legend(loc="upper left", fontsize=10)
        ax.yaxis.grid(visible=True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        output_file = output_path / "grouped_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved as '{output_file}'")

    # Print best performing model for each metric
    print("\n" + "=" * 80)
    print("Best Performing Models (Lower is Better):")
    print("=" * 80)
    for metric_name in metrics.values():
        if metric_name in summary_df.columns:
            best_idx = summary_df[metric_name].idxmin()
            if pd.notna(best_idx):
                best_model = summary_df.loc[best_idx, "Model"]
                best_value = summary_df.loc[best_idx, metric_name]
                print(f"{metric_name:30s}: {best_model:30s} ({best_value:.4f})")


def plot_egosafeshift(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Plots ego-safeshift metrics across different models and sample selection strategies.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
    """
    output_path = output_path / "ego_safeshift"
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metrics CSV
    metrics_filepath = Path(config.ego_safeshift_file)
    if not metrics_filepath.exists():
        log.error("Metrics file not found at %s", metrics_filepath)
        return
    metrics_df = pd.read_csv(metrics_filepath)

    # Filter for ego-safeshift-causal-benchmark experiments
    egosafeshift_df = metrics_df[metrics_df["Name"].str.contains(config.ego_safeshift_benchmark, na=False)].copy()
    print("Experiments on ego-safeshift-causal-benchmark:")
    print(egosafeshift_df[["Name", "State"]].to_string(index=False))
    print(f"\nTotal experiments found: {len(egosafeshift_df)}")

    # Extract model names
    egosafeshift_df["model_name"] = egosafeshift_df["Name"].str.replace(f"{config.ego_safeshift_benchmark}_", "")
    egosafeshift_df["model_name"] = egosafeshift_df["model_name"].map(lambda x: MODEL_NAME_MAP.get(str(x), str(x)))  # pyright: ignore[reportUnknownLambdaType]
    if config.show_run_id:
        egosafeshift_df["Model"] = (
            egosafeshift_df["model_name"].astype(str) + "[" + egosafeshift_df["ID"].astype(str) + "]"
        )
    else:
        egosafeshift_df["Model"] = egosafeshift_df["model_name"].astype(str)
    # Key metrics to compare
    metrics = {
        f"{split}/{metric}": f"{metric} ({split.split('-')[-1]},↓)"
        for metric, split in product(config.trajectory_forecasting_metrics, config.ego_safeshift_splits_to_compare)
    }
    log.info("Plotting sample selection sweep lineplots for metrics: %s", metrics)

    # Create a summary dataframe
    summary_data = []
    for _, row in egosafeshift_df.iterrows():
        model_metrics = {"Model": row["Model"]}
        for metric_col, metric_name in metrics.items():
            if metric_col in egosafeshift_df.columns:
                model_metrics[metric_name] = row[metric_col]
        summary_data.append(model_metrics)

    summary_df = pd.DataFrame(summary_data)
    print("Metrics Summary:")
    print(summary_df.to_string(index=False, float_format="{:.3f}".format))

    colormap = config.get("ego_safeshift_colormap", "tab10")

    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        },
    )

    # Plot benchmark comparison
    _plot_benchmark_comparison(summary_df, metrics, output_path, colormap)

    # Plot ID vs OOD comparison
    _plot_distribution_shift_comparison(
        summary_df, output_path, colormap, id_metric="brierFDE (id,↓)", ood_metric="brierFDE (ood,↓)"
    )

    # Plot grouped bar chart for key metrics comparison (commented out for now)
    metric_pairs = [
        ("brierFDE (id,↓)", "brierFDE (ood,↓)", "brierFDE"),
        ("minFDE6 (id,↓)", "minFDE6 (ood,↓)", "minFDE6"),
        ("minADE6 (id,↓)", "minADE6 (ood,↓)", "minADE6"),
        ("missRate (id,↓)", "missRate (ood,↓)", "missRate"),
    ]
    _plot_performance_gaps(summary_df, output_path, metric_pairs)

    # Create a grouped bar chart for comprehensive comparison
    _plot_grouped_bar_chart(
        summary_df, metrics, output_path, key_metrics_display=["brierFDE (id,↓)", "brierFDE (ood,↓)"]
    )

    print("\n✓ Analysis complete!")


def plot_causal_benchmark(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Plots causal benchmark metrics across different models and repeated runs,
    comparing two causal test splits and visualizing performance gaps.

    Args:
        config (DictConfig): Causal benchmark configuration.
        log (Logger): Logger for analysis output.
        output_path (Path): Directory to save plots.
    """
    output_path = output_path / "causal_benchmark"
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metrics CSV
    metrics_filepath = Path(config.causal_benchmark_file)
    if not metrics_filepath.exists():
        log.error("Causal benchmark CSV not found at %s", metrics_filepath)
        return
    metrics_df = pd.read_csv(metrics_filepath)
    # Ensure model name and run identifier exists
    if "Name" not in metrics_df.columns:
        log.error("CSV must contain a 'Name' column")
        return
    if "ID" not in metrics_df.columns:
        metrics_df["ID"] = np.arange(len(metrics_df))

    metrics_df["model_name"] = metrics_df["Name"].str.replace(f"{config.causal_benchmark}_", "")
    metrics_df["model_name"] = metrics_df["model_name"].map(lambda x: MODEL_NAME_MAP.get(str(x), str(x)))  # pyright: ignore[reportUnknownLambdaType]
    if config.show_run_id:
        metrics_df["Model"] = metrics_df["model_name"].astype(str) + "[" + metrics_df["ID"].astype(str) + "]"
    else:
        metrics_df["Model"] = metrics_df["model_name"].astype(str)

    print("\nCausal benchmark runs:")
    print(metrics_df[["model_name", "ID"]].to_string(index=False))
    print(f"\nTotal runs found: {len(metrics_df)}")

    # Metric column construction
    id_split, ood_split = config.causal_benchmark_splits_to_compare
    id_split_name = id_split.split("/")[-1]
    ood_split_name = ood_split.split("/")[-1]
    metrics = {
        f"{split}/{metric}": f"{metric} ({split.split('/')[-1]},↓)"
        for metric, split in product(
            config.trajectory_forecasting_metrics,
            config.causal_benchmark_splits_to_compare,
        )
    }

    log.info("Comparing causal splits: %s vs %s", id_split, ood_split)
    log.info("Metrics: %s", metrics)

    # Create summary dataframe (run-level, no aggregation)
    summary_data = []
    for _, row in metrics_df.iterrows():
        entry = {"Model": row["Model"]}
        for metric_col, display_name in metrics.items():
            if metric_col in metrics_df.columns:
                entry[display_name] = row[metric_col]
        summary_data.append(entry)

    summary_df = pd.DataFrame(summary_data)

    print("\nCausal Metrics Summary (run-level):")
    print(summary_df.to_string(index=False, float_format="{:.3f}".format))

    colormap = config.get("causal_benchmark_colormap", "tab20")

    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        },
    )

    # Benchmark comparison (run-level bars) - All metrics
    _plot_benchmark_comparison(summary_df, metrics, output_path, colormap)

    # Distribution Shift Comparison (brierFDE)
    _plot_distribution_shift_comparison(
        summary_df,
        output_path,
        colormap,
        id_metric=f"brierFDE ({id_split_name},↓)",
        ood_metric=f"brierFDE ({ood_split_name},↓)",
    )

    # Comprehensive Performance Gaps (all metrics)
    metric_pairs = [
        (f"{metric} ({id_split_name},↓)", f"{metric} ({ood_split_name},↓)", metric)
        for metric in config.trajectory_forecasting_metrics
    ]
    _plot_performance_gaps(summary_df, output_path, metric_pairs)

    # Group bar chart for key metrics comparison
    key_metrics_display = [
        f"{config.trajectory_forecasting_metrics[0]} ({id_split_name},↓)",
        f"{config.trajectory_forecasting_metrics[0]} ({ood_split_name},↓)",
    ]
    _plot_grouped_bar_chart(summary_df, metrics, output_path, key_metrics_display=key_metrics_display)

    print("\n✓ Causal benchmark analysis complete!")


def sample_selection_analysis(config: DictConfig, log: Logger) -> None:
    """Loads a CSV containing all model metrics downlaoded from MLflow from produces per-metric barplots with sample
    selection comparisons.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging anslysis information.
    """
    base_path = Path(config.base_path)
    output_path = base_path / "sample_selection_analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    # The 'all_runs.csv' file contains the metrics for all experiments and is exported from MLflow
    metrics_df = pd.read_csv(base_path / "all_runs.csv")

    # Models to analyze
    models = ["scenetf", "wayformer", "st-student", "st-teacher", "st-teacher-u"]
    sample_selection_strategies = ["none", "uniform-random", "token-random", "token-jaccsim", "token-jaccgum"]
    subset = ["waymo-mini-causal-", "waymo-remove-noncausal-"]

    for split, metric, model in product(config.splits_to_compare, config.metrics_to_compare, models):
        log.info("Comparing split: %s, metric: %s, model: %s", split, metric, model)

        # select columns that contain the split string
        metric_cols = [column for column in metrics_df.columns if metric in column]
        subset_cols = [column for column in metric_cols if any(sub in column for sub in subset)]
        valid_cols = ["Name"] + [column for column in subset_cols if split in column]
        if not valid_cols:
            log.info("No columns found for split %s; skipping.", split)
            continue

        # filter rows by model and sample selection strategy, keep only the split-related columns + model_name
        filtered_df = metrics_df[valid_cols].copy()
        if filtered_df.empty:
            log.info("Missing data; skipping")
            continue

        strategy_models = [model] + [
            model + f"_{ss_strategy}" for ss_strategy in sample_selection_strategies if ss_strategy != "none"
        ]
        filtered_df = filtered_df[filtered_df["Name"].isin(strategy_models)]
        # preserve the desired model order by making "Name" a categorical with the strategy_models ordering
        filtered_df["Name"] = pd.Categorical(filtered_df["Name"], categories=strategy_models, ordered=True)
        filtered_df = filtered_df.sort_values("Name").reset_index(drop=True)

        # Melt for seaborn plotting
        cols_to_melt = [col for col in filtered_df.columns if col != "Name"]
        df_melted = filtered_df.melt(id_vars=["Name"], value_vars=cols_to_melt, var_name="Metric", value_name="Value")
        if metric == "missRate":
            df_melted = df_melted[~df_melted.Metric.str.contains(metric + "6")]

        # skip if all values are NaN
        if df_melted["Value"].isna().all():  # pyright: ignore[reportGeneralTypeIssues]
            log.info("All values are NaN for split=%s, metric=%s, model=%s; skipping.", split, metric, model)
            continue

        # Create barplot comparing the model groups
        plt.subplots(figsize=(10, 6))
        sns.barplot(x="Metric", y="Value", hue="Name", data=df_melted, palette=config.palette)

        min_value, max_value = df_melted.min().Value, df_melted.max().Value
        plt.ylim(min_value - 0.1 * min_value, max(max_value + 0.1 * max_value, 1.0))

        plt.title(f"Sample Selection Analysis for {model}")
        plt.xlabel("Metrics")
        plt.ylabel("Metrics Values")
        plt.legend(title="Models", title_fontsize="12", fontsize="10", loc="upper left")

        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Show the plot
        output_filepath = output_path / f"{split}_{model}_{metric}.png"
        plt.savefig(output_filepath, dpi=200)
        plt.close()


def model_to_model_analysis(config: DictConfig, log: Logger) -> None:  # noqa: PLR0915
    """Loads a CSV containing all model metrics downlaoded from MLflow from produces per-metric barplots with model to
    model comparisons and generalization assessments.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging anslysis information.
    """
    base_path = Path(config.base_path)
    output_path = base_path / "model_to_model_analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    # The 'all_runs.csv' file contains the metrics for all experiments and is exported from MLflow
    metrics_df = pd.read_csv(base_path / "all_runs.csv")

    # Models to analyze
    models = ["scenetf", "wayformer", "st-student", "st-teacher", "st-teacher-u"]
    sample_selection_strategies = ["none", "uniform-random", "token-random", "token-jaccsim", "token-jaccgum"]
    subset = ["waymo-mini-causal-", "waymo-remove-noncausal-"]

    for metric in config.metrics_to_compare:
        generalization = pd.DataFrame({"models": models})

        for split, ss_strategy in product(config.splits_to_compare, sample_selection_strategies):
            log.info("Comparing split: %s, metric: %s, sample selection strategy: %s", split, metric, ss_strategy)

            # select columns that contain the split string
            metric_cols = [column for column in metrics_df.columns if metric in column]
            subset_cols = [column for column in metric_cols if any(sub in column for sub in subset)]
            valid_cols = ["Name"] + [column for column in subset_cols if split in column]
            if not valid_cols:
                log.info("No columns found for split %s; skipping.", split)
                continue

            # filter rows by model and sample selection strategy, keep only the split-related columns + model_name
            filtered_df = metrics_df[valid_cols].copy()
            if filtered_df.empty:
                log.info("Missing data; skipping")
                continue

            strategy_models = [model + f"_{ss_strategy}" for model in models] if ss_strategy != "none" else models
            filtered_df = filtered_df[filtered_df["Name"].isin(strategy_models)]
            # preserve the desired model order by making "Name" a categorical with the strategy_models ordering
            filtered_df["Name"] = pd.Categorical(filtered_df["Name"], categories=strategy_models, ordered=True)
            filtered_df = filtered_df.sort_values("Name").reset_index(drop=True)
            if filtered_df.empty:
                log.info("No rows match models for strategy %s; skipping.", ss_strategy or "none")
                continue

            # Compute generalization values
            unperturbed_values = filtered_df[filtered_df.columns[1]] + SMALL_EPSILON
            perturbed_values = filtered_df[filtered_df.columns[2]]
            generalization_values = 100 * (perturbed_values - unperturbed_values) / unperturbed_values.abs()
            generalization[ss_strategy] = generalization_values

            # Melt for seaborn plotting
            cols_to_melt = [col for col in filtered_df.columns if col != "Name"]
            df_melted = filtered_df.melt(
                id_vars=["Name"], value_vars=cols_to_melt, var_name="Metric", value_name="Value"
            )
            if metric == "missRate":
                df_melted = df_melted[~df_melted.Metric.str.contains(metric + "6")]

            # Create barplot comparing the model groups
            plt.subplots(figsize=(10, 6))
            sns.barplot(x="Metric", y="Value", hue="Name", data=df_melted, palette=config.palette)

            min_value, max_value = df_melted.min().Value, df_melted.max().Value
            plt.ylim(min_value - 0.1 * min_value, max(max_value + 0.1 * max_value, 1.0))
            # plt.yscale("log")
            # plt.ylim(bottom=0.0, top=10)

            plt.title(f"Sample Selection Strategy: {ss_strategy}")
            plt.xlabel("Metrics")
            plt.ylabel("Metrics Values")
            plt.legend(title="Models", title_fontsize="12", fontsize="10", loc="upper left")

            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()

            # Show the plot
            output_filepath = output_path / f"{split}_{metric}_ss-{ss_strategy}.png"
            plt.savefig(output_filepath, dpi=200)
            plt.close()

        # Generalization plot
        plt.subplots(figsize=(10, 6))

        generalization_melted = generalization.melt(
            id_vars=["models"], var_name="Strategy", value_name="Generalization"
        )
        sns.barplot(x="Strategy", y="Generalization", hue="models", data=generalization_melted, palette=config.palette)

        min_value, max_value = generalization_melted.min().Generalization, generalization_melted.max().Generalization
        plt.ylim(min_value - 0.1 * min_value, max(max_value + 0.1 * max_value, 1.0))

        plt.title(f"Generalization: {metric}")
        plt.xlabel("Strategy")
        plt.ylabel("Generalization")
        plt.legend(title="Models", title_fontsize="12", fontsize="10", loc="upper left")

        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Show the plot
        output_filepath = output_path / f"generalization_{metric}.png"
        plt.savefig(output_filepath, dpi=200)
        plt.close()


def group_analysis(config: DictConfig, log: Logger) -> None:
    """Loads a CSV containing model metrics from a specified group and produces per-metric barplots with model-to-model
    comparisons.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
    """
    base_path = Path(config.base_path)
    output_path = base_path / config.group_name
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(config.group_file)

    # Each metric in the dataframe follows the pattern 'split/metric_type' or 'split/subset/metric_type'. Below, we
    # filter the metrics by split and metric
    for split, metric in product(config.splits_to_compare, config.metrics_to_compare):
        log.info("Comparing split: %s and metric: %s", split, metric)

        # Get a sub dataframe that contains only the desired metric from the desired split
        filtered_df = metrics_df[metrics_df.Metric.str.contains(split) & metrics_df.Metric.str.contains(metric)]
        if len(filtered_df) == 0:
            log.info("Missing metric %s; skipping analysis.", metric)
            continue

        if metric == "missRate":
            filtered_df = filtered_df[~filtered_df.Metric.str.contains(metric + "6")]

        # Ad-hoc renaming 'split/dataset-subset-subsplit/metric' -> 'subset'
        filtered_df.Metric = filtered_df.Metric.str.extract(r"(?<=/)([^/]+)(?=/)", expand=False)
        filtered_df.Metric = filtered_df.Metric.str.extract(r"(?<=-)([^/]+)(?=-)", expand=False)
        df_melted = filtered_df.melt(id_vars=["Metric"], var_name="Model", value_name="Value")

        # Create barplot comparing the model groups
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Metric", y="Value", hue="Model", data=df_melted, palette=config.palette)

        # Scale the y-axis to highlight the metric differences. This just shows the y-range from 10% less than the min
        # metric value to 10% more than the max metric value.
        min_value, max_value = df_melted.min().Value, df_melted.max().Value
        plt.ylim(min_value - 0.1 * min_value, max(max_value + 0.1 * max_value, 1.0))

        plt.title(f"Split: {split}, Metric: {metric}")
        plt.xlabel("Metrics by Subset")
        plt.ylabel("Metrics Values")
        plt.legend(title="Models", title_fontsize="12", fontsize="10", loc="upper left")

        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Show the plot
        output_filepath = f"{output_path}/{split}_{metric}.png"
        plt.savefig(output_filepath, dpi=200)
