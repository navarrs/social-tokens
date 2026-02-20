import json
import math
from itertools import product
from logging import Logger
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from omegaconf import DictConfig

from scenetokens.utils.constants import SMALL_EPSILON


MODEL_NAME_MAP = {
    "wayformer": "Wayformer",
    "scenetransformer": "SceneTransformer",
    "scenetokens": "ST",
    "causal-scenetokens": "Causal-ST",
    "safe-scenetokens": "Safe-ST",
}

BENCHMARK_NAME_MAP = {
    "causal-benchmark-labeled": "CausalAgents",
    "ego-safeshift-causal-benchmark": "EgoSafeShift",
}

STRATEGY_NAME_MAP = {
    "random_drop": "Random",
    "token_random_drop": "Token(R)",
    "simple_token_jaccard_drop": "Token(SJ)",
    "simple_token_hamming_drop": "Token(SH)",
    "gumbel_token_jaccard_drop": "Token(GJ)",
    "gumbel_token_hamming_drop": "Token(GH)",
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
    plt.style.use("seaborn-v0_8-whitegrid")

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
        suffix = Path(file).stem.split("_")[0]  # contains the name of the selector
        metrics_dataframes[suffix] = metrics_df
        _plot_sample_selection_sweep_lineplot(config, log, output_path, metrics_df, f"_{suffix}")

    if len(metrics_dataframes) > 1:
        _plot_joint_sample_selection_sweep_lineplot(config, log, output_path, metrics_dataframes)


def _plot_sample_selection_sweep_heatmap(
    config: DictConfig, log: Logger, output_path: Path, metrics_df: pd.DataFrame, suffix: str = ""
) -> None:
    """Plot heatmaps comparing sample selection strategies across retention percentages for each (model, split, metric).

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_df (pd.DataFrame): DataFrame containing the metrics data.
        suffix (str): Suffix to append to output filenames to distinguish different metrics files.
    """
    output_path = output_path / "sample_selection_heatmaps"
    output_path.mkdir(parents=True, exist_ok=True)

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
                mapped_strategies = [STRATEGY_NAME_MAP.get(s, s) for s in strategies]
                ax.set_xticklabels(mapped_strategies, rotation=35, ha="right", rotation_mode="anchor")

                if k == 0:
                    ax.set_yticks(range(len(models)))
                    mapped_models = [MODEL_NAME_MAP.get(m, m) for m in models]
                    ax.set_yticklabels(mapped_models)
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
            output_file = output_path / f"{metric}_{subsplit}{suffix}.png"
            fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=9)
            fig.suptitle(f"{metric.replace('_', ' ')} — {split}", y=1.02)
            plt.tight_layout(rect=(0, 0, 0.9, 1))
            fig.savefig(output_file, dpi=200, bbox_inches="tight")
            plt.close(fig)

            log.info("Saved heatmaps to %s", output_file)


def _get_baseline_values(
    config: DictConfig, metrics_dfs: dict[str, pd.DataFrame]
) -> dict[str, dict[str, float | None]]:
    """Extract baseline metric values for each (model, split, metric) from the metrics dataframes.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        metrics_dfs (dict[str, pd.DataFrame]): Dictionary of DataFrames containing the metrics data for each strategy.

    Returns:
        dict[str, dict[str, float | None]]: Nested dictionary mapping model -> (split/metric) -> baseline value.
    """
    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    models = config.models_to_compare
    splits = config.sample_selection_splits_to_compare

    base_values = {model: {} for model in models}
    for model, split, metric in product(models, splits, metrics):
        column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric

        for _, metrics_df in metrics_dfs.items():
            base_name = f"{config.sample_selection_benchmark}_{model}"
            row = metrics_df[metrics_df["Name"] == base_name]

            # NOTE: this assumes lower is better for all metrics, which is true for our current metrics but may need to
            # be adjusted if we add new ones where higher is better
            current_metric = base_values[model].get(column, float("inf"))
            available_metric = float("inf")
            if not row.empty and column in row.columns:
                available_metric = row.iloc[0][column]
            base_values[model][column] = min(current_metric, available_metric)

    return base_values


def _plot_sample_selection_sweep_heatmap_baseline_gap(
    config: DictConfig,
    log: Logger,
    output_path: Path,
    metrics_dfs: dict[str, pd.DataFrame],
) -> None:
    """Plot heatmaps comparing sample selection strategies across retention percentages for each (model, split, metric).
    Shows the gap between each strategy and the baseline model.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_dfs (dict[str, pd.DataFrame]): Dictionary of DataFrames containing the metrics data for each strategy.
        suffix (str): Suffix to append to output filenames to distinguish different metrics files.
    """
    output_path = output_path / "sample_selection_heatmaps_baseline_gap"
    output_path.mkdir(parents=True, exist_ok=True)

    cmap = sns.color_palette(config.get("heatmap_colormap", "RdYlGn_r"), as_cmap=True)
    highlight_color = config.get("highlight_color", "dodgerblue")

    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    retention_pcts = list(map(float, config.sample_retention_percentages))
    models = config.models_to_compare
    strategies = config.sample_selection_strategies_to_compare
    splits = config.sample_selection_splits_to_compare
    log.info("Plotting sample selection sweep heatmaps (baseline gap) for metrics: %s", metrics)

    base_values = _get_baseline_values(config, metrics_dfs)

    num_models = len(models) * len(metrics_dfs)
    num_strategies = len(strategies)
    num_retention_pcts = len(retention_pcts)
    model_list = []
    for model, selector in product(models, metrics_dfs.keys()):
        model_name = f"{MODEL_NAME_MAP.get(model, model)} ({selector})"
        model_list.append(model_name)

    for split, metric in product(splits, metrics):
        subsplit = split.split("/")[-1]
        column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric
        log.info("Creating heatmap sweep plots for split=%s", split)

        heatmap_data = {}
        all_gaps = []
        # Create gap heatmaps (strategy vs baseline)
        for pct in retention_pcts:
            data = np.full((num_models, num_strategies), np.nan)
            for i, (model, metrics_df) in enumerate(product(models, metrics_dfs.values())):
                base_value = base_values[model].get(column)
                if base_value is None:
                    continue

                for j, strategy in enumerate(strategies):
                    run_name = f"{config.sample_selection_benchmark}_{model}_{strategy}_{pct}"
                    row = metrics_df[metrics_df["Name"] == run_name]
                    if not row.empty and column in row.columns:
                        strategy_value = row.iloc[0][column]
                        # Compute gap: (strategy_val - baseline_val) / baseline_val * 100
                        gap = ((strategy_value - base_value) / (abs(base_value) + SMALL_EPSILON)) * 100
                        data[i, j] = gap
                        all_gaps.append(gap)
            heatmap_data[pct] = data

        if not all_gaps:
            log.warning("No data found for metric=%s, split=%s", metric, split)
            continue

        vmin = np.nanmin(all_gaps)
        vmax = np.nanmax(all_gaps)

        # Round vmin and vmax to nearest 10% for better colorbar ticks
        # vmin = math.floor(vmin / 10) * 10
        # vmax = math.ceil(vmax / 10) * 10

        # Center colormap around 0
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs

        # Figure layout
        x_size = 4.0 * num_retention_pcts + 2.2
        y_size = 0.7 * num_models + 2.2
        fig, axes = plt.subplots(1, num_retention_pcts, figsize=(x_size, y_size), squeeze=False)
        axes = axes[0]

        # Plot strategy heatmaps
        for k, (ax, pct) in enumerate(zip(axes[:num_retention_pcts], retention_pcts, strict=False)):
            data = heatmap_data[pct]
            masked_data = np.ma.masked_invalid(data)
            im = ax.imshow(masked_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            im.cmap.set_bad(color="#eeeeee")

            ax.set_title(f"{int(pct * 100)}%", pad=6, fontsize=20)
            ax.set_xticks(range(num_strategies))
            mapped_strategies = [STRATEGY_NAME_MAP.get(s, s) for s in strategies]
            ax.set_xticklabels(mapped_strategies)  # , ha="right", rotation_mode="anchor")

            if k == 0:
                ax.set_yticks(range(num_models))
                ax.set_yticklabels(model_list)
                ax.tick_params(axis="y", pad=6)
            else:
                ax.set_yticks([])

            # Highlight best per row (minimum gap, closest to baseline)
            for i in range(data.shape[0]):
                row = data[i]
                if np.all(np.isnan(row)):
                    continue

                # j = np.nanargmin(np.abs(row))
                j = np.nanargmin(row)  # if we want to highlight the best strategy even if it's worse than baseline
                gap_val = row[j]

                # Highlight if gap is negative (improvement over baseline)
                if gap_val < 0:
                    edge_color = highlight_color
                    marker_color = highlight_color
                else:
                    edge_color = "black"
                    marker_color = "black"

                if config.add_rectangle_annotation:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=edge_color, linewidth=3))
                ax.plot(j, i, marker="*", ms=25, mec=marker_color, mew=1, c=marker_color, zorder=10)

            # Subtle grid
            ax.set_xticks(np.arange(-0.5, len(strategies), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
            # ax.grid(which="minor", color="black", alpha=0.1, linewidth=1)
            # ax.grid(which="major", color="black", alpha=0.1, linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

        # Colorbar
        cax = fig.add_axes(rect=(0.90, 0.11, 0.03, 0.7))
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Gap to Baseline (%)", fontsize=10)

        # Legend handles to show best strategies
        legend = [
            Line2D(
                [0],
                [0],
                marker="*",
                color=highlight_color,
                linestyle="None",
                markersize=10,
                label="Best strategy (closest to baseline)",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="black",
                linestyle="None",
                markersize=10,
                label="Best strategy (worse than baseline)",
            ),
        ]

        # Save figure
        output_file = output_path / f"{metric}_{subsplit}.png"
        fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=9)
        fig.suptitle(f"Gap to Baseline — {metric.replace('_', ' ')} {split}", y=1.02)
        plt.tight_layout(rect=(0, 0, 0.9, 1))
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

        log.info("Saved heatmaps to %s", output_file)


def _plot_sample_selection_sweep_distribution_gap(
    config: DictConfig,
    log: Logger,
    output_path: Path,
    metrics_dfs: dict[str, pd.DataFrame],
) -> None:
    """Plot heatmaps comparing sample selection strategies across retention percentages for each (model, metric).
    Shows the gap between two splits (split_b - split_a).

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_dfs (dict[str, pd.DataFrame]): Dictionary of DataFrames containing the metrics data for each strategy.
    """
    splits = config.sample_selection_splits_to_compare
    if len(splits) < 2:
        log.warning("Need at least two splits to compute distribution gap, got: %s", splits)
        return

    id_split, ood_split = splits[0], splits[1]
    id_subsplit = id_split.split("/")[-1]
    ood_subsplit = ood_split.split("/")[-1]

    output_path = output_path / "sample_selection_heatmaps_distribution_gap"
    output_path.mkdir(parents=True, exist_ok=True)

    cmap = sns.color_palette(config.get("heatmap_colormap", "RdYlGn_r"), as_cmap=True)
    highlight_color = config.get("highlight_color", "dodgerblue")

    metrics = config.trajectory_forecasting_metrics
    retention_pcts = list(map(float, config.sample_retention_percentages))
    models = config.models_to_compare
    strategies = config.sample_selection_strategies_to_compare
    log.info("Plotting sample selection sweep heatmaps (distribution gap) for metrics: %s", metrics)

    num_models = len(models) * len(metrics_dfs)
    num_strategies = len(strategies)
    num_retention_pcts = len(retention_pcts)
    model_list = []
    for model, selector in product(models, metrics_dfs.keys()):
        model_name = f"{MODEL_NAME_MAP.get(model, model)} ({selector})"
        model_list.append(model_name)

    for metric in metrics:
        id_column = f"{id_split}/{metric}"
        ood_column = f"{ood_split}/{metric}"
        log.info("Creating distribution gap heatmaps for metric=%s (%s vs %s)", metric, id_split, ood_split)

        # Compute baseline gaps for each model
        baseline_gaps = {}
        baseline_id_values = {}
        for model, metrics_df in product(models, metrics_dfs.values()):
            base_name = f"{config.sample_selection_benchmark}_{model}"
            base_row = metrics_df[metrics_df["Name"] == base_name]
            if not base_row.empty and id_column in base_row.columns and ood_column in base_row.columns:
                id_val = base_row.iloc[0][id_column]
                ood_val = base_row.iloc[0][ood_column]
                baseline_gap = ((ood_val - id_val) / (abs(id_val) + SMALL_EPSILON)) * 100
                baseline_gaps[model] = baseline_gap
                baseline_id_values[model] = id_val
            else:
                baseline_gaps[model] = None
                baseline_id_values[model] = None

        heatmap_data = {}
        all_gaps = []

        # Create gap heatmaps (split_b vs split_a)
        for pct in retention_pcts:
            data = np.full((num_models, num_strategies), np.nan)
            for i, (model, metrics_df) in enumerate(product(models, metrics_dfs.values())):
                for j, strategy in enumerate(strategies):
                    run_name = f"{config.sample_selection_benchmark}_{model}_{strategy}_{pct}"
                    row = metrics_df[metrics_df["Name"] == run_name]
                    if not row.empty and id_column in row.columns and ood_column in row.columns:
                        id_val = row.iloc[0][id_column]
                        ood_val = row.iloc[0][ood_column]
                        gap = ((ood_val - id_val) / (abs(id_val) + SMALL_EPSILON)) * 100
                        data[i, j] = gap
                        all_gaps.append(gap)
            heatmap_data[pct] = data

        if not all_gaps:
            log.warning("No data found for metric=%s (%s vs %s)", metric, id_split, ood_split)
            continue

        vmin = np.nanmin(all_gaps)
        vmax = np.nanmax(all_gaps)

        # Center colormap around 0
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs

        # Figure layout
        x_size = 4.0 * num_retention_pcts + 2.2
        y_size = 0.7 * num_models + 2.2
        fig, axes = plt.subplots(1, num_retention_pcts, figsize=(x_size, y_size), squeeze=False)
        axes = axes[0]

        # Plot strategy heatmaps
        for k, (ax, pct) in enumerate(zip(axes[:num_retention_pcts], retention_pcts, strict=False)):
            data = heatmap_data[pct]
            masked_data = np.ma.masked_invalid(data)
            im = ax.imshow(masked_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            im.cmap.set_bad(color="#eeeeee")

            ax.set_title(f"{int(pct * 100)}%", pad=6, fontsize=20)
            ax.set_xticks(range(num_strategies))
            mapped_strategies = [STRATEGY_NAME_MAP.get(s, s) for s in strategies]
            ax.set_xticklabels(mapped_strategies)

            if k == 0:
                ax.set_yticks(range(num_models))
                ax.set_yticklabels(model_list)
                ax.tick_params(axis="y", pad=6)
            else:
                ax.set_yticks([])

            # Highlight smallest gap per row AND experiments better than baseline
            for i in range(data.shape[0]):
                row = data[i]
                if np.all(np.isnan(row)):
                    continue

                # Get the model index (accounting for multiple selectors)
                model_idx = i % len(models)
                model = models[model_idx]
                baseline_gap = baseline_gaps.get(model)
                baseline_id_val = baseline_id_values.get(model)

                j = int(np.nanargmin(np.abs(row)))
                gap_val = row[j]

                # Get the ID value for this strategy/retention combination
                run_name = f"{config.sample_selection_benchmark}_{model}_{strategies[j]}_{pct}"
                metrics_df = list(metrics_dfs.values())[i // len(models)]  # Get corresponding metrics_df
                strategy_row = metrics_df[metrics_df["Name"] == run_name]
                strategy_id_val = None
                if not strategy_row.empty and id_column in strategy_row.columns:
                    strategy_id_val = strategy_row.iloc[0][id_column]

                # Determine marker color based on conditions
                marker_color = "black"
                edge_color = "black"

                # Magenta star if both gap and ID performance are better than baseline
                if (
                    baseline_gap is not None
                    and baseline_id_val is not None
                    and strategy_id_val is not None
                    and gap_val < baseline_gap
                    and strategy_id_val < baseline_id_val
                ):
                    marker_color = "magenta"
                    edge_color = "magenta"
                # Blue highlight if gap is better than baseline
                elif gap_val < baseline_gap:
                    edge_color = highlight_color
                    marker_color = highlight_color

                if config.add_rectangle_annotation:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=edge_color, linewidth=3))
                ax.plot(j, i, marker="*", ms=25, mec=marker_color, mew=1, c=marker_color, zorder=10)

            # Subtle grid
            ax.set_xticks(np.arange(-0.5, len(strategies), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
            ax.tick_params(which="minor", bottom=False, left=False)

        # Colorbar
        cax = fig.add_axes(rect=(0.90, 0.11, 0.03, 0.7))
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(f"Gap ({ood_subsplit} - {id_subsplit}) %", fontsize=10)

        # Legend handles to show best strategies
        legend = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="magenta",
                linestyle="None",
                markersize=10,
                label="Better gap AND better ID performance vs baseline",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color=highlight_color,
                linestyle="None",
                markersize=10,
                label="Performance gap better than baseline's ",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="black",
                linestyle="None",
                markersize=10,
                label="Smallest performance gap, but not better than baseline gap",
            ),
        ]

        # Save figure
        output_file = output_path / f"{metric}_{id_subsplit}_vs_{ood_subsplit}.png"
        fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=7)
        fig.suptitle(
            f"Split Gap — {metric.replace('_', ' ')} ({ood_subsplit} - {id_subsplit})",
            y=1.02,
        )
        plt.tight_layout(rect=(0, 0, 0.9, 1))
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

        log.info("Saved distribution gap heatmaps to %s", output_file)


def plot_sample_selection_sweep_heatmap(config: DictConfig, log: Logger, output_path: Path) -> None:  # noqa: PLR0915, PLR0912
    """For each split and metric, creates a figure with P heatmaps (one per retention percentage). Each heatmap shows:
    - rows: models
    - columns: strategies
    - color: metric value (lower is better)
    """
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
    metrics_dataframes = {}
    for file in config.sample_selection_files:
        log.info("Processing sample selection sweep lineplots for file: %s", file)
        metrics_filepath = Path(file)
        if not metrics_filepath.exists():
            log.error("Sample selection CSV not found at %s", metrics_filepath)
            return
        metrics_df = pd.read_csv(metrics_filepath)
        suffix = Path(file).stem.split("_")[0]  # contains the name of the selector
        metrics_dataframes[suffix] = metrics_df
        # _plot_sample_selection_sweep_heatmap(config, log, output_path, metrics_df, f"_{suffix}")

    # _plot_sample_selection_sweep_heatmap_baseline_gap(config, log, output_path, metrics_dataframes)
    _plot_sample_selection_sweep_distribution_gap(config, log, output_path, metrics_dataframes)


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


def run_benchmark_analysis(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Plots multiple In-Distribution (ID) vs Out-of-Distribution (OOD) benchmark analyses based on a CSV file
    containing model metrics.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    output_path = output_path / config.benchmark
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metrics CSV
    metrics_filepath = Path(config.benchmark_filepath)
    if not metrics_filepath.exists():
        log.error("Metrics file not found at %s", metrics_filepath)
        return
    metrics_df = pd.read_csv(metrics_filepath)

    # Ensure model name and run identifier exists
    if "Name" not in metrics_df.columns:
        log.error("CSV must contain a 'Name' column")
        return
    if "ID" not in metrics_df.columns:
        metrics_df["ID"] = np.arange(len(metrics_df))

    # Filter for ego-safeshift-causal-benchmark experiments
    benchmark_df = metrics_df[metrics_df["Name"].str.contains(config.benchmark, na=False)].copy()
    print(f"Experiments on {config.benchmark}:")
    print(benchmark_df[["Name", "State"]].to_string(index=False))
    print(f"\nTotal experiments found: {len(benchmark_df)}")

    # Extract model names
    benchmark_df["model_name"] = benchmark_df["Name"].str.replace(f"{config.benchmark}_", "")
    benchmark_df["model_name"] = benchmark_df["model_name"].map(lambda x: MODEL_NAME_MAP.get(str(x), str(x)))  # pyright: ignore[reportUnknownLambdaType]
    if config.show_run_id:
        benchmark_df["Model"] = benchmark_df["model_name"].astype(str) + "[" + benchmark_df["ID"].astype(str) + "]"
    else:
        benchmark_df["Model"] = benchmark_df["model_name"].astype(str)

    # Key metrics to compare
    id_split, ood_split = config.benchmark_splits_to_compare
    id_split_name = id_split.split("/")[-1]
    ood_split_name = ood_split.split("/")[-1]
    metrics = {
        f"{split}/{metric}": f"{metric} ({split.split('/')[-1]},↓)"
        for metric, split in product(
            config.trajectory_forecasting_metrics,
            config.benchmark_splits_to_compare,
        )
    }
    log.info("Comparing splits: %s vs %s", id_split, ood_split)
    log.info("Metrics: %s", metrics)

    # Create a summary dataframe
    summary_data = []
    for _, row in benchmark_df.iterrows():
        model_metrics = {"Model": row["Model"]}
        for metric_col, metric_name in metrics.items():
            if metric_col in benchmark_df.columns:
                model_metrics[metric_name] = row[metric_col]
        summary_data.append(model_metrics)

    summary_df = pd.DataFrame(summary_data)
    print("Metrics Summary:")
    print(summary_df.to_string(index=False, float_format="{:.3f}".format))

    colormap = config.get(f"{config.benchmark_colormap}", "tab10")

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
    # Distribution Shift Comparison (brierFDE)
    _plot_distribution_shift_comparison(
        summary_df,
        output_path,
        colormap,
        id_metric=f"brierFDE ({id_split_name},↓)",
        ood_metric=f"brierFDE ({ood_split_name},↓)",
    )

    # Plot grouped bar chart for key metrics comparison (commented out for now)
    metric_pairs = [
        (f"{metric} ({id_split_name},↓)", f"{metric} ({ood_split_name},↓)", metric)
        for metric in config.trajectory_forecasting_metrics
    ]
    _plot_performance_gaps(summary_df, output_path, metric_pairs)

    # Create a grouped bar chart for comprehensive comparison
    key_metrics_display = [
        f"{config.trajectory_forecasting_metrics[0]} ({id_split_name},↓)",
        f"{config.trajectory_forecasting_metrics[0]} ({ood_split_name},↓)",
    ]
    _plot_grouped_bar_chart(summary_df, metrics, output_path, key_metrics_display=key_metrics_display)

    # Generate LaTeX table
    _convert_to_tex_table(
        benchmark_df,
        BENCHMARK_NAME_MAP.get(config.benchmark) or config.benchmark,
        id_split,
        ood_split,
        config.trajectory_forecasting_metrics,
        output_path,
    )

    print("\n✓ Analysis complete!")


def _convert_to_tex_table(  # noqa: PLR0913, PLR0915
    benchmark_df: pd.DataFrame,
    benchmark_name: str,
    id_split: str,
    ood_split: str,
    metrics: list[str],
    output_path: Path | None,
    min_color_value: float = 20.0,
) -> str:
    """Converts the benchmark comparison DataFrame into a LaTeX table with performance gap annotations and coloring.

    Args:
        benchmark_df (pd.DataFrame): DataFrame containing model names and their corresponding metric values.
        benchmark_name (str): Display name of the benchmark for the table caption.
        id_split (str): Name of the In-Distribution split used in the metrics.
        ood_split (str): Name of the Out-of-Distribution split used in the metrics.
        metrics (list[str]): List of metric column names to include in the table.
        output_path (Path | None): Directory to save the generated LaTeX file. If None, the LaTeX string will be
            returned but not saved to a file.
        min_color_value (float): Minimum color intensity percentage for the gap coloring (0-100). Higher values will
            make the colors more vibrant even for smaller gaps.
    """
    # Precompute best ID/OOD and gap severity per metric
    best_id, best_ood, gap_stats = {}, {}, {}
    for metric in metrics:
        id_col = f"{id_split}/{metric}"
        id_vals = benchmark_df[id_col]
        best_id[metric] = id_vals.min()

        ood_col = f"{ood_split}/{metric}"
        ood_vals = benchmark_df[ood_col]
        best_ood[metric] = ood_vals.min()

        gaps = ((ood_vals - id_vals) / (id_vals + SMALL_EPSILON)) * 100
        gap_stats[metric] = (gaps.min(), gaps.max())  # best, worst

    # Build rows
    table_rows = []
    first_row = True

    for _, row in benchmark_df.iterrows():
        row_parts = []
        # Multirow benchmark label
        if first_row:
            row_parts.append(f"\\multirow{{{len(benchmark_df)}}}{{*}}{{\\texttt{{{benchmark_name}}}}}")
            first_row = False
        else:
            row_parts.append("")
        row_parts.append(str(row["Model"]))

        id_values, ood_values = [], []
        for metric in metrics:
            id_col = f"{id_split}/{metric}"
            ood_col = f"{ood_split}/{metric}"

            id_val = row[id_col]
            ood_val = row[ood_col]

            # In distribution value
            if pd.notna(id_val):
                id_str = f"{id_val:.3f}"
                if np.isclose(id_val, best_id[metric]):
                    id_str = f"\\textbf{{{id_str}}}"
            else:
                id_str = "---"
            id_values.append(id_str)

            # Out of Distribution value with gap annotation and coloring
            if pd.notna(id_val) and pd.notna(ood_val):
                gap = ((ood_val - id_val) / (id_val + SMALL_EPSILON)) * 100

                best_gap, worst_gap = gap_stats[metric]
                denom = max(abs(worst_gap - best_gap), SMALL_EPSILON)
                severity = abs(gap - best_gap) / denom
                severity = np.clip(severity, 0, 1)

                intensity = int(min_color_value + severity * (100 - min_color_value))  # min_color_value% - 100%

                color = "OrangeRed" if gap > 0 else "ForestGreen"
                gap_str = f"\\textcolor{{{color}!{intensity}}}{{{gap:+.2f}\\%}}"

                ood_str = f"{ood_val:.3f}"
                if np.isclose(ood_val, best_ood[metric]):
                    ood_str = f"\\textbf{{{ood_str}}}"

                ood_str = f"{ood_str} ({gap_str})"
            else:
                ood_str = "---"

            ood_values.append(ood_str)

        id_values.append("")  # Add empty column for spacing
        row_parts.extend(id_values)
        ood_values = ["", ood_values]  # Add empty column for spacing
        row_parts.extend(ood_values)

        table_rows.append(" & ".join(row_parts) + " \\\\")

    # Build LaTeX
    n_metrics = len(metrics)
    col_spec = "l l " + "c" * (2 * n_metrics) + "cc"  # Two extra columns for spacing between ID and OOD

    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append("\\setlength{\\tabcolsep}{4pt}")
    latex.append("\\caption{Distribution Shift Results}")
    latex.append("\\label{tab:distribution_shift_results}")

    latex.append("\\resizebox{\\textwidth}{!}{%")
    latex.append("\\begin{tabular}{" + col_spec + "}")
    latex.append("\\toprule")

    latex.append(
        f"\\multirow{{2}}{{*}}{{\\textbf{{Benchmark}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & "
        f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{In Distribution (Validation)}}}} & "
        f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{Out of Distribution (Test)}}}} \\\\"
    )

    latex.append(" & & " + " & ".join([*metrics, "", "", *metrics]) + " \\\\")
    latex.append("\\midrule")

    latex.extend(table_rows)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}%")
    latex.append("}")
    latex.append("\\end{table*}")

    latex_table = "\n".join(latex)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "results.tex").write_text(latex_table)

    return latex_table


def run_comparative_analysis(analysis_paths: dict[str, Path], output_path: Path) -> None:  # noqa: PLR0912, PLR0915
    """Runs a comparative analysis across the tokenization analyses of multiple experiments.

    Args:
        analysis_paths (dict[str, Path]): A dictionary mapping experiment names to their respective output paths.
        output_path (Path): The path where the comparative analysis results should be saved.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    filename_map = {
        "class_distribution_stats_best_mode.json": "Token Distribution",
        "group_uniqueness_stats_group_scenarios.json": "Group Uniqueness",
        "intergroup_uniqueness_stats.json": "Intergroup Uniqueness",
        "token_consistency_stats_hamming.json": "Token Consistency",
        "token_consistency_stats_jaccard.json": "Token Consistency",
    }
    metric_map = {
        "dead_tokens": "Dead Tokens",
        "mode_0_stats.mean": "Avg. Scenarios per Token",
        "mode_0_stats.median": "Median Scenarios per Token",
        "mode_0_stats.norm_mean": "Normalized Mean Scenarios per Token",
        "mode_0_stats.norm_median": "Normalized Median Scenarios per Token",
        "mode_0_stats.num_underutilized_tokens": "Underutilized Tokens",
        "num_total_tokens": "Vocabulary Size",
        "uniqueness.max": "Max. Vocabulary Overlap",
        "uniqueness.mean": "Avg. Vocabulary Overlap",
        "uniqueness.median": "Median Vocabulary Overlap",
        "uniqueness.min": "Min. Vocabulary Overlap",
        "uniqueness.std": "Std. Vocabulary Overlap",
        "intergroup_uniqueness.max": "Max. Intergroup Vocabulary Overlap",
        "intergroup_uniqueness.mean": "Avg. Intergroup Vocabulary Overlap",
        "intergroup_uniqueness.median": "Median Intergroup Vocabulary Overlap",
        "intergroup_uniqueness.min": "Min. Intergroup Vocabulary Overlap",
        "intergroup_uniqueness.std": "Std. Intergroup Vocabulary Overlap",
        "wasserstein_distance.max": "Max. Wasserstein Distance",
        "wasserstein_distance.mean": "Avg. Wasserstein Distance",
        "wasserstein_distance.median": "Median Wasserstein Distance",
        "wasserstein_distance.min": "Min. Wasserstein Distance",
        "wasserstein_distance.std": "Std. Wasserstein Distance",
        "diagonal.max": "Max. Consistency",
        "diagonal.mean": "Avg. Consistency",
        "diagonal.median": "Median Consistency",
        "diagonal.min": "Min. Consistency",
        "diagonal.std": "Std. Consistency",
    }
    # Run comparative analyses

    # This analysis produces a comparison of the tokenization distributions across the experiments.
    def _flatten_metrics(data: dict, prefix: str = "") -> dict[str, float | int | str | bool | None]:
        flat: dict[str, float | int | str | bool | None] = {}
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(_flatten_metrics(value, name))
            else:
                flat[name] = value
        return flat

    def _escape_latex(text: str) -> str:
        return (
            text.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
            .replace("$", "\\$")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("~", "\\textasciitilde{}")
            .replace("^", "\\textasciicircum{}")
        )

    def _format_value(value: float | str | bool | None) -> str:  # noqa: FBT001
        if value is None:
            return "--"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    experiment_names = sorted(analysis_paths.keys())
    model_names = [MODEL_NAME_MAP.get(name.split("_")[1], "Unknown Model") for name in experiment_names]
    rows: dict[tuple[str, str], dict[str, float | int | str | bool | None]] = {}

    for model_name, experiment_name in zip(model_names, experiment_names, strict=True):
        analysis_path = analysis_paths[experiment_name]

        for json_path in sorted(analysis_path.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                continue
            flat = _flatten_metrics(data)
            for metric_name, metric_value in flat.items():
                row_key = (json_path.name, metric_name)
                if row_key not in rows:
                    rows[row_key] = {}
                rows[row_key][model_name] = metric_value

    if not rows:
        print("No JSON metrics found for comparative analysis.")
        return

    table_path = output_path / "comparative_metrics_table.tex"
    columns = "l l " + " ".join(["c" for _ in model_names])
    header_cells = ["Analysis", "Metric"] + [_escape_latex(str(name)) for name in model_names]

    with table_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[h!]\n")
        handle.write("\\centering\n")
        handle.write("\\small\n")
        handle.write("\\setlength{\\tabcolsep}{4pt}\n")

        handle.write("\\caption{Tokenization Analysis}\n")
        handle.write("\\label{tab:tokenization_analysis}\n")

        handle.write("\\resizebox{\\columnwidth}{!}{%\n")
        handle.write("\\begin{tabular}{%s}\n" % columns)  # noqa: UP031
        handle.write("\\toprule\n")

        handle.write(" %s \\\\" % " & ".join([f"\\textbf{{{cell}}}" for cell in header_cells]))  # noqa: UP031
        handle.write("\n\\midrule\n")

        file_to_metrics: dict[str, list[str]] = {}
        for file_name, metric_name in rows:
            file_to_metrics.setdefault(file_name, []).append(metric_name)

        for file_name in sorted(file_to_metrics.keys()):
            metrics = sorted(file_to_metrics[file_name])
            for idx, metric_name in enumerate(metrics):
                values = rows[(file_name, metric_name)]
                if idx == 0:
                    file_cell = (
                        f"\\multirow{{{len(metrics)}}}{{*}}{{{_escape_latex(filename_map.get(file_name, file_name))}}}"
                    )
                else:
                    file_cell = ""
                row_values = [
                    file_cell,
                    _escape_latex(metric_map.get(metric_name, metric_name)),
                ]
                for model_name in model_names:
                    row_values.extend([_escape_latex(_format_value(values.get(model_name)))])
                handle.write(" %s \\\\\n" % " & ".join(row_values))  # noqa: UP031

            # only add midrule after each file group, not after the last one
            if file_name != sorted(file_to_metrics.keys())[-1]:
                handle.write("\n\\midrule\n")

        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")

        handle.write("}\n")
        handle.write("\\end{table}\n")
