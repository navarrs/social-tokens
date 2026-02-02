import math
from itertools import product
from logging import Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
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

    print("\n✓ Analysis complete!")


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
