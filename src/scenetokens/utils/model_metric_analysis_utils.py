import math
from collections.abc import Iterable
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
from numpy.typing import NDArray
from omegaconf import DictConfig

from scenetokens.utils.constants import EPSILON


MODEL_NAME_MAP = {
    "autobot": "AutoBot",
    "scenetransformer": "SceneTransformer",
    "wayformer": "Wayformer",
    "mtr": "MTR",
    "scenetokens": "ST",
    "causal-scenetokens": "Causal-ST",
    "safe-scenetokens": "Safe-ST",
}

MODEL_SIZE_MAP = {
    "AutoBot": "1.5M",
    "SceneTransformer": "7.6M",
    "Wayformer": "15.1M",
    "ST": "15.3M",
    "Causal-ST": "15.6M",
    "Safe-ST": "15.6M",
    "MTR": "27.2M",  # This is the size with d_model=256. The original MTR with d_model=512 has 65M parameters.
}

BENCHMARK_NAME_MAP = {
    "causal-benchmark-labeled": "CausalAgents",
    "ego-safeshift-causal-benchmark": "EgoSafeShift",
}

# STRATEGY_NAME_MAP = {
#     "random_drop": "Random",
#     "token_random_drop": "Token(R)",
#     "simple_token_jaccard_drop": "Token(SJ)",
#     "simple_token_hamming_drop": "Token(SH)",
#     "gumbel_token_jaccard_drop": "Token(GJ)",
#     "gumbel_token_hamming_drop": "Token(GH)",
#     "kmeans_random_drop": "KMeans(R)",
#     "simple_kmeans_cosine_drop": "KMeans(SC)",
#     "gumbel_kmeans_cosine_drop": "KMeans(GC)",
# }

STRATEGY_NAME_MAP = {
    "random_drop": "Random",
    "token_random_drop": "Token-R",
    "simple_token_jaccard_drop": "Token-SJ",
    "simple_token_hamming_drop": "Token-SH",
    "gumbel_token_jaccard_drop": "Token-GJ",
    "gumbel_token_hamming_drop": "Token-GH",
    "kmeans_random_drop": "KMeans-R",
    "simple_kmeans_cosine_drop": "KMeans-SC",
    "gumbel_kmeans_cosine_drop": "KMeans-GC",
}


# Maps CSV file stem prefixes to display labels used when multiple tokenizer files are active.
FILE_KEY_LABEL_MAP = {
    "st": "ST",
    "causalst": "Causal-ST",
    "safest": "Safe-ST",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _relative_gap_pct(value: float | NDArray, reference: float | NDArray) -> float | NDArray:
    """Compute ``(value - reference) / |reference| * 100``. Works for scalars and numpy arrays."""
    return ((value - reference) / (np.abs(reference) + EPSILON)) * 100


def _build_strategy_colormap(config: DictConfig, items: Iterable) -> dict:
    """Return a ``{item: color}`` mapping using the configured lineplot colormap."""
    cmap = plt.cm.get_cmap(config.get("lineplot_colormap", "tab10"))
    colors = [cmap(i) for i in range(cmap.N)]
    return {s: colors[i % len(colors)] for i, s in enumerate(items)}


def _collect_sweep_y_values(
    metrics_df: pd.DataFrame,
    sweep_prefix: str,
    retention_pcts: list[float],
    column: str,
    metric: str,
) -> tuple[NDArray, list[float]]:
    """Collect metric values across retention percentages for one sweep prefix.

    Args:
        metrics_df: DataFrame containing all experiment metrics.
        sweep_prefix: Experiment name prefix identifying this sweep (e.g. ``benchmark_model_strategy``).
        retention_pcts: Ordered list of retention-percentage floats.
        column: DataFrame column to read from.
        metric: Metric name (used to detect the ``Runtime`` special case).

    Returns:
        A tuple ``(y_arr, collected)`` where ``y_arr`` is a float array aligned to ``retention_pcts``
        (NaN for missing entries) and ``collected`` contains only the valid values.
    """
    y: list[float] = []
    collected: list[float] = []
    for pct in retention_pcts:
        row = metrics_df[metrics_df["Name"] == f"{sweep_prefix}_{pct}"]
        if not row.empty and column in row.columns:
            val = float(row.iloc[0][column])
            y.append(val)
            collected.append(val)
        else:
            y.append(float("nan"))
    y_arr = np.array(y, dtype=float)
    if metric == "Runtime":
        y_arr = y_arr / 3600.0
        collected = [v / 3600.0 for v in collected]
    return y_arr, collected


def _style_sweep_ax(ax: Axes, metric: str, retention_pcts: list[float]) -> None:
    """Apply standard styling to a sample-selection sweep lineplot axis."""
    ax.set_title(metric.replace("_", " "), pad=10)
    ax.set_xlabel("Data Retention (%)")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(retention_pcts)
    ax.set_xticklabels([f"{int(p * 100)}%" for p in retention_pcts])
    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend().remove()


def _set_yaxis_limits(
    ax: Axes,
    values: list[float],
    *,
    padding_factor: float = 0.5,
    lower_factor: float = 1.0,
    min_padding: float = 0.05,
) -> None:
    """Set y-axis limits with padding around the data range.

    Args:
        ax: Matplotlib axis to modify.
        values: Data values used to compute the range.
        padding_factor: Fraction of the data range to use as padding.
        lower_factor: Multiplier applied to padding on the lower end (useful for bar charts).
        min_padding: Minimum padding when the data range is zero.
    """
    if not values:
        return
    ymin, ymax = np.nanmin(values), np.nanmax(values)
    padding = padding_factor * (ymax - ymin) if ymax > ymin else min_padding
    ax.set_ylim(ymin - padding * lower_factor, ymax + padding)


def _load_sample_selection_dataframes(config: DictConfig, log: Logger) -> dict[str, pd.DataFrame] | None:
    """Load all sample-selection metrics CSVs listed in ``config.sample_selection_files``.

    Returns a ``{suffix: DataFrame}`` mapping, or ``None`` if any file is missing.
    The suffix is derived from the first ``_``-separated token of each filename stem.
    """
    metrics_dataframes: dict[str, pd.DataFrame] = {}
    for file in config.sample_selection_files:
        log.info("Loading sample selection file: %s", file)
        metrics_filepath = Path(file)
        if not metrics_filepath.exists():
            log.error("Sample selection CSV not found at %s", metrics_filepath)
            return None
        metrics_dataframes[Path(file).stem.split("_")[0]] = pd.read_csv(metrics_filepath)
    return metrics_dataframes


def _symmetric_vrange(values: list[float]) -> tuple[float, float]:
    """Return ``(-vabs, +vabs)`` where ``vabs = max(|min|, |max|)`` of *values*."""
    vabs = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
    return -vabs, vabs


def _flatten_metrics(data: dict, prefix: str = "") -> dict[str, float | int | str | bool | None]:
    """Recursively flatten a nested dict, joining keys with dots."""
    flat: dict[str, float | int | str | bool | None] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, name))
        else:
            flat[name] = value
    return flat


# ---------------------------------------------------------------------------
# Lineplot functions
# ---------------------------------------------------------------------------


def _plot_sample_selection_sweep_lineplot(
    config: DictConfig, log: Logger, output_path: Path, metrics_df: pd.DataFrame, suffix: str = ""
) -> None:
    """For each (model, split), create a figure with one subplot per metric.

    Each subplot shows metric values across retention percentages for all sample-selection strategies,
    plus a horizontal base-model reference line and a best-strategy star marker.

    Args:
        config: Model analysis configuration.
        log: Logger.
        output_path: Directory for generated plots.
        metrics_df: DataFrame with experiment metrics.
        suffix: Appended to output filenames to distinguish different metrics files.
    """
    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    log.info("Plotting sample selection sweep lineplots for metrics: %s", metrics)
    retention_pcts = list(map(float, config.sample_retention_percentages))
    colormap = _build_strategy_colormap(config, config.sample_selection_strategies_to_compare)

    for model, split in product(config.models_to_compare, config.sample_selection_splits_to_compare):
        subsplit = split.split("/")[-1]
        log.info("Creating sweep plot for model=%s, split=%s", model, split)

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
        for i, metric in enumerate(metrics):
            ax = axes[0, i]
            column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric

            best_value, best_strategy, best_x, best_y = np.inf, None, None, None
            all_y_values: list[float] = []

            for strategy in config.sample_selection_strategies_to_compare:
                sweep_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}"
                y, collected = _collect_sweep_y_values(metrics_df, sweep_prefix, retention_pcts, column, metric)
                all_y_values.extend(collected)

                if np.nanmin(y) < best_value:
                    best_value = np.nanmin(y)
                    best_strategy = strategy
                    idx = int(np.nanargmin(y))
                    best_x, best_y = retention_pcts[idx], y[idx]

                ax.plot(retention_pcts, y, marker="o", ms=6, lw=2.5, c=colormap[strategy], alpha=0.9, label=strategy)

            if best_strategy is not None:
                ax.plot(best_x, best_y, marker="*", ms=16, c=colormap[best_strategy], mec="k", zorder=10, label="Best")

            # Base model (no sample selection) reference line
            base_name = f"{config.sample_selection_benchmark}_{model}"
            base_df = metrics_df[metrics_df["Name"] == base_name]
            if not base_df.empty and column in base_df.columns:
                base_value = base_df[column].min()
                if metric == "Runtime":
                    base_value = base_value / 3600.0
                ax.axhline(base_value, ls="--", lw=2, c="black", alpha=0.7, label="Base model")
                all_y_values.append(base_value)

            _set_yaxis_limits(ax, all_y_values)
            _style_sweep_ax(ax, metric, retention_pcts)

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
    """For each (model, split, strategy), create a figure with one subplot per metric.

    Each subplot shows metric values across retention percentages for each metrics-file version as a
    separate line, allowing comparison across different selector implementations.

    Args:
        config: Model analysis configuration.
        log: Logger.
        output_path: Directory for generated plots.
        metrics_dataframes: ``{version_key: DataFrame}`` mapping.
    """
    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    log.info("Plotting joint sample selection sweep lineplots for metrics: %s", metrics)
    retention_pcts = list(map(float, config.sample_retention_percentages))
    colormap = _build_strategy_colormap(config, metrics_dataframes.keys())

    for model, split in product(config.models_to_compare, config.sample_selection_splits_to_compare):
        subsplit = split.split("/")[-1]

        for strategy in config.sample_selection_strategies_to_compare:
            log.info("Creating sweep plot for model=%s, split=%s, strategy=%s", model, split, strategy)
            sweep_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}"

            fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
            for i, metric in enumerate(metrics):
                ax = axes[0, i]
                column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric
                all_y_values: list[float] = []

                for version_key, version_df in metrics_dataframes.items():
                    y, collected = _collect_sweep_y_values(version_df, sweep_prefix, retention_pcts, column, metric)
                    all_y_values.extend(collected)
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

                _set_yaxis_limits(ax, all_y_values)
                _style_sweep_ax(ax, metric, retention_pcts)

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

    metrics_dataframes = _load_sample_selection_dataframes(config, log)
    if metrics_dataframes is None:
        return

    for suffix, metrics_df in metrics_dataframes.items():
        _plot_sample_selection_sweep_lineplot(config, log, output_path, metrics_df, f"_{suffix}")

    if len(metrics_dataframes) > 1:
        _plot_joint_sample_selection_sweep_lineplot(config, log, output_path, metrics_dataframes)


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------


def _identify_tokenizer_files(config: DictConfig, metrics_dfs: dict[str, pd.DataFrame]) -> set[str]:
    """Return the set of file keys whose strategy sets overlap with at least one other file.

    These are the "tokenizer files" (e.g. ``st``, ``causalst``, ``safest``) that contain the same
    strategy names but with results from different tokenizers.
    """
    strategies = list(config.sample_selection_strategies_to_compare)
    models = list(config.models_to_compare)
    benchmark = config.sample_selection_benchmark

    file_strategies: dict[str, set[str]] = {}
    for key, df in metrics_dfs.items():
        found: set[str] = set()
        for model, strategy in product(models, strategies):
            prefix = f"{benchmark}_{model}_{strategy}_"
            if df["Name"].str.startswith(prefix).any():
                found.add(strategy)
        file_strategies[key] = found

    tokenizer_keys: set[str] = set()
    keys = list(file_strategies)
    for i, ki in enumerate(keys):
        for kj in keys[i + 1 :]:
            if file_strategies[ki] & file_strategies[kj]:
                tokenizer_keys.add(ki)
                tokenizer_keys.add(kj)
    return tokenizer_keys


def _build_row_index(models: list[str], tokenizer_keys: list[str]) -> list[tuple[str, str | None]]:
    """Return ``(model, tokenizer_key)`` pairs that define the heatmap rows.

    When only one (or zero) tokenizer files are active the tokenizer dimension is collapsed and
    ``tokenizer_key`` is ``None`` or the single key (no label suffix is added in that case).
    """
    if len(tokenizer_keys) <= 1:
        tk = tokenizer_keys[0] if tokenizer_keys else None
        return [(m, tk) for m in models]
    return [(m, tk) for m in models for tk in tokenizer_keys]


def _lookup_strategy_value(  # noqa: PLR0913
    metrics_dfs: dict[str, pd.DataFrame],
    tokenizer_keys: set[str],
    run_name_prefix: str,
    tokenizer_key: str | None,
    column: str,
    *,
    multi_tokenizer: bool,
) -> float | None:
    """Return the first non-NaN metric value matching ``run_name_prefix`` in the appropriate dataframe(s).

    When ``multi_tokenizer`` is True and the strategy lives in a tokenizer file, only the dataframe
    for ``tokenizer_key`` is searched so results from different tokenizers are never mixed.
    Non-tokenizer strategies are searched across all non-tokenizer dataframes.
    """
    if multi_tokenizer and tokenizer_key is not None:
        # Determine whether this strategy is tokenizer-specific by checking if the prefix exists
        # in any tokenizer file other than the current one.
        is_tokenizer_strategy = any(
            metrics_dfs[k]["Name"].str.startswith(run_name_prefix).any()
            for k in tokenizer_keys
            if k != tokenizer_key and k in metrics_dfs
        ) or (tokenizer_key in metrics_dfs and metrics_dfs[tokenizer_key]["Name"].str.startswith(run_name_prefix).any())

        search_dfs: dict[str, pd.DataFrame]
        if is_tokenizer_strategy and tokenizer_key in metrics_dfs:
            search_dfs = {tokenizer_key: metrics_dfs[tokenizer_key]}
        else:
            search_dfs = {k: v for k, v in metrics_dfs.items() if k not in tokenizer_keys}
    else:
        search_dfs = metrics_dfs

    for df in search_dfs.values():
        matches = df[df["Name"].str.startswith(run_name_prefix)]
        if not matches.empty and column in matches.columns:
            val = matches.iloc[0][column]
            if pd.notna(val):
                return float(val)
    return None


# ---------------------------------------------------------------------------
# Heatmap functions
# ---------------------------------------------------------------------------


def _plot_sample_selection_sweep_heatmap(  # noqa: PLR0912, PLR0915
    config: DictConfig, log: Logger, output_path: Path, metrics_dfs: dict[str, pd.DataFrame], suffix: str = ""
) -> None:
    """Plot heatmaps comparing sample selection strategies across retention percentages for each (model, split, metric).

    Searches across all provided dataframes to find each (model, strategy, pct) combination,
    accommodating different naming suffixes used in different source files (e.g. ``_kmeans``).

    When multiple tokenizer files are active (e.g. ``st``, ``causalst``, ``safest``), rows expand to
    ``(model, tokenizer)`` pairs.  Non-tokenizer strategies (random, kmeans, dentp) repeat the same
    value across all tokenizer rows for the same model.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_dfs (dict[str, pd.DataFrame]): DataFrames keyed by file stem prefix (e.g. ``random``, ``st``).
        suffix (str): Suffix to append to output filenames.
    """
    output_path = output_path / "sample_selection_heatmaps"
    output_path.mkdir(parents=True, exist_ok=True)

    cmap = sns.color_palette(config.get("heatmap_colormap", "mako_r"), as_cmap=True)
    metrics = config.trajectory_forecasting_metrics + config.other_metrics

    log.info("Plotting sample selection sweep heatmaps for metrics: %s", metrics)
    retention_pcts = list(map(float, config.sample_retention_percentages))
    models = list(config.models_to_compare)
    strategies = list(config.sample_selection_strategies_to_compare)
    highlight_color = config.get("highlight_color", "dodgerblue")

    tokenizer_keys = _identify_tokenizer_files(config, metrics_dfs)
    # Preserve the order in which tokenizer files appear in the config so output is deterministic.
    ordered_tokenizer_keys = [k for k in metrics_dfs if k in tokenizer_keys]
    multi_tokenizer = len(ordered_tokenizer_keys) > 1
    row_index = _build_row_index(models, ordered_tokenizer_keys)
    num_rows = len(row_index)

    def _row_label(model: str, tk: str | None) -> str:
        base = MODEL_NAME_MAP.get(model, model)
        if multi_tokenizer and tk is not None:
            return f"{base} ({FILE_KEY_LABEL_MAP.get(tk, tk)})"
        return base

    for split in config.sample_selection_splits_to_compare:
        subsplit = split.split("/")[-1]
        log.info("Creating heatmap sweep plots for split=%s", split)

        for metric in metrics:
            column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric

            heatmap_data = {}
            all_values = []

            # Build strategy heatmaps — search appropriate dataframe(s) per (model, tokenizer, strategy)
            for pct in retention_pcts:
                data = np.full((num_rows, len(strategies)), np.nan)
                for i, (model, tk) in enumerate(row_index):
                    for j, strategy in enumerate(strategies):
                        run_name_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}_{pct}"
                        val = _lookup_strategy_value(
                            metrics_dfs, tokenizer_keys, run_name_prefix, tk, column, multi_tokenizer=multi_tokenizer
                        )
                        if val is not None:
                            data[i, j] = val
                            all_values.append(val)
                heatmap_data[pct] = data

            # Base-only heatmap — exact name match, repeated across tokenizer rows for the same model
            base_data = np.full((num_rows, 1), np.nan)
            base_values: dict[int, float | None] = {}
            for i, (model, _tk) in enumerate(row_index):
                base_name = f"{config.sample_selection_benchmark}_{model}"
                for df in metrics_dfs.values():
                    row = df[df["Name"] == base_name]
                    if not row.empty and column in row.columns:
                        val = float(row[column].min())
                        base_data[i, 0] = val
                        base_values[i] = val
                        all_values.append(val)
                        break
                if i not in base_values:
                    base_values[i] = None

            if not all_values:
                log.warning("No data found for metric=%s, split=%s", metric, split)
                continue

            vmin = np.nanmin(all_values)
            vmax = np.nanmax(all_values)

            # Figure layout — size so that each cell is square.
            # width_ratios are proportional to column counts so cell sizes match across subplots.
            # constrained_layout resolves colorbar placement, aspect="equal", and spacing together.
            num_pcts = len(retention_pcts)
            n_cols = len(strategies)
            cell = 0.75  # inches per cell
            label_margin = 2.2  # left margin for y-tick labels
            xtick_margin = 1.5  # bottom margin for rotated x-tick labels
            title_margin = 0.5  # top margin for subplot titles + suptitle
            cbar_margin = 0.8  # right margin for colorbar
            fig_w = cell * (num_pcts * n_cols + 1) + label_margin + cbar_margin
            fig_h = cell * num_rows + xtick_margin + title_margin
            marker_size = 25
            fig, axes = plt.subplots(
                1,
                num_pcts + 1,
                figsize=(fig_w, fig_h),
                squeeze=False,
                gridspec_kw={"width_ratios": [n_cols] * num_pcts + [1]},
                layout="constrained",
            )
            strategy_rotation = 0
            fig.get_layout_engine().set(wspace=0.02, w_pad=0.01)  # type: ignore[union-attr]
            axes = axes[0]

            row_labels = [_row_label(m, tk) for m, tk in row_index]

            # Plot strategy heatmaps
            for k, (ax, pct) in enumerate(zip(axes[:num_pcts], retention_pcts, strict=False)):
                data = heatmap_data[pct]
                masked_data = np.ma.masked_invalid(data)
                im = ax.imshow(masked_data, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
                im.cmap.set_bad(color="#eeeeee")

                ax.set_title(f"{int(pct * 100)}%", pad=6)
                ax.set_xticks(range(n_cols))
                ax.set_xticklabels(
                    [STRATEGY_NAME_MAP.get(s, s) for s in strategies],
                    rotation=strategy_rotation,
                    ha="center",
                    rotation_mode="anchor",
                )

                if k == 0:
                    ax.set_yticks(range(num_rows))
                    ax.set_yticklabels(row_labels)
                    ax.tick_params(axis="y", pad=6)
                else:
                    ax.set_yticks([])

                ax.tick_params(which="minor", bottom=False, left=False)

                # Gray outline star on every cell that equals or beats the baseline
                for i in range(data.shape[0]):
                    base_val = base_values.get(i)
                    if base_val is None:
                        continue
                    for j in range(data.shape[1]):
                        if not np.isnan(data[i, j]) and data[i, j] <= base_val:
                            ax.plot(j, i, marker="*", ms=marker_size, mec="lightgray", mew=1.5, c="none", zorder=5)

                # Highlight best per row
                for i in range(data.shape[0]):
                    row_data = data[i]
                    if np.all(np.isnan(row_data)):
                        continue

                    j = np.nanargmin(row_data)
                    best_val = row_data[j]
                    base_val = base_values.get(i)
                    edge_color, marker_color = "black", "black"
                    if base_val is not None and best_val <= base_val:
                        edge_color = highlight_color
                        marker_color = highlight_color

                    if config.add_rectangle_annotation:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=edge_color, linewidth=3))
                    ax.plot(j, i, marker="*", ms=marker_size, mec=marker_color, mew=1, c=marker_color, zorder=10)

            # Baseline heatmap
            ax_base = axes[-1]
            masked_base = np.ma.masked_invalid(base_data)
            im = ax_base.imshow(masked_base, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            im.cmap.set_bad(color="#eeeeee")
            ax_base.set_title("Baseline", pad=6)
            ax_base.set_xticks([])
            ax_base.set_yticks([])

            # Attach colorbar to the axes group so it auto-aligns and matches the heatmap height.
            # shrink pulls the bar height closer to the heatmap extent; fraction controls its width.
            cbar = fig.colorbar(im, ax=list(axes), pad=0.02, shrink=0.8, fraction=0.05, aspect=15)
            cbar.ax.tick_params(labelsize=9)

            # Legend handles to show best strategies
            legend = [
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color=highlight_color,
                    markersize=12,
                    label="Best strategy in group ≥ baseline",
                ),
                Line2D([0], [0], marker="*", color="black", markersize=12, label="Best strategy < baseline"),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="none",
                    mec="lightgray",
                    mew=1.5,
                    markersize=12,
                    label="Equals or beats baseline",
                ),
            ]

            output_file = output_path / f"{metric}_{subsplit}{suffix}.png"
            fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=9)
            fig.suptitle(f"{metric.replace('_', ' ')} — {split}")
            fig.savefig(output_file, dpi=200, bbox_inches="tight")
            plt.close(fig)

            log.info("Saved heatmaps to %s", output_file)


def _plot_sample_selection_sweep_heatmap_baseline_gap(  # noqa: PLR0912, PLR0915
    config: DictConfig,
    log: Logger,
    output_path: Path,
    metrics_dfs: dict[str, pd.DataFrame],
) -> None:
    """Plot heatmaps showing % gap to baseline for each (model, split, metric, retention_pct, strategy).

    Rows are ``(model, tokenizer)`` pairs when multiple tokenizer files are active.
    Non-tokenizer strategies repeat the same gap value across tokenizer rows for the same model.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_dfs (dict[str, pd.DataFrame]): Dictionary of DataFrames containing the metrics data for each strategy.
    """
    output_path = output_path / "sample_selection_heatmaps_baseline_gap"
    output_path.mkdir(parents=True, exist_ok=True)

    cmap = sns.color_palette(config.get("heatmap_colormap", "RdYlGn_r"), as_cmap=True)
    highlight_color = config.get("highlight_color", "dodgerblue")

    metrics = config.trajectory_forecasting_metrics + config.other_metrics
    retention_pcts = list(map(float, config.sample_retention_percentages))
    models = list(config.models_to_compare)
    strategies = list(config.sample_selection_strategies_to_compare)
    splits = config.sample_selection_splits_to_compare
    log.info("Plotting sample selection sweep heatmaps (baseline gap) for metrics: %s", metrics)

    tokenizer_keys = _identify_tokenizer_files(config, metrics_dfs)
    ordered_tokenizer_keys = [k for k in metrics_dfs if k in tokenizer_keys]
    multi_tokenizer = len(ordered_tokenizer_keys) > 1
    row_index = _build_row_index(models, ordered_tokenizer_keys)
    num_rows = len(row_index)
    num_retention_pcts = len(retention_pcts)
    n_cols = len(strategies)

    def _row_label(model: str, tk: str | None) -> str:
        base = MODEL_NAME_MAP.get(model, model)
        if multi_tokenizer and tk is not None:
            return f"{base} ({FILE_KEY_LABEL_MAP.get(tk, tk)})"
        return base

    for split, metric in product(splits, metrics):
        subsplit = split.split("/")[-1]
        column = f"{split}/{metric}" if metric in config.trajectory_forecasting_metrics else metric
        log.info("Creating baseline-gap heatmap plots for split=%s metric=%s", split, metric)

        # Fetch baseline values per model
        base_vals_per_model: dict[str, float | None] = {}
        for model in models:
            base_name = f"{config.sample_selection_benchmark}_{model}"
            for df in metrics_dfs.values():
                row = df[df["Name"] == base_name]
                if not row.empty and column in row.columns:
                    base_vals_per_model[model] = float(row[column].min())
                    break
            if model not in base_vals_per_model:
                base_vals_per_model[model] = None

        heatmap_data = {}
        all_gaps: list[float] = []

        # Build gap heatmaps using the same row/lookup logic as the main heatmap
        for pct in retention_pcts:
            data = np.full((num_rows, n_cols), np.nan)
            for i, (model, tk) in enumerate(row_index):
                base_val = base_vals_per_model.get(model)
                if base_val is None:
                    continue
                for j, strategy in enumerate(strategies):
                    run_name_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}_{pct}"
                    val = _lookup_strategy_value(
                        metrics_dfs, tokenizer_keys, run_name_prefix, tk, column, multi_tokenizer=multi_tokenizer
                    )
                    if val is not None:
                        gap = float(_relative_gap_pct(val, base_val))
                        data[i, j] = gap
                        all_gaps.append(gap)
            heatmap_data[pct] = data

        if not all_gaps:
            log.warning("No data found for metric=%s, split=%s", metric, split)
            continue

        vmin, vmax = _symmetric_vrange(all_gaps)

        # Figure layout — same cell-size approach as the main heatmap
        cell = 0.45
        label_margin = 2.2
        xtick_margin = 1.5
        title_margin = 0.5
        cbar_margin = 0.8
        fig_w = cell * num_retention_pcts * n_cols + label_margin + cbar_margin
        fig_h = cell * num_rows + xtick_margin + title_margin
        fig, axes = plt.subplots(1, num_retention_pcts, figsize=(fig_w, fig_h), squeeze=False, layout="constrained")
        fig.get_layout_engine().set(wspace=0.02, w_pad=0.01)  # type: ignore[union-attr]
        axes = axes[0]

        row_labels = [_row_label(m, tk) for m, tk in row_index]

        # Plot gap heatmaps
        im = None
        for k, (ax, pct) in enumerate(zip(axes, retention_pcts, strict=False)):
            data = heatmap_data[pct]
            masked_data = np.ma.masked_invalid(data)
            im = ax.imshow(masked_data, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            im.cmap.set_bad(color="#eeeeee")

            ax.set_title(f"{int(pct * 100)}%", pad=6)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(
                [STRATEGY_NAME_MAP.get(s, s) for s in strategies], rotation=35, ha="right", rotation_mode="anchor"
            )
            if k == 0:
                ax.set_yticks(range(num_rows))
                ax.set_yticklabels(row_labels)
                ax.tick_params(axis="y", pad=6)
            else:
                ax.set_yticks([])
            ax.tick_params(which="minor", bottom=False, left=False)

            # Gray outline star on every cell with gap <= 0 (equals or beats baseline)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if not np.isnan(data[i, j]) and data[i, j] <= 0:
                        ax.plot(j, i, marker="*", ms=18, mec="lightgray", mew=1.5, c="none", zorder=5)

            # Highlight best per row (most negative gap = most improvement over baseline)
            for i in range(data.shape[0]):
                row_data = data[i]
                if np.all(np.isnan(row_data)):
                    continue
                j = np.nanargmin(row_data)
                gap_val = row_data[j]
                marker_color = highlight_color if gap_val < 0 else "black"
                if config.add_rectangle_annotation:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=marker_color, linewidth=3))
                ax.plot(j, i, marker="*", ms=18, mec=marker_color, mew=1, c=marker_color, zorder=10)

        if im is not None:
            cbar = fig.colorbar(im, ax=list(axes), pad=0.02, shrink=0.8, fraction=0.05, aspect=15)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label("Gap to Baseline (%)", fontsize=9)

        legend = [
            Line2D([0], [0], marker="*", color=highlight_color, markersize=10, label="Best strategy beats baseline"),
            Line2D([0], [0], marker="*", color="black", markersize=10, label="Best strategy in group"),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                mec="black",
                alpha=0.1,
                mew=1.5,
                markersize=8,
                label="Equals or beats baseline",
            ),
        ]

        output_file = output_path / f"{metric}_{subsplit}.png"
        fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=9)
        fig.suptitle(f"Gap to Baseline — {metric.replace('_', ' ')} ({split})")
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

        log.info("Saved heatmaps to %s", output_file)


def _plot_sample_selection_sweep_distribution_gap(  # noqa: PLR0912, PLR0915
    config: DictConfig,
    log: Logger,
    output_path: Path,
    metrics_dfs: dict[str, pd.DataFrame],
) -> None:
    """Plot heatmaps showing the OOD-ID split gap for each (model, metric, retention_pct, strategy).

    Rows are ``(model, tokenizer)`` pairs when multiple tokenizer files are active.
    Non-tokenizer strategies repeat the same gap value across tokenizer rows for the same model.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
        metrics_dfs (dict[str, pd.DataFrame]): Dictionary of DataFrames containing the metrics data for each strategy.
    """
    splits = config.sample_selection_splits_to_compare
    if len(splits) < 2:  # noqa: PLR2004
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
    models = list(config.models_to_compare)
    strategies = list(config.sample_selection_strategies_to_compare)
    log.info("Plotting sample selection sweep heatmaps (distribution gap) for metrics: %s", metrics)

    tokenizer_keys = _identify_tokenizer_files(config, metrics_dfs)
    ordered_tokenizer_keys = [k for k in metrics_dfs if k in tokenizer_keys]
    multi_tokenizer = len(ordered_tokenizer_keys) > 1
    row_index = _build_row_index(models, ordered_tokenizer_keys)
    num_rows = len(row_index)
    num_retention_pcts = len(retention_pcts)
    n_cols = len(strategies)

    def _row_label(model: str, tk: str | None) -> str:
        base = MODEL_NAME_MAP.get(model, model)
        if multi_tokenizer and tk is not None:
            return f"{base} ({FILE_KEY_LABEL_MAP.get(tk, tk)})"
        return base

    for metric in metrics:
        id_column = f"{id_split}/{metric}"
        ood_column = f"{ood_split}/{metric}"
        log.info("Creating distribution gap heatmaps for metric=%s (%s vs %s)", metric, id_split, ood_split)

        # Compute baseline split-gap and ID value per model (exact name match, search all dfs)
        baseline_gaps: dict[str, float | None] = {}
        baseline_id_values: dict[str, float | None] = {}
        for model in models:
            base_name = f"{config.sample_selection_benchmark}_{model}"
            for df in metrics_dfs.values():
                base_row = df[df["Name"] == base_name]
                if not base_row.empty and id_column in base_row.columns and ood_column in base_row.columns:
                    id_val = float(base_row.iloc[0][id_column])
                    ood_val = float(base_row.iloc[0][ood_column])
                    baseline_gaps[model] = float(_relative_gap_pct(ood_val, id_val))
                    baseline_id_values[model] = id_val
                    break
            if model not in baseline_gaps:
                baseline_gaps[model] = None
                baseline_id_values[model] = None

        heatmap_data: dict[float, np.ndarray] = {}
        all_gaps: list[float] = []

        # Build split-gap heatmaps using the same row/lookup logic as the baseline-gap heatmap
        for pct in retention_pcts:
            data = np.full((num_rows, n_cols), np.nan)
            for i, (model, tk) in enumerate(row_index):
                for j, strategy in enumerate(strategies):
                    run_name_prefix = f"{config.sample_selection_benchmark}_{model}_{strategy}_{pct}"
                    id_val = _lookup_strategy_value(
                        metrics_dfs, tokenizer_keys, run_name_prefix, tk, id_column, multi_tokenizer=multi_tokenizer
                    )
                    ood_val = _lookup_strategy_value(
                        metrics_dfs, tokenizer_keys, run_name_prefix, tk, ood_column, multi_tokenizer=multi_tokenizer
                    )
                    if id_val is not None and ood_val is not None:
                        gap = float(_relative_gap_pct(ood_val, id_val))
                        data[i, j] = gap
                        all_gaps.append(gap)
            heatmap_data[pct] = data

        if not all_gaps:
            log.warning("No data found for metric=%s (%s vs %s)", metric, id_split, ood_split)
            continue

        vmin, vmax = _symmetric_vrange(all_gaps)

        # Figure layout — same cell-size approach as the other heatmaps
        cell = 0.45
        label_margin = 2.2
        xtick_margin = 1.5
        title_margin = 0.5
        cbar_margin = 0.8
        fig_w = cell * num_retention_pcts * n_cols + label_margin + cbar_margin
        fig_h = cell * num_rows + xtick_margin + title_margin
        fig, axes = plt.subplots(1, num_retention_pcts, figsize=(fig_w, fig_h), squeeze=False, layout="constrained")
        fig.get_layout_engine().set(wspace=0.02, w_pad=0.01)  # type: ignore[union-attr]
        axes = axes[0]

        row_labels = [_row_label(m, tk) for m, tk in row_index]

        im = None
        for k, (ax, pct) in enumerate(zip(axes, retention_pcts, strict=False)):
            data = heatmap_data[pct]
            masked_data = np.ma.masked_invalid(data)
            im = ax.imshow(masked_data, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
            im.cmap.set_bad(color="#eeeeee")

            ax.set_title(f"{int(pct * 100)}%", pad=6)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(
                [STRATEGY_NAME_MAP.get(s, s) for s in strategies], rotation=35, ha="right", rotation_mode="anchor"
            )
            if k == 0:
                ax.set_yticks(range(num_rows))
                ax.set_yticklabels(row_labels)
                ax.tick_params(axis="y", pad=6)
            else:
                ax.set_yticks([])
            ax.tick_params(which="minor", bottom=False, left=False)

            # Gray outline star on every cell whose split gap equals or beats the baseline gap
            for i, (model, _tk) in enumerate(row_index):
                baseline_gap = baseline_gaps.get(model)
                if baseline_gap is None:
                    continue
                for j in range(data.shape[1]):
                    if not np.isnan(data[i, j]) and data[i, j] <= baseline_gap:
                        ax.plot(j, i, marker="*", ms=18, mec="lightgray", mew=1.5, c="none", zorder=5)

            # Highlight the strategy with the smallest absolute gap per row
            for i, (model, tk) in enumerate(row_index):
                row_data = data[i]
                if np.all(np.isnan(row_data)):
                    continue

                baseline_gap = baseline_gaps.get(model)
                baseline_id_val = baseline_id_values.get(model)
                j = int(np.nanargmin(np.abs(row_data)))
                gap_val = row_data[j]

                # Look up the ID value for the best strategy to check combined improvement
                run_name_prefix = f"{config.sample_selection_benchmark}_{model}_{strategies[j]}_{pct}"
                strategy_id_val = _lookup_strategy_value(
                    metrics_dfs, tokenizer_keys, run_name_prefix, tk, id_column, multi_tokenizer=multi_tokenizer
                )

                # Magenta: both gap and ID performance beat baseline; blue: only gap beats baseline
                if (
                    baseline_gap is not None
                    and baseline_id_val is not None
                    and strategy_id_val is not None
                    and gap_val < baseline_gap
                    and strategy_id_val < baseline_id_val
                ):
                    marker_color = "magenta"
                elif baseline_gap is not None and gap_val < baseline_gap:
                    marker_color = highlight_color
                else:
                    marker_color = "black"

                if config.add_rectangle_annotation:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=marker_color, linewidth=3))
                ax.plot(j, i, marker="*", ms=18, mec=marker_color, mew=1, c=marker_color, zorder=10)

        if im is not None:
            cbar = fig.colorbar(im, ax=list(axes), pad=0.02, shrink=0.8, fraction=0.05, aspect=15)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label(f"Gap ({ood_subsplit} - {id_subsplit}) %", fontsize=9)

        magenta_label = "Better performance and gap than baseline"
        highlight_label = "Better gap than baseline"
        black_label = "Best in group, not better than baseline"
        legend = [
            Line2D([0], [0], marker="*", color="magenta", linestyle="None", markersize=10, label=magenta_label),
            Line2D([0], [0], marker="*", color=highlight_color, linestyle="None", markersize=10, label=highlight_label),
            Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=10, label=black_label),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                mec="black",
                alpha=0.1,
                mew=1.5,
                markersize=9,
                label="Equals or beats baseline gap",
            ),
        ]

        output_file = output_path / f"{metric}_{id_subsplit}_vs_{ood_subsplit}.png"
        fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=False, fontsize=7)
        fig.suptitle(f"Split Gap — {metric.replace('_', ' ')} ({ood_subsplit} - {id_subsplit})")
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

        log.info("Saved distribution gap heatmaps to %s", output_file)


def plot_sample_selection_sweep_heatmap(config: DictConfig, log: Logger, output_path: Path) -> None:
    """Creates heatmaps comparing sample selection sweeps for each (model, split, retention_percentage, metric).

    For each split and metric, generates P heatmaps (one per retention percentage) with rows as models, columns as
    strategies, and color representing metric values. Also generates baseline gap heatmaps when multiple dataframes are
    available.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        log (Logger): Logger for logging analysis information.
        output_path (Path): Directory to save the generated plots.
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

    metrics_dataframes = _load_sample_selection_dataframes(config, log)
    if metrics_dataframes is None:
        return

    _plot_sample_selection_sweep_heatmap(config, log, output_path, metrics_dataframes)

    # If multiple metrics files are available, create heatmaps showing gap to baseline across selectors
    if len(metrics_dataframes) > 1:
        _plot_sample_selection_sweep_heatmap_baseline_gap(config, log, output_path, metrics_dataframes)
        _plot_sample_selection_sweep_distribution_gap(config, log, output_path, metrics_dataframes)


# ---------------------------------------------------------------------------
# Benchmark analysis
# ---------------------------------------------------------------------------


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

        _set_yaxis_limits(ax, list(values), padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

    n_models = models.shape[0]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.5 * n_models * 3, 6))
    fig.suptitle("Distribution Shift Analysis", fontsize=14, fontweight="bold")

    _plot_bars(ax1, id_metric, "In-Distribution (ID) Performance")
    _plot_bars(ax2, ood_metric, "Out-of-Distribution (OOD) Performance")

    # Performance Gap (OOD - ID, relative %)
    id_values = summary_df[id_metric].to_numpy()
    ood_values = summary_df[ood_metric].to_numpy()
    gap_values: NDArray = _relative_gap_pct(ood_values, id_values)  # pyright: ignore[reportArgumentType, reportAssignmentType]

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
    num_metrics = len(metrics)
    n_models = summary_df["Model"].shape[0]
    n_cols = min(2, num_metrics)
    n_rows = math.ceil(num_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_models * n_cols, 4.0 * n_rows), constrained_layout=True)
    fig.suptitle("Model Performance Comparison", fontsize=20, fontweight="bold")

    axes = np.atleast_1d(axes).flatten()

    palette = sns.color_palette(colormap, len(summary_df))
    model_order = summary_df["Model"].to_numpy()

    for idx, metric_name in enumerate(metrics.values()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        values = summary_df[metric_name].to_numpy()
        bars = ax.bar(model_order, values, color=palette, edgecolor="black", linewidth=1.0, alpha=0.8)

        ax.set_title(metric_name, pad=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_axisbelow(True)

        _set_yaxis_limits(ax, list(values), padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

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

    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

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
                "percent": _relative_gap_pct(ood_vals, id_vals),
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
            ax1.bar(x + offset, gaps["absolute"], width, label=metric_name, alpha=0.8, edgecolor="black", linewidth=1)
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
    fig, ax = plt.subplots(figsize=(14, 7))

    available_metrics = [m for m in key_metrics_display if m in summary_df.columns]

    if available_metrics:
        x = np.arange(len(summary_df))
        width = 0.2
        all_values: list[float] = []
        for i, metric in enumerate(available_metrics):
            values = summary_df[metric].to_numpy()
            all_values.extend(values)
            ax.bar(x + width * i, values, width, label=metric, alpha=0.8, edgecolor="black", linewidth=1)

        _set_yaxis_limits(ax, all_values, padding_factor=0.15, lower_factor=0.4, min_padding=0.1)

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

    if "Name" not in metrics_df.columns:
        log.error("CSV must contain a 'Name' column")
        return
    if "ID" not in metrics_df.columns:
        metrics_df["ID"] = np.arange(len(metrics_df))

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

    _plot_benchmark_comparison(summary_df, metrics, output_path, colormap)

    _plot_distribution_shift_comparison(
        summary_df,
        output_path,
        colormap,
        id_metric=f"brierFDE ({id_split_name},↓)",
        ood_metric=f"brierFDE ({ood_split_name},↓)",
    )

    metric_pairs = [
        (f"{metric} ({id_split_name},↓)", f"{metric} ({ood_split_name},↓)", metric)
        for metric in config.trajectory_forecasting_metrics
    ]
    _plot_performance_gaps(summary_df, output_path, metric_pairs)

    key_metrics_display = [
        f"{config.trajectory_forecasting_metrics[0]} ({id_split_name},↓)",
        f"{config.trajectory_forecasting_metrics[0]} ({ood_split_name},↓)",
    ]
    _plot_grouped_bar_chart(summary_df, metrics, output_path, key_metrics_display=key_metrics_display)

    _distribution_shift_to_tex_table(
        benchmark_df,
        BENCHMARK_NAME_MAP.get(config.benchmark) or config.benchmark,
        id_split,
        ood_split,
        config.trajectory_forecasting_metrics,
        output_path,
    )

    print("\n✓ Analysis complete!")


def _distribution_shift_to_tex_table(  # noqa: PLR0912, PLR0913, PLR0915
    benchmark_df: pd.DataFrame,
    benchmark_name: str,
    id_split: str,
    ood_split: str,
    metrics: list[str],
    output_path: Path | None,
    min_color_value: float = 20.0,
) -> str:
    """Converts the distribution shift benchmark DataFrame into a LaTeX table with performance gap annotations/coloring.

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

        gaps: pd.Series = _relative_gap_pct(ood_vals, id_vals)  # pyright: ignore[reportAssignmentType, reportArgumentType]
        gap_stats[metric] = (gaps.min(), gaps.max())  # best, worst

    # Build rows
    table_rows = []
    first_row = True

    for _, row in benchmark_df.iterrows():
        row_parts = []
        if first_row:
            row_parts.append(f"\\multirow{{{len(benchmark_df)}}}{{*}}{{\\texttt{{{benchmark_name}}}}}")
            first_row = False
        else:
            row_parts.append("")
        row_parts.append(str(row["Model"]))

        # Model size
        if "model/params/total" in row and pd.notna(row["model/params/total"]):
            size_val = row["model/params/total"]
            size_str = f"{size_val:.2e}" if isinstance(size_val, (int, float)) else str(size_val)
        else:
            size_str = MODEL_SIZE_MAP.get(row["Model"], "---")
        row_parts.append(size_str)

        id_values, ood_values = [], []
        for metric in metrics:
            id_col = f"{id_split}/{metric}"
            ood_col = f"{ood_split}/{metric}"
            id_val = row[id_col]
            ood_val = row[ood_col]

            # In-distribution value
            if pd.notna(id_val):
                id_str = f"{id_val:.3f}"
                if np.isclose(id_val, best_id[metric]):
                    id_str = f"\\textbf{{{id_str}}}"
            else:
                id_str = "---"
            id_values.append(id_str)

            # Out-of-distribution value with gap annotation and coloring
            if pd.notna(id_val) and pd.notna(ood_val):
                gap = _relative_gap_pct(ood_val, id_val)

                best_gap, worst_gap = gap_stats[metric]
                denom = max(abs(worst_gap - best_gap), EPSILON)
                severity = np.clip(abs(gap - best_gap) / denom, 0, 1)
                intensity = int(min_color_value + severity * (100 - min_color_value))

                color = "OrangeRed" if gap > 0 else "ForestGreen"
                gap_str = f"\\textcolor{{{color}!{intensity}}}{{{gap:+.2f}\\%}}"

                ood_str = f"{ood_val:.3f}"
                if np.isclose(ood_val, best_ood[metric]):
                    ood_str = f"\\textbf{{{ood_str}}}"
                ood_str = f"{ood_str} ({gap_str})"
            else:
                ood_str = "---"

            ood_values.append(ood_str)

        id_values.append("")  # spacer column
        row_parts.extend(id_values)
        ood_values = ["", *ood_values]  # spacer column
        row_parts.extend(ood_values)
        table_rows.append(" & ".join(row_parts) + " \\\\")

    # Build LaTeX
    n_metrics = len(metrics)
    col_spec = "l l c " + "c" * (2 * n_metrics) + "cc"

    latex_lines: list[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Distribution Shift Results}",
        "\\label{tab:distribution_shift_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        (
            f"\\multirow{{2}}{{*}}{{\\textbf{{Benchmark}}}} & \\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & "
            f"\\multirow{{2}}{{*}}{{\\textbf{{Model Size}}}} & "
            f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{In Distribution (Validation)}}}} & "
            f"\\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{Out of Distribution (Test)}}}} \\\\"
        ),
        " & & & " + " & ".join([*metrics, "", "", *metrics]) + " \\\\",
        "\\midrule",
        *table_rows,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ]
    latex_table_str = "\n".join(latex_lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "results.tex").write_text(latex_table_str)

    return latex_table_str
