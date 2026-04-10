"""Shared helpers used across benchmark creation modules."""

import shutil
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

from scenetokens import utils


_LOGGER = utils.get_pylogger(__name__)


class Benchmark(Enum):
    CAUSAL_AGENTS = "causal_agents"
    EGO_SAFESHIFT = "ego_safeshift"
    SAFESHIFT = "safeshift"
    ENVIRONMENTS = "environments"


_DEFAULT_SPLITS: tuple[str, ...] = ("training", "validation", "testing")


def create_split_dirs(output_path: Path, splits: Iterable[str] = _DEFAULT_SPLITS) -> None:
    """Creates split subdirectories under output_path.

    Args:
        output_path: Root directory under which split subdirs are created.
        splits: Split names. Defaults to ("training", "validation", "testing").
    """
    for split in splits:
        (output_path / split).mkdir(parents=True, exist_ok=True)
        _LOGGER.info("Creating benchmark subdir: %s", output_path / split)


def collect_scenario_filepaths(data_path: Path) -> list[Path]:
    """Returns all .pkl scenario filepaths under data_path, excluding info files.

    Args:
        data_path: Root directory to search recursively.

    Returns:
        List of .pkl filepaths whose stem does not contain 'infos'.
    """
    return [fp for fp in data_path.rglob("*.pkl") if "infos" not in fp.stem]


def get_scenario_mapping(
    scenario_ids: list[str],
    output_data_path: Path,
    split: str,
) -> dict[str, Path]:
    """Creates a mapping from scenario IDs to output file paths.

    Args:
        scenario_ids: List of scenario IDs.
        output_data_path: Path to the output data.
        split: Data split (e.g., 'training', 'validation', 'testing').

    Returns:
        Mapping from scenario IDs to output file paths (each ending in .pkl).
    """
    return {scenario_id: output_data_path / split / f"{scenario_id}.pkl" for scenario_id in scenario_ids}


def copy_scenario(
    scenario_id: str,
    input_filepath: Path,
    output_filepath: Path,
    *,
    unlink_source: bool = False,
) -> None:
    """Copies a scenario file from input_filepath to output_filepath.

    Args:
        scenario_id: Scenario ID, used only for warning messages.
        input_filepath: Source file path.
        output_filepath: Destination file path.
        unlink_source: If True, delete the source file after a successful copy. Defaults to False.
    """
    if not input_filepath.exists():
        _LOGGER.warning("Scenario %s not found at %s.", scenario_id, input_filepath)
        return
    shutil.copy2(input_filepath, output_filepath)
    if unlink_source:
        input_filepath.unlink()
