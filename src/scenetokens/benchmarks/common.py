"""Shared helpers used across benchmark creation modules."""

import shutil
from enum import Enum
from pathlib import Path

from scenetokens import utils


_LOGGER = utils.get_pylogger(__name__)


class Benchmark(Enum):
    CAUSAL_AGENTS = "causal_agents"
    EGO_SAFESHIFT = "ego_safeshift"
    SAFESHIFT = "safeshift"
    ENVIRONMENTS = "environments"


_DEFAULT_SPLITS: list[str] = ["training", "validation", "testing"]


def create_split_dirs(output_path: Path, splits: list[str] = _DEFAULT_SPLITS) -> None:
    """Creates split subdirectories under output_path.

    Args:
        output_path: Root directory under which split subdirs are created.
        splits: List of split names. Defaults to ["training", "validation", "testing"].
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
        Mapping from scenario IDs to output file paths.
    """
    return {scenario_id: output_data_path / split / scenario_id for scenario_id in scenario_ids}


def copy_scenario(
    scenario_id: str,
    input_scenario_mapping: dict[str, Path],
    output_scenario_mapping: dict[str, Path],
) -> None:
    """Copies a scenario file from the input location to the output location.

    Args:
        scenario_id: Scenario ID used as the key in both mappings.
        input_scenario_mapping: Maps scenario IDs to source file paths.
        output_scenario_mapping: Maps scenario IDs to destination file paths.
    """
    if scenario_id not in input_scenario_mapping:
        _LOGGER.warning("Scenario %s not found in input mapping.", scenario_id)
        return
    shutil.copy2(input_scenario_mapping[scenario_id], output_scenario_mapping[scenario_id])
