import pickle
from pathlib import Path

import numpy as np
from characterization.schemas import (
    AgentData,
    DynamicMapData,
    Scenario,
    ScenarioMetadata,
    StaticMapData,
    TracksToPredict,
)
from characterization.utils.common import AgentType
from omegaconf import DictConfig

from scenetokens.datasets.base_dataset import BaseDataset
from scenetokens.utils import data_utils, pylogger


_LOGGER = pylogger.get_pylogger(__name__)


class WaymoDataset(BaseDataset):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

    def _deserialize_scenario(self, path: Path) -> dict:
        with path.open("rb") as f:
            return pickle.load(f)

    def _repack_scenario(self, scenario: dict) -> Scenario:
        """Repacks an input scenario in dictionary format into a GlobalScenario."""
        # Repack agent information from input scenario
        agent_data = self.repack_agent_data(scenario["track_infos"])

        # Repack static map information from input scenario
        static_map_data = self.repack_static_map_data(scenario["map_infos"])

        # Repack dynamic map information
        dynamic_map_data = self.repack_dynamic_map_data(scenario["dynamic_map_infos"])

        timestamps = scenario["timestamps_seconds"][: self.total_steps]
        freq = min(np.round(1 / np.mean(np.diff(timestamps))).item(), 10.0)

        # Select tracks to predict
        ego_vehicle_index = scenario["sdc_track_index"]
        tracks_to_predict = self.select_tracks_to_predict(scenario["tracks_to_predict"], ego_vehicle_index)

        # Repack meta information
        ego_index = scenario["sdc_track_index"]
        metadata = ScenarioMetadata(
            scenario_id=scenario["scenario_id"],
            timestamps_seconds=timestamps,
            frequency_hz=freq,
            current_time_index=self.current_time_idx,
            ego_vehicle_id=agent_data.agent_ids[ego_index],
            ego_vehicle_index=scenario["sdc_track_index"],
            track_length=self.total_steps,
            objects_of_interest=scenario["objects_of_interest"],
            dataset=f"waymo-{self.subset_data_tag}",
        )

        return Scenario(
            metadata=metadata,
            agent_data=agent_data,
            tracks_to_predict=tracks_to_predict,
            static_map_data=static_map_data,
            # NOTE: the model is not currently using dynamic map data.
            dynamic_map_data=dynamic_map_data,
        )

    def repack_agent_data(self, agent_data: dict) -> AgentData:
        """Packs agent information from Waymo format to AgentData format.

        Args:
            agent_data (dict): dictionary containing Waymo actor data:
                'object_id': indicating each agent IDs
                'object_type': indicating each agent type
                'trajs': tensor(num_agents, num_timesteps, num_features) containing each agents kinematic information.

        Returns:
            AgentData: pydantic validator encapsulating agent information.
        """
        # Mask timestamps based on desired sampling interval
        trajectory_sample_interval = self.config.get("trajectory_sample_interval", 1)
        frequency_mask = data_utils.generate_mask(self.past_len - 1, self.total_steps, trajectory_sample_interval)
        agent_data["trajs"][..., -1] *= frequency_mask[np.newaxis]

        object_types = [AgentType[n] for n in agent_data["object_type"]]
        return AgentData(
            agent_ids=agent_data["object_id"],
            agent_types=object_types,
            agent_trajectories=agent_data["trajs"],
        )

    @staticmethod
    def get_polyline_ids(polyline: dict, key: str) -> np.ndarray:
        """Extracts polyline indices from the polyline dictionary."""
        return np.array([value["id"] for value in polyline[key]], dtype=np.int32)

    @staticmethod
    def get_speed_limit_mph(polyline: dict, key: str) -> np.ndarray:
        """Extracts speed limit in mph from the polyline dictionary."""
        return np.array([value["speed_limit_mph"] for value in polyline[key]], dtype=np.float32)

    @staticmethod
    def get_polyline_idxs(polyline: dict, key: str) -> np.ndarray | None:
        polyline_idxs = np.array(
            [[value["polyline_index"][0], value["polyline_index"][1]] for value in polyline[key]],
            dtype=np.int32,
        )
        if polyline_idxs.shape[0] == 0:
            return None
        return polyline_idxs

    def repack_static_map_data(self, static_map_data: dict | None) -> StaticMapData | None:
        """Packs static map information from Waymo format to StaticMapData format.

        Args:
            static_map_data (dict | None): dictionary containing Waymo static scenario data:
                'all_polylines': contains all road data in the form of polylines in addition to specific road types (
                    'lane', 'road_line', 'road_edge', 'crosswalk', 'speed_bump', 'stop_sign')

        Returns:
            StaticMapData | None: pydantic validator encapsulating static map information.
        """
        if static_map_data is None:
            return None

        map_polylines = static_map_data["all_polylines"].astype(np.float32)  # shape: [N, 3] or [N, 3, 2]
        return StaticMapData(
            map_polylines=map_polylines,
            lane_ids=WaymoDataset.get_polyline_ids(static_map_data, "lane") if "lane" in static_map_data else None,
            lane_speed_limits_mph=WaymoDataset.get_speed_limit_mph(static_map_data, "lane")
            if "lane" in static_map_data
            else None,
            lane_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "lane")
            if "lane" in static_map_data
            else None,
            road_line_ids=WaymoDataset.get_polyline_ids(static_map_data, "road_line")
            if "road_line" in static_map_data
            else None,
            road_line_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "road_line")
            if "road_line" in static_map_data
            else None,
            road_edge_ids=WaymoDataset.get_polyline_ids(static_map_data, "road_edge")
            if "road_edge" in static_map_data
            else None,
            road_edge_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "road_edge")
            if "road_edge" in static_map_data
            else None,
            crosswalk_ids=WaymoDataset.get_polyline_ids(static_map_data, "crosswalk")
            if "crosswalk" in static_map_data
            else None,
            crosswalk_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "crosswalk")
            if "crosswalk" in static_map_data
            else None,
            speed_bump_ids=WaymoDataset.get_polyline_ids(static_map_data, "speed_bump")
            if "speed_bump" in static_map_data
            else None,
            speed_bump_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "speed_bump")
            if "speed_bump" in static_map_data
            else None,
            stop_sign_ids=WaymoDataset.get_polyline_ids(static_map_data, "stop_sign")
            if "stop_sign" in static_map_data
            else None,
            stop_sign_polyline_idxs=WaymoDataset.get_polyline_idxs(static_map_data, "stop_sign")
            if "stop_sign" in static_map_data
            else None,
            stop_sign_lane_ids=[
                stop_sign["lane_ids"] for stop_sign in static_map_data.get("stop_sign", {"lane_ids": []})
            ],
        )

    def repack_dynamic_map_data(self, dynamic_map_data: dict) -> DynamicMapData:
        """Packs dynamic map information from Waymo format to DynamicMapData format.

        Args:
            dynamic_map_data (dict): dictionary containing Waymo dynamic scenario data:
                'stop_points': traffic light stopping points.
                'lane_id': IDs of the lanes where the traffic light is.
                'state': state of the traffic light (e.g., red, etc).

        Returns:
            DynamicMapData: pydantic validator encapsulating static map information.
        """
        stop_points = dynamic_map_data["stop_point"][: self.total_steps]
        lane_id = dynamic_map_data["lane_id"][: self.total_steps]
        states = dynamic_map_data["state"][: self.total_steps]
        num_dynamic_stop_points = len(stop_points)
        if num_dynamic_stop_points == 0:
            stop_points = None
            lane_id = None
            states = None
        return DynamicMapData(stop_points=stop_points, lane_ids=lane_id, states=states)

    def select_tracks_to_predict(
        self, tracks_to_predict: dict, ego_vehicle_index: int | None = None
    ) -> TracksToPredict:
        if self.config.only_train_on_ego:
            if ego_vehicle_index is None:
                error_message = f"Invalid ego_vehicle_index value: {ego_vehicle_index}"
                raise ValueError(error_message)
            return TracksToPredict(
                track_index=[ego_vehicle_index], difficulty=[0], object_type=[AgentType.TYPE_VEHICLE]
            )
        return TracksToPredict(
            track_index=tracks_to_predict["track_index"],
            difficulty=tracks_to_predict["difficulty"],
            object_type=tracks_to_predict["object_type"],
        )

    def load_as_open_scenario(self, path: Path) -> Scenario:
        """Get the processed scenario in agent-centric format for HDF5 caching."""
        raw_scenario = self._deserialize_scenario(path)
        return self._repack_scenario(raw_scenario)
