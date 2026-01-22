from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from characterization.utils.common import AgentType
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from scenetokens.schemas.output_schemas import ModelOutput, ScenarioScores
from scenetokens.utils import metric_utils, save_cache
from scenetokens.utils.constants import KALMAN_DIFFICULTY, MILLION, TrajectoryType


class BaseModel(LightningModule, ABC):
    """Scenario Model wrapper based on: https://lightning.ai/docs/pytorch/latest/common/lightning_module.html"""

    def __init__(self, config: DictConfig) -> None:
        """Initializes the BaseModel class.

        Args:
            config (DictConfig): Configuration for the model.
        """
        super().__init__()
        self.config = config
        self.future_len = config.future_len
        self.past_len = config.past_len
        self.num_modes = config.num_modes
        self.max_num_agents = config.max_num_agents
        self.num_bivdist_params = 5  # (µ_x, µ_y, sig_x, sig_y, p)

        self.max_points_per_lane = config.max_points_per_lane
        self.max_num_roads = config.max_num_roads

        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.monitor = config.monitor
        self.criterion = config.criterion

        self.cache_batch = config.cache_batch
        self.cache_every_batch_idx = config.cache_every_batch_idx
        self.batch_cache_path = Path(config.batch_cache_path)
        self.sample_selection = config.get("sample_selection", False)
        self.batch_cache_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def forward(self, batch: dict) -> ModelOutput:
        """Model's forward pass.

        Args:
            batch (dict): dictionary containing the following input data batch:
                batch_size (int)
                input_dict (dict): dictionary containing the scenario information

        Returns:
            model_output (ModelOutput): model schema encapsulating all model outputs.
        """

    def configure_optimizers(self) -> dict:
        """Define and configure model optimizers and learning rate schedulers.

        Returns:
            optimizer_config (dict): dictionary containing the optimizer and LR scheduler specified within the model's
            configuration.
        """
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor,
            },
        }

    def print_and_get_num_params(self) -> int:
        """Returns the number of parameters in the model.

        Returns:
            total_parameters (int): total number of parameters in the model.
        """
        trainable_parameters = 0
        nontrainable_parameters = 0
        for parameter in self.parameters():
            num_params = parameter.numel()
            if parameter.requires_grad:
                trainable_parameters += num_params
            else:
                nontrainable_parameters += num_params

        total_parameters = trainable_parameters + nontrainable_parameters
        print("Total number of parameters %.2fM" % (total_parameters / MILLION))
        print("\tTrainable parameters: %.2fM" % (trainable_parameters / MILLION))
        print("\tNon-Trainable parameters: %.2fM" % (nontrainable_parameters / MILLION))
        return total_parameters

    def model_step(self, batch: dict, batch_idx: int, status: str) -> torch.Tensor:
        """Takes a model step, calculates the loss value and logs model outputs.

        Args:
            batch (dict): dictionary containing the batch information from the collate function.
            batch_idx (int): index of current batch.
            status (str): modeling status ('train', 'val', 'test')

        Returns:
            loss (torch.Tensor): a tensor containing the model's loss.
        """
        model_output = self.forward(batch)
        loss = self.criterion(model_output)
        self.log_info(batch["input_dict"], model_output, loss, status=status)
        if self.sample_selection:
            cache_filepath = Path(self.batch_cache_path, f"train_batch_{batch_idx}.pkl")
            save_cache(model_output, cache_filepath)
        elif status != "train" and self.cache_batch and batch_idx % self.cache_every_batch_idx:
            cache_filepath = Path(self.batch_cache_path, f"{status}_batch_{batch_idx}.pkl")
            save_cache(model_output, cache_filepath)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Performs a model step on a training batch.

        Args:
            batch (dict): dictionary containing the batch information from the collate function.
            batch_idx (int): index of current batch.

        Output
        ------
            loss (torch.Tensor): model's loss value.
        """
        return self.model_step(batch, batch_idx, status="train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Performs a model step on a validation batch.

        Args:
            batch (dict): dictionary containing the batch parameters.
            batch_idx (int): index of current batch.

        Output
        ------
            loss (torch.Tensor): model's loss value.
        """
        return self.model_step(batch, batch_idx, status="val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Performs a model step on a testing batch.

        Args:
            batch (dict): dictionary containing the batch parameters.
            batch_idx (int): index of current batch.

        Output
        ------
            loss (torch.Tensor): model's loss value.
        """
        return self.model_step(batch, batch_idx, status="test")

    @staticmethod
    def gather_input(inputs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gathers scenario information and masks for the ego-agent, other agents and map information.

        Args:
            inputs (dict): dictionary containing scenario data according to the collate_fn() from
                scenetokens/datasets/base_dataset.py

        Returns:
            dict[str, torch.Tensor]: dictionary containing model input as:
                'ego_in' (torch.tensor(B, H, D+1)): tensor containing ego-agent information
                'agents_in' (torch.tensor(B, H, N, D+1)): tensor containing other agents information
                'roads' (torch.tensor())
        """
        # Tensor dimensions:
        #   B -> batch size
        #   N -> max number agents in the scene
        #   H -> history length
        #   P -> number of map polylines
        #   D -> number of agent features + types
        #   M -> number of map features + types

        # Agent trajectories shape: (B, N, H, D)
        agents_in = inputs["obj_trajs"]
        # Index to predict shape: (B, 1, H, D)
        index_to_predict = inputs["track_index_to_predict"].view(-1, 1, 1, 1).repeat(1, 1, *agents_in.shape[-2:])
        # Ego agent input shape: (B, H, D)
        ego_in = torch.gather(agents_in, 1, index_to_predict).squeeze(1)

        # Agent masks shape: (B, N, H)
        agents_mask = inputs["obj_trajs_mask"]
        # Index to predict shape: (B, 1, H, D)
        index_to_predict = inputs["track_index_to_predict"].view(-1, 1, 1).repeat(1, 1, agents_mask.shape[-1])
        # Ego mask shape: (B, H)
        ego_mask = torch.gather(agents_mask, 1, index_to_predict).squeeze(1)

        # Combined agent trajectories shape: (B, N, H, D+1) -> (B, H, N, D+1)
        agents_in = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        # Combined ego trajectory shape: (B, H, D+1)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)

        # Road information shape: (B, P, M, D)
        roads = inputs["map_polylines"]
        # Road mask shape: (B, P, M)
        roads_mask = inputs["map_polylines_mask"].unsqueeze(-1)
        # Combined road information shape: (B, P, M, D+1)
        roads = torch.cat([roads, roads_mask], dim=-1)
        return ego_in, agents_in, roads

    @staticmethod
    def gather_scores(inputs: dict) -> ScenarioScores | None:
        """Gathers scenario scores for individual and interaction safety.

        Args:
            inputs (dict): dictionary containing scenario data according to the collate_fn() from
                scenetokens/datasets/base_dataset.py
        Returns:
            scenario_scores (ScenarioScores | None): pydantic validator containing scenario scores
                information.
        """
        scenario_scores = None
        if "individual_agent_scores" in inputs and "interaction_agent_scores" in inputs:
            individual_safety_scores = inputs["individual_agent_scores"].squeeze(-1)
            interaction_safety_scores = inputs["interaction_agent_scores"].squeeze(-1)
            scenario_scores = ScenarioScores(
                individual_agent_scores=individual_safety_scores,
                individual_scenario_score=inputs["individual_scene_scores"],
                interaction_agent_scores=interaction_safety_scores,
                interaction_scenario_score=inputs["interaction_scene_scores"],
            )
        return scenario_scores

    @staticmethod
    def split_by_trajectory_type(trajectory_types: np.ndarray, metric_dict: dict, metric_keys: list[str]) -> dict:
        """Splits a metric dictionary by trajectory type.

        Args:
            trajectory_types (np.ndarray(B)): array containing the trajectory types for each ego-agent in the batch.
            metric_dict (dict): dictionary containing the computed metrics.
            metric_keys (list[str]): list of metrics to split by trajectory type.

        Returns:
            split_dict (dict): a dictionary of metrics split by trajectory type.
        """
        split_dict = {}
        for traj_type in list(TrajectoryType):
            traj_type_idx = np.where(trajectory_types == traj_type.value)[0]
            if len(traj_type_idx) > 0:
                for key in metric_keys:
                    split_dict["traj_type/" + traj_type.name.lower() + "_" + key] = metric_dict[key][traj_type_idx]
        return split_dict

    @staticmethod
    def split_by_kalman_difficulty(kalman_difficulties: np.ndarray, metric_dict: dict, metric_keys: list[str]) -> dict:
        """Splits a metric dictionary by Kalman difficulty type.

        Args:
            kalman_difficulties (np.ndarray(B)): array containing the kalman_difficulties for each ego in the batch.
            metric_dict (dict): dictionary containing the computed metrics.
            metric_keys (list[str]): list of metrics to split by kalman_difficulties.

        Returns:
            split_dict (dict): a dictionary of metrics split by kalman_difficulties.
        """
        split_dict = {}
        for kalman_bucket, (low, high) in KALMAN_DIFFICULTY.items():
            in_range = np.logical_and(low <= kalman_difficulties, kalman_difficulties < high)
            kalman_diff_idx = np.where(in_range)[0]
            if len(kalman_diff_idx) > 0:
                for key in metric_keys:
                    split_dict["kalman/" + kalman_bucket + "_" + key] = metric_dict[key][kalman_diff_idx]
        return split_dict

    @staticmethod
    def split_by_agent_type(agent_types: np.ndarray, metric_dict: dict, metric_keys: list[str]) -> dict:
        """Splits a metric dictionary by agent type.

        Args:
            agent_types (np.ndarray(B)): array containing the agent types for each ego-agent in the batch.
            metric_dict (dict): dictionary containing the computed metrics.
            metric_keys (list[str]): list of metrics to split by agent type.

        Returns:
            split_dict (dict): a dictionary of metrics split by agent type.
        """
        split_dict = {}
        for agent_type in list(AgentType):
            agent_type_idx = np.where(agent_types == agent_type.value)[0]
            if len(agent_type_idx) > 0:
                for key in metric_keys:
                    split_dict["agent_types/" + agent_type.name.lower() + "_" + key] = metric_dict[key][agent_type_idx]
        return split_dict

    @staticmethod
    def split_by_dataset_name(dataset_names: np.ndarray, metric_dict: dict, metric_keys: list[str]) -> dict:
        """Splits a metric dictionary by dataset names.

        Args:
            dataset_names (np.ndarray(B)): array containing the dataset names for each ego-agent in the batch.
            metric_dict (dict): dictionary containing the computed metrics.
            metric_keys (list[str]): list of metrics to split by dataset names.

        Returns:
            split_dict (dict): a dictionary of metrics split by dataset names.
        """
        split_dict = {}
        unique_dataset_names = np.unique(dataset_names)
        for dataset_name in unique_dataset_names:
            dataset_idx = np.argwhere([n == str(dataset_name) for n in dataset_names])[:, 0]
            for key in metric_keys:
                split_dict[dataset_name + "/" + key] = metric_dict[key][dataset_idx]
        return split_dict

    @staticmethod
    def compute_metrics(inputs: dict, outputs: ModelOutput) -> dict[str, npt.NDArray[np.float64]]:
        """Computes task metrics on model outputs.

        Args:
            inputs (dict): dictionary containing input scenario information
            outputs (ModelOutput): pydantic validator with model output information.

        Returns:
            _type_: _description_
        """
        gt_traj = inputs["center_gt_trajs"].unsqueeze(1)  # .transpose(0, 1).unsqueeze(0)
        gt_traj_mask = inputs["center_gt_trajs_mask"].unsqueeze(1)
        center_gt_final_valid_idx = inputs["center_gt_final_valid_idx"]

        # Report trajectory metrics if trajectory outputs are available
        metric_dict = {}
        trajectory_ouput = outputs.trajectory_decoder_output
        if trajectory_ouput is not None:
            predicted_traj = trajectory_ouput.decoded_trajectories.value
            predicted_prob = trajectory_ouput.mode_probabilities.value

            batch_size, num_modes = predicted_prob.shape

            # Calculate metrics
            center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, num_modes, 1).to(torch.int64)
            # average and final displacement errors between the predicted and ground-truth trajectories
            ade, fde = metric_utils.compute_displacement_error(
                predicted_traj[:, :, :, :2],
                gt_traj[:, :, :, :2],
                gt_traj_mask,
                center_gt_final_valid_idx,
            )
            min_ade, _ = ade.min(axis=-1)
            min_fde, best_fde_idx = fde.min(axis=-1)

            # miss rate measues whether the final displacement is greater than a specified threshold
            miss_rate_all_modes = metric_utils.compute_miss_rate(fde)
            miss_rate_best_mode = metric_utils.compute_miss_rate(min_fde.unsqueeze(-1))

            predicted_prob = predicted_prob[torch.arange(batch_size), best_fde_idx]
            brier_fde = min_fde + torch.square(1 - predicted_prob)

            metric_dict = {
                "minADE6": min_ade.cpu().detach().numpy(),
                "minFDE6": min_fde.cpu().detach().numpy(),
                "missRate6": miss_rate_all_modes.cpu().detach().numpy(),
                "missRate": miss_rate_best_mode.cpu().detach().numpy(),
                "brierFDE": brier_fde.cpu().detach().numpy(),
            }

        # Report causal metrics if causal outputs are available
        causal_output = outputs.causal_output
        if causal_output is not None:
            labels = causal_output.causal_gt.value
            predictions = causal_output.causal_pred.value
            tp, tn, fp, fn = metric_utils.compute_binary_confusion_matrix(predictions, labels)
            metric_dict["causalTP"] = tp.cpu().detach().numpy()
            metric_dict["causalTN"] = tn.cpu().detach().numpy()
            metric_dict["causalFP"] = fp.cpu().detach().numpy()
            metric_dict["causalFN"] = fn.cpu().detach().numpy()

            precision, recall, f1_score = metric_utils.compute_accuracy(labels, predictions)
            metric_dict["precision"] = precision.cpu().detach().numpy()
            metric_dict["recall"] = recall.cpu().detach().numpy()
            metric_dict["f1Score"] = f1_score.cpu().detach().numpy()

        safety_output = outputs.safety_output
        if safety_output is not None:
            indvidual_labels = safety_output.individual_safety_gt.value.squeeze(-1)
            indvidual_predictions = safety_output.individual_safety_pred.value
            num_classes = safety_output.individual_safety_pred_probs.value.shape[-1]
            precision, recall, f1_score = metric_utils.compute_multiclass_accuracy(
                indvidual_labels, indvidual_predictions, num_classes
            )
            metric_dict["individualPrecision"] = precision.cpu().detach().numpy()
            metric_dict["individualRecall"] = recall.cpu().detach().numpy()
            metric_dict["individualF1Score"] = f1_score.cpu().detach().numpy()

            interaction_labels = safety_output.interaction_safety_gt.value.squeeze(-1)
            interaction_predictions = safety_output.interaction_safety_pred.value
            num_classes = safety_output.interaction_safety_pred_probs.value.shape[-1]
            precision, recall, f1_score = metric_utils.compute_multiclass_accuracy(
                interaction_labels, interaction_predictions, num_classes
            )
            metric_dict["interactionPrecision"] = precision.cpu().detach().numpy()
            metric_dict["interactionRecall"] = recall.cpu().detach().numpy()
            metric_dict["interactionF1Score"] = f1_score.cpu().detach().numpy()

        # TODO: review these metrics
        # If training a model with a scenario classification head, log perplexity and mutual information.
        # tokenization_outputs = outputs.tokenization_output
        # if tokenization_outputs is not None:
        #     scenario_class_probs = tokenization_outputs.token_probabilities.value
        #     assert scenario_class_probs is not None, "Token probabilities is None"
        #     # NOTE: the selected class does not have a ground truth value.
        #     selected_scenario_class = scenario_class_probs.argmax(dim=-1)

        #     # Perplexity will measure the uncertainty between the output probabilities w.r.t the selected class.
        #     perplexity = metric_utils.compute_perplexity(scenario_class_probs, selected_scenario_class)
        #     loss_dict["perplexity"] = perplexity.cpu().detach().numpy()

        #     # Mutual information will measure how related are the scenario probability distributions and their classes
        #     num_classes = scenario_class_probs.shape[-1]
        #     scenario_class_onehot = F.one_hot(selected_scenario_class, num_classes)
        #     mutual_information = metric_utils.compute_mutual_information(
        #         scenario_class_probs, scenario_class_onehot, normalize=True
        #     )
        #     loss_dict["mutualInformation"] = mutual_information.cpu().detach().numpy()
        return metric_dict

    def log_info(self, inputs: dict, outputs: ModelOutput, loss: torch.Tensor, status: str = "train") -> None:
        """Logs metric values after training and validation steps.

        Args:
            inputs (dict): dictionary containing input scenario information
            outputs (ModelOutput): pydantic validator with model output information.
            loss: (torch.Tensor): model's loss value.
            status (str): whether the info is logged from training step or validation step.
        """
        # Split based on dataset
        metric_dict = BaseModel.compute_metrics(inputs, outputs)
        metric_list = list(metric_dict.keys())

        # Separate the loss dictionary by sub-dataset name
        dataset_names = inputs["dataset_name"]
        new_dict = BaseModel.split_by_dataset_name(dataset_names, metric_dict, metric_list)
        metric_dict.update(new_dict)

        if status == "val" and self.config.get("eval", False):
            # Separate the loss dictionary by trajectory type
            trajectory_types = inputs["trajectory_type"].cpu().numpy()
            new_dict = BaseModel.split_by_trajectory_type(trajectory_types, metric_dict, metric_list)
            metric_dict.update(new_dict)

            # Split loss dict by kalman difficulty
            kalman_difficulties = inputs["kalman_difficulty"][:, -1].cpu().numpy()
            new_dict = BaseModel.split_by_kalman_difficulty(kalman_difficulties, metric_dict, metric_list)
            metric_dict.update(new_dict)

            agent_types = inputs["center_objects_type"]
            new_dict = BaseModel.split_by_agent_type(agent_types, metric_dict, metric_list)
            metric_dict.update(new_dict)

        # Take mean for each key but store original length before (useful for aggregation)
        size_dict = {key: len(value) for key, value in metric_dict.items()}
        metric_dict = {key: np.mean(value) for key, value in metric_dict.items()}

        # Log information
        total_loss = loss.cpu().detach().item()
        self.log(f"losses/{status}", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in metric_dict.items():
            batch_size = size_dict[k]
            self.log(status + "/" + k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # TODO: add support for visualization of scenarios
        # if self.local_rank == 0 and status == 'val' and batch_idx == 0:
        #     img = visualization.visualize_prediction(batch, prediction)
        #     wandb.log({"prediction": [wandb.Image(img)]})
