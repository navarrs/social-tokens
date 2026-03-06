"""Code for the SafeSceneTokens model."""

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F  # noqa: N812

from scenetokens.models.base_model import BaseModel
from scenetokens.models.components import common
from scenetokens.schemas.output_schemas import (
    ModelOutput,
    SafetyOutput,
    ScenarioEmbedding,
    TokenizationOutput,
    TrajectoryDecoderOutput,
)


class SafeSceneTokens(BaseModel):
    """SafeSceneTokens class.

    This model is similar to `SceneTokens` but adds an additional scenario classifier head and two safety-relevance
    classifiers derived from SafeShift's Scenario Characterization Scheme. Labels are preprocessed and inserted into
    the agent-centric scenario in `datasets/base_dataset.py`, which requires running the data processor with
    `autolabel_agents=true`.

    For more details on the ScenarioCharacterization strategy check: https://github.com/navarrs/ScenarioCharacterization.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the SafeSceneTokens model.

        Args:
            config (DictConfig): Configuration for the model.
        """
        super().__init__(config=config)
        # Agent and Road Encoders
        self.road_encoder = nn.Sequential(nn.Linear(self.config.map_input_size, self.config.hidden_size))
        self.agent_encoder = nn.Sequential(nn.Linear(self.config.agents_input_size, self.config.hidden_size))
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self.config.max_num_agents + 1), self.config.hidden_size)),
            requires_grad=True,
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_len, 1, self.config.hidden_size)),
            requires_grad=True,
        )

        # Scenario Encoder
        self.scenario_embedder = self.config.scenario_embedder

        # Scenario Classifier
        self.scenario_tokenizer = self.config.tokenizer

        # Trajectory decoder
        self.motion_decoder = self.config.motion_decoder

        # Agent safety classifier and tokenizer
        self.use_agent_tokenizer = self.config.use_agent_tokenizer
        if self.use_agent_tokenizer:
            self.agent_tokenizer = self.config.agent_tokenizer
        self.individual_safety_classifier = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.num_labels))
        self.interaction_safety_classifier = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.num_labels))

        self.selu = nn.SELU(inplace=True)
        self.criterion = self.config.criterion

        self.apply(common.initialize_weights_with_xavier)  # pyright: ignore[reportArgumentType]
        self.print_and_get_num_params()

    def forward(self, batch: dict) -> ModelOutput:
        """Run the model forward pass.

        Notation:
            B: batch size
            N: max number of non-ego agents in the scene
            H: history length
            F: future length
            M: number of predicted modes
            P: number of map polylines
            L: number of points per polyline
            Da: number of agent features
            Dr: number of road features
            Qe: number of encoded queries
            Qc: number of classifier queries
            Qd: number of decoded queries
            C: number of classes
            E: hidden size

        Args:
            batch (dict): dictionary containing the input data batch:
                batch_size (int)
                input_dict (dict): dictionary containing the scenario information

        Returns:
            ModelOutput: model outputs containing scenario embeddings, trajectory decoder outputs, tokenization outputs,
                safety outputs, ground-truth tensors, metadata, and optional scores.
        """
        inputs = batch["input_dict"]

        history_gt_trajs = inputs["obj_trajs"]
        history_gt_trajs_mask = inputs["obj_trajs_mask"].unsqueeze(-1)
        history_ground_truth = torch.cat([history_gt_trajs, history_gt_trajs_mask], dim=-1)

        center_gt_trajs = inputs["center_gt_trajs"][..., :2]
        center_gt_trajs_mask = inputs["center_gt_trajs_mask"].unsqueeze(-1)
        # Ground-truth trajectory shape: (B, F, 3), where 3 = (x, y, mask).
        future_ground_truth = torch.cat([center_gt_trajs, center_gt_trajs_mask], dim=-1).float()

        # Gathered input shapes
        #   ego_agent: (B, H, Da + 1)
        #   other_agents: (B, H, N, Da + 1)
        #   roads: (B, P, L, Dr + 1)
        ego_agent, other_agents, roads = BaseModel.gather_input(inputs)

        # Get scenario scores if available
        scenario_scores = BaseModel.gather_scores(inputs)

        # Processes and embeds historical information from self.embed() and produces a decoder embedding using a
        # trainable decoder query.
        scenario_embedder: ScenarioEmbedding = self.embed(ego_agent, other_agents, roads)

        # Select per-agent context for safety classification.
        context = scenario_embedder.scenario_dec.value
        agent_context = context[:, : self.max_num_agents]

        # Tokenize agent context if the branch is enabled.
        causal_tokenization_output = None
        if self.use_agent_tokenizer:
            causal_tokenization_output = self.agent_tokenizer(agent_context)

        # Get safety predictions
        # individual_safety_scores: (B, N, 1)
        # individual_safety_pred: (B, N, num_labels)
        individual_safety_scores = inputs["individual_agent_scores"].squeeze(-1)
        individual_safety_pred = self.individual_safety_classifier(agent_context)
        individual_safety_probs = F.softmax(individual_safety_pred, dim=-1)

        # interaction_safety_scores: (B, N, 1)
        # interaction_safety_pred: (B, N, num_labels)
        interaction_safety_scores = inputs["interaction_agent_scores"].squeeze(-1)
        interaction_safety_pred = self.interaction_safety_classifier(agent_context)
        interaction_safety_probs = F.softmax(interaction_safety_pred, dim=-1)

        safety_output = SafetyOutput(
            individual_safety_gt=individual_safety_scores,
            individual_safety_pred_probs=individual_safety_probs,
            individual_safety_pred=individual_safety_pred.argmax(dim=-1).to(torch.float),
            individual_safety_logits=individual_safety_pred,
            interaction_safety_gt=interaction_safety_scores,
            interaction_safety_pred_probs=interaction_safety_probs,
            interaction_safety_pred=interaction_safety_pred.argmax(dim=-1).to(torch.float),
            interaction_safety_logits=interaction_safety_pred,
        )

        # Classify the scenario using the selected tokenizer.
        scenario_context = context[:, self.max_num_agents :]
        tokenized_scenario: TokenizationOutput = self.scenario_tokenizer(scenario_context)
        tokens = None
        if self.config.token_conditioning:
            tokens = tokenized_scenario.token_indices.value
            tokens = F.one_hot(tokens, num_classes=self.config.tokenizer.num_tokens).detach()

        # Decode the scenario trajectories
        decoded_trajectories: TrajectoryDecoderOutput = self.motion_decoder(scenario_context, tokens)

        return ModelOutput(
            scenario_embedding=scenario_embedder,
            trajectory_decoder_output=decoded_trajectories,
            tokenization_output=tokenized_scenario,
            causal_tokenization_output=causal_tokenization_output,
            safety_output=safety_output,
            history_ground_truth=history_ground_truth,
            future_ground_truth=future_ground_truth,
            dataset_name=inputs["dataset_name"],
            scenario_id=inputs["scenario_id"],
            agent_ids=inputs["obj_ids"].squeeze(-1).squeeze(-1),
            scenario_scores=scenario_scores,
        )

    def embed(self, ego_agent: torch.Tensor, other_agents: torch.Tensor, roads: torch.Tensor) -> ScenarioEmbedding:
        """Encode scenario context from agents and map features.

        Notation:
            B: batch size
            N: max number of non-ego agents in the scene
            H: history length
            P: number of map polylines
            L: number of points per polyline
            Da: number of agent features
            Dr: number of road features
            E: hidden size

        Args:
            ego_agent (torch.Tensor): tensor with shape (B, H, Da + 1), containing ego-agent features and mask.
            other_agents (torch.Tensor): tensor with shape (B, H, N, Da + 1), containing other-agent features and masks.
            roads (torch.Tensor): tensor with shape (B, P, L, Dr + 1), containing map features and mask.

        Returns:
            ScenarioEmbedding: encoded and decoded scenario context used by the trajectory decoder.
        """
        # Ego information
        #   ego tensor shape: (B, H, D+1)
        #   ego mask shape: (B, H)
        ego_tensor = ego_agent[:, :, : self.config.agents_input_size]
        ego_masks = ego_agent[:, :, -1]
        # Other agents information
        #   agent tensor shape: (B, H, N, D+1)
        #   agent masks shape: (B, H, N)
        other_agents_tensor = other_agents[:, :, :, : self.config.agents_input_size]  # only opponent states
        other_agents_masks = other_agents[:, :, :, -1]
        # Combined agents information
        #   agents tensor shape: (B, H, 1+N, D)
        #   agents mask shape: (B, H, 1+N)
        agent_masks = torch.cat((torch.ones_like(ego_masks.unsqueeze(-1)), other_agents_masks), dim=-1)
        agent_masks_inv = (1.0 - agent_masks).to(torch.bool)  # Masked agent entries.
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), other_agents_tensor), dim=2)

        batch_size, _, num_agents, _ = agents_tensor.shape
        # Encode agent information
        #   agents_emb shape: (B, H, 1+N, hidden_size)
        #   agent_posemb shape: (1, 1, 1+N, hidden_size)
        #   temporal_posemb shape: (1, H, 1, hidden_size)
        #   pos_emb shape: (1, H, 1+N, hidden_size)
        #   agents_emb shape: (B, H * (1+N), hidden_size)
        agents_emb = self.agent_encoder(agents_tensor)
        agents_emb = self.selu(agents_emb)
        pos_emb = self.agents_positional_embedding[:, :, :num_agents] + self.temporal_positional_embedding
        agents_emb = (agents_emb + pos_emb).view(batch_size, -1, self.config.hidden_size)

        # Encode map information
        #   roads tensor shape: (B, P, L, Dr)
        #   roads mask shape: (B, P, L)
        #   roads emb shape: (B, P * L, hidden_size)
        roads_tensor = roads[:, : self.max_num_roads, :, : self.config.map_input_size]
        roads_mask = roads[:, : self.max_num_roads, :, -1]
        roads_inv = (1.0 - roads_mask).to(torch.bool)
        road_emb = self.road_encoder(roads_tensor).view(batch_size, -1, self.config.hidden_size)
        road_emb = self.selu(road_emb)

        # Process mixed information using the Perceiver encoder.
        #   mixed_input_features shape: (B, H * (1+N) + P * L, hidden_size)
        #   mixed_input_masks shape: (B, H * (1+N) + P * L)
        mixed_input_features = torch.concat([agents_emb, road_emb], dim=1)
        mixed_input_masks = torch.concat([agent_masks_inv.view(batch_size, -1), roads_inv.view(batch_size, -1)], dim=1)
        return self.scenario_embedder(mixed_input_features, mixed_input_masks)
