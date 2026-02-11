"""Code for the CausalSceneTokens model."""

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F  # noqa: N812

from scenetokens.models.base_model import BaseModel
from scenetokens.models.components import common
from scenetokens.schemas.output_schemas import (
    CausalOutput,
    ModelOutput,
    ScenarioEmbedding,
    TokenizationOutput,
    TrajectoryDecoderOutput,
)


class CausalSceneTokens(BaseModel):
    """CausalSceneTokens class.

    This model is similar to SceneTokens (models/scenetokens.py) but with an additional head for classifying the causal
    agents in the scene.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes the CausalSceneTokens class.

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

        #  Agent Causality Classifier + Tokenizer
        self.agent_tokenizer = self.config.agent_tokenizer
        self.causal_classifier = nn.Sequential(nn.Linear(self.config.hidden_size, 2))

        self.selu = nn.SELU(inplace=True)
        self.criterion = self.config.criterion

        self.apply(common.initialize_weights_with_xavier)
        self.print_and_get_num_params()

    def forward(self, batch: dict) -> ModelOutput:
        """Model's forward pass.
            B:  batch size
            N:  max number agents in the scene
            H:  history length
            F:  future length
            G:  GMM size
            M:  number of predicted modes
            P:  number of map polylines
            D:  number of agent features + types
            R:  number of road features + types
            Qe: number of encoded queries
            Qc: number of classifier queries
            Qd: number of decoded queries
            C:  number of classes
            E:  hidden size

        Args:
            batch (dict): dictionary containing the following input data batch:
                batch_size (int)
                input_dict (dict): dictionary containing the scenario information

        Returns:
            model_output (ModelOutput): model schema encapsulating all model outputs.
        """
        inputs = batch["input_dict"]
        history_gt_trajs = inputs["obj_trajs"]
        history_gt_trajs_mask = inputs["obj_trajs_mask"].unsqueeze(-1)
        history_ground_truth = torch.cat([history_gt_trajs, history_gt_trajs_mask], dim=-1)

        center_gt_trajs = inputs["center_gt_trajs"][..., :2]
        center_gt_trajs_mask = inputs["center_gt_trajs_mask"].unsqueeze(-1)
        # Ground truth trajectories shape: (B, F, 3) 3 is for x,y+mask
        future_ground_truth = torch.cat([center_gt_trajs, center_gt_trajs_mask], dim=-1)

        # Gathered input shapes
        #   ego_agent: (B, H, D + mask)
        #   other_agents: (B, N, H, D + mask)
        #   roads: (B, P, R, D + mask)
        ego_agent, other_agents, roads = BaseModel.gather_input(inputs)

        # Get scenario scores if available
        scenario_scores = BaseModel.gather_scores(inputs)

        # Processes and embeds historical information from self.embed() and produces a decoder embedding using a
        # trainable decoder query.
        scenario_embedder: ScenarioEmbedding = self.embed(ego_agent, other_agents, roads)

        # Classify the causal agents
        context = scenario_embedder.scenario_dec.value
        agent_context = context[:, : self.max_num_agents]

        # Classify the causal agents
        causal_tokenization_output: TokenizationOutput = self.agent_tokenizer(agent_context)

        # Get causal output and use it as a mask for the post scenario embedding
        # causal_pred shape (B, N, 2)
        # causal_idxs shape (B, N)
        causal_pred = self.causal_classifier(agent_context)
        causal_pred_probs = F.softmax(causal_pred, dim=-1)
        causal_output = CausalOutput(
            causal_gt=inputs["causal_idxs"],
            causal_pred_probs=causal_pred_probs,
            causal_pred=causal_pred_probs.argmax(dim=-1).to(torch.float),
            causal_logits=causal_pred,
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
            causal_output=causal_output,
            history_ground_truth=history_ground_truth,
            future_ground_truth=future_ground_truth,
            dataset_name=inputs["dataset_name"],
            scenario_id=inputs["scenario_id"],
            agent_ids=inputs["obj_ids"].squeeze(-1).squeeze(-1),
            scenario_scores=scenario_scores,
        )

    def embed(self, ego_agent: torch.Tensor, other_agents: torch.Tensor, roads: torch.Tensor) -> ScenarioEmbedding:
        """Encodes scenario context.
            B: batch size
            N: max number agents in the scene
            H: history length
            P: number of map polylines
            D: number of agent features + types
            M: number of map features + types

        Args:
            ego_agent (torch.tensor(B, H, D+1)): tensor containing ego-agent features (D) and mask (1) information.
            other_agents (torch.tensor(B, H, N, D+1)): tensor containing other agents features and mask information.
            roads (torch.tensor(B, P, M, D+1)): tensor containing map information.

        Returns:
            ScenarioEmbedding: a pydantic validator with
                scenario_enc (torch.tensor(B, Q, H)): tensor containing the embedded scene.
                scenario_dec (torch.tensor(B, Q, H)): tensor containing the refined embeddings of the scene.
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
        agent_masks_inv = (1.0 - agent_masks).to(torch.bool)  # only for agents.
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), other_agents_tensor), dim=2)

        batch_size, _, num_agents, _ = agents_tensor.shape
        # Encode agent information
        #   agents_emb shape: (B, H, 1+N, hidden_size)
        #   agents_posemb shape: (1, 1, 1+N, hidden_size)
        #   temporal_posembd shape: (1, H, 1, hidden_size)
        #   pos_emb shape: (1, H, 1+N, hidden_size)
        #   agents_emb shape: (B, H * (1+N), hidden_size)
        agents_emb = self.agent_encoder(agents_tensor)
        agents_emb = self.selu(agents_emb)
        pos_emb = self.agents_positional_embedding[:, :, :num_agents] + self.temporal_positional_embedding
        agents_emb = (agents_emb + pos_emb).view(batch_size, -1, self.config.hidden_size)

        # Encode map information
        #   roads tensor shape: (B, P, M, D)
        #   roads mask shape: (B, P, M)
        #   roads emb shape: (B, P * M, hidden_size)
        roads_tensor = roads[:, : self.max_num_roads, :, : self.config.map_input_size]
        roads_mask = roads[:, : self.max_num_roads, :, -1]
        roads_inv = (1.0 - roads_mask).to(torch.bool)
        road_emb = self.road_encoder(roads_tensor).view(batch_size, -1, self.config.hidden_size)
        road_emb = self.selu(road_emb)

        # Process mixed information using the perciever encoder
        #   mixed_input_features shape: (B, H * (1+N) + P * M, hidden_size)
        #   mixed_input_masks shape: (B, H * (1+N) + P * M)
        mixed_input_features = torch.concat([agents_emb, road_emb], dim=1)
        mixed_input_masks = torch.concat([agent_masks_inv.view(batch_size, -1), roads_inv.view(batch_size, -1)], dim=1)
        return self.scenario_embedder(mixed_input_features, mixed_input_masks)
