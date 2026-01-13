import torch
from easydict import EasyDict
from torch import nn

from scenetokens.models.base_model import BaseModel
from scenetokens.models.components import common
from scenetokens.schemas.output_schemas import ModelOutput, ScenarioEmbedding, TrajectoryDecoderOutput


class SceneTransformer(BaseModel):
    """Implementation of the SceneTransformer: https://arxiv.org/pdf/2106.08417.pdf"""

    def __init__(self, config: EasyDict) -> None:
        """Initializes the Wayformer class.

        Args:
            config (DictConfig): Configuration for the model.
        """
        super().__init__(config=config)
        # Agent and Road Encoders
        self.road_encoder = nn.Sequential(nn.Linear(config.map_input_size, config.hidden_size))
        self.agent_encoder = nn.Sequential(nn.Linear(config.agents_input_size, config.hidden_size))
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (config.max_num_agents + 1), config.hidden_size)),
            requires_grad=True,
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_len, 1, config.hidden_size)),
            requires_grad=True,
        )

        # NOTE: The scenario embedding, motion decoder and criterion and their respective hyper-parameters are directly
        # specified in the configuration (see `configs/model`) and instantiated through Hydra.

        # Scenario Encoder
        self.scenario_embedder = self.config.scenario_embedder

        # Trajectory decoder
        self.motion_decoder = self.config.motion_decoder
        self.criterion = self.config.criterion
        self.selu = nn.SELU(inplace=True)

        self.apply(common.initialize_weights_with_normal)
        self.print_and_get_num_params()

    def forward(self, batch: dict) -> ModelOutput:
        """Model's forward pass.
            B: batch size
            N:  max number agents in the scene
            H:  history length
            F:  future length
            G:  GMM size
            M:  number of predicted modes
            P:  number of map polylines
            D:  number of agent features + types
            R:  number of road features + types
            Qe: number of encoded queries
            Qd: number of decoded queries
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

        # Processes and embeddes historical information from self.encode() and produces a decoder embedding using a
        # the trainable decoder query.
        scenario_embedding: ScenarioEmbedding = self.embed(ego_agent, other_agents, roads)

        # Decode the scenario trajectories
        context = scenario_embedding.scenario_dec.value
        decoded_trajectories: TrajectoryDecoderOutput = self.motion_decoder(context)

        return ModelOutput(
            scenario_embedding=scenario_embedding,
            trajectory_decoder_output=decoded_trajectories,
            history_ground_truth=history_ground_truth,
            future_ground_truth=future_ground_truth,
            dataset_name=inputs["dataset_name"],
            scenario_id=inputs["scenario_id"],
            agent_ids=inputs["obj_ids"].squeeze(-1).squeeze(-1),
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
            other_agents (torch.tensor(B, H, N, D+1)): tensor containg other agents features and mask information.
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
        agents_emb = self.agent_encoder(agents_tensor)
        agents_emb = self.selu(agents_emb)
        pos_emb = self.agents_positional_embedding[:, :, :num_agents] + self.temporal_positional_embedding

        # Encode map information
        #   roads tensor shape: (B, P, M, D)
        #   roads mask shape: (B, P, M)
        #   roads emb shape: (B, P * M, hidden_size)
        roads_tensor = roads[:, : self.max_num_roads, :, : self.config.map_input_size]
        roads_mask = roads[:, : self.max_num_roads, :, -1]
        roads_inv = (1.0 - roads_mask).to(torch.bool).view(batch_size, -1)
        road_emb = self.road_encoder(roads_tensor).view(batch_size, -1, self.config.hidden_size)
        road_emb = self.selu(road_emb)

        # Make dimensions compatible with the FactorizedEmbedder
        # agents_emb: (B, 1+N, H, hidden_size)
        # agent_masks_inv: (B, 1+N, H)
        agents_emb = (agents_emb + pos_emb).transpose(1, 2)
        agent_masks_inv = agent_masks_inv.transpose(1, 2)
        return self.scenario_embedder(agents_emb, agent_masks_inv, road_emb, roads_inv)
