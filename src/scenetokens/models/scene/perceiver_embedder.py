"""Code for the PerceiverEmbedder component of the models. The PerceiverEmbedder takes in raw scenario features and
produces encoded and decoded scenario embeddings using the PerceiverIO architecture.
"""

import torch
from torch import nn

from scenetokens.models.components import common
from scenetokens.models.components.perceiver_io import PerceiverDecoder, PerceiverEncoder, TrainableQueryProvider
from scenetokens.schemas.output_schemas import ScenarioEmbedding


class PerceiverEmbedder(nn.Module):
    """PerceiverEmbedder class."""

    def __init__(
        self, num_encoder_queries: int, num_decoder_queries: int, hidden_size: int, query_init_scale: float
    ) -> None:
        """Initializes the Wayformer class.

        Args:
            num_encoder_queries (int): number of encoder queries for the PerceiverEncoder
            num_decoder_queries (int): number of encoder queries for the PerceiverDecoder
            hidden_size (int): size of the embedding space.
            query_init_scale (float): intialization scale value.
        """
        super().__init__()
        # Scenario Encoder
        self.perceiver_encoder = PerceiverEncoder(
            num_encoder_queries,
            hidden_size,
            num_cross_attention_qk_channels=hidden_size,
            num_cross_attention_v_channels=hidden_size,
            num_self_attention_qk_channels=hidden_size,
            num_self_attention_v_channels=hidden_size,
        )
        # Scenario Decoder
        scenario_decoder_query = TrainableQueryProvider(
            num_queries=num_decoder_queries,
            num_query_channels=hidden_size,
            init_scale=query_init_scale,
        )
        self.perceiver_decoder = PerceiverDecoder(scenario_decoder_query, hidden_size)
        self.apply(common.initialize_weights_with_xavier)

    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> ScenarioEmbedding:
        """Embeds the scenario features.
           B: batch size
           Q: number of queries
           H: hidden size
           D: scenario features

        Args:
            features (torch.tensor(B, H, D+1)): tensor containing scenario feature information.
            masks (torch.tensor(B, H, N, D+1)): tensor containg scenario mask information.

        Returns:
            ScenarioEmbedding: pydantic validator for the trajectory decoder with:
                scenario_enc (torch.tensor(B, Q, H)): encoded scenario using PerceiverIO
                scenario_dec (torch.tensor(B, Q, H)): decoded scenario using PerceiverIO
        """
        scenario_enc = self.perceiver_encoder(features, masks)
        scenario_dec = self.perceiver_decoder(scenario_enc)
        return ScenarioEmbedding(scenario_enc=scenario_enc, scenario_dec=scenario_dec)
