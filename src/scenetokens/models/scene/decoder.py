"""Code for the TrajectoryDecoder component of the models. The trajectory decoder takes in encoded scenario context and
and learned scenario tokens to produce multi-modal trajectory predictions for all agents in the scene.
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from scenetokens.models.components import common
from scenetokens.schemas.output_schemas import TrajectoryDecoderOutput


class TrajectoryDecoder(nn.Module):
    """TrajectoryDecoder class."""

    def __init__(
        self,
        num_modes: int,
        decoding_length: int,
        hidden_size: int,
        num_tokens: int | None = None,
        *,
        token_conditioning: bool = False,
    ) -> None:
        """Initializes the Trajectory Decoder
            NOTE: naive just refers to the tokenizer not having access to supervision signals.

        Args:
            num_modes (int): number of trajectories to predict.
            decoding_length (int): number of steps to decode.
            hidden_size (int): hidden size of the learnable queries.
            num_tokens (int): size of the token input.
            token_conditioning (bool): if True it will increase the input size of the decoders to conditon on tokens.
        """
        super().__init__()
        self.num_modes = num_modes
        self.token_conditioning = token_conditioning
        self.num_tokens = num_tokens

        self.num_bivdist_params = 5  # (µ_x, µ_y, sig_x, sig_y, p)
        self.decoding_length = decoding_length

        # Condition the trajectory decoder on the learned tokens
        if self.token_conditioning:
            assert num_tokens is not None, "num_tokens must be provided if token_conditioning is True."
            self.decoder_input_size = hidden_size + num_tokens
        else:
            self.decoder_input_size = hidden_size
        self.decoder_output = nn.Sequential(
            nn.Linear(self.decoder_input_size, self.num_bivdist_params * self.decoding_length)
        )
        self.decoder_probs = nn.Sequential(nn.Linear(self.decoder_input_size, 1))
        self.apply(common.initialize_weights_with_xavier)

    def forward(self, context: torch.Tensor, tokens: torch.Tensor | None = None) -> TrajectoryDecoderOutput:
        """Decodes encoded context.
           B: batch size
           Q: number of queries
           H: hidden size
           C: number of tokens/classes
           M: number of modes
           F: number of future/decoding steps

        Args:
            context (torch.tensor(B, Q, H)): tensor containing the scene encoded information.
            tokens (torch.tensor(B, Q, C)): tensor containing the scene encoded information.

        Returns:
            TrajectoryDecoderOutput: pydantic validator for the trajectory decoder with:
                decoded_trajectories (torch.tensor(B, M, F, 5)): decoded trajectories.
                mode_probabilities (torch.tensor(B, M)): probability scores for each mode.
        """
        batch_size, _, _ = context.shape

        # Concatenate token information if the decoder is conditioned on the scenario tokens.
        if self.token_conditioning and tokens is not None:
            assert tokens.shape[-1] == self.num_tokens, (
                f"Token shape[-1] {tokens.shape[-1]} != num tokens {self.num_tokens}"
            )
            # context shape: (B, Q, H+C)
            context = torch.cat([context[:, : self.num_modes], tokens], dim=2)

        # The trajectory decoder further processes M queries to produce M different predicted modes for all future (F)
        # steps in the scenario.
        #   dec_trajs shape: (B, M, F * 5) -> (B, M, F, 5)
        decoded_trajectories = self.decoder_output(context[:, : self.num_modes])
        decoded_trajectories = decoded_trajectories.reshape(batch_size, self.num_modes, self.decoding_length, -1)

        # The probability decoder further produces a probability score for each of the predicted modes, representing
        # the likelihood of the predicted trajectory.
        # Mode prediction shape: (B, num_modes, 1) -> (B, num_modes)
        mode_probabilities = self.decoder_probs(context[:, : self.num_modes])
        mode_probabilities = mode_probabilities.reshape(batch_size, self.num_modes)

        if len(np.argwhere(np.isnan(decoded_trajectories.detach().cpu().numpy()))) > 1:
            error_message = "Found nans during decoding step."
            raise ValueError(error_message)

        return TrajectoryDecoderOutput(
            decoded_trajectories=decoded_trajectories,
            mode_probabilities=F.softmax(mode_probabilities, dim=-1),
            mode_logits=mode_probabilities,
        )
