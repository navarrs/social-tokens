"""Code for the Reconstruction criterion."""

import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig

from scenetokens.models.criterion.base_criterion import Criterion
from scenetokens.schemas.output_schemas import ModelOutput, TokenizationOutput


class Reconstruction(Criterion):
    """Criterion for computing the reconstruction loss for both scenario and causal tokenization."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.reduction = config.get("reduction", "mean")
        self.reconstruction_weight = config.get("reconstruction_weight", 1.0)
        self.tokenization_weight = config.get("tokenization_weight", 1.0)

    def compute_tokenization_reconstruction(self, tokenization: TokenizationOutput) -> torch.Tensor:
        """Computes the toknization reconstruction loss as the MSE + quantization loss.

        Args:
            tokenization (TokenizationOutput): encapsulates the tokenization values.

        Returns:
            loss (torch.Tensor): the tokenization loss.
        """
        pre_ae_embedding = tokenization.input_embedding.value
        post_ae_embedding = tokenization.reconstructed_embedding.value

        # Enforces the decoder to produce the pre-encoded value
        loss = self.reconstruction_weight * F.mse_loss(pre_ae_embedding, post_ae_embedding, reduction=self.reduction)

        tokenization_loss = tokenization.loss
        if tokenization_loss is not None:
            return loss + self.tokenization_weight * tokenization_loss.value
        return loss

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Quantization Loss
           B: batch size
           Q: number of queries
           C: number of tokens/classes

        Args:
            model_output (ModelOutput): pydantic validator for model outputs.

        Returns:
            loss (torch.tensor): loss value.
        """
        scenario_tokenization_loss = None
        if model_output.tokenization_output is not None:
            scenario_tokenization_loss = self.compute_tokenization_reconstruction(model_output.tokenization_output)

        # If training with the tokenizers with auxiliary guidance, include the causal tokenization reconstruction.
        causal_tokenization_loss = None
        if model_output.causal_tokenization_output is not None:
            causal_tokenization_loss = self.compute_tokenization_reconstruction(model_output.causal_tokenization_output)

        assert (scenario_tokenization_loss is not None) or (causal_tokenization_loss is not None), (
            "Disable reconstruction loss if both thare are no tokenization outputs."
        )

        # TODO: handle this better
        if scenario_tokenization_loss is None:
            return causal_tokenization_loss.mean()
        if causal_tokenization_loss is None:
            return scenario_tokenization_loss.mean()

        # Final los shape: (1)
        return (scenario_tokenization_loss + causal_tokenization_loss).mean()
