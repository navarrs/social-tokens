import torch
from omegaconf import DictConfig

from scenetokens.models.criterion import (
    CausalClassification,
    Criterion,
    FocalCausalClassification,
    Reconstruction,
    TrajectoryPrediction,
)
from scenetokens.schemas.output_schemas import ModelOutput


class CausalSceneTokens(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.reconstruction_criterion = Reconstruction(config)
        self.trajpred_criterion = TrajectoryPrediction(config)

        # Classification loss is used for causal agent prediction. If 'use_focal_loss' will use FocalClassification
        # which focuses on imbalanced classes
        self.use_focal_loss = config.get("use_focal_loss", False)
        self.classification_loss = (
            FocalCausalClassification(config) if self.use_focal_loss else CausalClassification(config)
        )

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Computes the CausalSceneTokens loss which combines the quantization loss from scenario and agent
        tokenization, the trajectory prediction loss and the causal classifier loss.

        Args:
            model_output (ModelOutput): pydantic validator for model outputs.

        Returns:
            loss (torch.tensor): loss value.
        """
        reconstruction_loss = self.reconstruction_criterion(model_output)
        trajpred_loss = self.trajpred_criterion(model_output)
        classification_loss = self.classification_loss(model_output)
        return (reconstruction_loss + trajpred_loss + classification_loss).mean()
