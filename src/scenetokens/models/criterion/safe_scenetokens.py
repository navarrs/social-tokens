import torch
from omegaconf import DictConfig

from scenetokens.models.criterion import (
    Criterion,
    Reconstruction,
    SafetyClassification,
    TrajectoryPrediction,
)
from scenetokens.schemas.output_schemas import ModelOutput


class SafeSceneTokens(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.reconstruction_criterion = Reconstruction(config)
        self.trajpred_criterion = TrajectoryPrediction(config)

        # Classification loss is used for safety agent predictions
        config.safety_type = "individual"
        config.classification_weight = config.get("individual_classification_weight", 1.0)
        self.individual_safety_loss = SafetyClassification(config)

        config.safety_type = "interaction"
        config.classification_weight = config.get("interaction_classification_weight", 1.0)
        self.interaction_safety_loss = SafetyClassification(config)

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Computes the SafeSceneTokens loss which combines the quantization loss from scenario and agent tokenization,
        the trajectory prediction loss and the agent's individual and interaction safety classifier loss.

        Args:
            model_output (ModelOutput): pydantic validator for model outputs.

        Returns:
            loss (torch.tensor): loss value.
        """
        reconstruction_loss = self.reconstruction_criterion(model_output)
        trajpred_loss = self.trajpred_criterion(model_output)
        individual_safety_loss = self.individual_safety_loss(model_output)
        interaction_safety_loss = self.interaction_safety_loss(model_output)
        return (reconstruction_loss + trajpred_loss + individual_safety_loss + interaction_safety_loss).mean()
