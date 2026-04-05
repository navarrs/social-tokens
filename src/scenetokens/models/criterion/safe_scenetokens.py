"""Code for the SafeSceneTokens loss function."""

import torch
from omegaconf import DictConfig

from scenetokens.models.criterion import (
    Criterion,
    FocalSafetyClassification,
    Reconstruction,
    SafetyClassification,
    TrajectoryPrediction,
)
from scenetokens.schemas.output_schemas import ModelOutput


class SafeSceneTokens(Criterion):
    """Criterion for SafeSceneTokens."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.tokenize = config.get("tokenize", True)
        if self.tokenize:
            self.reconstruction_criterion = Reconstruction(config)

        self.predict_trajectories = config.get("predict_trajectories", True)
        if self.predict_trajectories:
            self.trajpred_criterion = TrajectoryPrediction(config)

        # Classification losses for safety-agent predictions.
        config.safety_type = "individual"
        config.classification_weight = config.get("individual_classification_weight", 1.0)
        self.individual_safety_loss = (
            FocalSafetyClassification(config) if config.get("use_focal_loss", False) else SafetyClassification(config)
        )

        config.safety_type = "interaction"
        config.classification_weight = config.get("interaction_classification_weight", 1.0)
        self.interaction_safety_loss = (
            FocalSafetyClassification(config) if config.get("use_focal_loss", False) else SafetyClassification(config)
        )

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Compute the SafeSceneTokens loss.

        Notation:
            L_rec: reconstruction loss
            L_traj: trajectory prediction loss
            L_ind: individual safety classification loss
            L_int: interaction safety classification loss

        Args:
            model_output (ModelOutput): Structured model outputs.

        Returns:
            torch.Tensor: Scalar loss value ``L_rec + L_traj + L_ind + L_int``.
        """
        individual_safety_loss = self.individual_safety_loss(model_output)
        interaction_safety_loss = self.interaction_safety_loss(model_output)
        safety_loss = individual_safety_loss + interaction_safety_loss

        if self.tokenize:
            reconstruction_loss = self.reconstruction_criterion(model_output)

            if self.predict_trajectories:
                return (reconstruction_loss + self.trajpred_criterion(model_output) + safety_loss).mean()

            return (reconstruction_loss + safety_loss).mean()

        if self.predict_trajectories:
            return (self.trajpred_criterion(model_output) + safety_loss).mean()

        return safety_loss.mean()
