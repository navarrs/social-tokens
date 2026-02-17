import torch
from omegaconf import DictConfig

from scenetokens.models.criterion import Criterion, Reconstruction, TrajectoryPrediction
from scenetokens.schemas.output_schemas import ModelOutput


class SceneTokens(Criterion):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.reconstruction_criterion = Reconstruction(config)
        self.trajpred_criterion = TrajectoryPrediction(config)

    def forward(self, model_output: ModelOutput) -> torch.Tensor:
        """Computes the Quantized SceneTokens loss which combines the quantization loss from the scenario tokenizer and
        the trajectory prediction loss from the scenario decoder.

        Args:
            model_output (ModelOutput): pydantic validator for model outputs.

        Returns:
            loss (torch.tensor): loss value.
        """
        reconstruction_loss = self.reconstruction_criterion(model_output)
        trajpred_loss = self.trajpred_criterion(model_output)
        return (reconstruction_loss + trajpred_loss).mean()
