"""This file includes relevant model output schemas.
NOTE: The output values don't have a specific dimension yet, since the exact output values are still in development.
"""

from typing import Any

from pydantic import BaseModel
from pydantic_tensor import Tensor
from pydantic_tensor.backend.torch import TorchTensor
from pydantic_tensor.types import Float, Int


class TokenizationOutput(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the output values of the Tokenizers defined in 'models/scene/tokenizers'.
    NOTE: The output values don't have a specific dimension yet, since the exact output values are still in development.

    Attributes:
        token_probabilities (TorchTensor(Float)): the probability mass over the tokens classes.
        token_indices (TorchTensor(Int)): the selected token classes.
        input_embedding (TorchTensor(Float)): the embedding values before encoding and/or quantizing.
        reconstructed_embedding (TorchTensor(Float) | None): the reconstructed embedding.
        quantized_embedding (TorchTensor(Float) | None): the discretized embedding if using quantization.
        loss (TorchTensor(Float) | None): the loss value if tokenizer is using a dedicated loss function.
    """

    num_tokens: int = 0
    token_probabilities: Tensor[TorchTensor, Any, Float] | None = None
    token_indices: Tensor[TorchTensor, Any, Int]
    input_embedding: Tensor[TorchTensor, Any, Float]
    reconstructed_embedding: Tensor[TorchTensor, Any, Float] | None = None
    quantized_embedding: Tensor[TorchTensor, Any, Float] | None = None
    loss: Tensor[TorchTensor, Any, Float] | None = None


class TrajectoryDecoderOutput(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the output values of the MotionDecoder defined in 'models/scene/decoder.py'.
    NOTE: The output values don't have a specific dimension yet, since the exact output values are still in development.

    Attributes:
        decoded_trajectories (TorchTensor(Float)): a set M decoded trajectories data.
        mode_probabilities (TorchTensor(Int)): the probability of each mode's (M) decoded trajectory.
    """

    decoded_trajectories: Tensor[TorchTensor, Any, Float]
    mode_probabilities: Tensor[TorchTensor, Any, Float]
    mode_logits: Tensor[TorchTensor, Any, Float] | None = None


class CausalOutput(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates causal output values.

    Attributes:
        causal_gt (TorchTensor(Float)): contains values in {0, 1} where 0 means not causal and 1 means causal.
        causal_pred (TorchTensor(Float)): contains values in {0, 1} indicating the predicted causal class.
        causal_pred_probs (TorchTensor(Float)): contains values in [0, 1] representing probability of being causal.
        causal_logits (TorchTensor(Float)): contains the causal logit values before applying a Sigmoid activation.
    """

    causal_gt: Tensor[TorchTensor, Any, Float]
    causal_pred: Tensor[TorchTensor, Any, Float]
    causal_pred_probs: Tensor[TorchTensor, Any, Float]
    causal_logits: Tensor[TorchTensor, Any, Float] | None = None


class ScenarioEmbedding(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates the output values of the ScenarioEmbedded defined in 'models/scene/embedder.py'.

    Attributes:
        scenario_enc (TorchTensor(Float)): an encoded embedding of a scenario if pre-processing is performed.
        scenario_dec (TorchTensor(Int)): a decoded embedding of a scenario
    """

    scenario_enc: Tensor[TorchTensor, Any, Float] | None = None
    scenario_dec: Tensor[TorchTensor, Any, Float]


class ModelOutput(BaseModel):  # pyright: ignore[reportUntypedBaseClass]
    """Encapsulates all model outputs decoded values of a scenario defined in 'models'.

    Attributes:
        motion_decoder_output (MotionDecoderOutput): an encoded embedding of a scenario if pre-processing is performed.
        tokenization_output (TokenizationOutput): a decoded embedding of a scenario
    """

    scenario_embedding: ScenarioEmbedding
    trajectory_decoder_output: TrajectoryDecoderOutput | None = None
    tokenization_output: TokenizationOutput | None = None
    causal_output: CausalOutput | None = None
    causal_tokenization_output: TokenizationOutput | None = None

    # Meta Information
    history_ground_truth: Tensor[TorchTensor, Any, Float]
    future_ground_truth: Tensor[TorchTensor, Any, Float]
    dataset_name: list[str]
    scenario_id: list[str]
    agent_ids: Tensor[TorchTensor, Any, Int]
    agent_scores: Tensor[TorchTensor, Any, Float] | None = None
    scene_score: Tensor[TorchTensor, Any, Float] | None = None
