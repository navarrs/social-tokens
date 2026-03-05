"""Vector-quantization tokenizer for scenario embeddings.

Reference: https://arxiv.org/abs/1711.00937
"""

import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize

from scenetokens.schemas.output_schemas import TokenizationOutput


class QuantizedTokenizer(nn.Module):
    """Discretize scenario query embeddings with a VQ codebook."""

    def __init__(  # noqa: PLR0913
        self,
        num_queries: int,
        hidden_size: int,
        num_tokens: int,
        quantization_weight: float = 0.25,
        commitment_weight: float = 0.25,
        reduction_factor: int = 2,
        *,
        normalize: bool = False,
    ) -> None:
        """Initialize the quantized tokenizer.

        Args:
            num_queries (int): Number of input scenario queries.
            hidden_size (int): Channel size of each query embedding.
            num_tokens (int): Number of codebook entries.
            quantization_weight (float): Stored weight for quantization-related regularization in downstream losses.
            commitment_weight (float): Commitment weight used by the VQ layer.
            reduction_factor (int): Compression factor for latent channels.
            normalize (bool): If `True`, apply `tanh` before encoding and after decoding.
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.num_queries = num_queries

        # Loss weights
        self.quantization_weight = quantization_weight
        self.commitment_weight = commitment_weight

        # Tokenizer follows a VQ-VAE-style bottleneck.
        reduced_hidden_size = hidden_size // reduction_factor

        # Optionally normalize and then encode the input embedding.
        if normalize:
            self.prepare_input = nn.Tanh()
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, reduced_hidden_size)
            )

            # Decode back to the original hidden size.
            self.decoder = nn.Sequential(
                nn.Linear(reduced_hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.Tanh()
            )
        else:
            self.prepare_input = nn.Identity()
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, reduced_hidden_size)
            )

            # Decode back to the original hidden size.
            self.decoder = nn.Sequential(
                nn.Linear(reduced_hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # Quantization codebook
        self.quantize = VectorQuantize(
            dim=reduced_hidden_size,
            codebook_size=num_tokens,
            commitment_weight=self.commitment_weight,
        )

    def forward(self, x: torch.Tensor) -> TokenizationOutput:
        """Tokenize scene embeddings with vector quantization.

        Notation:
            B: Batch size.
            Q: Number of scenario queries.
            H: Hidden size before quantization.
            Hr: Reduced hidden size used by the VQ codebook.
            C: Number of token classes (codebook entries).

        Args:
            x (torch.Tensor): Input embeddings with shape `(B, Q, H)`.

        Returns:
            TokenizationOutput: A container with:
                token_indices (torch.Tensor): Selected codebook indices with shape `(B, Q)`.
                input_embedding (torch.Tensor): Pre-quantization embeddings with shape `(B, Q, H)`.
                reconstructed_embedding (torch.Tensor): Decoded embeddings with shape `(B, Q, H)`.
                quantized_embedding (torch.Tensor): Quantized latent embeddings with shape `(B, Q, Hr)`.
                loss (torch.Tensor): VQ loss tensor returned by `VectorQuantize`.
        """
        # Input shape: (B, Q, H).
        batch_size, num_queries, _ = x.shape

        # Apply optional normalization before quantization.
        x = self.prepare_input(x)

        # Encode to reduced latent channels. Encoder output shape: (B, Q, Hr).
        embedding_prequant = self.encoder(x)

        # Quantize the latent embedding.
        # Quantized output shape: (B, Q, Hr).
        # Indices shape: (B, Q).
        # Quantization loss shape: scalar tensor.
        quantized_embedding, token_indices, vq_loss = self.quantize(embedding_prequant)

        # Decode the quantized embedding back to hidden size H.
        # Decoder output shape: (B, Q, H).
        embedding_postquant = self.decoder(quantized_embedding)
        embedding_postquant = embedding_postquant.view(batch_size, num_queries, -1)

        return TokenizationOutput(
            num_tokens=self.num_tokens,
            token_indices=token_indices,
            input_embedding=x,
            reconstructed_embedding=embedding_postquant,
            quantized_embedding=quantized_embedding,
            loss=vq_loss,
        )
