"""Transformer attention mechanism.

Role: Core
This module implements multi-head self-attention for transformer models.
"""

import torch
import torch.nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple heads
    for transformer architectures.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-head attention.

        Complexity: O(B*N²*D) where B=batch, N=sequence length, D=dimension

        Args:
            query: Query tensor [B, N, D]
            key: Key tensor [B, N, D]
            value: Value tensor [B, N, D]
            mask: Optional attention mask [B, N, N]

        Returns:
            Attention output [B, N, D]
        """
        batch_size = query.size(0)

        # Linear projections in batch from d_model => h x d_k
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)

        # Concatenate heads and apply final linear
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)

        return output

    @staticmethod
    def create_causal_mask(seq_len: int, device: str = 'cpu') -> torch.Tensor:
        """Create causal mask for autoregressive attention.

        Time: O(N²) where N is sequence length

        Args:
            seq_len: Sequence length
            device: Device to place tensor on

        Returns:
            Causal mask [N, N]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0

    @property
    def parameter_count(self) -> int:
        """Count total parameters. O(1)"""
        return sum(p.numel() for p in self.parameters())


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model

        # Compute positional encodings once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Complexity: O(B*N*D)

        Args:
            x: Input tensor [B, N, D]

        Returns:
            Encoded tensor [B, N, D]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
