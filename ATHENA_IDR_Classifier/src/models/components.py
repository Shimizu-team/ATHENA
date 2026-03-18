"""Core components: Positional Encoding and Bilinear Cross-Attention."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding (sinusoidal / learned / mixed).

    Parameters
    ----------
    d_model : int
        Model dimension.
    dropout : float
        Dropout rate.
    max_len : int
        Maximum sequence length.
    pe_type : str
        "sinusoidal": fixed sinusoidal PE only.
        "learned": learnable PE only.
        "mixed": fixed + learnable blend (default).
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        pe_type: str = "mixed",
    ):
        super().__init__()
        self.pe_type = pe_type
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

        if pe_type in ("learned", "mixed"):
            self.learned_pe = nn.Parameter(
                torch.randn(1, max_len, d_model) * 0.02
            )

        if pe_type == "mixed":
            self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Shape: [B, L, d_model]

        Returns
        -------
        torch.Tensor
            Shape: [B, L, d_model]
        """
        seq_len = x.size(1)

        if self.pe_type == "sinusoidal":
            x = x + self.pe[:, :seq_len, :]
        elif self.pe_type == "learned":
            x = x + self.learned_pe[:, :seq_len, :]
        else:  # mixed
            alpha = torch.sigmoid(self.alpha_raw)
            combined_pe = (
                alpha * self.pe[:, :seq_len, :]
                + (1 - alpha) * self.learned_pe[:, :seq_len, :]
            )
            x = x + combined_pe

        return self.dropout(x)


class BilinearCrossAttention(nn.Module):
    """Low-rank bilinear cross-attention.

    Q from x_q, K/V from x_kv.
    Score: score(q,k) = (Q @ U) @ (K @ V)^T / sqrt(rank)

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    rank : int
        Bilinear projection rank.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rank = rank

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.U = nn.Parameter(torch.randn(num_heads, self.d_k, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(num_heads, self.d_k, rank) * 0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute bilinear cross-attention.

        Parameters
        ----------
        x_q : torch.Tensor
            Query stream. Shape: [B, L, d_model]
        x_kv : torch.Tensor
            Key/Value stream. Shape: [B, L, d_model]
        padding_mask : torch.Tensor, optional
            Shape: [B, L], True=padding position (KV side).

        Returns
        -------
        torch.Tensor
            Shape: [B, L, d_model]
        """
        B, L, _ = x_q.shape

        Q = self.w_q(x_q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x_kv).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x_kv).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        Q_proj = torch.matmul(Q, self.U)  # [B, H, L, rank]
        K_proj = torch.matmul(K, self.V)  # [B, H, L, rank]
        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(self.rank)

        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, -1e4)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(context)
