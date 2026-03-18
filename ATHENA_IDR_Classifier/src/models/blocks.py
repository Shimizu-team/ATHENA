"""Building blocks: BilinearCrossAttnOnlyBlock, BiGRUBlock, MergeBlock."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.components import BilinearCrossAttention


class BilinearCrossAttnOnlyBlock(nn.Module):
    """Bidirectional bilinear cross-attention only (no self-attention or FFN).

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
        self.norm_seq = nn.LayerNorm(d_model)
        self.cross_attn_seq = BilinearCrossAttention(d_model, num_heads, rank, dropout)
        self.norm_struct = nn.LayerNorm(d_model)
        self.cross_attn_struct = BilinearCrossAttention(d_model, num_heads, rank, dropout)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_struct: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> tuple:
        """Bidirectional cross-attention forward pass.

        Parameters
        ----------
        x_seq : torch.Tensor
            Sequence stream. Shape: [B, L, d_model]
        x_struct : torch.Tensor
            Structure stream. Shape: [B, L, d_model]
        padding_mask : torch.Tensor, optional
            Shape: [B, L], True=padding position.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (x_seq, x_struct), each Shape: [B, L, d_model]
        """
        h_seq = self.norm_seq(x_seq)
        h_seq = self.cross_attn_seq(h_seq, x_struct, padding_mask=padding_mask)
        x_seq = x_seq + h_seq

        h_struct = self.norm_struct(x_struct)
        h_struct = self.cross_attn_struct(h_struct, x_seq, padding_mask=padding_mask)
        x_struct = x_struct + h_struct

        return x_seq, x_struct


class BiGRUBlock(nn.Module):
    """Bidirectional GRU block with residual connection + LayerNorm.

    Parameters
    ----------
    d_model : int
        Model dimension.
    gru_hidden_dim : int
        GRU hidden dimension (per direction).
    gru_num_layers : int
        Number of GRU layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(gru_hidden_dim * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape: [B, L, d_model]
        lengths : torch.Tensor, optional
            Shape: [B], actual lengths per sample.

        Returns
        -------
        torch.Tensor
            Shape: [B, L, d_model]
        """
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False,
            )
            gru_out, _ = self.gru(packed)
            gru_out, _ = pad_packed_sequence(
                gru_out, batch_first=True, total_length=x.size(1),
            )
        else:
            gru_out, _ = self.gru(x)

        h = self.proj(gru_out)
        return self.norm(x + h)


class MergeBlock(nn.Module):
    """Merge two streams into one: cat[x_seq, x_struct] -> Linear -> LayerNorm.

    Parameters
    ----------
    d_model : int
        Dimension per stream and output.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_struct: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_seq : torch.Tensor
            Sequence stream. Shape: [B, L, d_model]
        x_struct : torch.Tensor
            Structure stream. Shape: [B, L, d_model]

        Returns
        -------
        torch.Tensor
            Shape: [B, L, d_model]
        """
        return self.norm(self.proj(torch.cat([x_seq, x_struct], dim=-1)))
