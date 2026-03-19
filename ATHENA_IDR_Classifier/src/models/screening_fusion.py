"""Dual-stream fusion module (D-SB).

Separates ESM embeddings (sequence stream) from Boltz-2 features (structure stream).
Output: {"x_seq": [B, L, d_model], "x_struct": [B, L, d_model]}
"""

import torch
import torch.nn as nn


def _get_raw_scalars(batch: dict) -> torch.Tensor:
    """Concatenate scalar features into [B, L, n_scalars]."""
    components = [batch["plddt"], batch["pae_rowmean"], batch["pde_rowmean"]]
    return torch.stack(components, dim=-1)


class DSBFusion(nn.Module):
    """Dual-stream fusion: seq=LN(Linear(ESM, d)), struct=LN(Linear(cat[trunk, conf, scalars], d)).

    Parameters
    ----------
    d_esm : int
        ESM embedding dimension.
    d_boltz : int
        Boltz trunk/conf dimension.
    d_model : int
        Output dimension.
    n_scalars : int
        Number of scalar features (3).
    """

    def __init__(
        self,
        d_esm: int = 960,
        d_boltz: int = 384,
        d_model: int = 256,
        n_scalars: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.seq_proj = nn.Linear(d_esm, d_model)
        self.seq_norm = nn.LayerNorm(d_model)
        struct_dim = d_boltz * 2 + n_scalars
        self.struct_proj = nn.Linear(struct_dim, d_model)
        self.struct_norm = nn.LayerNorm(d_model)

    def forward(self, batch: dict) -> dict:
        x_seq = self.seq_norm(self.seq_proj(batch["esm_embedding"]))
        raw_scalars = _get_raw_scalars(batch)
        struct_in = torch.cat([
            batch["trunk_s"], batch["conf_s"], raw_scalars,
        ], dim=-1)
        x_struct = self.struct_norm(self.struct_proj(struct_in))
        return {"x_seq": x_seq, "x_struct": x_struct}
