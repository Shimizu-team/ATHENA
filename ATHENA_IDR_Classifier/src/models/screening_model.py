"""ScreeningModel: Bilinear cross-attention IDR prediction model (LC1 architecture).

Inference-only version. No LoRA or CRF support.
"""

import logging

import torch
import torch.nn as nn

from src.models.blocks import (
    BiGRUBlock,
    BilinearCrossAttnOnlyBlock,
    MergeBlock,
)
from src.models.components import BilinearCrossAttention
from src.models.components import PositionalEncoding
from src.models.screening_fusion import DSBFusion

logger = logging.getLogger(__name__)


ACTIVATION_REGISTRY = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
}


def _build_classifier(
    d_model: int,
    n_layers: int = 1,
    activation: str = "gelu",
    dropout: float = 0.1,
) -> nn.Sequential:
    """Build configurable MLP classification head.

    Parameters
    ----------
    d_model : int
        Input dimension.
    n_layers : int
        Number of hidden layers.
    activation : str
        Activation function name.
    dropout : float
        Dropout rate.

    Returns
    -------
    nn.Sequential
        Classification head.
    """
    act_fn = ACTIVATION_REGISTRY[activation]

    layers = []
    in_dim = d_model
    for i in range(n_layers):
        out_dim = in_dim // 2
        layers.extend([
            nn.Linear(in_dim, out_dim),
            act_fn(),
            nn.Dropout(dropout),
        ])
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def _get_nested(d: dict, dotted_key: str, default=None):
    """Get a value from a nested dict using dot-separated key."""
    keys = dotted_key.split(".")
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


class ScreeningModel(nn.Module):
    """IDR prediction model with bilinear cross-attention (LC1 architecture).

    Architecture: biGRU(seq) + biGRU(struct) -> [PE] -> BilinearCrossAttn -> Merge -> biGRU -> Classifier

    Parameters
    ----------
    config : dict
        Configuration dictionary with nested keys under 'model.*'.
    """

    def __init__(self, config: dict):
        super().__init__()

        def _get(key, default=None):
            return _get_nested(config, key, default)

        # Model hyperparameters
        self.d_model = _get("model.d_model", 512)
        d_esm = _get("model.d_esm", 960)
        d_boltz = _get("model.d_boltz", 384)
        num_heads = _get("model.num_heads", 8)
        dropout = _get("model.dropout", 0.1)
        gru_hidden_dim = _get("model.gru_hidden_dim", 256)
        gru_num_layers = _get("model.gru_num_layers", 1)
        bilinear_rank = _get("model.bilinear_rank", 32)

        # Fusion: D-SB (dual-stream)
        n_scalars = _get("model.n_scalars", 3)
        self.fusion = DSBFusion(
            d_esm=d_esm,
            d_boltz=d_boltz,
            d_model=self.d_model,
            n_scalars=n_scalars,
        )

        # Pre-GRU (each stream)
        self.pre_gru_seq = BiGRUBlock(
            self.d_model, gru_hidden_dim, gru_num_layers, dropout,
        )
        self.pre_gru_struct = BiGRUBlock(
            self.d_model, gru_hidden_dim, gru_num_layers, dropout,
        )

        # Positional Encoding (optional)
        pe_type = _get("model.pe_type", None)
        if pe_type is not None:
            max_len = _get("model.max_position_len", 5000)
            self.pos_encoder_seq = PositionalEncoding(
                self.d_model, dropout, max_len, pe_type,
            )
            self.pos_encoder_struct = PositionalEncoding(
                self.d_model, dropout, max_len, pe_type,
            )
        else:
            self.pos_encoder_seq = None
            self.pos_encoder_struct = None

        # Bilinear Cross-Attention
        self.cross_attn_block = BilinearCrossAttnOnlyBlock(
            self.d_model, num_heads, rank=bilinear_rank, dropout=dropout,
        )

        # Merge -> single stream
        self.merge = MergeBlock(self.d_model)

        # Post-GRU
        self.post_gru = BiGRUBlock(
            self.d_model, gru_hidden_dim, gru_num_layers, dropout,
        )

        # Final norm
        self.final_norm = nn.LayerNorm(self.d_model)

        # Classification head
        head_n_layers = _get("model.head.n_layers", 1)
        head_activation = _get("model.head.activation", "silu")
        self.classifier = _build_classifier(
            self.d_model, n_layers=head_n_layers,
            activation=head_activation, dropout=dropout,
        )

        # Log parameter count
        total = sum(p.numel() for p in self.parameters())
        logger.info("ScreeningModel [LC1 + D-SB]: %d params", total)

    def forward(self, batch: dict) -> dict:
        """Forward pass.

        Parameters
        ----------
        batch : dict
            Batch from collate_fn.

        Returns
        -------
        dict
            {"logits": [B, L, 1]}
        """
        padding_mask = batch.get("padding_mask", None)
        lengths = batch.get("lengths", None)
        fusion_out = self.fusion(batch)

        x_seq = fusion_out["x_seq"]
        x_struct = fusion_out["x_struct"]

        # Pre-GRU (each stream independently)
        x_seq = self.pre_gru_seq(x_seq, lengths)
        x_struct = self.pre_gru_struct(x_struct, lengths)

        # Positional Encoding
        if self.pos_encoder_seq is not None:
            x_seq = self.pos_encoder_seq(x_seq)
            x_struct = self.pos_encoder_struct(x_struct)

        # Cross-Attention (bidirectional)
        x_seq, x_struct = self.cross_attn_block(x_seq, x_struct, padding_mask)

        # Merge -> single stream
        x = self.merge(x_seq, x_struct)

        # Post-GRU
        x = self.post_gru(x, lengths)

        # Final norm + classifier
        x = self.final_norm(x)
        logits = self.classifier(x)  # [B, L, 1]

        return {"logits": logits}
