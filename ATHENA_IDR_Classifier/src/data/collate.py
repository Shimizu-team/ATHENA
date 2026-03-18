"""Collate function and sample builder for inference."""

import numpy as np
import torch

from src.data.boltz_loader import BoltzFeatureLoader


def build_sample(
    protein_id: str,
    esm_embedding: torch.Tensor,
    boltz_loader: BoltzFeatureLoader,
    pae_max: float = 31.75,
    pde_max: float = 31.75,
) -> dict:
    """Build a sample dict for a single protein.

    Parameters
    ----------
    protein_id : str
        Protein ID.
    esm_embedding : torch.Tensor
        ESM embedding [L, d_esm].
    boltz_loader : BoltzFeatureLoader
        Boltz-2 feature loader.
    pae_max : float
        PAE normalization max.
    pde_max : float
        PDE normalization max.

    Returns
    -------
    dict
        Sample dict compatible with collate_fn.
    """
    seq_len = esm_embedding.shape[0]

    if isinstance(esm_embedding, np.ndarray):
        esm_embedding = torch.from_numpy(esm_embedding).float()
    elif esm_embedding.dtype != torch.float32:
        esm_embedding = esm_embedding.float()

    trunk_s = torch.from_numpy(boltz_loader.load_trunk_s(protein_id)).float()
    conf_s = torch.from_numpy(boltz_loader.load_conf_s(protein_id)).float()
    scalars = boltz_loader.load_scalar_features(protein_id)
    plddt = torch.from_numpy(scalars["plddt"]).float()
    pae_rowmean = torch.from_numpy(scalars["pae_rowmean"]).float() / pae_max
    pde_rowmean = torch.from_numpy(scalars["pde_rowmean"]).float() / pde_max

    min_len = min(seq_len, trunk_s.shape[0])
    esm_embedding = esm_embedding[:min_len]
    trunk_s = trunk_s[:min_len]
    conf_s = conf_s[:min_len]
    plddt = plddt[:min_len]
    pae_rowmean = pae_rowmean[:min_len]
    pde_rowmean = pde_rowmean[:min_len]

    dummy_label = torch.zeros(min_len, dtype=torch.long)

    return {
        "protein_id": protein_id,
        "esm_embedding": esm_embedding,
        "label": dummy_label,
        "trunk_s": trunk_s,
        "conf_s": conf_s,
        "plddt": plddt,
        "pae_rowmean": pae_rowmean,
        "pde_rowmean": pde_rowmean,
        "length": min_len,
    }


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length samples into a padded batch.

    Parameters
    ----------
    batch : list[dict]
        List of sample dicts from build_sample.

    Returns
    -------
    dict
        Batched tensors with padding.
    """
    protein_ids = [s["protein_id"] for s in batch]
    lengths = torch.tensor([s["length"] for s in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    B = len(batch)

    d_esm = batch[0]["esm_embedding"].shape[-1]
    d_boltz = batch[0]["trunk_s"].shape[-1]

    esm_embedding = torch.zeros(B, max_len, d_esm)
    label = torch.full((B, max_len), fill_value=-100, dtype=torch.long)
    trunk_s = torch.zeros(B, max_len, d_boltz)
    conf_s = torch.zeros(B, max_len, d_boltz)
    plddt = torch.zeros(B, max_len)
    pae_rowmean = torch.zeros(B, max_len)
    pde_rowmean = torch.zeros(B, max_len)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)

    for i, sample in enumerate(batch):
        L = sample["length"]
        esm_embedding[i, :L] = sample["esm_embedding"]
        label[i, :L] = sample["label"]
        trunk_s[i, :L] = sample["trunk_s"]
        conf_s[i, :L] = sample["conf_s"]
        plddt[i, :L] = sample["plddt"]
        pae_rowmean[i, :L] = sample["pae_rowmean"]
        pde_rowmean[i, :L] = sample["pde_rowmean"]
        padding_mask[i, :L] = False

    return {
        "protein_ids": protein_ids,
        "esm_embedding": esm_embedding,
        "label": label,
        "trunk_s": trunk_s,
        "conf_s": conf_s,
        "plddt": plddt,
        "pae_rowmean": pae_rowmean,
        "pde_rowmean": pde_rowmean,
        "lengths": lengths,
        "padding_mask": padding_mask,
    }
