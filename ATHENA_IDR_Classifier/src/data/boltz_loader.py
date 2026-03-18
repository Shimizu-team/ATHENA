"""Boltz-2 feature loader (inference-only, single-model mode)."""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class BoltzFeatureLoader:
    """Load Boltz-2 features from prediction directories.

    Parameters
    ----------
    prediction_dirs : list[str]
        Boltz-2 prediction result directories.
        Each should contain {protein_id}/ subdirectories.
    """

    def __init__(self, prediction_dirs: list):
        self._prediction_dirs = prediction_dirs
        self._protein_dir_map = self._build_protein_dir_map()
        logger.info(
            "BoltzFeatureLoader: %d directories, %d proteins",
            len(prediction_dirs), len(self._protein_dir_map),
        )

    def _build_protein_dir_map(self) -> dict:
        """Build protein_id -> directory path map."""
        protein_map = {}
        for pred_dir in self._prediction_dirs:
            if not os.path.isdir(pred_dir):
                logger.warning("Directory not found: %s", pred_dir)
                continue
            for entry in os.listdir(pred_dir):
                entry_path = os.path.join(pred_dir, entry)
                if os.path.isdir(entry_path):
                    protein_map[entry] = entry_path
        return protein_map

    def _get_protein_dir(self, protein_id: str) -> str:
        """Get directory path for a protein ID."""
        if protein_id not in self._protein_dir_map:
            raise FileNotFoundError(
                f"Boltz-2 predictions not found: {protein_id}"
            )
        return self._protein_dir_map[protein_id]

    def load_trunk_s(self, protein_id: str) -> np.ndarray:
        """Load trunk_s embedding. Returns [L, 384]."""
        protein_dir = self._get_protein_dir(protein_id)
        path = os.path.join(protein_dir, "embeddings", "trunk_s.npz")
        return np.load(path)["trunk_s"]

    def load_conf_s(self, protein_id: str) -> np.ndarray:
        """Load conf_s embedding. Returns [L, 384]."""
        protein_dir = self._get_protein_dir(protein_id)
        path = os.path.join(protein_dir, "embeddings", "conf_s.npz")
        return np.load(path)["conf_s"]

    def load_plddt(self, protein_id: str) -> np.ndarray:
        """Load pLDDT scores. Returns [L]."""
        protein_dir = self._get_protein_dir(protein_id)
        filename = f"plddt_{protein_id}_model_0.npz"
        return np.load(os.path.join(protein_dir, filename))["plddt"]

    def load_pae(self, protein_id: str) -> np.ndarray:
        """Load PAE matrix. Returns [L, L]."""
        protein_dir = self._get_protein_dir(protein_id)
        filename = f"pae_{protein_id}_model_0.npz"
        return np.load(os.path.join(protein_dir, filename))["pae"]

    def load_pde(self, protein_id: str) -> np.ndarray:
        """Load PDE matrix. Returns [L, L]."""
        protein_dir = self._get_protein_dir(protein_id)
        filename = f"pde_{protein_id}_model_0.npz"
        return np.load(os.path.join(protein_dir, filename))["pde"]

    def load_scalar_features(self, protein_id: str) -> dict:
        """Load scalar features (plddt, pae_rowmean, pde_rowmean).

        Returns
        -------
        dict
            'plddt': [L], 'pae_rowmean': [L], 'pde_rowmean': [L]
        """
        plddt = self.load_plddt(protein_id)
        pae = self.load_pae(protein_id)
        pde = self.load_pde(protein_id)
        return {
            "plddt": plddt,
            "pae_rowmean": pae.mean(axis=1),
            "pde_rowmean": pde.mean(axis=1),
        }

    def has_features(self, protein_id: str) -> bool:
        """Check if all required Boltz-2 features exist for this protein."""
        if protein_id not in self._protein_dir_map:
            return False
        protein_dir = self._protein_dir_map[protein_id]
        required = [
            os.path.join(protein_dir, "embeddings", "trunk_s.npz"),
            os.path.join(protein_dir, "embeddings", "conf_s.npz"),
            os.path.join(protein_dir, f"plddt_{protein_id}_model_0.npz"),
            os.path.join(protein_dir, f"pae_{protein_id}_model_0.npz"),
            os.path.join(protein_dir, f"pde_{protein_id}_model_0.npz"),
        ]
        return all(os.path.exists(p) for p in required)

    def available_proteins(self) -> set:
        """Return set of available protein IDs."""
        return set(self._protein_dir_map.keys())
