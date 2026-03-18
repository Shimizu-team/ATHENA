# IDR Prediction — Inference Only

Per-residue intrinsically disordered region (IDR) prediction using protein language model (ESM-C) embeddings and Boltz-2 structure features.

This is a simplified inference-only version of the full training pipeline. It includes pre-trained model weights and pre-computed features for 100 example proteins from the CAID3-PDB dataset.

## Quick Start

```bash
# Install dependencies
pip install torch numpy biopython pyyaml

# Predict all 100 example proteins
python predict.py \
    --fasta data/example_sequences.fasta \
    --embeddings data/embeddings/example_esmc_300m.pt \
    --boltz-dir data/boltz2_predictions/predictions \
    --output-dir output/

# Predict specific proteins
python predict.py \
    --fasta data/example_sequences.fasta \
    --embeddings data/embeddings/example_esmc_300m.pt \
    --boltz-dir data/boltz2_predictions/predictions \
    --protein-ids DP03748 DP03749 \
    --output-dir output/
```

## TSUBAME Quick Start

```bash
module load miniconda
eval "$(conda shell.bash hook)"
conda activate esm-c_lora

python predict.py \
    --fasta data/example_sequences.fasta \
    --embeddings data/embeddings/example_esmc_300m.pt \
    --boltz-dir data/boltz2_predictions/predictions \
    --output-dir output/
```

## Using Your Own Proteins

To predict new proteins, you need three inputs with **matching protein IDs** across all of them:

1. **FASTA file** — standard format (`>PROTEIN_ID\nSEQUENCE`)
2. **ESM-C embeddings** — `.pt` file
3. **Boltz-2 features** — directory of per-protein subdirectories

### Step 1: Prepare your FASTA

```
>P12345
MKFLILLFNILCLFPVLAADNHGVSMQAYSRL
>Q67890
MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGG
```

The protein ID is the first word after `>`. This ID must be used consistently across all files.

### Step 2: Compute ESM-C embeddings

Install the ESM SDK (`pip install esm`), then run on a GPU:

```python
import re
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

model = ESMC.from_pretrained("esmc_300m").to("cuda")
model.eval()

sequences = {"P12345": "MKFLIL...", "Q67890": "MAEGEI..."}  # from your FASTA

embeddings = {}
for pid, seq in sequences.items():
    clean_seq = re.sub(r"[BOUZ]", "X", seq)  # replace non-standard AAs
    protein = ESMProtein(sequence=clean_seq)
    protein_tensor = model.encode(protein)
    with torch.no_grad():
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
    # IMPORTANT: slice off BOS/EOS tokens → shape [L, 960]
    embeddings[pid] = output.embeddings[0, 1:-1, :].cpu()

torch.save(embeddings, "my_esmc_300m_embeddings.pt")
```

**Output**: a `.pt` dict of `{protein_id: tensor[L, 960]}` (float32, CPU). Must use ESM-C **300m** (960-dim). ESM-C 600m will not work.

### Step 3: Compute Boltz-2 features

Install Boltz-2 (`pip install boltz`) and use the full pipeline repository's script:

```bash
# From the full training repository
python scripts/compute_boltz_embeddings.py \
    --config configs/boltz2.yaml \
    --fasta your_sequences.fasta \
    --fasta-format standard \
    --output-dir ./boltz2_output/
```

This produces the required directory structure under `boltz2_output/predictions/`:

```
predictions/
└── P12345/
    ├── embeddings/
    │   ├── trunk_s.npz       # "trunk_s" key, shape [L, 384]
    │   └── conf_s.npz        # "conf_s" key,  shape [L, 384]
    ├── plddt_P12345_model_0.npz   # "plddt" key, shape [L]
    ├── pae_P12345_model_0.npz     # "pae" key,   shape [L, L]
    └── pde_P12345_model_0.npz     # "pde" key,   shape [L, L]
```

**Note**: Extracting `trunk_s` and `conf_s` requires the custom `BoltzEmbeddingExtractor` from the full repository — standard Boltz-2 only outputs structures and confidence metrics, not the intermediate trunk/confidence representations.

### Step 4: Run inference

```bash
python predict.py \
    --fasta your_sequences.fasta \
    --embeddings my_esmc_300m_embeddings.pt \
    --boltz-dir boltz2_output/predictions \
    --output-dir output/
```

Proteins missing either ESM-C embeddings or Boltz-2 features will be skipped with a warning.

## Output Format

Predictions are saved in both JSON and TSV formats:

**JSON** (`predictions.json`):
```json
{
  "DP03748": {
    "length": 180,
    "threshold": 0.4302,
    "fraction_disordered": 0.15,
    "probabilities": [0.12, 0.87, ...],
    "predictions": [0, 1, ...]
  }
}
```

**TSV** (`predictions.tsv`):
```
protein_id  position  amino_acid  probability  prediction
DP03748     1         M           0.1234       0
DP03748     2         R           0.8765       1
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--fasta` | (required) | FASTA file with query sequences |
| `--embeddings` | (required) | Pre-computed ESM-C embeddings `.pt` file |
| `--boltz-dir` | (required) | Boltz-2 predictions directory |
| `--checkpoint` | `weights/best_model.pt` | Model checkpoint path |
| `--protein-ids` | all | Specific protein IDs to predict |
| `--output-dir` | `./output` | Output directory |
| `--batch-size` | 8 | Inference batch size |
| `--threshold` | auto (0.4302) | Classification threshold |
| `--format` | both | Output format: json, tsv, or both |

## Model Architecture

LC1: Bilinear cross-attention with dual-stream fusion.

```
ESM-C (960d) ─→ Linear ─→ BiGRU ─→ ┐
                                     ├─ BilinearCrossAttn ─→ Merge ─→ BiGRU ─→ MLP ─→ P(disorder)
Boltz-2 (771d) → Linear ─→ BiGRU ─→ ┘
```

## Directory Structure

```
├── predict.py              # Entry point
├── requirements.txt        # Minimal dependencies
├── weights/
│   ├── best_model.pt       # Pre-trained checkpoint
│   └── eval_test_metrics.json
├── data/
│   ├── example_sequences.fasta
│   ├── embeddings/
│   │   └── example_esmc_300m.pt
│   └── boltz2_predictions/
│       └── predictions/
│           ├── DP03748/
│           ├── DP03749/
│           └── ... (100 proteins)
└── src/
    ├── models/
    │   ├── screening_model.py
    │   ├── screening_fusion.py
    │   ├── blocks.py
    │   └── components.py
    └── data/
        ├── boltz_loader.py
        └── collate.py
```
