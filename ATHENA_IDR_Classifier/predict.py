"""IDR (Intrinsically Disordered Region) prediction from FASTA.

Loads a trained model checkpoint and pre-computed features to predict
per-residue disorder probabilities for protein sequences.

Usage:
    # Predict all 100 example proteins (using bundled data)
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
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.boltz_loader import BoltzFeatureLoader
from src.data.collate import build_sample, collate_fn
from src.models.screening_model import ScreeningModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_fasta(fasta_path: str) -> dict:
    """Parse a simple FASTA file into {protein_id: sequence}.

    Parameters
    ----------
    fasta_path : str
        Path to FASTA file.

    Returns
    -------
    dict[str, str]
        protein_id -> amino acid sequence.
    """
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    return sequences


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="IDR prediction from FASTA using pre-computed features",
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="Path to FASTA file with query sequences",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to pre-computed pLM embeddings .pt file",
    )
    parser.add_argument(
        "--boltz-dir",
        type=str,
        required=True,
        help="Boltz-2 predictions directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: weights/best_model.pt)",
    )
    parser.add_argument(
        "--protein-ids",
        nargs="+",
        type=str,
        default=None,
        help="Specific protein IDs to predict (default: all in FASTA)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for predictions (default: ./output)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (default: use optimal from training)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "tsv", "both"],
        default="both",
        help="Output format (default: both)",
    )
    return parser.parse_args()


def main():
    """Main prediction pipeline."""
    args = parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.checkpoint is None:
        args.checkpoint = os.path.join(script_dir, "weights", "best_model.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load sequences
    logger.info("Loading FASTA: %s", args.fasta)
    all_sequences = parse_fasta(args.fasta)

    if args.protein_ids:
        sequences = {
            pid: all_sequences[pid]
            for pid in args.protein_ids
            if pid in all_sequences
        }
        missing = set(args.protein_ids) - set(sequences.keys())
        if missing:
            logger.warning("Protein IDs not found in FASTA: %s", sorted(missing))
    else:
        sequences = all_sequences

    logger.info("Sequences to predict: %d", len(sequences))

    # 2. Load pre-computed embeddings
    logger.info("Loading pre-computed embeddings: %s", args.embeddings)
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)

    # 3. Build Boltz loader
    boltz_loader = BoltzFeatureLoader(prediction_dirs=[args.boltz_dir])

    # 4. Build samples
    samples = []
    skipped = []
    for pid in sequences:
        if pid not in embeddings:
            logger.warning("No embedding for %s, skipping", pid)
            skipped.append(pid)
            continue
        if not boltz_loader.has_features(pid):
            logger.warning("No Boltz-2 features for %s, skipping", pid)
            skipped.append(pid)
            continue
        sample = build_sample(pid, embeddings[pid], boltz_loader)
        samples.append(sample)

    if not samples:
        logger.error("No valid samples. Check embeddings and Boltz-2 data.")
        sys.exit(1)

    logger.info("Valid samples: %d (skipped: %d)", len(samples), len(skipped))

    # 5. Load model
    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ScreeningModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 6. Get threshold
    threshold = args.threshold
    if threshold is None:
        metrics_path = os.path.join(
            os.path.dirname(args.checkpoint), "eval_test_metrics.json",
        )
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            threshold = metrics.get("optimal_threshold", 0.5)
            logger.info("Optimal threshold from training: %.4f", threshold)
        else:
            threshold = 0.5
            logger.info("Using default threshold: %.4f", threshold)

    # 7. Run inference
    logger.info("Running inference...")
    probabilities = {}

    for i in range(0, len(samples), args.batch_size):
        batch_samples = samples[i : i + args.batch_size]
        batch = collate_fn(batch_samples)

        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    output = model(batch_device)
            else:
                output = model(batch_device)

        logits = output["logits"].squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        lengths = batch["lengths"].numpy()

        for j, pid in enumerate(batch["protein_ids"]):
            L = int(lengths[j])
            probabilities[pid] = probs[j, :L]

    # 8. Save results
    logger.info("Saving results to %s", args.output_dir)

    if args.format in ("json", "both"):
        results = {}
        for pid, probs_arr in probabilities.items():
            disordered = (probs_arr >= threshold).astype(int)
            results[pid] = {
                "length": len(probs_arr),
                "threshold": threshold,
                "fraction_disordered": float(disordered.mean()),
                "probabilities": probs_arr.tolist(),
                "predictions": disordered.tolist(),
            }

        json_path = os.path.join(args.output_dir, "predictions.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("JSON saved: %s", json_path)

    if args.format in ("tsv", "both"):
        tsv_path = os.path.join(args.output_dir, "predictions.tsv")
        with open(tsv_path, "w") as f:
            f.write("protein_id\tposition\tamino_acid\tprobability\tprediction\n")
            for pid, probs_arr in probabilities.items():
                seq = sequences.get(pid, "X" * len(probs_arr))
                for pos, (aa, prob) in enumerate(zip(seq, probs_arr), start=1):
                    pred = 1 if prob >= threshold else 0
                    f.write(f"{pid}\t{pos}\t{aa}\t{prob:.4f}\t{pred}\n")
        logger.info("TSV saved: %s", tsv_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Prediction Summary (threshold=%.4f):", threshold)
    for pid, probs_arr in probabilities.items():
        frac = float((probs_arr >= threshold).mean())
        logger.info(
            "  %s: %d residues, %.1f%% disordered",
            pid, len(probs_arr), frac * 100,
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
