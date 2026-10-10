import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from config import (
    Config,
    DEFAULT_BASE_MODEL_REVISION,
    EXPECTED_POOLING_TYPES,
    parse_args,
)


def load_base_model(base_model_name: str, revision: Optional[str]):
    """Load the base model and its remote code at a fixed Hub revision."""
    return AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        revision=revision,
        trust_remote_code=True
    )


def verify_pooling_types(base_model) -> None:
    """Make sure the base model pools the way the classifier head was trained."""
    pooler = getattr(base_model, "pooler", None)
    pooling_types = getattr(pooler, "pooling_types", None)
    if pooling_types is None:
        print("Warning: could not read the pooling types of the base model; "
              f"expected {list(EXPECTED_POOLING_TYPES)}.")
        return
    if tuple(pooling_types) != EXPECTED_POOLING_TYPES:
        raise ValueError(
            f"The base model pools with {list(pooling_types)}, but the ATHENA IDP classifier "
            f"was trained with {list(EXPECTED_POOLING_TYPES)}. Use the pinned base model "
            f"revision ({DEFAULT_BASE_MODEL_REVISION})."
        )


def setup_xlora_inference_model(
    base_model_name: str,
    adapter_paths: Dict[str, str],
    adapter_weights: List[float],
    device: str = None,
    base_model_revision: Optional[str] = DEFAULT_BASE_MODEL_REVISION
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the base model
    base_model = load_base_model(base_model_name, base_model_revision)
    verify_pooling_types(base_model)

    tokenizer = base_model.tokenizer

    if not adapter_paths:
        raise ValueError(
            "At least one LoRA adapter is required (--adapter_paths). The trained "
            "classification head is stored inside the adapter (PEFT modules_to_save); "
            "the base model alone has an untrained head."
        )

    if len(adapter_paths) == 1:
        adapter_name = list(adapter_paths.keys())[0]
        adapter_path = adapter_paths[adapter_name]
        print(f"Loading single LoRA adapter '{adapter_name}'...")
        
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            adapter_name=adapter_name
        )
        print(f"Single LoRA adapter '{adapter_name}' loaded successfully.")
        
    else:
        if len(adapter_weights) != len(adapter_paths):
            raise ValueError("Number of adapters and weights must match.")
            
        first_adapter_name = list(adapter_paths.keys())[0]
        first_adapter_path = adapter_paths[first_adapter_name]
        
        print(f"Loading base model '{base_model_name}' with first adapter '{first_adapter_name}'...")
        model = PeftModel.from_pretrained(
            base_model,
            first_adapter_path,
            adapter_name=first_adapter_name,
            use_xlora=True 
        )

        for name, path in list(adapter_paths.items())[1:]:
            print(f"Loading adapter '{name}'...")
            model.load_adapter(path, adapter_name=name)

        adapter_names = list(adapter_paths.keys())
        print(f"\nCombining adapters with weights: {dict(zip(adapter_names, adapter_weights))}")
        model.add_weighted_adapter(
            adapters=adapter_names,
            weights=adapter_weights,
            adapter_name="combined_inference" 
        )

        model.set_adapter("combined_inference")

    print("\nModel setup complete.")
    return model, tokenizer

def fasta2dict(filename):
    sequences = {}
    current_header, current_seq = None, []
    with open(filename, 'r') as file:
        for line in file:
            line = line.rstrip()
            if not line: continue
            if line.startswith('>'):
                if current_header is not None:
                    sequences[current_header] = ''.join(current_seq)
                    current_seq = []
                raw = line[1:]
                if raw.startswith('sp|'):
                    seq_id = raw.split('|')[1]
                elif raw.startswith('tr|'):
                    parts = raw.split('|')
                    seq_id = parts[1]                 
                elif '|' in raw:
                    seq_id = raw.split('|',1)[0]
                else:
                    seq_id = raw.split(None,1)[0]
                current_header = seq_id
            else:
                current_seq.append(line)
        if current_header is not None:
            sequences[current_header] = ''.join(current_seq)
    print(f"> {len(sequences)} sequences loaded from {filename}")
    return sequences

def min_max_scale(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max scale scores to [0, 1] across all sequences in the dict."""
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class InferenceDataset(Dataset):
    def __init__(self, seq_ids, sequences):
        self.seq_ids = seq_ids
        self.sequences = sequences

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        return {
            "seq_id": self.seq_ids[idx],
            "sequence": self.sequences[idx]
        }


@torch.no_grad()
def predict(conf: Config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    adapter_paths = dict(conf.adapter_paths) if conf.adapter_paths else None

    model, tokenizer = setup_xlora_inference_model(
        base_model_name=conf.base_model,
        adapter_paths=adapter_paths,
        adapter_weights=conf.adapter_weights,
        device=device,
        base_model_revision=conf.base_model_revision
    )

    # The classification head is restored by PeftModel.from_pretrained together
    # with the LoRA weights (adapter_config.json: modules_to_save = ["classifier",
    # "score"]). The legacy classifier_params.pth shipped next to the adapter is an
    # untrained, default-initialised nn.Linear that was never part of the forward
    # pass; it is intentionally not loaded.
    if conf.classifier_params_path:
        print("Note: --classifier_params_path is ignored. The trained classifier head "
              "is loaded from the LoRA adapter (modules_to_save).")

    model.to(device)
    model.eval()
    print(f"Model is ready on {device}")
    
    sequences = fasta2dict(conf.fasta_path)
    
    final_predictions = {} 
    
    seq_ids = list(sequences.keys())
    sequence_list = list(sequences.values())
    
    # Prepare the dataset and data loader
    inference_dataset = InferenceDataset(seq_ids, sequence_list)
    inference_dataloader = DataLoader(inference_dataset, batch_size=conf.batch_size, shuffle=False)

    print(f"Starting prediction for {len(seq_ids)} sequences with batch size {conf.batch_size}...")

    with torch.no_grad():
        for batch in tqdm(inference_dataloader, desc="Predicting in batches"):
            batch_ids = batch["seq_id"]
            batch_sequences = batch["sequence"]

            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                truncation=True,
                max_length=conf.max_length,
                padding="max_length"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            
            if conf.output_type == "before_softmax":
                raw_scores = outputs.logits.cpu()
                for i, seq_id in enumerate(batch_ids):
                    final_predictions[seq_id] = raw_scores[i].tolist()
            
            else:
                probabilities = torch.softmax(outputs.logits, dim=-1).cpu()
                prob_for_label_1 = probabilities[:, 1]
                for i, seq_id in enumerate(batch_ids):
                    final_predictions[seq_id] = prob_for_label_1[i].item()
    os.makedirs(conf.output_dir, exist_ok=True)
    
    if conf.output_type == "before_softmax":
        idp_scores_before_softmax = {k: v[1] for k, v in final_predictions.items()}
        structured_scores_before_softmax = {k: v[0] for k, v in final_predictions.items()}
        
        output_idp_path = os.path.join(conf.output_dir, f"IDP_score_before_softmax_{conf.title}.pt")
        output_structured_path = os.path.join(conf.output_dir, f"Structured_score_before_softmax_{conf.title}.pt")
        
        torch.save(idp_scores_before_softmax, output_idp_path)
        torch.save(structured_scores_before_softmax, output_structured_path)
        
        print(f"\nIDP scores (before softmax) saved to: {output_idp_path}")
        print(f"Structured scores (before softmax) saved to: {output_structured_path}")

        if conf.min_max_scaling:
            athena_scores = min_max_scale(idp_scores_before_softmax)
            output_athena_path = os.path.join(conf.output_dir, f"ATHENA_score_{conf.title}.pt")
            torch.save(athena_scores, output_athena_path)
            print(f"Min-max scaled IDP scores (ATHENA score) saved to: {output_athena_path}")

        print("\n--- IDP Score (Before Softmax) Results (Top 5) ---")
        for i, (seq_id, score) in enumerate(idp_scores_before_softmax.items()):
            if i >= 5: break
            print(f"  - Sequence ID: {seq_id}, IDP Score (Before Softmax): {score:.4f}")
        
    else:
        output_path = os.path.join(conf.output_dir, f"IDP_score_{conf.title}.pt")
        torch.save(final_predictions, output_path)
        print(f"\nPredictions (IDP Score) saved successfully to: {output_path}")
        
        print("\n--- IDP Score (After Softmax) Results (Top 5) ---")
        for i, (seq_id, IDP_score) in enumerate(final_predictions.items()):
            if i >= 5: break
            print(f"  - Sequence ID: {seq_id}, IDP Score: {IDP_score:.4f}")
    
    return final_predictions


def main():
    args = parse_args()
    conf = Config(args)
    print(vars(conf))
    predict(conf) 

if __name__ == "__main__":
    main()