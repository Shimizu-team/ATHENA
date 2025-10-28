import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoConfig)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.utils.data import Dataset
from config import *
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset
import argparse 


def setup_xlora_inference_model(
    base_model_name: str,
    adapter_paths: Dict[str, str],
    adapter_weights: List[float],
    device: str = None
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ベースモデルの読み込み
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    tokenizer = base_model.tokenizer

    base_linear = next(
        m for m in base_model.classifier.modules()
        if isinstance(m, nn.Linear)
    )
    orig_in_features = base_linear.in_features    

    if adapter_paths is None:
        print("No LoRA adapters specified. Using base model only.")
        model = base_model
        
    elif len(adapter_paths) == 1:
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
    return model, tokenizer, orig_in_features

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

class Linear_idp_classification_head(nn.Module):
    def __init__(self, emb_dim, num_labels):
        super().__init__()
        self.Linear_idp_classification_head = nn.Linear(emb_dim, num_labels)
    def forward(self, features, **kwargs):
        return self.Linear_idp_classification_head(features)

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
def predict(conf: argparse.Namespace): # <--- RENAMED

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    adapter_paths = dict(conf.adapter_paths) if conf.adapter_paths else None
    
    model, tokenizer, orig_in_features = setup_xlora_inference_model(
        base_model_name=conf.base_model,
        adapter_paths=adapter_paths,
        adapter_weights=conf.adapter_weights,
        device=device
    )
    
    model.classifier = Linear_idp_classification_head(orig_in_features, conf.num_labels)
    cls_path = os.path.join(conf.classifier_params_path, "classifier_params.pth")
    
    if not os.path.exists(cls_path):
        raise FileNotFoundError(f"Classifier weights not found at: {cls_path}")
        
    model.classifier.load_state_dict(torch.load(cls_path, map_location='cpu'))
    print(f"Loaded classifier weights from {cls_path}")

    model.to(device)
    model.eval()
    print(f"Model is ready on {device}")
    
    sequences = fasta2dict(conf.fasta_path)
    
    final_predictions = {} 
    
    seq_ids = list(sequences.keys())
    sequence_list = list(sequences.values())
    
    # データセットとデータローダーの準備
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