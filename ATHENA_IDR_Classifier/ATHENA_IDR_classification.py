import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import gc
import csv
import argparse
import random
from datetime import datetime

# --- Imports from ESM Script ---
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.sdk import batch_executor

# --- Imports from IDR Script ---
# This script assumes you have a file named 'Transformer_LSTM.py'
# in the same directory, which defines your model class.
try:
    from Transformer_LSTM import EnhancedLSTMTransformerIDRPredictor
except ImportError:
    print("Error: Could not import 'EnhancedLSTMTransformerIDRPredictor' from 'Transformer_LSTM.py'.")
    print("Please ensure 'Transformer_LSTM.py' is in the same directory.")
    # Define a placeholder class to allow the script to be read,
    # but it will fail at model loading.
    class EnhancedLSTMTransformerIDRPredictor(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("Actual 'EnhancedLSTMTransformerIDRPredictor' class not found.")

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def read_fasta(fasta_path):
    """
    Basic FASTA file parser.
    Yields (header, sequence) tuples.
    """
    header, sequence = None, []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    yield header, ''.join(sequence)
                header = line
                sequence = []
            else:
                sequence.append(line)
        if header:
            yield header, ''.join(sequence)

def extract_id(header):
    """
    Extracts a unique protein ID from a FASTA header.
    Example: >sp|P12345|... -> P12345
    """
    try:
        # Assumes a UniProt-like format: >db|ID|...
        return header.split('|')[1]
    except IndexError:
        # Fallback for simple headers: >MyProteinID
        return header.lstrip('>').split()[0]


# =============================================================================
# == STEP 1: EMBEDDING GENERATION (Refactored from Script 1)
# =============================================================================

def embed_sequence(client, sequence: str):
    """Embed a protein sequence using an ESM client (API or local)"""
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    return output


def generate_embeddings(args, device):
    """
    Loads protein sequences and generates embeddings using ESM.
    Returns a dictionary of residue embeddings.
    Optionally saves embeddings to disk if args.save_embeddings_dir is set.
    """
    torch.cuda.empty_cache()

    # --- 1. Load Protein Sequences ---
    print("▶ Loading FASTA file...")
    sequences = read_fasta(args.fasta_file)
    Protein_dict = {}
    for header, seq in sequences:
        Protein_dict[extract_id(header)] = seq
    print(f"Loaded {len(Protein_dict)} sequences.")

    # --- 2. Initialize ESM-C Model ---
    is_api_model = "esmc-" in args.esm_model_name
    if is_api_model:
        print(f"Initializing ESM3 Forge API client for model: {args.esm_model_name}")
        if not args.esm_api_token:
            raise ValueError("ESM API token (--esm_api_token) is required for API-based models.")
        client = ESM3ForgeInferenceClient(model=args.esm_model_name, url="https://forge.evolutionaryscale.ai", token=args.esm_api_token)
    else:
        print(f"Loading local ESM model: {args.esm_model_name}")
        client = ESMC.from_pretrained(args.esm_model_name).to(device)
        client.eval() # Set local model to eval mode

    # --- 3. Warm up model ---
    print("▶ Warming up model...")
    try:
        dummy_seq = next(iter(Protein_dict.values()))
        with torch.no_grad():
             _ = embed_sequence(client, dummy_seq)
        print("▶ Warm-up complete")
    except Exception as e:
        print(f"▶ Warm-up exception: {e}")
    torch.cuda.empty_cache()

    # --- 4. Process Proteins in Batches ---
    all_residue_embeddings = {}
    all_cls_embeddings = {} # In case user wants to save them
    failed_proteins = []

    # Sort proteins by length for better memory management
    protein_items = sorted(list(Protein_dict.items()), key=lambda x: len(x[1]))
    protein_ids = [x[0] for x in protein_items]
    Protein_dict = {pid: Protein_dict[pid] for pid in protein_ids}  # Reorder
    
    total_proteins = len(protein_ids)
    processed_count = 0
    
    for batch_start in range(0, total_proteins, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_proteins)
        batch_protein_ids = protein_ids[batch_start:batch_end]
        batch_sequences = [Protein_dict[pid] for pid in batch_protein_ids]
        
        print(f"Processing batch {batch_start//args.batch_size + 1}/{(total_proteins-1)//args.batch_size + 1}: Proteins {batch_start} to {batch_end-1}")
        
        batch_residue_dict = {}
        batch_cls_dict = {}
        
        try:
            # Process batch using executor
            with torch.no_grad(): # Disable gradients for embedding
                with batch_executor() as executor:
                    outputs = executor.execute_batch(
                        user_func=embed_sequence, 
                        client=client, 
                        sequence=batch_sequences
                    )
            
            # Process each output
            success_count = 0
            for i, output in enumerate(outputs):
                protein_id = batch_protein_ids[i]
                try:
                    # CLS token embedding
                    cls_emb = output.embeddings[0][0].cpu()
                    # Residue embeddings (CLS and EOS excluded)
                    res_emb = output.embeddings[0][1:-1].cpu()
                    
                    batch_cls_dict[protein_id] = cls_emb
                    batch_residue_dict[protein_id] = res_emb
                    success_count += 1
                    
                except Exception as e:
                    print(f"Failed to process protein {protein_id} due to error: {type(e).__name__} {str(e)}")
                    failed_proteins.append(protein_id)

            print(f"Batch stats: {success_count} successful, {len(batch_protein_ids) - success_count} failed")
            
            # Update the master dictionaries
            all_residue_embeddings.update(batch_residue_dict)
            all_cls_embeddings.update(batch_cls_dict)
            
            processed_count += len(batch_protein_ids)
            
            # --- 5. Optionally Save Embeddings to Disk ---
            if args.save_embeddings_dir:
                ensure_dir(args.save_embeddings_dir + '/') # Ensure base dir exists
                
                start_idx = batch_start
                end_idx = batch_end - 1
                
                # Save residue embeddings
                if batch_residue_dict:
                    res_save_path = os.path.join(args.save_embeddings_dir, f"residue_emb_{start_idx}_to_{end_idx}.pt")
                    torch.save(batch_residue_dict, res_save_path)
                    print(f"Saved residue embeddings to {res_save_path}")
                    
                # Save CLS embeddings
                if batch_cls_dict:
                    cls_save_path = os.path.join(args.save_embeddings_dir, f"cls_emb_{start_idx}_to_{end_idx}.pt")
                    torch.save(batch_cls_dict, cls_save_path)
                    print(f"Saved CLS embeddings to {cls_save_path}")
            
            # Clear batch data and cache
            del batch_residue_dict, batch_cls_dict, outputs
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as batch_error:
            print(f"!! Critical Error in batch {batch_start}-{batch_end}: {batch_error}")
            print(f"Skipping this batch.")
            failed_proteins.extend(batch_protein_ids)
            processed_count += len(batch_protein_ids)
            torch.cuda.empty_cache()
            gc.collect()


    print(f"\nEmbedding generation complete.")
    print(f"Successfully processed {len(all_residue_embeddings)}/{total_proteins} proteins.")
    if failed_proteins:
        print(f"Failed proteins ({len(failed_proteins)}): {failed_proteins}")

    return all_residue_embeddings


# =============================================================================
# == STEP 2: IDR PREDICTION (From Script 2)
# =============================================================================

class InferenceDataset(Dataset):
    def __init__(self, inputs_dict):
        self.protein_ids = list(inputs_dict.keys())
        self.inputs_dict = inputs_dict
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        return {
            'protein_id': protein_id,
            'embedding': self.inputs_dict[protein_id]
        }

def collate_fn_inference(batch):
    batch = sorted(batch, key=lambda x: len(x['embedding']), reverse=True)
    protein_ids = [item['protein_id'] for item in batch]
    lengths = torch.tensor([len(item['embedding']) for item in batch])
    max_len = lengths[0].item()
    
    # Check embedding dim
    if batch[0]['embedding'].dim() == 1:
        # Handle case where a sequence might have length 1 (unlikely but possible)
        embedding_dim = batch[0]['embedding'].shape[0]
        padded_embeddings = torch.zeros(len(batch), max_len, embedding_dim)
        for i, item in enumerate(batch):
            seq_len = lengths[i].item()
            if seq_len > 0:
                 padded_embeddings[i, :seq_len, :] = item['embedding'].view(seq_len, embedding_dim)
    else:
        embedding_dim = batch[0]['embedding'].shape[1]
        padded_embeddings = torch.zeros(len(batch), max_len, embedding_dim)
        for i, item in enumerate(batch):
            seq_len = lengths[i].item()
            if seq_len > 0:
                padded_embeddings[i, :seq_len] = item['embedding']
    
    return {
        'protein_ids': protein_ids,
        'embeddings': padded_embeddings,
        'lengths': lengths
    }

def inference_on_new_data_csv(model, new_inputs_dict, device, batch_size):
    """
    Performs inference on a dictionary of new protein embeddings and returns data
    ready for CSV output.
    """
    model.eval() # Set model to evaluation mode
    
    inference_dataset = InferenceDataset(new_inputs_dict)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_inference,
        num_workers=0,
        pin_memory=False
    )
    
    csv_rows = []
    
    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, batch in enumerate(inference_loader):
            protein_ids = batch['protein_ids']
            embeddings = batch['embeddings'].to(device)
            lengths = batch['lengths']
            
            # Forward pass
            logits = model(embeddings, lengths)
            
            # Calculate probabilities (for class 1 - disordered)
            probs = torch.softmax(logits, dim=-1)[:, :, 1]
            
            # Get hard predictions
            preds = torch.argmax(logits, dim=-1)
            
            # Process each sequence in the batch
            for i, protein_id in enumerate(protein_ids):
                seq_len = lengths[i].item()
                if seq_len == 0: continue # Skip empty sequences
                
                protein_preds = preds[i, :seq_len].cpu().numpy().tolist()
                protein_probs = probs[i, :seq_len].cpu().numpy().tolist()
                
                for residue_idx in range(seq_len):
                    csv_rows.append({
                        'Protein ID': protein_id,
                        'Residue Index': residue_idx,
                        'Predicted Label': protein_preds[residue_idx],
                        'Disordered Probability': f"{protein_probs[residue_idx]:.4f}"
                    })
                
    return csv_rows


# =============================================================================
# == MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='End-to-End Protein IDR Prediction Pipeline')
    
    # --- Input/Output Arguments ---
    parser.add_argument('--fasta_file', type=str, required=True,
                        help='Path to the input FASTA file containing protein sequences.')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save the final IDR predictions CSV file.')
    parser.add_argument('--idr_model_path', type=str, required=True,
                        help='Path to the pre-trained IDR model state_dict (.pt file).')
    
    # --- ESM Embedding Arguments ---
    parser.add_argument('--esm_model_name', type=str, default='esmc_300m',
                        help='Name of the ESM model to use (e.g., "esmc_300m" or path to local model).')
    parser.add_argument('--esm_api_token', type=str, default=None,
                        help='ESM3 Forge API token (required if using an API model).')
    
    # --- Runtime Arguments ---
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for both embedding generation and IDR prediction.')
    parser.add_argument('--device', type=str, default='auto', 
                        help='Device to use (cuda/cpu/auto).')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility.')
    
    # --- Optional Saving Arguments ---
    parser.add_argument('--save_embeddings_dir', type=str, default=None,
                        help='(Optional) Directory to save intermediate protein embeddings (.pt files).')
                        
    args = parser.parse_args()
    
    # --- 1. Setup ---
    set_seed(args.seed)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Start time: {datetime.now().isoformat()}")

    # --- 2. Step 1: Generate Embeddings ---
    print("\n--- STEP 1: Generating Protein Embeddings ---")
    all_residue_embeddings = generate_embeddings(args, device)
    
    if not all_residue_embeddings:
        print("No embeddings were generated. Exiting.")
        return

    # --- 3. Step 2: Load IDR Prediction Model ---
    print("\n--- STEP 2: Loading IDR Prediction Model ---")
    
    # These parameters MUST match the saved model
    # TODO: Ideally, save these in the checkpoint and load them
    model_config = {
        'input_dim': 960, # This must match the output dim of your ESM model
        'hidden_dim': 64,
        'num_heads': 2,
        'num_lstm_layers': 3,
        'num_transformer_layers': 4,
        'dropout': 0.4599119369393788,
        'forward_expansion': 2
    }
    
    model = EnhancedLSTMTransformerIDRPredictor(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_heads=model_config['num_heads'],
        num_lstm_layers=model_config['num_lstm_layers'],
        num_transformer_layers=model_config['num_transformer_layers'],
        dropout=model_config['dropout'],
        forward_expansion=model_config['forward_expansion']
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.idr_model_path, map_location=device)['model_state_dict'])
        print(f"Successfully loaded model from {args.idr_model_path}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Please ensure 'idr_model_path' is correct and the model_config matches the saved model.")
        return
        
    # --- 4. Step 3: Run IDR Prediction ---
    print("\n--- STEP 3: Running IDR Prediction ---")
    inference_rows = inference_on_new_data_csv(
        model, 
        all_residue_embeddings, 
        device, 
        args.batch_size
    )
    print("--- Inference Complete ---")

    # --- 5. Step 4: Save Results ---
    if inference_rows:
        ensure_dir(args.output_csv)
        fieldnames = ['Protein ID', 'Residue Index', 'Predicted Label', 'Disordered Probability']
        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inference_rows)
        print(f"\n--- STEP 4: Results Saved ---")
        print(f"Inference results for {len(all_residue_embeddings)} proteins saved to '{args.output_csv}'")
    else:
        print("\nNo predictions were generated. CSV file not created.")

    print(f"End time: {datetime.now().isoformat()}")

if __name__ == '__main__':
    main()