import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_recall_curve
import os
from tqdm import tqdm
import random
import math
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Custom dataset class for variable-length protein sequences
class ProteinIDRDataset(Dataset):
    def __init__(self, inputs_dict, labels_dict):
        self.protein_ids = list(inputs_dict.keys())
        self.inputs_dict = inputs_dict
        self.labels_dict = labels_dict
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        return {
            'protein_id': protein_id,
            'embedding': self.inputs_dict[protein_id],
            'label': self.labels_dict[protein_id]
        }

# Custom collate function to handle variable-length sequences
def collate_fn(batch):
    # Sort batch by sequence length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: len(x['embedding']), reverse=True)
    
    protein_ids = [item['protein_id'] for item in batch]
    
    # Get sequence lengths
    lengths = torch.tensor([len(item['embedding']) for item in batch])
    
    # Pad sequences
    max_len = lengths[0].item()
    embedding_dim = batch[0]['embedding'].shape[1]
    
    # Create padded tensors
    padded_embeddings = torch.zeros(len(batch), max_len, embedding_dim)
    padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    # Fill padded tensors with actual data
    for i, item in enumerate(batch):
        seq_len = lengths[i].item()
        padded_embeddings[i, :seq_len] = item['embedding']
        padded_labels[i, :seq_len] = item['label']
    
    return {
        'protein_ids': protein_ids,
        'embeddings': padded_embeddings,
        'labels': padded_labels,
        'lengths': lengths
    }

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy for regularization"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = nn.functional.log_softmax(pred, dim=1)
        return (-one_hot * log_prob).sum(dim=1).mean()

class EnhancedPositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable parameters"""
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(EnhancedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Fixed positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
        # Learnable position embedding
        self.learned_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Combination weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        seq_len = x.size(1)
        # Combine fixed and learnable positional encodings
        combined_pe = self.alpha * self.pe[:, :seq_len, :] + (1 - self.alpha) * self.learned_pe[:, :seq_len, :]
        x = x + combined_pe
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """Simplified multi-head self-attention (removed relative positioning for now)"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Add head and query dimensions
            scores.masked_fill_(mask, -1e4)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(context)

class EnhancedLSTMTransformerIDRPredictor(nn.Module):
    """Enhanced model with additional features - FIXED VERSION"""
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_lstm_layers=2, 
                 num_transformer_layers=4, dropout=0.1, forward_expansion=4,
                 use_residual_lstm=True, use_layer_scale=True):
        super(EnhancedLSTMTransformerIDRPredictor, self).__init__()
        
        self.use_residual_lstm = use_residual_lstm
        
        # Input projection if needed
        self.input_projection = nn.Linear(input_dim, hidden_dim * 2) if input_dim != hidden_dim * 2 else nn.Identity()
        
        # Multi-layer bidirectional LSTM with residual connections
        self.lstm_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            lstm_input_size = hidden_dim * 2 if i == 0 else hidden_dim * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0
                )
            )
        
        lstm_output_dim = hidden_dim * 2
        
        # Layer normalization and dropout
        self.lstm_layer_norms = nn.ModuleList([nn.LayerNorm(lstm_output_dim) for _ in range(num_lstm_layers)])
        self.lstm_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_lstm_layers)])
        
        # Enhanced positional encoding
        self.pos_encoder = EnhancedPositionalEncoding(lstm_output_dim, dropout)
        
        # Custom transformer layers with layer scale
        self.transformer_layers = nn.ModuleList()
        
        # FIX: Store layer scale parameters separately, not in ModuleDict
        if use_layer_scale:
            self.attn_layer_scales = nn.ParameterList()
            self.ffn_layer_scales = nn.ParameterList()
        else:
            self.attn_layer_scales = None
            self.ffn_layer_scales = None
        
        for _ in range(num_transformer_layers):
            # Custom transformer layer
            layer = nn.ModuleDict({
                'self_attn': MultiHeadSelfAttention(lstm_output_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(lstm_output_dim),
                'ffn': nn.Sequential(
                    nn.Linear(lstm_output_dim, lstm_output_dim * forward_expansion),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(lstm_output_dim * forward_expansion, lstm_output_dim),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(lstm_output_dim)
            })
            self.transformer_layers.append(layer)
            
            # FIX: Add layer scale parameters to separate ParameterLists
            if use_layer_scale:
                self.attn_layer_scales.append(nn.Parameter(torch.ones(lstm_output_dim) * 1e-4))
                self.ffn_layer_scales.append(nn.Parameter(torch.ones(lstm_output_dim) * 1e-4))
        
        # Final layers
        self.final_norm = nn.LayerNorm(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, lengths):
        # Input projection
        x = self.input_projection(x)
        
        # Multi-layer LSTM with residual connections
        for i, (lstm, norm, dropout) in enumerate(zip(self.lstm_layers, self.lstm_layer_norms, self.lstm_dropouts)):
            # Pack padded sequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # LSTM forward pass
            packed_output, _ = lstm(packed_x)
            
            # Unpack
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Residual connection (if not first layer and dimensions match)
            if self.use_residual_lstm and i > 0 and lstm_output.shape == x.shape:
                lstm_output = lstm_output + x
            
            # Layer norm and dropout
            x = dropout(norm(lstm_output))
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask
        batch_size, max_len, _ = x.shape
        mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=x.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = False
        
        # Custom transformer layers
        for i, layer in enumerate(self.transformer_layers):
            # Self-attention
            attn_output = layer['self_attn'](x, mask=mask)
            
            # FIX: Layer scale using separate parameter lists
            if self.attn_layer_scales is not None:
                attn_output = self.attn_layer_scales[i] * attn_output
            x = layer['norm1'](x + attn_output)
            
            # Feed-forward network
            ffn_output = layer['ffn'](x)
            
            # FIX: Layer scale using separate parameter lists
            if self.ffn_layer_scales is not None:
                ffn_output = self.ffn_layer_scales[i] * ffn_output
            x = layer['norm2'](x + ffn_output)
        
        # Final processing
        x = self.final_norm(x)
        x = self.dropout(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
