#!/usr/bin/env python
"""
gru4rec_baseline_quality_diagram.py

Generates comprehensive visualization diagnostics for GRU4REC baseline model:
  1. Recommendation Hit Rate by Session Position
  2. Precision@K and Recall@K curves
  3. MRR (Mean Reciprocal Rank) distribution
  4. Session Length vs Recommendation Quality

Supports PyTorch .pt model files and evaluates on recommendation sequences.
Automatically falls back among common model filenames if the one given is missing.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")


def resolve_model_path(path):
    """Find the best available model file"""
    candidates = [
        path,
        "artefacts/gru4rec_baseline.pt",
        "artefacts/gru4rec.pt",
        "models/gru4rec_baseline.pt"
    ]
    
    for candidate in candidates:
        if Path(candidate).exists():
            print(f"Using model: {candidate}")
            return candidate
    
    raise FileNotFoundError(f"No GRU4REC model found in: {candidates}")


def load_config(config_path="config.yaml"):
    """Load configuration"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}


def load_sequences_data(seq_path):
    """Load recommendation sequences data"""
    try:
        df = pd.read_parquet(seq_path)
        print(f"Loaded {len(df):,} sequences from {seq_path}")
        return df
    except Exception as e:
        print(f"Error loading sequences: {e}")
        return pd.DataFrame()


def load_item_mapping(item2idx_path):
    """Load item to index mapping"""
    try:
        with open(item2idx_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load item mapping from {item2idx_path}: {e}")
        return {}


class GRU4RecDataset(Dataset):
    """Dataset for GRU4Rec evaluation"""
    
    def __init__(self, sequences, item2idx, max_len=50):
        self.sequences = sequences
        self.item2idx = item2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences.iloc[idx]['item_seq']
        
        # Convert to indices
        seq_indices = [self.item2idx.get(str(item), 0) for item in seq]
        
        # Pad or truncate
        if len(seq_indices) > self.max_len:
            seq_indices = seq_indices[-self.max_len:]
        else:
            seq_indices = [0] * (self.max_len - len(seq_indices)) + seq_indices
            
        return torch.tensor(seq_indices, dtype=torch.long)


class SimpleGRU4Rec(nn.Module):
    """Simple GRU4Rec model for evaluation"""
    
    def __init__(self, vocab_size, emb_dim=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        return self.output(gru_out[:, -1, :])  # Last timestep


def load_model(model_path, vocab_size=1000):
    """Load GRU4Rec model"""
    try:
        # Try loading the model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('config', {})
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                config = checkpoint.get('config', {})
            else:
                state_dict = checkpoint
                config = {}
        else:
            # Direct model
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
            config = {}
            
        # Extract model dimensions from state dict
        emb_dim = state_dict['embedding.weight'].shape[1] if 'embedding.weight' in state_dict else 64
        vocab_size = state_dict['embedding.weight'].shape[0] if 'embedding.weight' in state_dict else vocab_size
        
        # Try to infer hidden size from GRU weights
        gru_keys = [k for k in state_dict.keys() if 'gru' in k and 'weight_ih_l0' in k]
        if gru_keys:
            hidden_size = state_dict[gru_keys[0]].shape[0] // 3  # GRU has 3 gates
        else:
            hidden_size = 128
            
        print(f"Model config: vocab_size={vocab_size}, emb_dim={emb_dim}, hidden_size={hidden_size}")
        
        # Create and load model
        model = SimpleGRU4Rec(vocab_size, emb_dim, hidden_size)
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Warning: Could not load full state dict: {e}")
            # Try partial loading
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Return dummy model for visualization
        model = SimpleGRU4Rec(vocab_size, 64, 128)
        model.eval()
        return model


def evaluate_recommendations(model, sequences_df, item2idx, k_values=[1, 5, 10, 20]):
    """Evaluate recommendation quality"""
    
    results = {
        'hit_rates': {k: [] for k in k_values},
        'mrr_scores': [],
        'session_lengths': [],
        'precision_at_k': {k: [] for k in k_values},
        'recall_at_k': {k: [] for k in k_values},
        'position_hits': defaultdict(list)
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("Evaluating recommendation quality...")
    
    # Sample sequences for evaluation (to avoid memory issues)
    eval_sequences = sequences_df.sample(min(1000, len(sequences_df)), random_state=42)
    
    with torch.no_grad():
        for idx, row in eval_sequences.iterrows():
            seq = row['item_seq']
            
            if len(seq) < 2:
                continue
                
            # Use all but last item as input, last item as target
            input_seq = seq[:-1]
            target_item = seq[-1]
            
            # Convert to model input
            input_indices = [item2idx.get(str(item), 0) for item in input_seq]
            
            # Pad sequence
            max_len = 50
            if len(input_indices) > max_len:
                input_indices = input_indices[-max_len:]
            else:
                input_indices = [0] * (max_len - len(input_indices)) + input_indices
                
            input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
            
            try:
                # Get predictions
                outputs = model(input_tensor)
                scores = torch.softmax(outputs, dim=-1).cpu().numpy()[0]
                
                # Get top-k recommendations
                top_indices = np.argsort(scores)[::-1]
                
                # Convert back to item IDs
                idx2item = {v: int(k) for k, v in item2idx.items()}
                top_items = [idx2item.get(idx, 0) for idx in top_indices]
                
                # Calculate metrics
                target_idx = item2idx.get(str(target_item), -1)
                session_length = len(seq)
                
                results['session_lengths'].append(session_length)
                
                # Hit rates and position tracking
                for k in k_values:
                    top_k = top_items[:k]
                    hit = target_item in top_k
                    results['hit_rates'][k].append(hit)
                    
                    if hit:
                        position = top_k.index(target_item) + 1
                        results['position_hits'][k].append(position)
                
                # MRR calculation
                if target_item in top_items:
                    rank = top_items.index(target_item) + 1
                    mrr = 1.0 / rank
                    results['mrr_scores'].append(mrr)
                else:
                    results['mrr_scores'].append(0.0)
                    
            except Exception as e:
                print(f"Error processing sequence {idx}: {e}")
                continue
    
    return results


def plot_diagnostics(results, out_path):
    """Generate comprehensive diagnostic plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GRU4REC Baseline Model - Recommendation Quality Diagnostics', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Hit Rate by K
    ax1 = axes[0, 0]
    k_values = sorted(results['hit_rates'].keys())
    hit_rates = [np.mean(results['hit_rates'][k]) * 100 for k in k_values]
    
    bars = ax1.bar(range(len(k_values)), hit_rates, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
    ax1.set_title('Hit Rate @ K', fontweight='bold', fontsize=12)
    ax1.set_xlabel('K (Top-K Recommendations)')
    ax1.set_ylabel('Hit Rate (%)')
    ax1.set_xticks(range(len(k_values)))
    ax1.set_xticklabels([f'@{k}' for k in k_values])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, hit_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. MRR Distribution
    ax2 = axes[0, 1]
    mrr_scores = [score for score in results['mrr_scores'] if score > 0]
    
    if mrr_scores:
        ax2.hist(mrr_scores, bins=20, alpha=0.7, color='#FF6B6B', edgecolor='black')
        ax2.axvline(np.mean(mrr_scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean MRR: {np.mean(mrr_scores):.3f}')
        ax2.axvline(np.median(mrr_scores), color='orange', linestyle='--', linewidth=2,
                   label=f'Median MRR: {np.median(mrr_scores):.3f}')
    
    ax2.set_title('Mean Reciprocal Rank (MRR) Distribution', fontweight='bold', fontsize=12)
    ax2.set_xlabel('MRR Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Session Length vs Hit Rate
    ax3 = axes[1, 0]
    if results['session_lengths'] and results['hit_rates'][10]:  # Use HR@10
        df_temp = pd.DataFrame({
            'session_length': results['session_lengths'],
            'hit_rate': results['hit_rates'][10]
        })
        
        # Bin session lengths
        df_temp['length_bin'] = pd.cut(df_temp['session_length'], bins=5)
        grouped = df_temp.groupby('length_bin')['hit_rate'].mean()
        
        x_pos = range(len(grouped))
        bars = ax3.bar(x_pos, grouped.values * 100, alpha=0.7, color='#45B7D1')
        
        ax3.set_title('Hit Rate@10 by Session Length', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Session Length Bins')
        ax3.set_ylabel('Hit Rate@10 (%)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' 
                            for interval in grouped.index], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, grouped.values * 100):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom')
    
    # 4. Precision/Recall Curves
    ax4 = axes[1, 1]
    k_range = range(1, 21)
    precisions = []
    recalls = []
    
    for k in k_range:
        if k in results['hit_rates']:
            precision = np.mean(results['hit_rates'][k])
            recall = np.mean(results['hit_rates'][k])  # For single item, P=R
            precisions.append(precision * 100)
            recalls.append(recall * 100)
    
    if precisions and recalls:
        ax4.plot(k_range[:len(precisions)], precisions, 'o-', color='#FF6B6B', 
                linewidth=2, label='Precision@K', markersize=6)
        ax4.plot(k_range[:len(recalls)], recalls, 's-', color='#4ECDC4', 
                linewidth=2, label='Recall@K', markersize=6)
    
    ax4.set_title('Precision@K and Recall@K', fontweight='bold', fontsize=12)
    ax4.set_xlabel('K')
    ax4.set_ylabel('Score (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"GRU4REC baseline diagnostics saved to: {out_path}")


def main():
    # Configuration
    config = load_config()
    
    # Paths
    model_path = resolve_model_path("artefacts/gru4rec_baseline.pt")
    sequences_path = config.get('app', {}).get('reco_sequences_path', 'data/processed/reco_sequences.parquet')
    item2idx_path = config.get('app', {}).get('item2idx_path', 'artefacts/item2idx.json')
    out_path = "reports/gru4rec_baseline_quality_diagram.png"
    
    # Load data
    sequences_df = load_sequences_data(sequences_path)
    item2idx = load_item_mapping(item2idx_path)
    
    if sequences_df.empty:
        print("No sequences data available for evaluation")
        return
    
    # Load model
    vocab_size = len(item2idx) + 1 if item2idx else 1000
    model = load_model(model_path, vocab_size)
    
    # Evaluate
    results = evaluate_recommendations(model, sequences_df, item2idx)
    
    # Generate diagnostics
    plot_diagnostics(results, out_path)
    
    # Print summary
    print("\n" + "="*60)
    print("GRU4REC BASELINE MODEL EVALUATION SUMMARY")
    print("="*60)
    
    if results['hit_rates'][10]:
        print(f"Hit Rate@1:  {np.mean(results['hit_rates'][1])*100:.1f}%")
        print(f"Hit Rate@5:  {np.mean(results['hit_rates'][5])*100:.1f}%")
        print(f"Hit Rate@10: {np.mean(results['hit_rates'][10])*100:.1f}%")
        print(f"Hit Rate@20: {np.mean(results['hit_rates'][20])*100:.1f}%")
        print(f"Mean MRR:    {np.mean(results['mrr_scores']):.3f}")
        print(f"Sessions Evaluated: {len(results['session_lengths'])}")
    else:
        print("No evaluation results available")


if __name__ == "__main__":
    main()
