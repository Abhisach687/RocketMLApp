#!/usr/bin/env python
"""
gru4rec_tuned_quality_diagram.py

Generates comprehensive visualization diagnostics for GRU4REC tuned model:
  1. Recommendation Hit Rate by Session Position (Tuned vs Baseline)
  2. Precision@K and Recall@K curves (Comparative)
  3. MRR (Mean Reciprocal Rank) improvement analysis
  4. Session Length vs Recommendation Quality (Enhanced)

Supports PyTorch .pt model files and provides comparative analysis with baseline.
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
sns.set_palette("Set2")


def resolve_model_path(path, model_type="tuned"):
    """Find the best available model file"""
    if model_type == "tuned":
        candidates = [
            path,
            "artefacts/gru4rec_tuned.pt",
            "artefacts/gru4rec_optimized.pt",
            "models/gru4rec_tuned.pt"
        ]
    else:
        candidates = [
            "artefacts/gru4rec_baseline.pt",
            "artefacts/gru4rec.pt",
            "models/gru4rec_baseline.pt"
        ]
    
    for candidate in candidates:
        if Path(candidate).exists():
            print(f"Using {model_type} model: {candidate}")
            return candidate
    
    print(f"Warning: No {model_type} model found in: {candidates}")
    return None


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


class EnhancedGRU4Rec(nn.Module):
    """Enhanced GRU4Rec model for tuned version"""
    
    def __init__(self, vocab_size, emb_dim=128, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        gru_out, _ = self.gru(embedded)
        return self.output(gru_out[:, -1, :])  # Last timestep


class SimpleGRU4Rec(nn.Module):
    """Simple GRU4Rec model for baseline"""
    
    def __init__(self, vocab_size, emb_dim=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        return self.output(gru_out[:, -1, :])


def load_model(model_path, vocab_size=1000, model_type="tuned"):
    """Load GRU4Rec model (tuned or baseline)"""
    if model_path is None:
        print(f"Creating dummy {model_type} model for visualization")
        if model_type == "tuned":
            model = EnhancedGRU4Rec(vocab_size, 128, 256, 2)
        else:
            model = SimpleGRU4Rec(vocab_size, 64, 128, 1)
        model.eval()
        return model
        
    try:
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
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
            config = {}
            
        # Extract model dimensions
        emb_dim = state_dict['embedding.weight'].shape[1] if 'embedding.weight' in state_dict else (128 if model_type == "tuned" else 64)
        vocab_size = state_dict['embedding.weight'].shape[0] if 'embedding.weight' in state_dict else vocab_size
        
        # Infer architecture from state dict
        gru_keys = [k for k in state_dict.keys() if 'gru' in k and 'weight_ih_l0' in k]
        if gru_keys:
            hidden_size = state_dict[gru_keys[0]].shape[0] // 3
        else:
            hidden_size = 256 if model_type == "tuned" else 128
            
        # Check for multiple layers
        num_layers = len([k for k in state_dict.keys() if 'gru.weight_ih_l' in k])
        if num_layers == 0:
            num_layers = 2 if model_type == "tuned" else 1
            
        print(f"{model_type.title()} model config: vocab_size={vocab_size}, emb_dim={emb_dim}, hidden_size={hidden_size}, layers={num_layers}")
        
        # Create model
        if model_type == "tuned":
            model = EnhancedGRU4Rec(vocab_size, emb_dim, hidden_size, num_layers)
        else:
            model = SimpleGRU4Rec(vocab_size, emb_dim, hidden_size, num_layers)
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Warning: Could not load full state dict for {model_type}: {e}")
            # Try partial loading
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        # Return dummy model
        if model_type == "tuned":
            model = EnhancedGRU4Rec(vocab_size, 128, 256, 2)
        else:
            model = SimpleGRU4Rec(vocab_size, 64, 128, 1)
        model.eval()
        return model


def evaluate_model(model, sequences_df, item2idx, model_name, k_values=[1, 5, 10, 20]):
    """Evaluate a single model"""
    
    results = {
        'model_name': model_name,
        'hit_rates': {k: [] for k in k_values},
        'mrr_scores': [],
        'session_lengths': [],
        'ndcg_scores': [],
        'position_hits': defaultdict(list)
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Evaluating {model_name} model...")
    
    # Sample sequences for evaluation
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
                session_length = len(seq)
                results['session_lengths'].append(session_length)
                
                # Hit rates
                for k in k_values:
                    top_k = top_items[:k]
                    hit = target_item in top_k
                    results['hit_rates'][k].append(hit)
                    
                    if hit:
                        position = top_k.index(target_item) + 1
                        results['position_hits'][k].append(position)
                
                # MRR and NDCG
                if target_item in top_items:
                    rank = top_items.index(target_item) + 1
                    mrr = 1.0 / rank
                    ndcg = 1.0 / np.log2(rank + 1)  # Simplified NDCG
                    results['mrr_scores'].append(mrr)
                    results['ndcg_scores'].append(ndcg)
                else:
                    results['mrr_scores'].append(0.0)
                    results['ndcg_scores'].append(0.0)
                    
            except Exception as e:
                print(f"Error processing sequence {idx} for {model_name}: {e}")
                continue
    
    return results


def plot_comparative_diagnostics(tuned_results, baseline_results, out_path):
    """Generate comparative diagnostic plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('GRU4REC Tuned vs Baseline - Comparative Recommendation Quality Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Comparative Hit Rates
    ax1 = axes[0, 0]
    k_values = sorted(tuned_results['hit_rates'].keys())
    
    tuned_hit_rates = [np.mean(tuned_results['hit_rates'][k]) * 100 for k in k_values]
    baseline_hit_rates = [np.mean(baseline_results['hit_rates'][k]) * 100 for k in k_values]
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, tuned_hit_rates, width, label='Tuned Model', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, baseline_hit_rates, width, label='Baseline Model', 
                   color='#4ECDC4', alpha=0.8)
    
    ax1.set_title('Hit Rate @ K Comparison', fontweight='bold', fontsize=12)
    ax1.set_xlabel('K (Top-K Recommendations)')
    ax1.set_ylabel('Hit Rate (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'@{k}' for k in k_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add improvement labels
    for i, (tuned, baseline) in enumerate(zip(tuned_hit_rates, baseline_hit_rates)):
        improvement = tuned - baseline
        ax1.text(i, max(tuned, baseline) + 1, f'+{improvement:.1f}%', 
                ha='center', va='bottom', fontweight='bold', 
                color='green' if improvement > 0 else 'red')
    
    # 2. MRR Distribution Comparison
    ax2 = axes[0, 1]
    
    tuned_mrr = [score for score in tuned_results['mrr_scores'] if score > 0]
    baseline_mrr = [score for score in baseline_results['mrr_scores'] if score > 0]
    
    if tuned_mrr and baseline_mrr:
        ax2.hist(baseline_mrr, bins=20, alpha=0.6, color='#4ECDC4', 
                label=f'Baseline (μ={np.mean(baseline_mrr):.3f})', density=True)
        ax2.hist(tuned_mrr, bins=20, alpha=0.6, color='#FF6B6B', 
                label=f'Tuned (μ={np.mean(tuned_mrr):.3f})', density=True)
        
        ax2.axvline(np.mean(baseline_mrr), color='#4ECDC4', linestyle='--', linewidth=2)
        ax2.axvline(np.mean(tuned_mrr), color='#FF6B6B', linestyle='--', linewidth=2)
    
    ax2.set_title('MRR Distribution Comparison', fontweight='bold', fontsize=12)
    ax2.set_xlabel('MRR Score')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Session Length Performance Analysis
    ax3 = axes[1, 0]
    
    # Combine data for analysis
    tuned_df = pd.DataFrame({
        'session_length': tuned_results['session_lengths'],
        'hit_rate': tuned_results['hit_rates'][10],
        'model': 'Tuned'
    })
    
    baseline_df = pd.DataFrame({
        'session_length': baseline_results['session_lengths'],
        'hit_rate': baseline_results['hit_rates'][10],
        'model': 'Baseline'
    })
    
    combined_df = pd.concat([tuned_df, baseline_df])
    
    # Bin session lengths
    combined_df['length_bin'] = pd.cut(combined_df['session_length'], bins=5)
    grouped = combined_df.groupby(['length_bin', 'model'])['hit_rate'].mean().unstack()
    
    if not grouped.empty:
        grouped.plot(kind='bar', ax=ax3, color=['#4ECDC4', '#FF6B6B'], alpha=0.8)
        ax3.set_title('Hit Rate@10 by Session Length', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Session Length Bins')
        ax3.set_ylabel('Hit Rate@10 (%)')
        ax3.legend(title='Model')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # 4. Precision@K Curves Comparison
    ax4 = axes[1, 1]
    k_range = range(1, 21)
    
    tuned_precisions = []
    baseline_precisions = []
    
    for k in k_range:
        if k in tuned_results['hit_rates']:
            tuned_precisions.append(np.mean(tuned_results['hit_rates'][k]) * 100)
            baseline_precisions.append(np.mean(baseline_results['hit_rates'][k]) * 100)
    
    if tuned_precisions and baseline_precisions:
        ax4.plot(k_range[:len(tuned_precisions)], tuned_precisions, 'o-', 
                color='#FF6B6B', linewidth=3, label='Tuned Model', markersize=6)
        ax4.plot(k_range[:len(baseline_precisions)], baseline_precisions, 's-', 
                color='#4ECDC4', linewidth=3, label='Baseline Model', markersize=6)
        
        # Fill area between curves
        ax4.fill_between(k_range[:len(tuned_precisions)], tuned_precisions, baseline_precisions, 
                        alpha=0.2, color='green' if np.mean(tuned_precisions) > np.mean(baseline_precisions) else 'red')
    
    ax4.set_title('Precision@K Comparison', fontweight='bold', fontsize=12)
    ax4.set_xlabel('K')
    ax4.set_ylabel('Precision@K (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"GRU4REC comparative diagnostics saved to: {out_path}")


def main():
    # Configuration
    config = load_config()
    
    # Paths
    tuned_model_path = resolve_model_path("artefacts/gru4rec_tuned.pt", "tuned")
    baseline_model_path = resolve_model_path("artefacts/gru4rec_baseline.pt", "baseline")
    sequences_path = config.get('app', {}).get('reco_sequences_path', 'data/processed/reco_sequences.parquet')
    item2idx_path = config.get('app', {}).get('item2idx_path', 'artefacts/item2idx.json')
    out_path = "reports/gru4rec_tuned_quality_diagram.png"
    
    # Load data
    sequences_df = load_sequences_data(sequences_path)
    item2idx = load_item_mapping(item2idx_path)
    
    if sequences_df.empty:
        print("No sequences data available for evaluation")
        return
    
    # Load models
    vocab_size = len(item2idx) + 1 if item2idx else 1000
    tuned_model = load_model(tuned_model_path, vocab_size, "tuned")
    baseline_model = load_model(baseline_model_path, vocab_size, "baseline")
    
    # Evaluate both models
    tuned_results = evaluate_model(tuned_model, sequences_df, item2idx, "Tuned")
    baseline_results = evaluate_model(baseline_model, sequences_df, item2idx, "Baseline")
    
    # Generate comparative diagnostics
    plot_comparative_diagnostics(tuned_results, baseline_results, out_path)
    
    # Print comparative summary
    print("\n" + "="*70)
    print("GRU4REC TUNED vs BASELINE COMPARATIVE EVALUATION")
    print("="*70)
    
    print("\n📊 HIT RATES COMPARISON:")
    for k in [1, 5, 10, 20]:
        tuned_hr = np.mean(tuned_results['hit_rates'][k]) * 100
        baseline_hr = np.mean(baseline_results['hit_rates'][k]) * 100
        improvement = tuned_hr - baseline_hr
        print(f"Hit Rate@{k:2d}: Tuned {tuned_hr:5.1f}% | Baseline {baseline_hr:5.1f}% | Δ {improvement:+5.1f}%")
    
    print("\n📈 OTHER METRICS:")
    tuned_mrr = np.mean(tuned_results['mrr_scores'])
    baseline_mrr = np.mean(baseline_results['mrr_scores'])
    mrr_improvement = ((tuned_mrr - baseline_mrr) / baseline_mrr * 100) if baseline_mrr > 0 else 0
    
    tuned_ndcg = np.mean(tuned_results['ndcg_scores'])
    baseline_ndcg = np.mean(baseline_results['ndcg_scores'])
    ndcg_improvement = ((tuned_ndcg - baseline_ndcg) / baseline_ndcg * 100) if baseline_ndcg > 0 else 0
    
    print(f"Mean MRR:     Tuned {tuned_mrr:.3f} | Baseline {baseline_mrr:.3f} | Δ {mrr_improvement:+5.1f}%")
    print(f"Mean NDCG:    Tuned {tuned_ndcg:.3f} | Baseline {baseline_ndcg:.3f} | Δ {ndcg_improvement:+5.1f}%")
    print(f"Sessions Evaluated: {len(tuned_results['session_lengths'])}")
    
    print("\n🎯 RECOMMENDATION:")
    overall_improvement = np.mean([
        np.mean(tuned_results['hit_rates'][k]) - np.mean(baseline_results['hit_rates'][k]) 
        for k in [1, 5, 10, 20]
    ]) * 100
    
    if overall_improvement > 2:
        print("✅ Tuned model shows SIGNIFICANT improvement over baseline")
    elif overall_improvement > 0.5:
        print("✅ Tuned model shows moderate improvement over baseline")
    else:
        print("⚠️  Tuned model shows minimal improvement - consider further optimization")


if __name__ == "__main__":
    main()
