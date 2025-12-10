"""
Deliverable 3: Simple Neural Language Model
Train LSTM-based language models for Finnish and Hindi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import time
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class Vocabulary:
    """Vocabulary class for mapping words to indices"""
    
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        
    def build(self, sentences):
        """Build vocabulary from sentences"""
        # Count word frequencies
        for sent in sentences:
            self.word_freq.update(sent)
        
        # Add special tokens
        special_tokens = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
        
        # Add words meeting frequency threshold
        idx = len(special_tokens)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Words below threshold: {sum(1 for f in self.word_freq.values() if f < self.min_freq)}")
        
        return self
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk_token])
    
    def decode(self, idx):
        return self.idx2word.get(idx, self.unk_token)
    
    @property
    def pad_idx(self):
        return self.word2idx[self.pad_token]
    
    @property
    def unk_idx(self):
        return self.word2idx[self.unk_token]
    
    @property
    def sos_idx(self):
        return self.word2idx[self.sos_token]
    
    @property
    def eos_idx(self):
        return self.word2idx[self.eos_token]


class LanguageModelDataset(Dataset):
    """Dataset for language modeling"""
    
    def __init__(self, sentences, vocab, seq_length=30):
        self.vocab = vocab
        self.seq_length = seq_length
        self.data = self._prepare_data(sentences)
        
    def _prepare_data(self, sentences):
        """Convert sentences to sequences of indices"""
        all_indices = []
        
        for sent in sentences:
            # Add SOS and EOS tokens
            indices = [self.vocab.sos_idx]
            indices += [self.vocab.encode(w) for w in sent]
            indices += [self.vocab.eos_idx]
            all_indices.extend(indices)
        
        return torch.tensor(all_indices, dtype=torch.long)
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        # Input: sequence of tokens
        # Target: same sequence shifted by 1
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


class LSTMLanguageModel(nn.Module):
    """LSTM-based Language Model"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Tie weights (optional but often helpful)
        if embed_dim == hidden_dim:
            self.fc.weight = self.embedding.weight
        
    def forward(self, x, hidden=None):
        # x: (batch_size, seq_length)
        
        # Embedding
        embeds = self.dropout(self.embedding(x))  # (batch, seq, embed_dim)
        
        # LSTM
        lstm_out, hidden = self.lstm(embeds, hidden)  # (batch, seq, hidden_dim)
        
        # Output projection
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # (batch, seq, vocab_size)
        
        return logits, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


def load_conllu(filepath):
    """Load sentences from CoNLL-U file"""
    sentences = []
    current = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            elif not line:
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split('\t')
                if parts[0].isdigit():
                    current.append(parts[1].lower())
    
    if current:
        sentences.append(current)
    
    return sentences


def calculate_perplexity(model, dataloader, criterion):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            hidden = model.init_hidden(batch_size)
            logits, _ = model(x, hidden)
            
            # Flatten for loss calculation
            logits_flat = logits.view(-1, model.vocab_size)
            y_flat = y.view(-1)
            
            loss = criterion(logits_flat, y_flat)
            
            # Count non-padding tokens
            non_pad = (y_flat != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def train_model(model, train_loader, dev_loader, epochs=50, lr=0.001, patience=5):
    """Train the language model with early stopping"""
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {
        'train_loss': [],
        'train_ppl': [],
        'dev_ppl': [],
        'lr': []
    }
    
    best_dev_ppl = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\nTraining for up to {epochs} epochs with patience={patience}")
    print("-" * 60)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        total_loss = 0
        total_tokens = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            optimizer.zero_grad()
            
            hidden = model.init_hidden(batch_size)
            logits, _ = model(x, hidden)
            
            # Flatten
            logits_flat = logits.view(-1, model.vocab_size)
            y_flat = y.view(-1)
            
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            non_pad = (y_flat != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
        
        # Calculate metrics
        train_loss = total_loss / total_tokens
        train_ppl = np.exp(train_loss)
        dev_ppl = calculate_perplexity(model, dev_loader, criterion)
        
        # Update scheduler
        scheduler.step(dev_ppl)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['dev_ppl'].append(dev_ppl)
        history['lr'].append(current_lr)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:8.2f} | "
              f"Dev PPL: {dev_ppl:8.2f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        # Early stopping check
        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}! Best Dev PPL: {best_dev_ppl:.2f}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, best_dev_ppl


def plot_learning_curves(history, language, save_path):
    """Plot and save learning curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_ppl']) + 1)
    
    # Plot 1: Training Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{language} - Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Perplexity
    axes[1].plot(epochs, history['train_ppl'], 'b-', label='Train PPL')
    axes[1].plot(epochs, history['dev_ppl'], 'r-', label='Dev PPL')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title(f'{language} - Perplexity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title(f'{language} - Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to: {save_path}")
    plt.close()


def run_experiment(language, train_file, dev_file, test_file, config):
    """Run complete experiment for a language"""
    
    print(f"\n{'='*70}")
    print(f"NEURAL LANGUAGE MODEL: {language.upper()}")
    print(f"{'='*70}")
    
    # Load data
    print("\nLoading data...")
    train_sents = load_conllu(train_file)
    dev_sents = load_conllu(dev_file)
    test_sents = load_conllu(test_file)
    
    print(f"Train: {len(train_sents)} sentences")
    print(f"Dev: {len(dev_sents)} sentences")
    print(f"Test: {len(test_sents)} sentences")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = Vocabulary(min_freq=config['min_freq'])
    vocab.build(train_sents)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = LanguageModelDataset(train_sents, vocab, seq_length=config['seq_length'])
    dev_dataset = LanguageModelDataset(dev_sents, vocab, seq_length=config['seq_length'])
    test_dataset = LanguageModelDataset(test_sents, vocab, seq_length=config['seq_length'])
    
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Dev sequences: {len(dev_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Create model
    print("\nCreating model...")
    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    history, best_dev_ppl = train_model(
        model, train_loader, dev_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        patience=config['patience']
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    test_ppl = calculate_perplexity(model, test_loader, criterion)
    print(f"Test Perplexity: {test_ppl:.2f}")
    
    # Plot learning curves
    plot_learning_curves(history, language, f'learning_curves_{language}.png')
    
    # Save model
    model_path = f'model_{language}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': config,
        'history': history
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    results = {
        'language': language,
        'vocab_size': len(vocab),
        'best_dev_ppl': best_dev_ppl,
        'test_ppl': test_ppl,
        'epochs_trained': len(history['train_ppl']),
        'final_train_ppl': history['train_ppl'][-1],
        'history': history
    }
    
    return results


def main():
    """Main function"""
    
    # Configuration
    config = {
        'embed_dim': 256,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'seq_length': 30,
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 50,
        'patience': 5,
        'min_freq': 2
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # File paths
    finnish_train = "data/ud/UD_Finnish-TDT/fi_tdt-ud-train.conllu"
    finnish_dev = "data/ud/UD_Finnish-TDT/fi_tdt-ud-dev.conllu"
    finnish_test = "data/ud/UD_Finnish-TDT/fi_tdt-ud-test.conllu"
    
    hindi_train = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
    hindi_dev = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu"
    hindi_test = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-test.conllu"
    
    # Run experiments
    finnish_results = run_experiment("Finnish", finnish_train, finnish_dev, finnish_test, config)
    hindi_results = run_experiment("Hindi", hindi_train, hindi_dev, hindi_test, config)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: NEURAL LANGUAGE MODEL RESULTS")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<25} {'Finnish':<15} {'Hindi':<15}")
    print("-" * 55)
    print(f"{'Vocabulary Size':<25} {finnish_results['vocab_size']:<15,} {hindi_results['vocab_size']:<15,}")
    print(f"{'Epochs Trained':<25} {finnish_results['epochs_trained']:<15} {hindi_results['epochs_trained']:<15}")
    print(f"{'Final Train PPL':<25} {finnish_results['final_train_ppl']:<15.2f} {hindi_results['final_train_ppl']:<15.2f}")
    print(f"{'Best Dev PPL':<25} {finnish_results['best_dev_ppl']:<15.2f} {hindi_results['best_dev_ppl']:<15.2f}")
    print(f"{'Test PPL':<25} {finnish_results['test_ppl']:<15.2f} {hindi_results['test_ppl']:<15.2f}")
    
    # Comparison with n-gram baseline
    print(f"\n{'='*70}")
    print("COMPARISON WITH N-GRAM BASELINE")
    print(f"{'='*70}")
    
    # N-gram results from Deliverable 2
    ngram_results = {
        'Finnish': {'2-gram': 266.52, '3-gram': 1257.93, '4-gram': 3401.37},
        'Hindi': {'2-gram': 192.36, '3-gram': 861.14, '4-gram': 2834.04}
    }
    
    print(f"\n{'Model':<20} {'Finnish Test PPL':<20} {'Hindi Test PPL':<20}")
    print("-" * 60)
    print(f"{'2-gram':<20} {ngram_results['Finnish']['2-gram']:<20.2f} {ngram_results['Hindi']['2-gram']:<20.2f}")
    print(f"{'3-gram':<20} {ngram_results['Finnish']['3-gram']:<20.2f} {ngram_results['Hindi']['3-gram']:<20.2f}")
    print(f"{'4-gram':<20} {ngram_results['Finnish']['4-gram']:<20.2f} {ngram_results['Hindi']['4-gram']:<20.2f}")
    print(f"{'Neural LM (LSTM)':<20} {finnish_results['test_ppl']:<20.2f} {hindi_results['test_ppl']:<20.2f}")
    
    # Improvement calculation
    finnish_improvement = (ngram_results['Finnish']['2-gram'] - finnish_results['test_ppl']) / ngram_results['Finnish']['2-gram'] * 100
    hindi_improvement = (ngram_results['Hindi']['2-gram'] - hindi_results['test_ppl']) / ngram_results['Hindi']['2-gram'] * 100
    
    print(f"\nImprovement over best n-gram (2-gram):")
    print(f"  Finnish: {finnish_improvement:.1f}%")
    print(f"  Hindi: {hindi_improvement:.1f}%")
    
    # Save all results
    all_results = {
        'config': config,
        'finnish': {
            'vocab_size': finnish_results['vocab_size'],
            'epochs_trained': finnish_results['epochs_trained'],
            'final_train_ppl': finnish_results['final_train_ppl'],
            'best_dev_ppl': finnish_results['best_dev_ppl'],
            'test_ppl': finnish_results['test_ppl']
        },
        'hindi': {
            'vocab_size': hindi_results['vocab_size'],
            'epochs_trained': hindi_results['epochs_trained'],
            'final_train_ppl': hindi_results['final_train_ppl'],
            'best_dev_ppl': hindi_results['best_dev_ppl'],
            'test_ppl': hindi_results['test_ppl']
        },
        'ngram_baseline': ngram_results
    }
    
    with open('neural_lm_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to: neural_lm_results.json")


if __name__ == "__main__":
    main()