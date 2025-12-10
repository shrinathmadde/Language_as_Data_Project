"""
Deliverable 4: Analyze Model Variants (Simplified)
4 key modifications - one per category:
1. Input representation: BPE Tokenization
2. Model architecture: LSTM + Self-Attention
3. Training process: Label Smoothing
4. Sampling: Different decoding strategies
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
from tokenizers import Tokenizer

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# DATA LOADING AND VOCABULARY
# =============================================================================

class Vocabulary:
    """Word-level vocabulary"""
    
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        
    def build(self, sentences):
        for sent in sentences:
            self.word_freq.update(sent)
        
        special_tokens = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
        
        idx = len(special_tokens)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
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
    def sos_idx(self):
        return self.word2idx[self.sos_token]
    
    @property
    def eos_idx(self):
        return self.word2idx[self.eos_token]


class BPEVocabulary:
    """BPE tokenizer wrapper"""
    
    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.sos_token = "[CLS]"
        self.eos_token = "[SEP]"
        
    def __len__(self):
        return self.tokenizer.get_vocab_size()
    
    def encode_sentence(self, sentence):
        text = " ".join(sentence)
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    @property
    def pad_idx(self):
        return self.tokenizer.token_to_id(self.pad_token)
    
    @property
    def sos_idx(self):
        return self.tokenizer.token_to_id(self.sos_token)
    
    @property
    def eos_idx(self):
        return self.tokenizer.token_to_id(self.eos_token)
    
    def decode(self, idx):
        return self.tokenizer.id_to_token(idx)


class LanguageModelDataset(Dataset):
    """Word-level dataset"""
    
    def __init__(self, sentences, vocab, seq_length=30):
        self.vocab = vocab
        self.seq_length = seq_length
        self.data = self._prepare_data(sentences)
        
    def _prepare_data(self, sentences):
        all_indices = []
        for sent in sentences:
            indices = [self.vocab.sos_idx]
            indices += [self.vocab.encode(w) for w in sent]
            indices += [self.vocab.eos_idx]
            all_indices.extend(indices)
        return torch.tensor(all_indices, dtype=torch.long)
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


class BPELanguageModelDataset(Dataset):
    """BPE-level dataset"""
    
    def __init__(self, sentences, vocab, seq_length=30):
        self.vocab = vocab
        self.seq_length = seq_length
        self.data = self._prepare_data(sentences)
        
    def _prepare_data(self, sentences):
        all_indices = []
        for sent in sentences:
            indices = [self.vocab.sos_idx]
            indices += self.vocab.encode_sentence(sent)
            indices += [self.vocab.eos_idx]
            all_indices.extend(indices)
        return torch.tensor(all_indices, dtype=torch.long)
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


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


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class LSTMLanguageModel(nn.Module):
    """Baseline LSTM Language Model"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits, hidden
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class LSTMWithAttention(nn.Module):
    """LSTM with Self-Attention (Category 2: Architecture)"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # Causal mask for autoregressive attention
        seq_len = lstm_out.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
        
        # Self-attention with residual
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, attn_mask=attn_mask)
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits, hidden
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def calculate_perplexity(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            hidden = model.init_hidden(batch_size)
            logits, _ = model(x, hidden)
            
            logits_flat = logits.view(-1, model.vocab_size)
            y_flat = y.view(-1)
            
            loss = criterion(logits_flat, y_flat)
            
            non_pad = (y_flat != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def train_model(model, train_loader, dev_loader, config, verbose=True):
    """Train model with early stopping"""
    
    if config.get('label_smoothing', 0) > 0:
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history = {'train_ppl': [], 'dev_ppl': []}
    best_dev_ppl = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        model.train()
        total_loss = 0
        total_tokens = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            optimizer.zero_grad()
            hidden = model.init_hidden(batch_size)
            logits, _ = model(x, hidden)
            
            logits_flat = logits.view(-1, model.vocab_size)
            y_flat = y.view(-1)
            
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            non_pad = (y_flat != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
        
        train_ppl = np.exp(total_loss / total_tokens)
        dev_ppl = calculate_perplexity(model, dev_loader, criterion)
        
        scheduler.step(dev_ppl)
        
        history['train_ppl'].append(train_ppl)
        history['dev_ppl'].append(dev_ppl)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  Epoch {epoch+1:2d} | Train PPL: {train_ppl:8.2f} | Dev PPL: {dev_ppl:8.2f} | Time: {elapsed:.1f}s")
        
        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                if verbose:
                    print(f"  Early stopping! Best Dev PPL: {best_dev_ppl:.2f}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, best_dev_ppl


# =============================================================================
# SAMPLING STRATEGIES (Category 4)
# =============================================================================

def sample_greedy(model, vocab, start_tokens, max_length=30):
    """Greedy decoding"""
    model.eval()
    
    tokens = [vocab.sos_idx] + [vocab.encode(t) for t in start_tokens]
    generated = list(start_tokens)
    
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([tokens[-30:]]).to(device)
            hidden = model.init_hidden(1)
            logits, _ = model(x, hidden)
            
            next_token = torch.argmax(logits[0, -1, :]).item()
            
            if next_token == vocab.eos_idx:
                break
            
            tokens.append(next_token)
            generated.append(vocab.decode(next_token))
    
    return " ".join(generated)


def sample_temperature(model, vocab, start_tokens, max_length=30, temperature=1.0):
    """Temperature sampling"""
    model.eval()
    
    tokens = [vocab.sos_idx] + [vocab.encode(t) for t in start_tokens]
    generated = list(start_tokens)
    
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([tokens[-30:]]).to(device)
            hidden = model.init_hidden(1)
            logits, _ = model(x, hidden)
            
            probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == vocab.eos_idx:
                break
            
            tokens.append(next_token)
            generated.append(vocab.decode(next_token))
    
    return " ".join(generated)


def sample_top_k(model, vocab, start_tokens, max_length=30, k=10):
    """Top-k sampling"""
    model.eval()
    
    tokens = [vocab.sos_idx] + [vocab.encode(t) for t in start_tokens]
    generated = list(start_tokens)
    
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([tokens[-30:]]).to(device)
            hidden = model.init_hidden(1)
            logits, _ = model(x, hidden)
            
            top_k_logits, top_k_indices = torch.topk(logits[0, -1, :], k)
            probs = torch.softmax(top_k_logits, dim=-1)
            
            idx = torch.multinomial(probs, 1).item()
            next_token = top_k_indices[idx].item()
            
            if next_token == vocab.eos_idx:
                break
            
            tokens.append(next_token)
            generated.append(vocab.decode(next_token))
    
    return " ".join(generated)


def sample_top_p(model, vocab, start_tokens, max_length=30, p=0.9):
    """Top-p (nucleus) sampling"""
    model.eval()
    
    tokens = [vocab.sos_idx] + [vocab.encode(t) for t in start_tokens]
    generated = list(start_tokens)
    
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([tokens[-30:]]).to(device)
            hidden = model.init_hidden(1)
            logits, _ = model(x, hidden)
            
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            
            cutoff = torch.where(cumsum >= p)[0]
            cutoff_idx = cutoff[0].item() + 1 if len(cutoff) > 0 else len(probs)
            
            nucleus_probs = sorted_probs[:cutoff_idx]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            
            idx = torch.multinomial(nucleus_probs, 1).item()
            next_token = sorted_indices[idx].item()
            
            if next_token == vocab.eos_idx:
                break
            
            tokens.append(next_token)
            generated.append(vocab.decode(next_token))
    
    return " ".join(generated)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiments(language, train_sents, dev_sents, test_sents, bpe_path):
    """Run all 4 variant experiments for a language"""
    
    print(f"\n{'#'*70}")
    print(f"# {language.upper()}")
    print(f"{'#'*70}")
    
    results = {}
    
    # Build word vocabulary
    vocab = Vocabulary(min_freq=2)
    vocab.build(train_sents)
    print(f"\nWord vocabulary: {len(vocab)}")
    
    # Word-level datasets
    train_data = LanguageModelDataset(train_sents, vocab, seq_length=30)
    dev_data = LanguageModelDataset(dev_sents, vocab, seq_length=30)
    test_data = LanguageModelDataset(test_sents, vocab, seq_length=30)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)
    
    base_config = {'learning_rate': 0.001, 'epochs': 30, 'patience': 5}
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # -------------------------------------------------------------------------
    # BASELINE
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BASELINE: LSTM")
    print(f"{'='*60}")
    
    baseline = LSTMLanguageModel(len(vocab), 256, 256, 2, 0.3).to(device)
    print(f"Parameters: {sum(p.numel() for p in baseline.parameters()):,}")
    
    history, best_dev = train_model(baseline, train_loader, dev_loader, base_config)
    test_ppl = calculate_perplexity(baseline, test_loader, criterion)
    print(f"Test PPL: {test_ppl:.2f}")
    
    results['baseline'] = {
        'test_ppl': test_ppl,
        'best_dev_ppl': best_dev,
        'params': sum(p.numel() for p in baseline.parameters()),
        'epochs': len(history['train_ppl'])
    }
    
    # -------------------------------------------------------------------------
    # CATEGORY 1: INPUT REPRESENTATION - BPE
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CATEGORY 1 (Input): BPE TOKENIZATION")
    print(f"{'='*60}")
    
    if Path(bpe_path).exists():
        bpe_vocab = BPEVocabulary(bpe_path)
        print(f"BPE vocabulary: {len(bpe_vocab)}")
        
        bpe_train = BPELanguageModelDataset(train_sents, bpe_vocab, seq_length=30)
        bpe_dev = BPELanguageModelDataset(dev_sents, bpe_vocab, seq_length=30)
        bpe_test = BPELanguageModelDataset(test_sents, bpe_vocab, seq_length=30)
        
        bpe_train_loader = DataLoader(bpe_train, batch_size=64, shuffle=True)
        bpe_dev_loader = DataLoader(bpe_dev, batch_size=64)
        bpe_test_loader = DataLoader(bpe_test, batch_size=64)
        
        bpe_model = LSTMLanguageModel(len(bpe_vocab), 256, 256, 2, 0.3).to(device)
        print(f"Parameters: {sum(p.numel() for p in bpe_model.parameters()):,}")
        
        history, best_dev = train_model(bpe_model, bpe_train_loader, bpe_dev_loader, base_config)
        test_ppl = calculate_perplexity(bpe_model, bpe_test_loader, criterion)
        print(f"Test PPL: {test_ppl:.2f}")
        
        results['bpe'] = {
            'test_ppl': test_ppl,
            'best_dev_ppl': best_dev,
            'params': sum(p.numel() for p in bpe_model.parameters()),
            'vocab_size': len(bpe_vocab),
            'epochs': len(history['train_ppl'])
        }
    else:
        print(f"BPE tokenizer not found: {bpe_path}")
        results['bpe'] = None
    
    # -------------------------------------------------------------------------
    # CATEGORY 2: ARCHITECTURE - LSTM + Attention
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CATEGORY 2 (Architecture): LSTM + SELF-ATTENTION")
    print(f"{'='*60}")
    
    attn_model = LSTMWithAttention(len(vocab), 256, 256, 2, 0.3).to(device)
    print(f"Parameters: {sum(p.numel() for p in attn_model.parameters()):,}")
    
    history, best_dev = train_model(attn_model, train_loader, dev_loader, base_config)
    test_ppl = calculate_perplexity(attn_model, test_loader, criterion)
    print(f"Test PPL: {test_ppl:.2f}")
    
    results['attention'] = {
        'test_ppl': test_ppl,
        'best_dev_ppl': best_dev,
        'params': sum(p.numel() for p in attn_model.parameters()),
        'epochs': len(history['train_ppl'])
    }
    
    # -------------------------------------------------------------------------
    # CATEGORY 3: TRAINING - Label Smoothing
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CATEGORY 3 (Training): LABEL SMOOTHING (0.1)")
    print(f"{'='*60}")
    
    ls_config = {'learning_rate': 0.001, 'epochs': 30, 'patience': 5, 'label_smoothing': 0.1}
    
    ls_model = LSTMLanguageModel(len(vocab), 256, 256, 2, 0.3).to(device)
    print(f"Parameters: {sum(p.numel() for p in ls_model.parameters()):,}")
    
    history, best_dev = train_model(ls_model, train_loader, dev_loader, ls_config)
    test_ppl = calculate_perplexity(ls_model, test_loader, criterion)
    print(f"Test PPL: {test_ppl:.2f}")
    
    results['label_smoothing'] = {
        'test_ppl': test_ppl,
        'best_dev_ppl': best_dev,
        'params': sum(p.numel() for p in ls_model.parameters()),
        'epochs': len(history['train_ppl'])
    }
    
    # -------------------------------------------------------------------------
    # CATEGORY 4: SAMPLING STRATEGIES
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CATEGORY 4 (Sampling): DIFFERENT DECODING STRATEGIES")
    print(f"{'='*60}")
    
    # Use baseline model for sampling
    if language.lower() == 'finnish':
        prompts = [['hän', 'on'], ['se', 'oli']]
    else:
        prompts = [['वह', 'है'], ['यह', 'एक']]
    
    sampling_results = {}
    
    for prompt in prompts:
        prompt_str = " ".join(prompt)
        print(f"\nPrompt: '{prompt_str}'")
        print("-" * 50)
        
        sampling_results[prompt_str] = {}
        
        greedy = sample_greedy(baseline, vocab, prompt)
        print(f"Greedy:       {greedy}")
        sampling_results[prompt_str]['greedy'] = greedy
        
        temp_low = sample_temperature(baseline, vocab, prompt, temperature=0.7)
        print(f"Temp=0.7:     {temp_low}")
        sampling_results[prompt_str]['temp_0.7'] = temp_low
        
        temp_high = sample_temperature(baseline, vocab, prompt, temperature=1.5)
        print(f"Temp=1.5:     {temp_high}")
        sampling_results[prompt_str]['temp_1.5'] = temp_high
        
        topk = sample_top_k(baseline, vocab, prompt, k=10)
        print(f"Top-k (k=10): {topk}")
        sampling_results[prompt_str]['top_k'] = topk
        
        topp = sample_top_p(baseline, vocab, prompt, p=0.9)
        print(f"Top-p (p=0.9): {topp}")
        sampling_results[prompt_str]['top_p'] = topp
    
    results['sampling'] = sampling_results
    
    return results


def main():
    print("="*70)
    print("DELIVERABLE 4: MODEL VARIANTS (4 CATEGORIES)")
    print("="*70)
    
    # File paths
    finnish_files = {
        'train': "data/ud/UD_Finnish-TDT/fi_tdt-ud-train.conllu",
        'dev': "data/ud/UD_Finnish-TDT/fi_tdt-ud-dev.conllu",
        'test': "data/ud/UD_Finnish-TDT/fi_tdt-ud-test.conllu",
        'bpe': "tokenizer_Finnish_BPE.json"
    }
    
    hindi_files = {
        'train': "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu",
        'dev': "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu",
        'test': "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-test.conllu",
        'bpe': "tokenizer_Hindi_BPE.json"
    }
    
    # Load data
    print("\nLoading Finnish...")
    fi_train = load_conllu(finnish_files['train'])
    fi_dev = load_conllu(finnish_files['dev'])
    fi_test = load_conllu(finnish_files['test'])
    
    print("Loading Hindi...")
    hi_train = load_conllu(hindi_files['train'])
    hi_dev = load_conllu(hindi_files['dev'])
    hi_test = load_conllu(hindi_files['test'])
    
    # Run experiments
    finnish_results = run_experiments("Finnish", fi_train, fi_dev, fi_test, finnish_files['bpe'])
    hindi_results = run_experiments("Hindi", hi_train, hi_dev, hi_test, hindi_files['bpe'])
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: ALL VARIANTS")
    print(f"{'='*70}")
    
    print(f"\n{'Variant':<25} {'Category':<20} {'Finnish PPL':<15} {'Hindi PPL':<15}")
    print("-" * 75)
    
    variants = [
        ('baseline', 'Baseline'),
        ('bpe', '1. Input Rep'),
        ('attention', '2. Architecture'),
        ('label_smoothing', '3. Training')
    ]
    
    for var, cat in variants:
        fi_ppl = finnish_results[var]['test_ppl'] if finnish_results.get(var) else 'N/A'
        hi_ppl = hindi_results[var]['test_ppl'] if hindi_results.get(var) else 'N/A'
        
        if isinstance(fi_ppl, float):
            print(f"{var:<25} {cat:<20} {fi_ppl:<15.2f} {hi_ppl:<15.2f}")
        else:
            print(f"{var:<25} {cat:<20} {fi_ppl:<15} {hi_ppl:<15}")
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    var_names = ['Baseline', 'BPE\n(Input)', 'Attention\n(Arch)', 'Label Smooth\n(Training)']
    x = np.arange(len(var_names))
    width = 0.35
    
    fi_ppls = [finnish_results[v[0]]['test_ppl'] if finnish_results.get(v[0]) else 0 for v in variants]
    hi_ppls = [hindi_results[v[0]]['test_ppl'] if hindi_results.get(v[0]) else 0 for v in variants]
    
    ax.bar(x - width/2, fi_ppls, width, label='Finnish', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, hi_ppls, width, label='Hindi', alpha=0.8, color='darkorange')
    
    ax.set_ylabel('Test Perplexity')
    ax.set_title('Model Variants Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(var_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (fi, hi) in enumerate(zip(fi_ppls, hi_ppls)):
        if fi > 0:
            ax.text(i - width/2, fi + 2, f'{fi:.1f}', ha='center', va='bottom', fontsize=9)
        if hi > 0:
            ax.text(i + width/2, hi + 2, f'{hi:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('deliverable4_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: deliverable4_comparison.png")
    plt.close()
    
    # Save results
    save_results = {
        'finnish': {k: v for k, v in finnish_results.items() if k != 'sampling'},
        'hindi': {k: v for k, v in hindi_results.items() if k != 'sampling'},
        'finnish_sampling': finnish_results.get('sampling', {}),
        'hindi_sampling': hindi_results.get('sampling', {})
    }
    
    with open('deliverable4_results.json', 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print("Results saved: deliverable4_results.json")


if __name__ == "__main__":
    main()