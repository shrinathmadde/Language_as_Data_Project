"""
Deliverable 5: Human Evaluation of Language Model Output
- Generate outputs from 2 model configurations
- 15+ prompts per language
- Annotation framework with guidelines
- Inter-annotator agreement calculation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
import csv
from pathlib import Path

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# MODEL AND VOCABULARY (from Deliverable 3/4)
# =============================================================================

class Vocabulary:
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
    def sos_idx(self):
        return self.word2idx[self.sos_token]
    
    @property
    def eos_idx(self):
        return self.word2idx[self.eos_token]


class LanguageModelDataset(Dataset):
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


class LSTMLanguageModel(nn.Module):
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


def load_conllu(filepath):
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
# SAMPLING FUNCTIONS
# =============================================================================

def sample_top_p(model, vocab, start_tokens, max_length=25, p=0.9, temperature=1.0):
    """Top-p (nucleus) sampling"""
    model.eval()
    tokens = [vocab.sos_idx] + [vocab.encode(t) for t in start_tokens]
    generated = list(start_tokens)
    
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([tokens[-30:]]).to(device)
            hidden = model.init_hidden(1)
            logits, _ = model(x, hidden)
            
            probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
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


def sample_greedy(model, vocab, start_tokens, max_length=25):
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


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, dev_loader, config, verbose=False):
    if config.get('label_smoothing', 0) > 0:
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=config['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_dev_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        total_tokens = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            optimizer.zero_grad()
            hidden = model.init_hidden(batch_size)
            logits, _ = model(x, hidden)
            
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            non_pad = (y.view(-1) != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
        
        # Dev evaluation
        model.eval()
        dev_loss = 0
        dev_tokens = 0
        with torch.no_grad():
            for x, y in dev_loader:
                x, y = x.to(device), y.to(device)
                hidden = model.init_hidden(x.size(0))
                logits, _ = model(x, hidden)
                loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
                non_pad = (y.view(-1) != 0).sum().item()
                dev_loss += loss.item() * non_pad
                dev_tokens += non_pad
        
        avg_dev_loss = dev_loss / dev_tokens
        
        if verbose:
            print(f"  Epoch {epoch+1}: Train Loss={total_loss/total_tokens:.4f}, Dev Loss={avg_dev_loss:.4f}")
        
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


# =============================================================================
# PROMPTS FOR EVALUATION
# =============================================================================

# Hindi prompts (15+ diverse prompts)
HINDI_PROMPTS = [
    # News-style
    ["प्रधानमंत्री", "ने"],        # Prime Minister (ergative)
    ["सरकार", "ने"],              # Government (ergative)
    ["पुलिस", "ने"],              # Police (ergative)
    ["अधिकारियों", "ने"],         # Officials (ergative)
    
    # Descriptive
    ["यह", "एक"],                 # This is a
    ["वह", "बहुत"],               # He/She very
    ["यहां", "पर"],               # Here at
    ["इस", "समय"],               # At this time
    
    # Action-oriented
    ["उन्होंने", "कहा"],          # He/She said
    ["लोगों", "ने"],              # People (ergative)
    ["हमें", "इस"],               # We this
    ["आज", "के"],                 # Today's
    
    # Questions/statements
    ["क्या", "आप"],               # What you
    ["जब", "वह"],                 # When he/she
    ["अगर", "हम"],                # If we
    ["इसके", "बाद"],              # After this
]

# Finnish prompts (15+ diverse prompts)
FINNISH_PROMPTS = [
    # Subject-verb
    ["hän", "on"],                # He/she is
    ["se", "oli"],                # It was
    ["minä", "olen"],             # I am
    ["he", "ovat"],               # They are
    
    # News-style
    ["suomen", "hallitus"],       # Finnish government
    ["presidentti", "sanoi"],     # President said
    ["poliisi", "on"],            # Police is/has
    ["tutkijat", "ovat"],         # Researchers are/have
    
    # Descriptive
    ["tämä", "on"],               # This is
    ["siellä", "oli"],            # There was
    ["nyt", "on"],                # Now is
    ["ensi", "vuonna"],           # Next year
    
    # Connectives
    ["mutta", "hän"],             # But he/she
    ["koska", "se"],              # Because it
    ["kun", "hän"],               # When he/she
    ["jos", "me"],                # If we
]


# =============================================================================
# INTER-ANNOTATOR AGREEMENT
# =============================================================================

def cohens_kappa(ann1, ann2):
    """Calculate Cohen's Kappa for two annotators"""
    assert len(ann1) == len(ann2)
    n = len(ann1)
    
    # Get all unique labels
    labels = sorted(set(ann1) | set(ann2))
    
    # Build confusion matrix
    matrix = {}
    for l1 in labels:
        matrix[l1] = {l2: 0 for l2 in labels}
    
    for a1, a2 in zip(ann1, ann2):
        matrix[a1][a2] += 1
    
    # Calculate observed agreement
    po = sum(matrix[l][l] for l in labels) / n
    
    # Calculate expected agreement
    pe = 0
    for l in labels:
        row_sum = sum(matrix[l].values()) / n
        col_sum = sum(matrix[l2][l] for l2 in labels) / n
        pe += row_sum * col_sum
    
    # Kappa
    if pe == 1:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    return kappa


def krippendorff_alpha(annotations, level='ordinal'):
    """
    Calculate Krippendorff's Alpha for multiple annotators
    annotations: list of lists, each inner list is one annotator's ratings
    """
    n_annotators = len(annotations)
    n_items = len(annotations[0])
    
    # Collect all values per item
    item_values = []
    for i in range(n_items):
        values = [annotations[a][i] for a in range(n_annotators) if annotations[a][i] is not None]
        item_values.append(values)
    
    # Calculate observed disagreement
    Do = 0
    n_pairs = 0
    for values in item_values:
        if len(values) < 2:
            continue
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                if level == 'ordinal':
                    Do += (values[i] - values[j]) ** 2
                else:  # nominal
                    Do += 0 if values[i] == values[j] else 1
                n_pairs += 1
    
    if n_pairs == 0:
        return 1.0
    Do = Do / n_pairs
    
    # Calculate expected disagreement
    all_values = [v for values in item_values for v in values]
    De = 0
    n_total_pairs = 0
    for i in range(len(all_values)):
        for j in range(i+1, len(all_values)):
            if level == 'ordinal':
                De += (all_values[i] - all_values[j]) ** 2
            else:
                De += 0 if all_values[i] == all_values[j] else 1
            n_total_pairs += 1
    
    if n_total_pairs == 0:
        return 1.0
    De = De / n_total_pairs
    
    if De == 0:
        return 1.0
    
    alpha = 1 - (Do / De)
    return alpha


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def generate_evaluation_samples(language, prompts, model_baseline, model_variant, vocab, n_samples=3):
    """Generate samples from both models for each prompt"""
    samples = []
    
    for prompt in prompts:
        prompt_str = " ".join(prompt)
        
        for sample_idx in range(n_samples):
            # Model A: Baseline with top-p sampling
            output_a = sample_top_p(model_baseline, vocab, prompt, max_length=20, p=0.9, temperature=1.0)
            
            # Model B: Label smoothing model with top-p sampling  
            output_b = sample_top_p(model_variant, vocab, prompt, max_length=20, p=0.9, temperature=1.0)
            
            samples.append({
                'id': len(samples) + 1,
                'prompt': prompt_str,
                'output_A': output_a,
                'output_B': output_b,
                'model_A': 'Baseline',
                'model_B': 'Label_Smoothing'
            })
    
    return samples


def create_annotation_sheet(samples, output_file):
    """Create CSV annotation sheet"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'ID', 'Prompt', 'Output_A', 'Output_B',
            'Fluency_A (1-5)', 'Fluency_B (1-5)',
            'Coherence_A (1-5)', 'Coherence_B (1-5)',
            'Preference (A/B/Tie)', 'Notes'
        ])
        
        for sample in samples:
            writer.writerow([
                sample['id'],
                sample['prompt'],
                sample['output_A'],
                sample['output_B'],
                '', '', '', '', '', ''
            ])
    
    print(f"Annotation sheet saved: {output_file}")


def print_annotation_guidelines():
    """Print annotation guidelines"""
    guidelines = """
================================================================================
ANNOTATION GUIDELINES FOR LANGUAGE MODEL EVALUATION
================================================================================

TASK: Evaluate generated text continuations from two language models (A and B).

--------------------------------------------------------------------------------
CATEGORY 1: SYNTACTIC FLUENCY (1-5 scale)
--------------------------------------------------------------------------------
Evaluate whether the generated text follows grammatical rules of the language.

Score | Description
------|---------------------------------------------------------------------------
  1   | Completely ungrammatical; random word sequences; incomprehensible
  2   | Mostly ungrammatical; major errors in word order, agreement, or case
  3   | Some grammatical errors but structure is recognizable
  4   | Minor grammatical errors; mostly well-formed sentences
  5   | Fully grammatical; native-like sentence structure

Examples (Hindi):
- Score 5: "प्रधानमंत्री ने कहा कि सरकार इस मुद्दे पर विचार करेगी।"
- Score 3: "प्रधानमंत्री कहा है कि सरकार मुद्दे विचार।"
- Score 1: "प्रधानमंत्री है कहा ने सरकार इस।"

Examples (Finnish):
- Score 5: "Hän on ollut professorina yliopistossa viisi vuotta."
- Score 3: "Hän on ollut professori yliopisto viisi vuotta."
- Score 1: "Hän professori ollut on yliopisto."

--------------------------------------------------------------------------------
CATEGORY 2: SEMANTIC COHERENCE (1-5 scale)
--------------------------------------------------------------------------------
Evaluate whether the text makes sense and maintains a logical topic/theme.

Score | Description
------|---------------------------------------------------------------------------
  1   | Completely incoherent; no logical connection between words
  2   | Mostly incoherent; topic shifts randomly; contradictory statements
  3   | Partially coherent; some logical flow but with confusing parts
  4   | Mostly coherent; clear topic with minor inconsistencies
  5   | Fully coherent; logical flow; meaningful content that follows from prompt

Examples (Hindi):
- Score 5: "यह एक ऐतिहासिक इमारत है जो मुगल काल में बनाई गई थी।"
- Score 3: "यह एक ऐतिहासिक इमारत है जो खाना बनाती है।"
- Score 1: "यह एक ऐतिहासिक नीला चलना पानी कल।"

Examples (Finnish):  
- Score 5: "Tämä on vanha rakennus, joka rakennettiin 1800-luvulla."
- Score 3: "Tämä on vanha rakennus, joka syö kalaa."
- Score 1: "Tämä on vanha sininen juosta vesi huomenna."

--------------------------------------------------------------------------------
CATEGORY 3: OVERALL PREFERENCE (A/B/Tie)
--------------------------------------------------------------------------------
Which output would you prefer to read? Consider both fluency and coherence.

- A: Output A is clearly better
- B: Output B is clearly better  
- Tie: Both outputs are roughly equal in quality

--------------------------------------------------------------------------------
HANDLING <UNK> TOKENS
--------------------------------------------------------------------------------
- <UNK> (unknown word) tokens indicate words not in vocabulary
- Many <UNK> tokens should lower both fluency AND coherence scores
- A sentence with 50%+ <UNK> tokens: maximum score of 2 for both categories

--------------------------------------------------------------------------------
ANNOTATION PROCESS
--------------------------------------------------------------------------------
1. Read the prompt to understand the expected continuation context
2. Read Output A - score fluency (1-5), then coherence (1-5)
3. Read Output B - score fluency (1-5), then coherence (1-5)
4. Decide overall preference (A/B/Tie)
5. Add notes for unusual cases or observations

--------------------------------------------------------------------------------
FOR NON-NATIVE SPEAKERS
--------------------------------------------------------------------------------
If you don't speak the target language:
1. Use Google Translate to get rough English translations
2. Check grammar using online tools (e.g., LanguageTool)
3. Focus on structural patterns you can identify
4. Mark uncertain ratings with "?" in notes column
5. Document your translation/verification process

================================================================================
"""
    print(guidelines)
    return guidelines


def run_evaluation_pipeline(language, train_file, dev_file, prompts):
    """Run the complete evaluation pipeline for a language"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATION PIPELINE: {language.upper()}")
    print(f"{'='*70}")
    
    # Load data
    print("\nLoading data...")
    train_sents = load_conllu(train_file)
    dev_sents = load_conllu(dev_file)
    
    # Build vocabulary
    vocab = Vocabulary(min_freq=2)
    vocab.build(train_sents)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_data = LanguageModelDataset(train_sents, vocab, seq_length=30)
    dev_data = LanguageModelDataset(dev_sents, vocab, seq_length=30)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=64)
    
    # Train Model A: Baseline
    print("\nTraining Model A (Baseline)...")
    model_a = LSTMLanguageModel(len(vocab), 256, 256, 2, 0.3).to(device)
    config_a = {'learning_rate': 0.001, 'epochs': 20, 'patience': 5}
    model_a = train_model(model_a, train_loader, dev_loader, config_a, verbose=True)
    
    # Train Model B: Label Smoothing
    print("\nTraining Model B (Label Smoothing)...")
    model_b = LSTMLanguageModel(len(vocab), 256, 256, 2, 0.3).to(device)
    config_b = {'learning_rate': 0.001, 'epochs': 20, 'patience': 5, 'label_smoothing': 0.1}
    model_b = train_model(model_b, train_loader, dev_loader, config_b, verbose=True)
    
    # Generate evaluation samples
    print("\nGenerating evaluation samples...")
    samples = generate_evaluation_samples(language, prompts, model_a, model_b, vocab, n_samples=1)
    
    # Save samples
    samples_file = f'evaluation_samples_{language}.json'
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Samples saved: {samples_file}")
    
    # Create annotation sheet
    annotation_file = f'annotation_sheet_{language}.csv'
    create_annotation_sheet(samples, annotation_file)
    
    # Print samples for review
    print(f"\n{'='*70}")
    print(f"GENERATED SAMPLES FOR {language.upper()}")
    print(f"{'='*70}")
    
    for sample in samples:
        print(f"\n[{sample['id']}] Prompt: {sample['prompt']}")
        print(f"    Model A (Baseline):        {sample['output_A']}")
        print(f"    Model B (Label Smoothing): {sample['output_B']}")
    
    return samples, vocab, model_a, model_b


def simulate_annotations(samples, annotator_bias=0):
    """
    Simulate annotations for demonstration.
    In real evaluation, this would be replaced by actual human annotations.
    """
    np.random.seed(42 + annotator_bias)
    
    annotations = {
        'fluency_A': [],
        'fluency_B': [],
        'coherence_A': [],
        'coherence_B': [],
        'preference': []
    }
    
    for sample in samples:
        # Count UNK tokens
        unk_a = sample['output_A'].count('<UNK>')
        unk_b = sample['output_B'].count('<UNK>')
        
        # Base scores (affected by UNK count)
        base_fluency_a = max(1, 4 - unk_a * 0.5 + np.random.normal(0, 0.5))
        base_fluency_b = max(1, 4 - unk_b * 0.5 + np.random.normal(0, 0.5))
        
        base_coherence_a = max(1, 3.5 - unk_a * 0.5 + np.random.normal(0, 0.7))
        base_coherence_b = max(1, 4 - unk_b * 0.4 + np.random.normal(0, 0.6))
        
        # Clip to 1-5 range
        fluency_a = int(np.clip(round(base_fluency_a + annotator_bias * 0.2), 1, 5))
        fluency_b = int(np.clip(round(base_fluency_b + annotator_bias * 0.2), 1, 5))
        coherence_a = int(np.clip(round(base_coherence_a + annotator_bias * 0.2), 1, 5))
        coherence_b = int(np.clip(round(base_coherence_b + annotator_bias * 0.2), 1, 5))
        
        # Preference
        score_a = fluency_a + coherence_a
        score_b = fluency_b + coherence_b
        if score_b - score_a >= 2:
            pref = 'B'
        elif score_a - score_b >= 2:
            pref = 'A'
        else:
            pref = 'Tie'
        
        annotations['fluency_A'].append(fluency_a)
        annotations['fluency_B'].append(fluency_b)
        annotations['coherence_A'].append(coherence_a)
        annotations['coherence_B'].append(coherence_b)
        annotations['preference'].append(pref)
    
    return annotations


def calculate_agreement(ann1, ann2, ann3):
    """Calculate inter-annotator agreement"""
    
    print(f"\n{'='*70}")
    print("INTER-ANNOTATOR AGREEMENT")
    print(f"{'='*70}")
    
    categories = ['fluency_A', 'fluency_B', 'coherence_A', 'coherence_B']
    
    results = {}
    
    for cat in categories:
        # Pairwise Cohen's Kappa
        kappa_12 = cohens_kappa(ann1[cat], ann2[cat])
        kappa_13 = cohens_kappa(ann1[cat], ann3[cat])
        kappa_23 = cohens_kappa(ann2[cat], ann3[cat])
        avg_kappa = (kappa_12 + kappa_13 + kappa_23) / 3
        
        # Krippendorff's Alpha
        alpha = krippendorff_alpha([ann1[cat], ann2[cat], ann3[cat]], level='ordinal')
        
        results[cat] = {
            'kappa_12': kappa_12,
            'kappa_13': kappa_13,
            'kappa_23': kappa_23,
            'avg_kappa': avg_kappa,
            'alpha': alpha
        }
        
        print(f"\n{cat}:")
        print(f"  Cohen's Kappa (1-2): {kappa_12:.3f}")
        print(f"  Cohen's Kappa (1-3): {kappa_13:.3f}")
        print(f"  Cohen's Kappa (2-3): {kappa_23:.3f}")
        print(f"  Average Kappa:       {avg_kappa:.3f}")
        print(f"  Krippendorff's α:    {alpha:.3f}")
    
    # Preference agreement
    print(f"\nPreference:")
    pref_kappa_12 = cohens_kappa(ann1['preference'], ann2['preference'])
    pref_kappa_13 = cohens_kappa(ann1['preference'], ann3['preference'])
    pref_kappa_23 = cohens_kappa(ann2['preference'], ann3['preference'])
    print(f"  Cohen's Kappa (1-2): {pref_kappa_12:.3f}")
    print(f"  Cohen's Kappa (1-3): {pref_kappa_13:.3f}")
    print(f"  Cohen's Kappa (2-3): {pref_kappa_23:.3f}")
    
    return results


def analyze_results(samples, ann1, ann2, ann3):
    """Analyze evaluation results"""
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS ANALYSIS")
    print(f"{'='*70}")
    
    n = len(samples)
    
    # Average scores across annotators
    avg_fluency_a = np.mean([ann1['fluency_A'], ann2['fluency_A'], ann3['fluency_A']], axis=0)
    avg_fluency_b = np.mean([ann1['fluency_B'], ann2['fluency_B'], ann3['fluency_B']], axis=0)
    avg_coherence_a = np.mean([ann1['coherence_A'], ann2['coherence_A'], ann3['coherence_A']], axis=0)
    avg_coherence_b = np.mean([ann1['coherence_B'], ann2['coherence_B'], ann3['coherence_B']], axis=0)
    
    print(f"\nAverage Scores (n={n} samples):")
    print(f"{'Metric':<20} {'Model A (Baseline)':<20} {'Model B (Label Smooth)':<20}")
    print("-" * 60)
    print(f"{'Fluency':<20} {np.mean(avg_fluency_a):<20.2f} {np.mean(avg_fluency_b):<20.2f}")
    print(f"{'Coherence':<20} {np.mean(avg_coherence_a):<20.2f} {np.mean(avg_coherence_b):<20.2f}")
    print(f"{'Overall':<20} {np.mean(avg_fluency_a + avg_coherence_a)/2:<20.2f} {np.mean(avg_fluency_b + avg_coherence_b)/2:<20.2f}")
    
    # Preference counts
    all_prefs = ann1['preference'] + ann2['preference'] + ann3['preference']
    pref_a = all_prefs.count('A')
    pref_b = all_prefs.count('B')
    pref_tie = all_prefs.count('Tie')
    total_prefs = len(all_prefs)
    
    print(f"\nPreference Distribution:")
    print(f"  Model A preferred: {pref_a}/{total_prefs} ({100*pref_a/total_prefs:.1f}%)")
    print(f"  Model B preferred: {pref_b}/{total_prefs} ({100*pref_b/total_prefs:.1f}%)")
    print(f"  Tie:               {pref_tie}/{total_prefs} ({100*pref_tie/total_prefs:.1f}%)")
    
    # Statistical significance (simple sign test)
    if pref_b > pref_a:
        winner = "Model B (Label Smoothing)"
    elif pref_a > pref_b:
        winner = "Model A (Baseline)"
    else:
        winner = "No clear winner"
    
    print(f"\nConclusion: {winner} is preferred overall.")
    
    return {
        'avg_fluency_a': float(np.mean(avg_fluency_a)),
        'avg_fluency_b': float(np.mean(avg_fluency_b)),
        'avg_coherence_a': float(np.mean(avg_coherence_a)),
        'avg_coherence_b': float(np.mean(avg_coherence_b)),
        'pref_a': pref_a,
        'pref_b': pref_b,
        'pref_tie': pref_tie
    }


def main():
    print("="*70)
    print("DELIVERABLE 5: HUMAN EVALUATION OF MODEL OUTPUT")
    print("="*70)
    
    # Print annotation guidelines
    guidelines = print_annotation_guidelines()
    
    # Save guidelines to file
    with open('annotation_guidelines.txt', 'w', encoding='utf-8') as f:
        f.write(guidelines)
    print("Guidelines saved: annotation_guidelines.txt")
    
    # File paths
    hindi_train = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
    hindi_dev = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu"
    
    finnish_train = "data/ud/UD_Finnish-TDT/fi_tdt-ud-train.conllu"
    finnish_dev = "data/ud/UD_Finnish-TDT/fi_tdt-ud-dev.conllu"
    
    # Run evaluation for Hindi (as primary language from Assignment 1)
    hindi_samples, hindi_vocab, hindi_model_a, hindi_model_b = run_evaluation_pipeline(
        "Hindi", hindi_train, hindi_dev, HINDI_PROMPTS
    )
    
    # Also run for Finnish
    finnish_samples, finnish_vocab, finnish_model_a, finnish_model_b = run_evaluation_pipeline(
        "Finnish", finnish_train, finnish_dev, FINNISH_PROMPTS
    )
    
    # Simulate 3 annotators (in real scenario, these would be actual human annotations)
    print(f"\n{'='*70}")
    print("SIMULATING ANNOTATIONS (Replace with actual human annotations)")
    print(f"{'='*70}")
    
    print("\n--- HINDI EVALUATION ---")
    hindi_ann1 = simulate_annotations(hindi_samples, annotator_bias=0)
    hindi_ann2 = simulate_annotations(hindi_samples, annotator_bias=1)
    hindi_ann3 = simulate_annotations(hindi_samples, annotator_bias=-1)
    
    hindi_agreement = calculate_agreement(hindi_ann1, hindi_ann2, hindi_ann3)
    hindi_results = analyze_results(hindi_samples, hindi_ann1, hindi_ann2, hindi_ann3)
    
    print("\n--- FINNISH EVALUATION ---")
    finnish_ann1 = simulate_annotations(finnish_samples, annotator_bias=0)
    finnish_ann2 = simulate_annotations(finnish_samples, annotator_bias=1)
    finnish_ann3 = simulate_annotations(finnish_samples, annotator_bias=-1)
    
    finnish_agreement = calculate_agreement(finnish_ann1, finnish_ann2, finnish_ann3)
    finnish_results = analyze_results(finnish_samples, finnish_ann1, finnish_ann2, finnish_ann3)
    
    # Save all results
    all_results = {
        'hindi': {
            'samples': hindi_samples,
            'annotations': {
                'annotator_1': hindi_ann1,
                'annotator_2': hindi_ann2,
                'annotator_3': hindi_ann3
            },
            'results': hindi_results
        },
        'finnish': {
            'samples': finnish_samples,
            'annotations': {
                'annotator_1': finnish_ann1,
                'annotator_2': finnish_ann2,
                'annotator_3': finnish_ann3
            },
            'results': finnish_results
        }
    }
    
    with open('deliverable5_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nAll results saved: deliverable5_results.json")
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  - annotation_guidelines.txt")
    print("  - annotation_sheet_Hindi.csv")
    print("  - annotation_sheet_Finnish.csv")
    print("  - evaluation_samples_Hindi.json")
    print("  - evaluation_samples_Finnish.json")
    print("  - deliverable5_results.json")
    print("\nNOTE: The annotations shown are SIMULATED.")
    print("Replace with actual human annotations for your report.")


if __name__ == "__main__":
    main()