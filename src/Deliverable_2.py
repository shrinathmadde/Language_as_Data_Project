"""
Deliverable 2: N-Gram Language Model Baseline
"""

import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import math

class NgramLanguageModel:
    """N-gram language model with add-k smoothing"""
    
    def __init__(self, n=3, k=0.01):
        self.n = n
        self.k = k  # Smoothing parameter
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        self.unk_token = "<UNK>"
        self.start_token = "<s>"
        self.end_token = "</s>"
        
    def load_conllu(self, filepath):
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
    
    def build_vocab(self, sentences, min_freq=2):
        """Build vocabulary with frequency threshold"""
        word_freq = Counter()
        for sent in sentences:
            word_freq.update(sent)
        
        self.vocab = {w for w, c in word_freq.items() if c >= min_freq}
        self.vocab.add(self.unk_token)
        self.vocab.add(self.start_token)
        self.vocab.add(self.end_token)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Words below threshold: {sum(1 for c in word_freq.values() if c < min_freq)}")
        
        return self.vocab
    
    def preprocess_sentence(self, sentence):
        """Add start/end tokens and replace OOV with UNK"""
        tokens = [self.start_token] * (self.n - 1)
        tokens += [w if w in self.vocab else self.unk_token for w in sentence]
        tokens += [self.end_token]
        return tokens
    
    def train(self, sentences):
        """Train n-gram model"""
        print(f"Training {self.n}-gram model on {len(sentences)} sentences...")
        
        for sentence in sentences:
            tokens = self.preprocess_sentence(sentence)
            
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                word = tokens[i]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
        
        print(f"Unique {self.n}-grams: {sum(len(v) for v in self.ngram_counts.values())}")
        print(f"Unique contexts: {len(self.context_counts)}")
    
    def probability(self, word, context):
        """Calculate smoothed probability P(word|context)"""
        context = tuple(context)
        count_ngram = self.ngram_counts[context][word]
        count_context = self.context_counts[context]
        
        # Add-k smoothing
        prob = (count_ngram + self.k) / (count_context + self.k * len(self.vocab))
        return prob
    
    def sentence_log_probability(self, sentence):
        """Calculate log probability of a sentence"""
        tokens = self.preprocess_sentence(sentence)
        log_prob = 0.0
        
        for i in range(self.n - 1, len(tokens)):
            context = tokens[i - self.n + 1:i]
            word = tokens[i]
            prob = self.probability(word, context)
            log_prob += math.log2(prob)
        
        return log_prob, len(tokens) - (self.n - 1)
    
    def perplexity(self, sentences):
        """Calculate perplexity on a set of sentences"""
        total_log_prob = 0.0
        total_tokens = 0
        oov_count = 0
        total_words = 0
        
        for sentence in sentences:
            # Count OOV before preprocessing
            for w in sentence:
                total_words += 1
                if w not in self.vocab:
                    oov_count += 1
            
            log_prob, num_tokens = self.sentence_log_probability(sentence)
            total_log_prob += log_prob
            total_tokens += num_tokens
        
        avg_log_prob = total_log_prob / total_tokens
        ppl = 2 ** (-avg_log_prob)
        oov_rate = 100 * oov_count / total_words
        
        return ppl, oov_rate, total_tokens


def run_experiment(language, train_file, dev_file, test_file):
    """Run n-gram experiment for a language"""
    print(f"\n{'='*60}")
    print(f"N-GRAM LANGUAGE MODEL: {language.upper()}")
    print(f"{'='*60}")
    
    results = {}
    
    for n in [2, 3, 4]:
        print(f"\n{'-'*40}")
        print(f"{n}-gram Model")
        print(f"{'-'*40}")
        
        model = NgramLanguageModel(n=n, k=0.01)
        
        # Load data
        train_sents = model.load_conllu(train_file)
        dev_sents = model.load_conllu(dev_file)
        test_sents = model.load_conllu(test_file)
        
        print(f"Train: {len(train_sents)} sentences")
        print(f"Dev: {len(dev_sents)} sentences")
        print(f"Test: {len(test_sents)} sentences")
        
        # Build vocab and train
        model.build_vocab(train_sents, min_freq=2)
        model.train(train_sents)
        
        # Evaluate
        dev_ppl, dev_oov, _ = model.perplexity(dev_sents)
        test_ppl, test_oov, _ = model.perplexity(test_sents)
        
        print(f"\nResults:")
        print(f"  Dev  Perplexity: {dev_ppl:.2f} (OOV: {dev_oov:.2f}%)")
        print(f"  Test Perplexity: {test_ppl:.2f} (OOV: {test_oov:.2f}%)")
        
        results[n] = {
            'dev_ppl': dev_ppl,
            'test_ppl': test_ppl,
            'dev_oov': dev_oov,
            'test_oov': test_oov
        }
    
    return results


def main():
    # File paths
    finnish_train = "data/ud/UD_Finnish-TDT/fi_tdt-ud-train.conllu"
    finnish_dev = "data/ud/UD_Finnish-TDT/fi_tdt-ud-dev.conllu"
    finnish_test = "data/ud/UD_Finnish-TDT/fi_tdt-ud-test.conllu"
    
    hindi_train = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
    hindi_dev = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu"
    hindi_test = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-test.conllu"
    
    # Run experiments
    finnish_results = run_experiment("Finnish", finnish_train, finnish_dev, finnish_test)
    hindi_results = run_experiment("Hindi", hindi_train, hindi_dev, hindi_test)
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<10} {'Finnish Dev':<15} {'Finnish Test':<15} {'Hindi Dev':<15} {'Hindi Test':<15}")
    print("-" * 70)
    for n in [2, 3, 4]:
        print(f"{n}-gram     {finnish_results[n]['dev_ppl']:<15.2f} {finnish_results[n]['test_ppl']:<15.2f} "
              f"{hindi_results[n]['dev_ppl']:<15.2f} {hindi_results[n]['test_ppl']:<15.2f}")


if __name__ == "__main__":
    main()