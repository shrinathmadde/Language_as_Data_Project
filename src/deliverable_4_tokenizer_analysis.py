"""
Deliverable 4: Morphological Analysis using Sub-word Tokenizers
Train and analyze Unigram or BPE tokenizers for German and Hindi
"""

import os
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.normalizers import NFD, StripAccents, Lowercase, NFKC
from tokenizers.pre_tokenizers import Whitespace, Metaspace

sns.set_style("whitegrid")


class TokenizerAnalyzer:
    """Train and analyze sub-word tokenizers"""

    def __init__(self, language_name, ud_train_file, ud_test_file, vocab_size=5000, model_type='BPE'):
        self.language = language_name
        self.ud_train_file = ud_train_file
        self.ud_test_file = ud_test_file
        self.vocab_size = vocab_size
        self.model_type = model_type  # 'BPE' or 'Unigram'
        self.tokenizer = None
        self.train_words = []
        self.test_words = []
        self.word_frequencies = Counter()

    def load_conllu_data(self, conllu_file, is_train=True):
        """Extract words from CoNLL-U file"""
        print(f"\nLoading data from {conllu_file}...")
        words = []
        sentences = []
        current_sentence = []

        with open(conllu_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                elif not line:
                    if current_sentence:
                        sentences.append(' '.join(current_sentence))
                        words.extend(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split('\t')
                    if parts[0].isdigit():
                        token = parts[1]
                        current_sentence.append(token)

        if current_sentence:
            sentences.append(' '.join(current_sentence))
            words.extend(current_sentence)

        print(f"Loaded {len(words)} words from {len(sentences)} sentences")

        if is_train:
            self.train_words = words
            self.word_frequencies = Counter(words)
            # Save sentences to temporary file for training
            temp_file = f"temp_{self.language}_train.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sentences))
            return temp_file, sentences
        else:
            self.test_words = words
            return None, sentences

    def create_tokenizer(self, normalize=True, strip_accents=False, lowercase=False):
        """
        Create and configure a tokenizer with specified normalization options

        Args:
            normalize: Apply NFKC normalization
            strip_accents: Remove accent marks
            lowercase: Convert to lowercase
        """
        print(f"\n{'='*60}")
        print(f"Creating {self.model_type} Tokenizer for {self.language}")
        print(f"{'='*60}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Normalization settings:")
        print(f"  - NFKC normalization: {normalize}")
        print(f"  - Strip accents: {strip_accents}")
        print(f"  - Lowercase: {lowercase}")

        # Create base tokenizer
        if self.model_type == 'BPE':
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
                show_progress=True
            )
        else:  # Unigram
            tokenizer = Tokenizer(Unigram())
            trainer = UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
                show_progress=True,
                unk_token="[UNK]"
            )

        # Configure normalizers
        normalizer_sequence = []
        if normalize:
            normalizer_sequence.append(NFKC())
        if strip_accents:
            normalizer_sequence.append(NFD())
            normalizer_sequence.append(StripAccents())
        if lowercase:
            normalizer_sequence.append(Lowercase())

        if normalizer_sequence:
            tokenizer.normalizer = normalizers.Sequence(normalizer_sequence)

        # Configure pre-tokenizer
        tokenizer.pre_tokenizer = Whitespace()

        self.tokenizer = tokenizer
        self.trainer = trainer

        return tokenizer

    def train_tokenizer(self, train_file):
        """Train the tokenizer on the training data"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not created. Call create_tokenizer() first.")

        print(f"\nTraining {self.model_type} tokenizer...")
        self.tokenizer.train([train_file], self.trainer)
        print("Training complete!")

        # Save tokenizer
        save_path = f"tokenizer_{self.language}_{self.model_type}.json"
        self.tokenizer.save(save_path)
        print(f"Tokenizer saved to: {save_path}")

    def analyze_tokenization(self, test_sentences, frequent_threshold=100, rare_threshold=5):
        """
        Analyze tokenizer performance on test data

        Args:
            test_sentences: List of sentences to analyze
            frequent_threshold: Minimum frequency to be considered "frequent"
            rare_threshold: Maximum frequency to be considered "rare"
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Train the tokenizer first.")

        print(f"\n{'='*60}")
        print(f"Analyzing Tokenization for {self.language}")
        print(f"{'='*60}")

        # Categorize words by frequency
        frequent_words = [w for w, c in self.word_frequencies.items() if c >= frequent_threshold]
        rare_words = [w for w, c in self.word_frequencies.items() if 1 < c <= rare_threshold]
        hapax_words = [w for w, c in self.word_frequencies.items() if c == 1]
        unseen_words = [w for w in self.test_words if w not in self.word_frequencies]

        print(f"\nWord categories in training data:")
        print(f"  Frequent words (>={frequent_threshold}): {len(frequent_words)}")
        print(f"  Rare words (2-{rare_threshold}): {len(rare_words)}")
        print(f"  Hapax (frequency=1): {len(hapax_words)}")
        print(f"  Unseen in test: {len(set(unseen_words))}")

        # Analyze tokenization for different word categories
        stats = {}

        for category_name, word_list in [
            ('frequent', frequent_words[:100]),
            ('rare', rare_words[:100]),
            ('hapax', hapax_words[:100]),
            ('unseen', list(set(unseen_words))[:100])
        ]:
            if not word_list:
                continue

            token_lengths = []
            tokens_per_word = []
            all_tokens = []

            for word in word_list:
                encoding = self.tokenizer.encode(word)
                tokens = encoding.tokens
                all_tokens.extend(tokens)
                tokens_per_word.append(len(tokens))
                avg_token_len = np.mean([len(t) for t in tokens]) if tokens else 0
                token_lengths.append(avg_token_len)

            stats[category_name] = {
                'avg_tokens_per_word': np.mean(tokens_per_word) if tokens_per_word else 0,
                'std_tokens_per_word': np.std(tokens_per_word) if tokens_per_word else 0,
                'avg_token_length': np.mean(token_lengths) if token_lengths else 0,
                'token_counter': Counter(all_tokens)
            }

        # Print statistics
        print(f"\n{'Category':<15} {'Tokens/Word':<15} {'Avg Token Len':<15}")
        print("-" * 45)
        for category, data in stats.items():
            print(f"{category:<15} "
                  f"{data['avg_tokens_per_word']:<15.2f} "
                  f"{data['avg_token_length']:<15.2f}")

        # Analyze full test corpus
        all_tokens = []
        all_tokens_per_word = []

        print(f"\nAnalyzing full test corpus...")
        for sentence in test_sentences[:1000]:  # Sample for performance
            encoding = self.tokenizer.encode(sentence)
            tokens = encoding.tokens
            all_tokens.extend(tokens)
            words = sentence.split()
            all_tokens_per_word.extend([len(self.tokenizer.encode(w).tokens) for w in words])

        token_freq = Counter(all_tokens)

        print(f"\nOverall test corpus statistics:")
        print(f"  Total tokens generated: {len(all_tokens)}")
        print(f"  Unique tokens: {len(token_freq)}")
        print(f"  Average tokens per word: {np.mean(all_tokens_per_word):.2f}")
        print(f"  Std tokens per word: {np.std(all_tokens_per_word):.2f}")

        print(f"\nTop 20 most frequent tokens:")
        for i, (token, count) in enumerate(token_freq.most_common(20), 1):
            print(f"  {i:2d}. {token:20s} {count:>7,}")

        return stats, token_freq

    def analyze_morphology(self, examples=None):
        """
        Analyze morphological segmentation with specific examples

        Args:
            examples: Dict of {description: [word_list]} for testing specific patterns
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")

        print(f"\n{'='*60}")
        print(f"Morphological Segmentation Analysis for {self.language}")
        print(f"{'='*60}")

        if examples is None:
            # Use default examples based on language
            if self.language == "German":
                examples = {
                    'Plurals (-en, -e, -er)': ['Hund', 'Hunde', 'Hunden', 'Kind', 'Kinder', 'Kindern'],
                    'Past participle (ge-...-t)': ['machen', 'gemacht', 'lernen', 'gelernt', 'spielen', 'gespielt'],
                    'Compounds': ['Haustür', 'Haus', 'Tür', 'Kindergarten', 'Kinder', 'Garten'],
                    'Diminutive (-chen)': ['Haus', 'Häuschen', 'Buch', 'Büchlein'],
                }
            elif self.language == "Hindi":
                examples = {
                    'Plurals': ['लड़का', 'लड़के', 'लड़की', 'लड़कियाँ'],
                    'Case markers': ['घर', 'घर में', 'घर से', 'घर को'],
                    'Verb forms': ['जाना', 'जाता', 'जाती', 'गया', 'गई'],
                }
            else:
                examples = {}

        for pattern_type, word_list in examples.items():
            print(f"\n{pattern_type}:")
            for word in word_list:
                encoding = self.tokenizer.encode(word)
                tokens = encoding.tokens
                print(f"  {word:20s} → {' | '.join(tokens)}")

        # Check consistency of segmentation
        print(f"\n{'='*60}")
        print("Segmentation Consistency Analysis")
        print(f"{'='*60}")

        # Find words with common affixes in training data
        if self.language == "German":
            # Check plural -en consistency
            en_words = [w for w in self.train_words if w.endswith('en') and len(w) > 4][:10]
            print("\nWords ending in '-en' (potential plurals):")
            for word in en_words:
                encoding = self.tokenizer.encode(word)
                tokens = encoding.tokens
                print(f"  {word:20s} → {' | '.join(tokens)}")

        return examples

    def visualize_token_distribution(self, token_freq):
        """Create visualization of token frequency distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Token frequency distribution
        frequencies = sorted(token_freq.values(), reverse=True)
        ax1.plot(range(len(frequencies)), frequencies, 'o-', markersize=2, alpha=0.6)
        ax1.set_xlabel('Token Rank')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{self.language} - Token Frequency Distribution')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Token length distribution
        token_lengths = [len(token) for token in token_freq.keys()]
        ax2.hist(token_lengths, bins=range(1, max(token_lengths)+1), alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Token Length (characters)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'{self.language} - Token Length Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'tokenizer_analysis_{self.language}_{self.model_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {filename}")


def main():
    """Main analysis pipeline for both languages"""

    print("="*80)
    print("DELIVERABLE 4: MORPHOLOGICAL ANALYSIS WITH SUB-WORD TOKENIZERS")
    print("="*80)

    # Define paths
    german_train = "data/ud/UD_German-GSD/de_gsd-ud-train.conllu"
    german_test = "data/ud/UD_German-GSD/de_gsd-ud-test.conllu"
    hindi_train = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
    hindi_test = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-test.conllu"

    # Choose tokenizer type: 'BPE' or 'Unigram'
    model_type = 'BPE'  # Change to 'Unigram' to try that model
    vocab_size = 5000

    # ===== GERMAN ANALYSIS =====
    print("\n" + "="*80)
    print("GERMAN TOKENIZER")
    print("="*80)

    german_analyzer = TokenizerAnalyzer(
        language_name="German",
        ud_train_file=german_train,
        ud_test_file=german_test,
        vocab_size=vocab_size,
        model_type=model_type
    )

    # Load data
    train_file, train_sentences = german_analyzer.load_conllu_data(german_train, is_train=True)
    _, test_sentences = german_analyzer.load_conllu_data(german_test, is_train=False)

    # Create and train tokenizer
    # For German, we'll use basic normalization but no accent stripping
    german_analyzer.create_tokenizer(normalize=True, strip_accents=False, lowercase=False)
    german_analyzer.train_tokenizer(train_file)

    # Analyze tokenization
    german_stats, german_token_freq = german_analyzer.analyze_tokenization(test_sentences)

    # Morphological analysis
    german_analyzer.analyze_morphology()

    # Visualize
    german_analyzer.visualize_token_distribution(german_token_freq)

    # Clean up temp file
    if os.path.exists(train_file):
        os.remove(train_file)

    # ===== HINDI ANALYSIS =====
    print("\n" + "="*80)
    print("HINDI TOKENIZER")
    print("="*80)

    hindi_analyzer = TokenizerAnalyzer(
        language_name="Hindi",
        ud_train_file=hindi_train,
        ud_test_file=hindi_test,
        vocab_size=vocab_size,
        model_type=model_type
    )

    # Load data
    train_file, train_sentences = hindi_analyzer.load_conllu_data(hindi_train, is_train=True)
    _, test_sentences = hindi_analyzer.load_conllu_data(hindi_test, is_train=False)

    # Create and train tokenizer
    # For Hindi, we use normalization but typically don't strip accents (vowel marks are crucial)
    hindi_analyzer.create_tokenizer(normalize=True, strip_accents=False, lowercase=False)
    hindi_analyzer.train_tokenizer(train_file)

    # Analyze tokenization
    hindi_stats, hindi_token_freq = hindi_analyzer.analyze_tokenization(test_sentences)

    # Morphological analysis
    hindi_analyzer.analyze_morphology()

    # Visualize
    hindi_analyzer.visualize_token_distribution(hindi_token_freq)

    # Clean up temp file
    if os.path.exists(train_file):
        os.remove(train_file)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - tokenizer_German_{model_type}.json")
    print(f"  - tokenizer_Hindi_{model_type}.json")
    print(f"  - tokenizer_analysis_German_{model_type}.png")
    print(f"  - tokenizer_analysis_Hindi_{model_type}.png")


if __name__ == "__main__":
    main()
