"""
Deliverable 3: Corpus Statistics Analysis
A comparative analysis of word distributions between German and Hindi corpora
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import re

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class CorpusAnalyzer:
    """Analyze corpus statistics for language comparison"""

    def __init__(self, corpus_file, language_name, ud_file=None):
        self.corpus_file = corpus_file
        self.language = language_name
        self.ud_file = ud_file
        self.word_freq = {}
        self.sentences = []
        self.tokens = []

    def load_wordlist(self):
        """Load wordlist from CSV file"""
        print(f"\nLoading {self.language} corpus from {self.corpus_file}...")
        df = pd.read_csv(self.corpus_file, skiprows=2)  # Skip header rows
        self.word_freq = dict(zip(df['Item'], df['Frequency']))
        print(f"Loaded {len(self.word_freq)} unique word forms")
        return self.word_freq

    def load_ud_sentences(self):
        """Load sentences from Universal Dependencies CoNLL-U file"""
        if not self.ud_file or not Path(self.ud_file).exists():
            print(f"Warning: UD file not found: {self.ud_file}")
            return []

        print(f"Loading sentences from {self.ud_file}...")
        sentences = []
        current_sentence = []

        with open(self.ud_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                elif not line:  # Empty line marks sentence boundary
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split('\t')
                    if parts[0].isdigit():  # Valid token line
                        token = parts[1]  # Word form
                        current_sentence.append(token)

        if current_sentence:
            sentences.append(current_sentence)

        self.sentences = sentences
        self.tokens = [token for sent in sentences for token in sent]
        print(f"Loaded {len(sentences)} sentences with {len(self.tokens)} tokens")
        return sentences

    def basic_statistics(self):
        """Calculate basic corpus statistics"""
        stats = {}

        # From wordlist
        stats['num_types'] = len(self.word_freq)
        stats['num_tokens'] = sum(self.word_freq.values())
        stats['type_token_ratio'] = stats['num_types'] / stats['num_tokens']

        # Hapax legomena (words appearing once)
        stats['hapax_legomena'] = sum(1 for freq in self.word_freq.values() if freq == 1)
        stats['hapax_percentage'] = 100 * stats['hapax_legomena'] / stats['num_types']

        # From UD sentences
        if self.sentences:
            stats['num_sentences'] = len(self.sentences)
            sentence_lengths_words = [len(sent) for sent in self.sentences]
            sentence_lengths_chars = [sum(len(word) for word in sent) for sent in self.sentences]

            stats['avg_sentence_length_words'] = np.mean(sentence_lengths_words)
            stats['std_sentence_length_words'] = np.std(sentence_lengths_words)
            stats['p10_sentence_length_words'] = np.percentile(sentence_lengths_words, 10)
            stats['p90_sentence_length_words'] = np.percentile(sentence_lengths_words, 90)

            stats['avg_sentence_length_chars'] = np.mean(sentence_lengths_chars)
            stats['std_sentence_length_chars'] = np.std(sentence_lengths_chars)

            # Word form length
            word_lengths = [len(word) for word in self.tokens]
            stats['avg_word_length'] = np.mean(word_lengths)
            stats['std_word_length'] = np.std(word_lengths)
            stats['p10_word_length'] = np.percentile(word_lengths, 10)
            stats['p90_word_length'] = np.percentile(word_lengths, 90)

        return stats

    def most_frequent_words(self, n=20):
        """Get the n most frequent words"""
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:n]

    def zipf_analysis(self):
        """Analyze Zipf's law: frequency vs rank"""
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        ranks = np.arange(1, len(sorted_words) + 1)
        frequencies = np.array([freq for _, freq in sorted_words])
        return ranks, frequencies

    def ngram_analysis(self, n=2, top_k=20):
        """Extract and count n-grams from sentences"""
        if not self.sentences:
            return []

        ngrams = []
        for sentence in self.sentences:
            if len(sentence) >= n:
                for i in range(len(sentence) - n + 1):
                    ngram = tuple(sentence[i:i+n])
                    ngrams.append(ngram)

        ngram_counts = Counter(ngrams)
        return ngram_counts.most_common(top_k)

    def closed_class_analysis(self, word_list):
        """
        Analyze frequency of closed word class members
        word_list: list of words to search for (e.g., articles, conjunctions)
        """
        results = {}
        total_tokens = sum(self.word_freq.values())

        for word in word_list:
            freq = self.word_freq.get(word, 0)
            relative_freq = (freq / total_tokens) * 100 if total_tokens > 0 else 0
            results[word] = {
                'frequency': freq,
                'relative_frequency': relative_freq
            }

        return results

    def print_statistics(self, stats):
        """Print statistics in a formatted way"""
        print(f"\n{'='*60}")
        print(f"Statistics for {self.language} Corpus")
        print(f"{'='*60}")

        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:.2f}")
            else:
                print(f"{key:.<40} {value:,}")


def compare_corpora(analyzer1, analyzer2):
    """Generate comparative visualizations between two corpora"""

    # 1. Zipf's Law comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ranks1, freqs1 = analyzer1.zipf_analysis()
    ranks2, freqs2 = analyzer2.zipf_analysis()

    ax1.loglog(ranks1[:1000], freqs1[:1000], 'o-', alpha=0.6, label=analyzer1.language, markersize=2)
    ax1.loglog(ranks2[:1000], freqs2[:1000], 'o-', alpha=0.6, label=analyzer2.language, markersize=2)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Frequency')
    ax1.set_title("Zipf's Law: Frequency vs Rank (log-log)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Top frequent words comparison
    top_words1 = dict(analyzer1.most_frequent_words(15))
    top_words2 = dict(analyzer2.most_frequent_words(15))

    x = np.arange(15)
    width = 0.35

    ax2.barh(x, list(top_words1.values())[:15], width, label=analyzer1.language, alpha=0.8)
    ax2.barh(x + width, list(top_words2.values())[:15], width, label=analyzer2.language, alpha=0.8)
    ax2.set_xlabel('Frequency')
    ax2.set_title('Top 15 Most Frequent Words')
    ax2.legend()
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig('comparison_zipf_and_frequency.png', dpi=300, bbox_inches='tight')
    print("\nSaved: comparison_zipf_and_frequency.png")

    # 3. Word length distribution
    if analyzer1.tokens and analyzer2.tokens:
        fig, ax = plt.subplots(figsize=(10, 6))

        word_lengths1 = [len(word) for word in analyzer1.tokens[:10000]]  # Sample for performance
        word_lengths2 = [len(word) for word in analyzer2.tokens[:10000]]

        ax.hist(word_lengths1, bins=range(1, 25), alpha=0.5, label=analyzer1.language, density=True)
        ax.hist(word_lengths2, bins=range(1, 25), alpha=0.5, label=analyzer2.language, density=True)
        ax.set_xlabel('Word Length (characters)')
        ax.set_ylabel('Density')
        ax.set_title('Word Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('comparison_word_length_distribution.png', dpi=300, bbox_inches='tight')
        print("Saved: comparison_word_length_distribution.png")

    # 4. Sentence length distribution
    if analyzer1.sentences and analyzer2.sentences:
        fig, ax = plt.subplots(figsize=(10, 6))

        sent_lengths1 = [len(sent) for sent in analyzer1.sentences]
        sent_lengths2 = [len(sent) for sent in analyzer2.sentences]

        ax.hist(sent_lengths1, bins=50, alpha=0.5, label=analyzer1.language, density=True)
        ax.hist(sent_lengths2, bins=50, alpha=0.5, label=analyzer2.language, density=True)
        ax.set_xlabel('Sentence Length (words)')
        ax.set_ylabel('Density')
        ax.set_title('Sentence Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('comparison_sentence_length_distribution.png', dpi=300, bbox_inches='tight')
        print("Saved: comparison_sentence_length_distribution.png")


def main():
    """Main analysis pipeline"""

    # Define file paths
    german_corpus = "data/corpo/wordlist_deu_news_2008_10K_20251112220311.csv"
    hindi_corpus = "data/corpo/wordlist_hin_news_2019_20251112220411.csv"
    german_ud = "data/ud/UD_German-GSD/de_gsd-ud-train.conllu"
    hindi_ud = "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"

    # Create analyzers
    german = CorpusAnalyzer(german_corpus, "German", german_ud)
    hindi = CorpusAnalyzer(hindi_corpus, "Hindi", hindi_ud)

    # Load data
    german.load_wordlist()
    german.load_ud_sentences()

    hindi.load_wordlist()
    hindi.load_ud_sentences()

    # Calculate statistics
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)

    german_stats = german.basic_statistics()
    german.print_statistics(german_stats)

    hindi_stats = hindi.basic_statistics()
    hindi.print_statistics(hindi_stats)

    # Compare type-token ratios
    print(f"\n{'='*80}")
    print("TYPE-TOKEN RATIO COMPARISON")
    print(f"{'='*80}")
    print(f"German TTR: {german_stats['type_token_ratio']:.6f}")
    print(f"Hindi TTR:  {hindi_stats['type_token_ratio']:.6f}")
    print(f"Difference: {abs(german_stats['type_token_ratio'] - hindi_stats['type_token_ratio']):.6f}")

    # Most frequent words
    print(f"\n{'='*80}")
    print("TOP 20 MOST FREQUENT WORDS")
    print(f"{'='*80}")

    print(f"\n{german.language}:")
    for i, (word, freq) in enumerate(german.most_frequent_words(20), 1):
        print(f"{i:2d}. {word:15s} {freq:>10,}")

    print(f"\n{hindi.language}:")
    for i, (word, freq) in enumerate(hindi.most_frequent_words(20), 1):
        print(f"{i:2d}. {word:15s} {freq:>10,}")

    # N-gram analysis
    print(f"\n{'='*80}")
    print("BIGRAM ANALYSIS")
    print(f"{'='*80}")

    german_bigrams = german.ngram_analysis(n=2, top_k=15)
    print(f"\n{german.language} - Top 15 Bigrams:")
    for i, (bigram, count) in enumerate(german_bigrams, 1):
        print(f"{i:2d}. {' '.join(bigram):30s} {count:>7,}")

    hindi_bigrams = hindi.ngram_analysis(n=2, top_k=15)
    print(f"\n{hindi.language} - Top 15 Bigrams:")
    for i, (bigram, count) in enumerate(hindi_bigrams, 1):
        print(f"{i:2d}. {' '.join(bigram):30s} {count:>7,}")

    # Trigram analysis
    print(f"\n{'='*80}")
    print("TRIGRAM ANALYSIS")
    print(f"{'='*80}")

    german_trigrams = german.ngram_analysis(n=3, top_k=10)
    print(f"\n{german.language} - Top 10 Trigrams:")
    for i, (trigram, count) in enumerate(german_trigrams, 1):
        print(f"{i:2d}. {' '.join(trigram):40s} {count:>7,}")

    hindi_trigrams = hindi.ngram_analysis(n=3, top_k=10)
    print(f"\n{hindi.language} - Top 10 Trigrams:")
    for i, (trigram, count) in enumerate(hindi_trigrams, 1):
        print(f"{i:2d}. {' '.join(trigram):40s} {count:>7,}")

    # Closed word class analysis - Conjunctions
    print(f"\n{'='*80}")
    print("CLOSED WORD CLASS ANALYSIS: CONJUNCTIONS")
    print(f"{'='*80}")

    german_conjunctions = ['und', 'oder', 'aber', 'denn', 'sondern']
    hindi_conjunctions = ['और', 'या', 'लेकिन', 'परन्तु', 'किन्तु']

    german_conj_freq = german.closed_class_analysis(german_conjunctions)
    hindi_conj_freq = hindi.closed_class_analysis(hindi_conjunctions)

    print(f"\n{german.language} Conjunctions:")
    for word, data in german_conj_freq.items():
        print(f"  {word:15s}: {data['frequency']:>10,} ({data['relative_frequency']:.4f}%)")

    print(f"\n{hindi.language} Conjunctions:")
    for word, data in hindi_conj_freq.items():
        print(f"  {word:15s}: {data['frequency']:>10,} ({data['relative_frequency']:.4f}%)")

    # Generate comparative visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    compare_corpora(german, hindi)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
