# Deliverable 3: Corpus Statistics Analysis

## Overview
A comprehensive comparative analysis of word distributions and linguistic statistics between German and Hindi corpora. This script analyzes both word frequency lists and Universal Dependencies (UD) treebank data to extract meaningful patterns and differences between the two languages.

## Purpose
- Compare statistical properties of German and Hindi corpora
- Analyze word frequency distributions (Zipf's law)
- Examine sentence and word length distributions
- Study n-gram patterns (bigrams and trigrams)
- Analyze closed word classes (e.g., conjunctions)
- Generate comparative visualizations

## Requirements
```bash
pip install pandas numpy matplotlib seaborn
```

## Input Data Required

### 1. Word Frequency Lists (CSV format)
- **German**: `data/corpo/wordlist_deu_news_2008_10K_20251112220311.csv`
- **Hindi**: `data/corpo/wordlist_hin_news_2019_20251112220411.csv`

Format: CSV files with columns `Item` (word) and `Frequency` (count)

### 2. Universal Dependencies CoNLL-U Files
- **German**: `data/ud/UD_German-GSD/de_gsd-ud-train.conllu`
- **Hindi**: `data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu`

## Usage

### Basic Execution
```bash
python deliverable_3_corpus_statistics.py
```

### Customization
Edit the file paths in the `main()` function (lines 242-246):
```python
german_corpus = "path/to/german/wordlist.csv"
hindi_corpus = "path/to/hindi/wordlist.csv"
german_ud = "path/to/german/treebank.conllu"
hindi_ud = "path/to/hindi/treebank.conllu"
```

## Core Components

### Class: `CorpusAnalyzer`
Main class for analyzing corpus statistics.

**Key Methods:**
- `load_wordlist()` - Load word frequency data from CSV
- `load_ud_sentences()` - Extract sentences from CoNLL-U files
- `basic_statistics()` - Calculate core corpus metrics
- `most_frequent_words(n)` - Get top N frequent words
- `zipf_analysis()` - Analyze frequency-rank distribution
- `ngram_analysis(n, top_k)` - Extract and count n-grams
- `closed_class_analysis(word_list)` - Analyze specific word frequencies

### Function: `compare_corpora(analyzer1, analyzer2)`
Generate comparative visualizations between two languages.

## Statistics Computed

### Basic Statistics
- **Number of types** (unique words)
- **Number of tokens** (total words)
- **Type-token ratio** (lexical diversity)
- **Hapax legomena** (words appearing once)
- **Hapax percentage**

### Sentence Statistics (from UD data)
- Number of sentences
- Average sentence length (words and characters)
- Standard deviation of sentence length
- 10th and 90th percentile sentence lengths

### Word Statistics (from UD data)
- Average word length
- Standard deviation of word length
- 10th and 90th percentile word lengths

### N-gram Analysis
- Top 15 bigrams (2-word sequences)
- Top 10 trigrams (3-word sequences)
- Frequency counts for each n-gram

### Closed Word Class Analysis
- Frequency and relative frequency of specific words
- Example conjunctions:
  - German: und, oder, aber, denn, sondern
  - Hindi: और, या, लेकिन, परन्तु, किन्तु

## Output

### Console Output
The script prints detailed statistics including:
- Basic corpus statistics for both languages
- Type-token ratio comparison
- Top 20 most frequent words
- Top 15 bigrams
- Top 10 trigrams
- Conjunction frequency analysis

### Visualizations

#### 1. `comparison_zipf_and_frequency.png`
Two-panel figure:
- **Left panel**: Zipf's law plot (frequency vs rank on log-log scale)
- **Right panel**: Bar chart comparing top 15 most frequent words

#### 2. `comparison_word_length_distribution.png`
Histogram comparing word length distributions between German and Hindi

#### 3. `comparison_sentence_length_distribution.png`
Histogram comparing sentence length distributions (in words)

## Example Output
```
============================================================
Statistics for German Corpus
============================================================
num_types................................... 10,000
num_tokens.................................. 1,234,567
type_token_ratio............................ 0.008100
hapax_legomena.............................. 3,456
hapax_percentage............................ 34.56
num_sentences............................... 13,814
avg_sentence_length_words................... 15.23
avg_word_length............................. 5.67

============================================================
TYPE-TOKEN RATIO COMPARISON
============================================================
German TTR: 0.008100
Hindi TTR:  0.012345
Difference: 0.004245

============================================================
TOP 20 MOST FREQUENT WORDS
============================================================

German:
 1. die               123,456
 2. der               98,765
 3. und               87,654
...
```

## Key Insights

### Zipf's Law
The script validates Zipf's law, which states that the frequency of a word is inversely proportional to its rank. The log-log plot typically shows a linear relationship.

### Type-Token Ratio (TTR)
- **Lower TTR**: More repetitive text, less lexical diversity
- **Higher TTR**: More varied vocabulary
- Useful for comparing morphological complexity between languages

### Morphological Complexity
Languages with richer morphology (like Hindi) typically show:
- Higher type-token ratios
- More hapax legomena
- Longer average word lengths

## Code Structure

**Lines 19-163**: `CorpusAnalyzer` class definition
**Lines 165-237**: `compare_corpora()` function for visualizations
**Lines 239-349**: `main()` function orchestrating the analysis

## Customization Options

### Analyze Different Conjunctions
Modify lines 326-327 and 329-330:
```python
german_conjunctions = ['your', 'words', 'here']
hindi_conjunctions = ['आपके', 'शब्द', 'यहाँ']
```

### Change N-gram Size
Modify the `ngram_analysis()` calls:
```python
# For 4-grams instead of trigrams
german_4grams = german.ngram_analysis(n=4, top_k=10)
```

### Adjust Visualization Settings
Modify lines 14-16 for plot styling:
```python
sns.set_style("darkgrid")  # Different style
plt.rcParams['figure.figsize'] = (16, 8)  # Larger figures
```

## Performance Notes
- Processing time depends on corpus size
- Large corpora may take several minutes to analyze
- Visualizations are saved at 300 DPI for publication quality

## Related Scripts
- **Prerequisite**: Run `treebanks_download.py` to get UD data
- **Next**: `deliverable_4_tokenizer_analysis.py` for morphological analysis
- **Next**: `deliverable_5_dependency_parsing.py` for syntactic analysis
