# Deliverable 4: Morphological Analysis using Sub-word Tokenizers

## Overview
Train and analyze sub-word tokenizers (BPE or Unigram) for German and Hindi to understand morphological segmentation patterns. This script explores how different tokenization algorithms handle word formation, affixes, and morphological complexity across languages.

## Purpose
- Train BPE (Byte Pair Encoding) or Unigram tokenizers on German and Hindi
- Analyze how tokenizers segment words based on frequency
- Compare tokenization patterns for frequent vs rare vs unseen words
- Study morphological consistency (plurals, verb forms, compounds)
- Evaluate tokenizer performance on different word categories

## Requirements
```bash
pip install tokenizers numpy matplotlib seaborn
```

## Input Data Required

### Universal Dependencies CoNLL-U Files
- **German Train**: `data/ud/UD_German-GSD/de_gsd-ud-train.conllu`
- **German Test**: `data/ud/UD_German-GSD/de_gsd-ud-test.conllu`
- **Hindi Train**: `data/ud/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu`
- **Hindi Test**: `data/ud/UD_Hindi-HDTB/hi_hdtb-ud-test.conllu`

## Usage

### Basic Execution
```bash
python deliverable_4_tokenizer_analysis.py
```

### Customization

#### Choose Tokenizer Type
Edit line 336:
```python
model_type = 'BPE'  # or 'Unigram'
```

#### Adjust Vocabulary Size
Edit line 337:
```python
vocab_size = 5000  # or any desired size
```

#### Modify Normalization Settings
In the `create_tokenizer()` calls (lines 358, 393):
```python
analyzer.create_tokenizer(
    normalize=True,      # Apply NFKC normalization
    strip_accents=False, # Remove accent marks
    lowercase=False      # Convert to lowercase
)
```

## Core Components

### Class: `TokenizerAnalyzer`
Main class for training and analyzing tokenizers.

**Initialization Parameters:**
- `language_name`: Language identifier (e.g., "German")
- `ud_train_file`: Path to training CoNLL-U file
- `ud_test_file`: Path to test CoNLL-U file
- `vocab_size`: Target vocabulary size (default: 5000)
- `model_type`: 'BPE' or 'Unigram'

**Key Methods:**
- `load_conllu_data()` - Extract words from CoNLL-U files
- `create_tokenizer()` - Initialize tokenizer with normalization options
- `train_tokenizer()` - Train the tokenizer on training data
- `analyze_tokenization()` - Analyze performance on different word categories
- `analyze_morphology()` - Study morphological segmentation patterns
- `visualize_token_distribution()` - Create frequency and length visualizations

## Analysis Features

### 1. Word Category Analysis
Words are categorized by frequency in training data:
- **Frequent words**: Frequency ≥ 100
- **Rare words**: Frequency between 2-5
- **Hapax words**: Frequency = 1
- **Unseen words**: Words only in test set

For each category, computes:
- Average tokens per word
- Standard deviation of tokens per word
- Average token length

### 2. Morphological Pattern Analysis

#### German Examples
- **Plurals**: -en, -e, -er suffixes
  - e.g., Hund → Hunde → Hunden
- **Past Participles**: ge-...-t prefix/suffix pattern
  - e.g., machen → gemacht
- **Compounds**: Word composition
  - e.g., Haustür → Haus + Tür
- **Diminutives**: -chen suffix
  - e.g., Haus → Häuschen

#### Hindi Examples
- **Plurals**: Various plural markers
  - e.g., लड़का → लड़के
- **Case Markers**: Postpositions
  - e.g., घर में, घर से, घर को
- **Verb Forms**: Tense/aspect markers
  - e.g., जाना → जाता → गया

### 3. Consistency Analysis
Checks whether the tokenizer consistently segments:
- Words with common affixes
- Morphological variants of the same root
- Related word forms

## Output

### Console Output

#### Tokenizer Creation
```
============================================================
Creating BPE Tokenizer for German
============================================================
Vocabulary size: 5000
Normalization settings:
  - NFKC normalization: True
  - Strip accents: False
  - Lowercase: False
```

#### Training Progress
```
Training BPE tokenizer...
Training complete!
Tokenizer saved to: tokenizer_German_BPE.json
```

#### Analysis Results
```
============================================================
Analyzing Tokenization for German
============================================================

Word categories in training data:
  Frequent words (>=100): 523
  Rare words (2-5): 8,234
  Hapax (frequency=1): 15,678
  Unseen in test: 2,345

Category         Tokens/Word     Avg Token Len
---------------------------------------------
frequent         1.23            4.56
rare             2.45            3.21
hapax            3.12            2.87
unseen           3.45            2.65

Top 20 most frequent tokens:
   1. der                  12,345
   2. die                  11,234
   3. und                   9,876
  ...
```

#### Morphological Segmentation
```
============================================================
Morphological Segmentation Analysis for German
============================================================

Plurals (-en, -e, -er):
  Hund                 → Hund
  Hunde                → Hund | e
  Hunden               → Hund | en
  Kind                 → Kind
  Kinder               → Kind | er
```

### Saved Files

#### 1. Trained Tokenizer Models (JSON)
- `tokenizer_German_BPE.json` (or Unigram)
- `tokenizer_Hindi_BPE.json` (or Unigram)

These can be loaded later for reuse:
```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer_German_BPE.json")
```

#### 2. Visualizations (PNG)
- `tokenizer_analysis_German_BPE.png`
- `tokenizer_analysis_Hindi_BPE.png`

Each visualization contains two panels:
- **Left**: Token frequency distribution (rank vs frequency)
- **Right**: Token length distribution (histogram)

### Temporary Files
The script creates temporary training files:
- `temp_German_train.txt`
- `temp_Hindi_train.txt`

These are automatically cleaned up after training.

## Key Insights

### BPE vs Unigram
- **BPE**: Merges most frequent character pairs iteratively
  - Good for compound words
  - Deterministic segmentation

- **Unigram**: Probabilistic language model approach
  - Better for agglutinative languages
  - Can produce multiple segmentations

### Frequency Effects
- **Frequent words**: Usually segmented as single tokens
- **Rare words**: Broken into more sub-word units
- **Unseen words**: Most fragmented, relies on character-level patterns

### Morphological Complexity
- **German**: Compound word formation well-captured
- **Hindi**: Agglutinative morphology requires more tokens per word

## Code Structure

**Lines 21-320**: `TokenizerAnalyzer` class
- Lines 76-130: Tokenizer creation and configuration
- Lines 132-145: Training functionality
- Lines 146-237: Tokenization analysis
- Lines 239-293: Morphological pattern analysis
- Lines 295-319: Visualization

**Lines 322-417**: `main()` function
- Separate pipelines for German and Hindi
- Sequential processing: load → create → train → analyze → visualize

## Advanced Usage

### Custom Morphological Examples
```python
german_examples = {
    'Your Category': ['word1', 'word2', 'word3'],
    'Another Category': ['word4', 'word5']
}
german_analyzer.analyze_morphology(examples=german_examples)
```

### Different Frequency Thresholds
In `analyze_tokenization()` call:
```python
stats, token_freq = analyzer.analyze_tokenization(
    test_sentences,
    frequent_threshold=200,  # Higher threshold
    rare_threshold=3         # Lower threshold
)
```

### Sample Fewer Test Sentences
Modify line 218 for faster processing:
```python
for sentence in test_sentences[:500]:  # Only 500 instead of 1000
```

## Performance Notes
- Training time: ~30 seconds per language (depends on corpus size)
- Analysis time: ~1-2 minutes per language
- Memory usage: Moderate (~500MB for vocab_size=5000)
- Temporary files are cleaned up automatically

## Troubleshooting

### Issue: "Tokenizer not created"
**Solution**: Ensure `create_tokenizer()` is called before `train_tokenizer()`

### Issue: "No training data"
**Solution**: Check that CoNLL-U files exist and are readable

### Issue: "Empty sentences"
**Solution**: Verify CoNLL-U files are properly formatted (not truncated)

## Related Scripts
- **Prerequisite**: Run `treebanks_download.py` to get UD data
- **Previous**: `deliverable_3_corpus_statistics.py` for corpus analysis
- **Next**: `deliverable_5_dependency_parsing.py` for syntactic analysis

## References
- BPE: [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909)
- Unigram: [Kudo (2018)](https://arxiv.org/abs/1804.10959)
- HuggingFace Tokenizers: https://github.com/huggingface/tokenizers
