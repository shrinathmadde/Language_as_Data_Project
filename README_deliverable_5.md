# Deliverable 5: Syntactic Analysis using Dependency Parsers

## Overview
Perform comprehensive syntactic analysis using SpaCy dependency parsers. This script analyzes dependency tree structures, computes tree statistics, evaluates parser accuracy against gold standard annotations, and explores syntactic patterns in German (and potentially other languages).

## Purpose
- Parse sentences using pre-trained SpaCy models
- Compute dependency tree statistics (depth, node degree, etc.)
- Analyze syntactic patterns by POS tags
- Evaluate parser performance (UAS, LAS, Label Accuracy)
- Visualize syntactic structures and distributions
- Test parser behavior on ambiguous sentences

## Requirements
```bash
pip install spacy numpy matplotlib seaborn

# Download SpaCy models
python -m spacy download de_core_news_sm  # German (small)
# OR
python -m spacy download de_core_news_md  # German (medium)
# OR
python -m spacy download de_core_news_lg  # German (large)
```

### Note on Hindi
Standard SpaCy does not include pre-trained Hindi models. Options:
1. Train a custom model using SpaCy
2. Use alternative parsers (Stanza, Trankit, UDPipe)
3. Focus on German for this deliverable (as currently configured)

## Input Data Required

### Universal Dependencies CoNLL-U Files
- **German Test**: `data/ud/UD_German-GSD/de_gsd-ud-test.conllu`

The CoNLL-U file provides gold standard annotations for evaluation.

## Usage

### Basic Execution
```bash
python deliverable_5_dependency_parsing.py
```

### Customization

#### Choose SpaCy Model
Edit line 509:
```python
german_model = "de_core_news_sm"  # or de_core_news_md, de_core_news_lg
```

Model comparison:
- **sm (small)**: ~15MB, faster, lower accuracy
- **md (medium)**: ~50MB, balanced
- **lg (large)**: ~500MB, slower, higher accuracy

#### Adjust Number of Sentences
Edit line 523:
```python
german_analyzer.load_gold_conllu(max_sentences=500)  # Default: 500
```

#### Test Different Ambiguous Sentences
Edit lines 544-548:
```python
ambiguous_sentences = [
    "Your ambiguous sentence here.",
    "Another test case.",
]
```

## Core Components

### Class: `DependencyAnalyzer`
Main class for dependency parsing analysis.

**Initialization Parameters:**
- `language_name`: Language identifier (e.g., "German")
- `model_name`: SpaCy model name (e.g., "de_core_news_sm")
- `conllu_file`: Path to gold standard CoNLL-U file (optional)

**Key Methods:**
- `load_spacy_model()` - Load SpaCy parser model
- `load_gold_conllu()` - Load gold standard dependency trees
- `parse_sentences()` - Parse sentences using SpaCy
- `compute_tree_statistics()` - Calculate tree metrics
- `evaluate_parser()` - Compare against gold standard
- `show_examples()` - Display example parses
- `visualize_statistics()` - Create visualization plots

## Analysis Features

### 1. Tree Structure Statistics

#### Tree Depth
- Maximum distance from root to any leaf node
- Indicates sentence complexity and embedding depth

#### Node Degree
- Number of children for each node
- Distribution shows branching patterns

#### Distance to Root
- Path length from each token to sentence root
- Analyzed separately for different POS tags
- Shows syntactic centrality of word classes

### 2. Syntactic Pattern Analysis

#### Leaf Nodes
- Tokens with no dependents (zero children)
- Typically: nouns, adjectives, adverbs, punctuation

#### Ancestor Analysis
- Common parent POS tags for specific word types
- Example: What do NOUNs typically depend on?

#### Descendant Analysis
- Common child POS tags for specific word types
- Example: What typically depends on VERBs?

#### Dependency Relations
- Frequency of different dependency types
- Common patterns: nsubj, obj, det, case, etc.

### 3. Parser Evaluation Metrics

#### UAS (Unlabeled Attachment Score)
Percentage of tokens with correct head attachment (ignoring label).

Formula: `correct_heads / total_tokens × 100`

#### LAS (Labeled Attachment Score)
Percentage of tokens with both correct head and correct dependency label.

Formula: `correct_both / total_tokens × 100`

#### Label Accuracy (LS)
Percentage of tokens with correct dependency label (ignoring head).

Formula: `correct_labels / total_tokens × 100`

### 4. Ambiguity Testing
The script tests parser behavior on structurally ambiguous sentences:
- **PP Attachment**: "Ich sehe den Mann mit dem Fernrohr"
  - Who has the telescope? The man or the speaker?
- **Instrument Ambiguity**: "Die Polizei erschoss den Mann mit der Waffe"
  - Who had the weapon?
- **Locative Ambiguity**: "Der Hund beißt den Mann im Park"
  - Who is in the park? Dog, man, or both?

## Output

### Console Output

#### Model Loading
```
============================================================
GERMAN DEPENDENCY PARSING
============================================================

Loading SpaCy model: de_core_news_sm
Model loaded successfully!
Pipeline components: ['tok2vec', 'morphologizer', 'parser', 'lemmatizer', 'ner']
```

#### Gold Standard Loading
```
Loading gold standard from data/ud/UD_German-GSD/de_gsd-ud-test.conllu...
Loaded 500 gold standard trees
```

#### Tree Statistics
```
============================================================
Dependency Tree Statistics - German
============================================================

Tree Depth:
  Average: 5.34
  Std Dev: 1.89
  Min: 2
  Max: 14

Node Degree Distribution:
  Degree 0:  8,234 nodes (45.23%)
  Degree 1:  4,567 nodes (25.12%)
  Degree 2:  2,891 nodes (15.89%)
  ...

Average Distance to Root by POS:
  VERB        : 0.98
  AUX         : 1.23
  ROOT        : 0.00
  NOUN        : 2.45
  ...

Most Common Leaf Node POS Tags:
  NOUN        :  3,456 (34.56%)
  PUNCT       :  2,345 (23.45%)
  ADJ         :  1,234 (12.34%)
  ...
```

#### Dependency Relations
```
Most Common Dependency Relations:
  punct               :  2,456 (12.34%)
  case                :  1,987 (9.87%)
  det                 :  1,765 (8.76%)
  nsubj               :  1,543 (7.65%)
  ...
```

#### Parser Evaluation
```
============================================================
Parser Evaluation - German
============================================================

Evaluation Results (18,234 tokens):
  Unlabeled Attachment Score (UAS): 89.45%
  Labeled Attachment Score (LAS):   86.23%
  Label Accuracy Score (LS):        91.67%
```

#### Example Parses
```
============================================================
Example Dependency Parses - German
============================================================

Example 1: Der Hund läuft schnell durch den Park.
------------------------------------------------------------
Token           POS      Head            Dep Rel         Children
------------------------------------------------------------
Der             DET      Hund            det
Hund            NOUN     läuft           nsubj           Der
läuft           VERB     ROOT            ROOT            Hund, schnell, durch, .
schnell         ADV      läuft           advmod
durch           ADP      läuft           obl             Park
den             DET      Park            det
Park            NOUN     durch           pobj            den
.               PUNCT    läuft           punct
```

### Saved Visualizations

#### 1. `dependency_analysis_German.png`
Six-panel figure containing:
1. **Tree Depth Distribution**: Histogram of parse tree depths
2. **Node Degree Distribution**: Bar chart of child counts
3. **Distance to Root by POS**: Average distances for different POS tags
4. **Top Dependency Relations**: Most frequent dependency types
5. **Leaf Node POS Distribution**: POS tags of terminal nodes
6. **POS Tag Distribution**: Overall POS tag frequencies

#### 2. `dependency_comparison.png` (if comparing languages)
Four-panel comparative figure:
1. Tree depth comparison (overlaid histograms)
2. Distance to root by POS (grouped bar chart)
3. Node degree distribution comparison
4. Summary statistics table

## Key Insights

### Tree Depth
- **Shallow trees** (depth 2-4): Simple sentences
- **Deep trees** (depth 7+): Complex sentences with multiple embeddings

### Node Degree
- Most nodes have 0-2 children (binary branching tendency)
- High-degree nodes (3+ children) are often verbs with multiple arguments

### POS Distance Patterns
- **VERBs**: Closest to root (often ARE the root)
- **Function words** (DET, ADP): Medium distance
- **Content words** (NOUN, ADJ): Further from root

### Common Dependency Relations
German-specific patterns:
- High frequency of `case` (prepositions)
- Frequent `det` (articles)
- Common `nsubj` and `obj` (subject/object)

## Code Structure

**Lines 17-404**: `DependencyAnalyzer` class
- Lines 28-38: Model loading
- Lines 40-88: Gold standard loading
- Lines 90-109: Sentence parsing
- Lines 111-184: Tree statistics computation
- Lines 186-243: Statistics printing
- Lines 244-309: Parser evaluation
- Lines 311-329: Example display
- Lines 331-403: Visualization

**Lines 406-494**: `compare_languages()` function
**Lines 497-562**: `main()` function

## Advanced Usage

### Adding Hindi or Other Languages
1. Install/train a SpaCy model for the language
2. Add analyzer initialization:
```python
hindi_analyzer = DependencyAnalyzer(
    "Hindi",
    "your_hindi_model",
    "data/ud/UD_Hindi-HDTB/hi_hdtb-ud-test.conllu"
)
```
3. Follow the same pipeline as German

### Custom Sentence Parsing
```python
analyzer = DependencyAnalyzer("German", "de_core_news_sm")
analyzer.load_spacy_model()

custom_sentences = ["Your sentence here.", "Another one."]
analyzer.parse_sentences(sentences=custom_sentences)
```

### Visualize Specific Sentences
```python
# After parsing
from spacy import displacy

doc = analyzer.nlp("Das ist ein Test.")
displacy.serve(doc, style="dep")
```

## Performance Notes
- Parsing speed: ~100-1000 sentences/second (depends on model size)
- Memory usage:
  - Small model: ~200MB
  - Large model: ~1GB
- Evaluation time: ~1-2 seconds for 500 sentences

## Troubleshooting

### Issue: "Model not found"
**Solution**:
```bash
python -m spacy download de_core_news_sm
```

### Issue: "Mismatched token counts"
**Solution**: Some CoNLL-U sentences may have multi-word tokens that SpaCy tokenizes differently. These are automatically skipped in evaluation.

### Issue: "Low accuracy scores"
**Solution**:
- SpaCy may use different annotation conventions than UD
- Try a larger model (md or lg)
- Some discrepancy is normal (80-90% is typical)

## Understanding Dependency Relations

Common UD relations:
- **nsubj**: Nominal subject
- **obj**: Object
- **iobj**: Indirect object
- **obl**: Oblique nominal
- **det**: Determiner
- **case**: Case marking (preposition)
- **amod**: Adjectival modifier
- **advmod**: Adverbial modifier
- **aux**: Auxiliary verb
- **cc**: Coordinating conjunction
- **conj**: Conjunct

Full list: https://universaldependencies.org/u/dep/

## Related Scripts
- **Prerequisite**: Run `treebanks_download.py` to get UD data
- **Previous**: `deliverable_3_corpus_statistics.py` for corpus analysis
- **Previous**: `deliverable_4_tokenizer_analysis.py` for morphological analysis

## References
- Universal Dependencies: https://universaldependencies.org/
- SpaCy: https://spacy.io/
- Dependency Grammar: Tesnière (1959), Nivre et al. (2016)
