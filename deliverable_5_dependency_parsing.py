"""
Deliverable 5: Syntactic Analysis using Dependency Parsers
Comparative analysis of syntactic structures using SpaCy parsers
"""

import spacy
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sns.set_style("whitegrid")


class DependencyAnalyzer:
    """Analyze dependency parse trees"""

    def __init__(self, language_name, model_name, conllu_file=None):
        self.language = language_name
        self.model_name = model_name
        self.conllu_file = conllu_file
        self.nlp = None
        self.parsed_sentences = []
        self.gold_trees = []  # For evaluation against UD gold standard

    def load_spacy_model(self):
        """Load SpaCy model for the language"""
        print(f"\nLoading SpaCy model: {self.model_name}")
        try:
            self.nlp = spacy.load(self.model_name)
            print(f"Model loaded successfully!")
            print(f"Pipeline components: {self.nlp.pipe_names}")
        except OSError:
            print(f"Model {self.model_name} not found!")
            print(f"Please install it using: python -m spacy download {self.model_name}")
            sys.exit(1)

    def load_gold_conllu(self, max_sentences=1000):
        """Load gold standard trees from CoNLL-U file for evaluation"""
        if not self.conllu_file or not Path(self.conllu_file).exists():
            print(f"Warning: CoNLL-U file not found: {self.conllu_file}")
            return []

        print(f"\nLoading gold standard from {self.conllu_file}...")
        gold_trees = []
        current_tree = []
        current_text = []

        with open(self.conllu_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                elif not line:
                    if current_tree:
                        gold_trees.append({
                            'tokens': current_tree,
                            'text': ' '.join(current_text)
                        })
                        current_tree = []
                        current_text = []
                        if len(gold_trees) >= max_sentences:
                            break
                else:
                    parts = line.split('\t')
                    if parts[0].isdigit():
                        token_info = {
                            'id': int(parts[0]),
                            'form': parts[1],
                            'lemma': parts[2],
                            'upos': parts[3],
                            'head': int(parts[6]),
                            'deprel': parts[7]
                        }
                        current_tree.append(token_info)
                        current_text.append(parts[1])

        if current_tree and len(gold_trees) < max_sentences:
            gold_trees.append({
                'tokens': current_tree,
                'text': ' '.join(current_text)
            })

        self.gold_trees = gold_trees
        print(f"Loaded {len(gold_trees)} gold standard trees")
        return gold_trees

    def parse_sentences(self, sentences=None):
        """Parse sentences using SpaCy"""
        if self.nlp is None:
            raise ValueError("SpaCy model not loaded!")

        if sentences is None:
            sentences = [tree['text'] for tree in self.gold_trees]

        print(f"\nParsing {len(sentences)} sentences...")
        self.parsed_sentences = []

        for i, sent in enumerate(sentences):
            if (i + 1) % 100 == 0:
                print(f"  Parsed {i+1}/{len(sentences)} sentences...")

            doc = self.nlp(sent)
            self.parsed_sentences.append(doc)

        print(f"Parsing complete!")
        return self.parsed_sentences

    def tree_depth(self, token, current_depth=0):
        """Calculate depth of a subtree rooted at token"""
        if not list(token.children):
            return current_depth
        return max(self.tree_depth(child, current_depth + 1) for child in token.children)

    def distance_to_root(self, token):
        """Calculate distance from token to root"""
        distance = 0
        current = token
        while current.head != current:
            distance += 1
            current = current.head
        return distance

    def compute_tree_statistics(self):
        """Compute various statistics about dependency trees"""
        if not self.parsed_sentences:
            raise ValueError("No parsed sentences available!")

        stats = {
            'tree_depths': [],
            'node_degrees': [],
            'distances_to_root': defaultdict(list),  # by POS tag
            'leaf_pos': [],
            'ancestors': defaultdict(list),  # ancestors by POS
            'descendants': defaultdict(list),  # descendants by POS
            'dep_relations': Counter(),
            'pos_tags': Counter()
        }

        print(f"\nComputing tree statistics for {len(self.parsed_sentences)} sentences...")

        for doc in self.parsed_sentences:
            # Find root
            root = [token for token in doc if token.head == token][0]

            # Tree depth
            depth = self.tree_depth(root)
            stats['tree_depths'].append(depth)

            # Analyze each token
            for token in doc:
                # Node degree (number of children)
                degree = len(list(token.children))
                stats['node_degrees'].append(degree)

                # POS tag
                stats['pos_tags'][token.pos_] += 1

                # Dependency relation
                stats['dep_relations'][token.dep_] += 1

                # Distance to root by POS
                distance = self.distance_to_root(token)
                stats['distances_to_root'][token.pos_].append(distance)

                # Leaf nodes (no children)
                if degree == 0:
                    stats['leaf_pos'].append(token.pos_)

                # Ancestors (path to root)
                current = token.head
                while current != current.head:
                    stats['ancestors'][token.pos_].append(current.pos_)
                    current = current.head
                if token.head != token:  # Not root
                    stats['ancestors'][token.pos_].append(current.pos_)

                # Descendants (children)
                for child in token.children:
                    stats['descendants'][token.pos_].append(child.pos_)

        return stats

    def print_statistics(self, stats):
        """Print formatted statistics"""
        print(f"\n{'='*60}")
        print(f"Dependency Tree Statistics - {self.language}")
        print(f"{'='*60}")

        # Tree depth
        print(f"\nTree Depth:")
        print(f"  Average: {np.mean(stats['tree_depths']):.2f}")
        print(f"  Std Dev: {np.std(stats['tree_depths']):.2f}")
        print(f"  Min: {np.min(stats['tree_depths'])}")
        print(f"  Max: {np.max(stats['tree_depths'])}")

        # Node degrees
        print(f"\nNode Degree Distribution:")
        degree_counts = Counter(stats['node_degrees'])
        for degree in sorted(degree_counts.keys())[:6]:
            count = degree_counts[degree]
            pct = 100 * count / len(stats['node_degrees'])
            print(f"  Degree {degree}: {count:>6,} nodes ({pct:>5.2f}%)")

        # Distance to root by POS
        print(f"\nAverage Distance to Root by POS:")
        pos_distances = {pos: np.mean(dists) for pos, dists in stats['distances_to_root'].items()}
        for pos, avg_dist in sorted(pos_distances.items(), key=lambda x: x[1])[:10]:
            print(f"  {pos:<12s}: {avg_dist:.2f}")

        # Most common leaf nodes
        print(f"\nMost Common Leaf Node POS Tags:")
        leaf_counter = Counter(stats['leaf_pos'])
        for pos, count in leaf_counter.most_common(10):
            pct = 100 * count / len(stats['leaf_pos'])
            print(f"  {pos:<12s}: {count:>6,} ({pct:>5.2f}%)")

        # Most common ancestors
        print(f"\nMost Common Ancestors (by child POS):")
        for child_pos in ['NOUN', 'VERB', 'ADJ']:
            if child_pos in stats['ancestors']:
                ancestors = Counter(stats['ancestors'][child_pos])
                print(f"  {child_pos}:")
                for anc_pos, count in ancestors.most_common(5):
                    print(f"    → {anc_pos:<12s}: {count:>6,}")

        # Most common descendants
        print(f"\nMost Common Descendants (by parent POS):")
        for parent_pos in ['NOUN', 'VERB', 'ADJ']:
            if parent_pos in stats['descendants']:
                descendants = Counter(stats['descendants'][parent_pos])
                print(f"  {parent_pos}:")
                for desc_pos, count in descendants.most_common(5):
                    print(f"    → {desc_pos:<12s}: {count:>6,}")

        # Most common dependency relations
        print(f"\nMost Common Dependency Relations:")
        for dep, count in stats['dep_relations'].most_common(15):
            pct = 100 * count / sum(stats['dep_relations'].values())
            print(f"  {dep:<20s}: {count:>6,} ({pct:>5.2f}%)")

    def evaluate_parser(self):
        """
        Evaluate parser against gold standard
        Compute UAS, LAS, and Label Accuracy
        """
        if not self.gold_trees or not self.parsed_sentences:
            print("Warning: Cannot evaluate - missing gold trees or parsed sentences")
            return None

        print(f"\n{'='*60}")
        print(f"Parser Evaluation - {self.language}")
        print(f"{'='*60}")

        total_tokens = 0
        correct_heads = 0  # For UAS
        correct_labels = 0  # For Label Accuracy
        correct_both = 0  # For LAS

        for gold, parsed_doc in zip(self.gold_trees, self.parsed_sentences):
            # Create mapping from parsed doc
            parsed_tokens = list(parsed_doc)

            # Check if lengths match
            if len(gold['tokens']) != len(parsed_tokens):
                continue

            for gold_token, parsed_token in zip(gold['tokens'], parsed_tokens):
                total_tokens += 1

                # Get gold head and label
                gold_head = gold_token['head']
                gold_deprel = gold_token['deprel']

                # Get parsed head (convert to 1-indexed)
                parsed_head = parsed_token.head.i + 1 if parsed_token.head != parsed_token else 0
                if parsed_token == parsed_doc[0]:  # Root handling
                    parsed_head = 0

                parsed_deprel = parsed_token.dep_

                # Compare
                if gold_head == parsed_head:
                    correct_heads += 1

                if gold_deprel == parsed_deprel:
                    correct_labels += 1

                if gold_head == parsed_head and gold_deprel == parsed_deprel:
                    correct_both += 1

        # Calculate scores
        uas = 100 * correct_heads / total_tokens if total_tokens > 0 else 0
        las = 100 * correct_both / total_tokens if total_tokens > 0 else 0
        label_acc = 100 * correct_labels / total_tokens if total_tokens > 0 else 0

        print(f"\nEvaluation Results ({total_tokens} tokens):")
        print(f"  Unlabeled Attachment Score (UAS): {uas:.2f}%")
        print(f"  Labeled Attachment Score (LAS):   {las:.2f}%")
        print(f"  Label Accuracy Score (LS):        {label_acc:.2f}%")

        return {
            'uas': uas,
            'las': las,
            'label_accuracy': label_acc,
            'total_tokens': total_tokens
        }

    def show_examples(self, n=3, ambiguous=None):
        """Show example parses"""
        print(f"\n{'='*60}")
        print(f"Example Dependency Parses - {self.language}")
        print(f"{'='*60}")

        examples = ambiguous if ambiguous else self.parsed_sentences[:n]

        for i, doc in enumerate(examples[:n], 1):
            print(f"\nExample {i}: {doc.text}")
            print("-" * 60)
            print(f"{'Token':<15} {'POS':<8} {'Head':<15} {'Dep Rel':<15} {'Children'}")
            print("-" * 60)

            for token in doc:
                children = [child.text for child in token.children]
                head_text = token.head.text if token.head != token else "ROOT"
                print(f"{token.text:<15} {token.pos_:<8} {head_text:<15} "
                      f"{token.dep_:<15} {', '.join(children)}")

    def visualize_statistics(self, stats):
        """Create visualizations of tree statistics"""
        fig = plt.figure(figsize=(15, 10))

        # 1. Tree depth distribution
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(stats['tree_depths'], bins=range(0, max(stats['tree_depths'])+2), alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Tree Depth')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{self.language} - Tree Depth Distribution')
        ax1.grid(True, alpha=0.3)

        # 2. Node degree distribution
        ax2 = plt.subplot(2, 3, 2)
        degree_counts = Counter(stats['node_degrees'])
        degrees = sorted(degree_counts.keys())[:10]
        counts = [degree_counts[d] for d in degrees]
        ax2.bar(degrees, counts, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Node Degree (# children)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{self.language} - Node Degree Distribution')
        ax2.grid(True, alpha=0.3)

        # 3. Distance to root by POS
        ax3 = plt.subplot(2, 3, 3)
        pos_to_plot = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP']
        pos_present = [pos for pos in pos_to_plot if pos in stats['distances_to_root']]
        avg_distances = [np.mean(stats['distances_to_root'][pos]) for pos in pos_present]
        ax3.barh(pos_present, avg_distances, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Average Distance to Root')
        ax3.set_title(f'{self.language} - Avg Distance to Root')
        ax3.grid(True, alpha=0.3)

        # 4. Top dependency relations
        ax4 = plt.subplot(2, 3, 4)
        top_deps = stats['dep_relations'].most_common(12)
        deps, counts = zip(*top_deps)
        ax4.barh(range(len(deps)), counts, alpha=0.7, edgecolor='black')
        ax4.set_yticks(range(len(deps)))
        ax4.set_yticklabels(deps)
        ax4.set_xlabel('Frequency')
        ax4.set_title(f'{self.language} - Top Dependency Relations')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3)

        # 5. Leaf node POS distribution
        ax5 = plt.subplot(2, 3, 5)
        leaf_counter = Counter(stats['leaf_pos'])
        top_leaves = leaf_counter.most_common(10)
        leaves, counts = zip(*top_leaves)
        ax5.barh(range(len(leaves)), counts, alpha=0.7, edgecolor='black')
        ax5.set_yticks(range(len(leaves)))
        ax5.set_yticklabels(leaves)
        ax5.set_xlabel('Frequency')
        ax5.set_title(f'{self.language} - Leaf Node POS Tags')
        ax5.invert_yaxis()
        ax5.grid(True, alpha=0.3)

        # 6. POS tag distribution
        ax6 = plt.subplot(2, 3, 6)
        top_pos = stats['pos_tags'].most_common(12)
        pos, counts = zip(*top_pos)
        ax6.bar(range(len(pos)), counts, alpha=0.7, edgecolor='black')
        ax6.set_xticks(range(len(pos)))
        ax6.set_xticklabels(pos, rotation=45, ha='right')
        ax6.set_ylabel('Frequency')
        ax6.set_title(f'{self.language} - POS Tag Distribution')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'dependency_analysis_{self.language}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {filename}")


def compare_languages(analyzer1, analyzer2, stats1, stats2):
    """Create comparative visualizations between two languages"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Tree depth comparison
    ax1 = axes[0, 0]
    ax1.hist(stats1['tree_depths'], bins=20, alpha=0.5, label=analyzer1.language, density=True)
    ax1.hist(stats2['tree_depths'], bins=20, alpha=0.5, label=analyzer2.language, density=True)
    ax1.set_xlabel('Tree Depth')
    ax1.set_ylabel('Density')
    ax1.set_title('Tree Depth Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Average distance to root comparison
    ax2 = axes[0, 1]
    pos_to_plot = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']
    pos_present = [pos for pos in pos_to_plot
                   if pos in stats1['distances_to_root'] and pos in stats2['distances_to_root']]

    x = np.arange(len(pos_present))
    width = 0.35

    dist1 = [np.mean(stats1['distances_to_root'][pos]) for pos in pos_present]
    dist2 = [np.mean(stats2['distances_to_root'][pos]) for pos in pos_present]

    ax2.bar(x - width/2, dist1, width, label=analyzer1.language, alpha=0.8)
    ax2.bar(x + width/2, dist2, width, label=analyzer2.language, alpha=0.8)
    ax2.set_xlabel('POS Tag')
    ax2.set_ylabel('Average Distance to Root')
    ax2.set_title('Distance to Root by POS')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pos_present)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Node degree comparison
    ax3 = axes[1, 0]
    degree_counts1 = Counter(stats1['node_degrees'])
    degree_counts2 = Counter(stats2['node_degrees'])

    max_degree = max(max(degree_counts1.keys()), max(degree_counts2.keys()))
    degrees = list(range(min(7, max_degree + 1)))

    freq1 = [100 * degree_counts1.get(d, 0) / len(stats1['node_degrees']) for d in degrees]
    freq2 = [100 * degree_counts2.get(d, 0) / len(stats2['node_degrees']) for d in degrees]

    x = np.arange(len(degrees))
    ax3.bar(x - width/2, freq1, width, label=analyzer1.language, alpha=0.8)
    ax3.bar(x + width/2, freq2, width, label=analyzer2.language, alpha=0.8)
    ax3.set_xlabel('Node Degree')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Node Degree Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels(degrees)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_data = [
        ['Metric', analyzer1.language, analyzer2.language],
        ['Avg Tree Depth', f"{np.mean(stats1['tree_depths']):.2f}",
         f"{np.mean(stats2['tree_depths']):.2f}"],
        ['Avg Node Degree', f"{np.mean(stats1['node_degrees']):.2f}",
         f"{np.mean(stats2['node_degrees']):.2f}"],
        ['Most Common POS', stats1['pos_tags'].most_common(1)[0][0],
         stats2['pos_tags'].most_common(1)[0][0]],
        ['Most Common Dep', stats1['dep_relations'].most_common(1)[0][0],
         stats2['dep_relations'].most_common(1)[0][0]],
    ]

    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    plt.savefig('dependency_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparative visualization saved: dependency_comparison.png")


def main():
    """Main analysis pipeline"""

    print("="*80)
    print("DELIVERABLE 5: DEPENDENCY PARSING AND SYNTACTIC ANALYSIS")
    print("="*80)

    # SpaCy models to use
    # German: de_core_news_sm, de_core_news_md, de_core_news_lg
    # Hindi: Not available in standard SpaCy
    # For this example, we'll use German. You can add other languages as needed.

    german_model = "de_core_news_sm"  # or de_core_news_md, de_core_news_lg
    german_test = "data/ud/UD_German-GSD/de_gsd-ud-test.conllu"

    # Note: Hindi doesn't have a standard SpaCy model
    # You would need to train one or use an alternative
    # For demonstration, we'll focus on German

    # ===== GERMAN ANALYSIS =====
    print("\n" + "="*80)
    print("GERMAN DEPENDENCY PARSING")
    print("="*80)

    german_analyzer = DependencyAnalyzer("German", german_model, german_test)
    german_analyzer.load_spacy_model()
    german_analyzer.load_gold_conllu(max_sentences=500)
    german_analyzer.parse_sentences()

    # Compute statistics
    german_stats = german_analyzer.compute_tree_statistics()
    german_analyzer.print_statistics(german_stats)

    # Show examples
    german_analyzer.show_examples(n=3)

    # Evaluate parser
    german_eval = german_analyzer.evaluate_parser()

    # Visualize
    german_analyzer.visualize_statistics(german_stats)

    # Optional: Test with ambiguous sentences
    print(f"\n{'='*60}")
    print("Testing with Ambiguous Sentences")
    print(f"{'='*60}")

    ambiguous_sentences = [
        "Ich sehe den Mann mit dem Fernrohr.",  # PP attachment ambiguity
        "Die Polizei erschoss den Mann mit der Waffe.",  # Instrument ambiguity
        "Der Hund beißt den Mann im Park."  # Locative ambiguity
    ]

    print("\nParsing ambiguous sentences...")
    ambiguous_parsed = [german_analyzer.nlp(sent) for sent in ambiguous_sentences]
    german_analyzer.show_examples(n=3, ambiguous=ambiguous_parsed)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - dependency_analysis_German.png")

    print("\nNote: Hindi parsing would require a trained model.")
    print("You can train your own using SpaCy or use alternative parsers.")


if __name__ == "__main__":
    main()
