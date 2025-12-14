import csv
import json
import numpy as np

# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def cohens_kappa(ann1, ann2):
    """Calculate Cohen's Kappa for two annotators"""
    # Ensure lists are the same length
    min_len = min(len(ann1), len(ann2))
    if min_len == 0:
        return 0.0
    
    ann1 = ann1[:min_len]
    ann2 = ann2[:min_len]
    n = min_len
    
    # Get all unique labels
    labels = sorted(list(set(ann1) | set(ann2)))
    
    # Build confusion matrix
    matrix = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    
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
    """Calculate Krippendorff's Alpha for multiple annotators"""
    # Find the minimum length across all non-empty annotators to avoid IndexErrors
    lengths = [len(a) for a in annotations if len(a) > 0]
    if not lengths:
        return 0.0
    min_len = min(lengths)
    
    # Truncate all annotations to the minimum length
    cleaned_annotations = [a[:min_len] for a in annotations if len(a) >= min_len]
    
    if not cleaned_annotations: 
        return 0.0
        
    n_annotators = len(cleaned_annotations)
    n_items = min_len
    
    # Collect all values per item
    item_values = []
    for i in range(n_items):
        values = [cleaned_annotations[a][i] for a in range(n_annotators)]
        item_values.append(values)
    
    # Calculate observed disagreement
    Do = 0
    n_pairs = 0
    for values in item_values:
        if len(values) < 2:
            continue
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                try:
                    if level == 'ordinal':
                        val_i = float(values[i])
                        val_j = float(values[j])
                        Do += (val_i - val_j) ** 2
                    else:  # nominal
                        Do += 0 if values[i] == values[j] else 1
                    n_pairs += 1
                except ValueError:
                    continue
    
    if n_pairs == 0:
        return 1.0
    Do = Do / n_pairs
    
    # Calculate expected disagreement
    all_values = [v for values in item_values for v in values]
    De = 0
    n_total_pairs = 0
    for i in range(len(all_values)):
        for j in range(i+1, len(all_values)):
            try:
                if level == 'ordinal':
                    val_i = float(all_values[i])
                    val_j = float(all_values[j])
                    De += (val_i - val_j) ** 2
                else:
                    De += 0 if all_values[i] == all_values[j] else 1
                n_total_pairs += 1
            except ValueError:
                continue
    
    if n_total_pairs == 0:
        return 1.0
    De = De / n_total_pairs
    
    if De == 0:
        return 1.0
    
    alpha = 1 - (Do / De)
    return alpha

# =============================================================================
# DATA LOADING
# =============================================================================

def read_annotation_csv(filepath):
    """Reads a filled annotation CSV and returns the dictionary structure."""
    annotations = {
        'fluency_A': [],
        'fluency_B': [],
        'coherence_A': [],
        'coherence_B': [],
        'preference': []
    }
    
    print(f"Reading {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # FIX: Handle "5.0" strings by converting to float first, then int
                    annotations['fluency_A'].append(int(float(row['Fluency_A (1-5)'])))
                    annotations['fluency_B'].append(int(float(row['Fluency_B (1-5)'])))
                    annotations['coherence_A'].append(int(float(row['Coherence_A (1-5)'])))
                    annotations['coherence_B'].append(int(float(row['Coherence_B (1-5)'])))
                    annotations['preference'].append(row['Preference (A/B/Tie)'].strip())
                except (ValueError, KeyError) as e:
                    # Skip rows only if they are genuinely empty or broken
                    if any(row.values()): 
                        print(f"  Skipping row {row.get('ID', '?')} in {filepath}: {e}")
                    continue
    except FileNotFoundError:
        print(f"  Error: File {filepath} not found.")
        return None

    return annotations

# =============================================================================
# EVALUATION LOGIC
# =============================================================================

def calculate_agreement(ann1, ann2, ann3):
    """Calculate inter-annotator agreement"""
    
    print(f"\n{'='*70}")
    print("INTER-ANNOTATOR AGREEMENT")
    print(f"{'='*70}")
    
    # Check if we have valid data
    if not (ann1 and ann2 and ann3):
        print("Error: One or more annotation files failed to load.")
        return {}
        
    categories = ['fluency_A', 'fluency_B', 'coherence_A', 'coherence_B']
    results = {}
    
    for cat in categories:
        # Check lengths
        l1, l2, l3 = len(ann1[cat]), len(ann2[cat]), len(ann3[cat])
        if l1 == 0 or l2 == 0 or l3 == 0:
            print(f"Skipping {cat}: Missing data in one or more files.")
            continue
            
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
    
    results['preference'] = {
        'kappa_12': pref_kappa_12,
        'kappa_13': pref_kappa_13,
        'kappa_23': pref_kappa_23
    }
    
    return results

def analyze_results(ann1, ann2, ann3):
    """Analyze evaluation results"""
    
    if not (ann1 and ann2 and ann3):
        return {}
        
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS ANALYSIS")
    print(f"{'='*70}")
    
    # Ensure equal lengths for averaging
    min_len = min(len(ann1['fluency_A']), len(ann2['fluency_A']), len(ann3['fluency_A']))
    
    # Helper to slice list
    def sl(lst): return lst[:min_len]

    avg_fluency_a = np.mean([sl(ann1['fluency_A']), sl(ann2['fluency_A']), sl(ann3['fluency_A'])])
    avg_fluency_b = np.mean([sl(ann1['fluency_B']), sl(ann2['fluency_B']), sl(ann3['fluency_B'])])
    avg_coherence_a = np.mean([sl(ann1['coherence_A']), sl(ann2['coherence_A']), sl(ann3['coherence_A'])])
    avg_coherence_b = np.mean([sl(ann1['coherence_B']), sl(ann2['coherence_B']), sl(ann3['coherence_B'])])
    
    print(f"\nAverage Scores (n={min_len} samples x 3 annotators):")
    print(f"{'Metric':<20} {'Model A (Baseline)':<20} {'Model B (Label Smooth)':<20}")
    print("-" * 60)
    print(f"{'Fluency':<20} {avg_fluency_a:<20.2f} {avg_fluency_b:<20.2f}")
    print(f"{'Coherence':<20} {avg_coherence_a:<20.2f} {avg_coherence_b:<20.2f}")
    print(f"{'Overall':<20} {(avg_fluency_a + avg_coherence_a)/2:<20.2f} {(avg_fluency_b + avg_coherence_b)/2:<20.2f}")
    
    # Preference counts
    all_prefs = ann1['preference'][:min_len] + ann2['preference'][:min_len] + ann3['preference'][:min_len]
    pref_a = all_prefs.count('A')
    pref_b = all_prefs.count('B')
    pref_tie = all_prefs.count('Tie')
    total_prefs = len(all_prefs)
    
    if total_prefs > 0:
        print(f"\nPreference Distribution:")
        print(f"  Model A preferred: {pref_a}/{total_prefs} ({100*pref_a/total_prefs:.1f}%)")
        print(f"  Model B preferred: {pref_b}/{total_prefs} ({100*pref_b/total_prefs:.1f}%)")
        print(f"  Tie:               {pref_tie}/{total_prefs} ({100*pref_tie/total_prefs:.1f}%)")
        
        if pref_b > pref_a:
            winner = "Model B (Label Smoothing)"
        elif pref_a > pref_b:
            winner = "Model A (Baseline)"
        else:
            winner = "No clear winner"
        print(f"\nConclusion: {winner} is preferred overall.")
        
        return {
            'winner': winner,
            'pref_a': pref_a,
            'pref_b': pref_b
        }
    return {}

# =============================================================================
# MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # TODO: CONFIRM THESE FILENAMES MATCH YOUR FILES
    # -------------------------------------------------------------------------
    files = [
        "A_annotation_sheet_Finnish.csv", 
        "B_annotation_sheet_Finnish.csv", 
        "C_annotation_sheet_Finnish.csv"
    ]
    # -------------------------------------------------------------------------

    print(f"Processing evaluation for: {files}")
    
    ann1 = read_annotation_csv(files[0])
    ann2 = read_annotation_csv(files[1])
    ann3 = read_annotation_csv(files[2])

    if ann1 and ann2 and ann3:
        agreement_stats = calculate_agreement(ann1, ann2, ann3)
        analysis_stats = analyze_results(ann1, ann2, ann3)

        output_data = {
            "files_used": files,
            "agreement": agreement_stats,
            "results": analysis_stats
        }

        output_filename = "evaluation_results_Finnish.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_filename}")
    else:
        print("Failed to run evaluation due to file errors.")

if __name__ == "__main__":
    main()