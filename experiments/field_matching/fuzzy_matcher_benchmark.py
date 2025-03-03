#!/usr/bin/env python3
"""
Fuzzy Field Matching Benchmark

This script benchmarks different fuzzy matching algorithms for field identifier matching
with configurable thresholds. It evaluates performance and accuracy metrics.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz as rfuzz
from thefuzz import fuzz as tfuzz
import jellyfish
from tqdm import tqdm
import random
import string
import os
from sklearn.metrics import precision_recall_fscore_support

# Create output directory
os.makedirs("results", exist_ok=True)

# Sample data generation
def generate_test_data(num_samples=1000, error_rate=0.2, max_errors=3):
    """Generate test data with controlled error rates for benchmarking."""
    original_fields = []
    modified_fields = []
    
    # Common field prefixes in Norwegian regulatory documents
    prefixes = ['§', 'pkt.', 'kap.', 'art.', 'BRA', 'BYA', 'BFA', 'f_', 'o_', 
                'p_', 'b_', 'PBL', 'TEK', 'SAK', 'DOK', 'KPA']
    
    # Generate original fields
    for i in range(num_samples):
        # Create realistic field identifiers
        if random.random() < 0.7:  # 70% have prefixes
            prefix = random.choice(prefixes)
            number = f"{random.randint(1, 50)}"
            if random.random() < 0.5:
                number += f".{random.randint(1, 20)}"
            if random.random() < 0.3:
                number += f".{random.randint(1, 10)}"
            field = f"{prefix}{number}"
        else:
            # Some fields are just descriptive
            words = ['regulering', 'plan', 'bygning', 'areal', 'høyde', 
                    'bredde', 'avstand', 'grense', 'formål', 'utnyttelse']
            field = random.choice(words) + random.choice(['grad', 'faktor', 'indeks', 'område', 'sone'])
        
        original_fields.append(field)
        
        # Create modified version with controlled errors
        if random.random() < error_rate:
            # Apply 1 to max_errors modifications
            num_errors = random.randint(1, max_errors)
            modified = list(field)
            
            for _ in range(num_errors):
                error_type = random.choice(['insert', 'delete', 'replace', 'transpose', 'case'])
                
                if error_type == 'insert' and len(modified) > 0:
                    pos = random.randint(0, len(modified))
                    char = random.choice(string.ascii_letters + string.digits + '.-_§')
                    modified.insert(pos, char)
                    
                elif error_type == 'delete' and len(modified) > 1:
                    pos = random.randint(0, len(modified) - 1)
                    modified.pop(pos)
                    
                elif error_type == 'replace' and len(modified) > 0:
                    pos = random.randint(0, len(modified) - 1)
                    char = random.choice(string.ascii_letters + string.digits + '.-_§')
                    modified[pos] = char
                    
                elif error_type == 'transpose' and len(modified) > 1:
                    pos = random.randint(0, len(modified) - 2)
                    modified[pos], modified[pos + 1] = modified[pos + 1], modified[pos]
                    
                elif error_type == 'case' and len(modified) > 0:
                    pos = random.randint(0, len(modified) - 1)
                    if modified[pos].isalpha():
                        modified[pos] = modified[pos].swapcase()
            
            modified_fields.append(''.join(modified))
        else:
            # No errors, keep original
            modified_fields.append(field)
    
    # Create ground truth matches
    matches = [(original, modified) for original, modified in zip(original_fields, modified_fields)]
    
    return original_fields, modified_fields, matches

# Matching algorithms
def levenshtein_distance(s1, s2):
    """Calculate normalized Levenshtein distance."""
    if not s1 and not s2:
        return 0.0
    distance = jellyfish.levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len) if max_len > 0 else 1.0

def jaro_winkler_similarity(s1, s2):
    """Calculate Jaro-Winkler similarity."""
    return jellyfish.jaro_winkler_similarity(s1, s2)

def fuzzywuzzy_ratio(s1, s2):
    """Calculate FuzzyWuzzy ratio."""
    return fuzz.ratio(s1, s2) / 100.0

def fuzzywuzzy_partial_ratio(s1, s2):
    """Calculate FuzzyWuzzy partial ratio."""
    return fuzz.partial_ratio(s1, s2) / 100.0

def rapidfuzz_ratio(s1, s2):
    """Calculate RapidFuzz ratio."""
    return rfuzz.ratio(s1, s2) / 100.0

def thefuzz_ratio(s1, s2):
    """Calculate TheFuzz ratio."""
    return tfuzz.ratio(s1, s2) / 100.0

# Benchmark function
def benchmark_matchers(original_fields, modified_fields, ground_truth, thresholds):
    """Benchmark different matching algorithms with various thresholds."""
    algorithms = {
        'Levenshtein': levenshtein_distance,
        'Jaro-Winkler': jaro_winkler_similarity,
        'FuzzyWuzzy Ratio': fuzzywuzzy_ratio,
        'FuzzyWuzzy Partial': fuzzywuzzy_partial_ratio,
        'RapidFuzz Ratio': rapidfuzz_ratio,
        'TheFuzz Ratio': thefuzz_ratio
    }
    
    results = []
    
    for algo_name, algo_func in tqdm(algorithms.items(), desc="Testing algorithms"):
        for threshold in tqdm(thresholds, desc=f"Testing {algo_name} thresholds", leave=False):
            start_time = time.time()
            
            # Perform matching
            matches = []
            for orig in original_fields:
                for mod in modified_fields:
                    similarity = algo_func(orig, mod)
                    if similarity >= threshold:
                        matches.append((orig, mod))
            
            # Calculate metrics
            execution_time = time.time() - start_time
            
            # Convert matches to sets for comparison
            predicted_matches = set(matches)
            true_matches = set(ground_truth)
            
            # Calculate precision, recall, F1
            true_positives = len(predicted_matches.intersection(true_matches))
            false_positives = len(predicted_matches - true_matches)
            false_negatives = len(true_matches - predicted_matches)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'Algorithm': algo_name,
                'Threshold': threshold,
                'Execution Time (s)': execution_time,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'True Positives': true_positives,
                'False Positives': false_positives,
                'False Negatives': false_negatives
            })
    
    return pd.DataFrame(results)

# Visualization functions
def plot_precision_recall_curve(results):
    """Plot precision-recall curves for different algorithms."""
    plt.figure(figsize=(12, 8))
    
    for algo in results['Algorithm'].unique():
        algo_results = results[results['Algorithm'] == algo]
        plt.plot(algo_results['Recall'], algo_results['Precision'], 'o-', label=algo)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Different Matching Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/precision_recall_curve.png', dpi=300)
    plt.close()

def plot_f1_vs_threshold(results):
    """Plot F1 score vs threshold for different algorithms."""
    plt.figure(figsize=(12, 8))
    
    for algo in results['Algorithm'].unique():
        algo_results = results[results['Algorithm'] == algo]
        plt.plot(algo_results['Threshold'], algo_results['F1 Score'], 'o-', label=algo)
    
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold for Different Matching Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/f1_vs_threshold.png', dpi=300)
    plt.close()

def plot_execution_time(results):
    """Plot execution time for different algorithms."""
    plt.figure(figsize=(12, 8))
    
    # Group by algorithm and calculate mean execution time
    algo_times = results.groupby('Algorithm')['Execution Time (s)'].mean().reset_index()
    
    # Sort by execution time
    algo_times = algo_times.sort_values('Execution Time (s)')
    
    plt.barh(algo_times['Algorithm'], algo_times['Execution Time (s)'])
    plt.xlabel('Mean Execution Time (s)')
    plt.ylabel('Algorithm')
    plt.title('Mean Execution Time for Different Matching Algorithms')
    plt.grid(True, axis='x')
    plt.savefig('results/execution_time.png', dpi=300)
    plt.close()

def main():
    # Parameters
    num_samples = 1000
    error_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    all_results = []
    
    for error_rate in error_rates:
        print(f"Testing with error rate: {error_rate}")
        
        # Generate test data
        original_fields, modified_fields, ground_truth = generate_test_data(
            num_samples=num_samples, error_rate=error_rate
        )
        
        # Run benchmark
        results = benchmark_matchers(original_fields, modified_fields, ground_truth, thresholds)
        
        # Add error rate to results
        results['Error Rate'] = error_rate
        all_results.append(results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save results to CSV
    combined_results.to_csv('results/fuzzy_matching_benchmark.csv', index=False)
    
    # Create visualizations
    for error_rate in error_rates:
        error_results = combined_results[combined_results['Error Rate'] == error_rate]
        
        # Create directory for this error rate
        os.makedirs(f"results/error_rate_{error_rate}", exist_ok=True)
        
        # Plot precision-recall curve
        plt.figure(figsize=(12, 8))
        for algo in error_results['Algorithm'].unique():
            algo_results = error_results[error_results['Algorithm'] == algo]
            plt.plot(algo_results['Recall'], algo_results['Precision'], 'o-', label=algo)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Error Rate: {error_rate})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/error_rate_{error_rate}/precision_recall_curve.png', dpi=300)
        plt.close()
        
        # Plot F1 vs threshold
        plt.figure(figsize=(12, 8))
        for algo in error_results['Algorithm'].unique():
            algo_results = error_results[error_results['Algorithm'] == algo]
            plt.plot(algo_results['Threshold'], algo_results['F1 Score'], 'o-', label=algo)
        
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(f'F1 Score vs Threshold (Error Rate: {error_rate})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/error_rate_{error_rate}/f1_vs_threshold.png', dpi=300)
        plt.close()
    
    # Plot overall execution time
    plot_execution_time(combined_results)
    
    # Find best algorithm and threshold for each error rate
    best_configs = []
    for error_rate in error_rates:
        error_results = combined_results[combined_results['Error Rate'] == error_rate]
        best_row = error_results.loc[error_results['F1 Score'].idxmax()]
        best_configs.append({
            'Error Rate': error_rate,
            'Best Algorithm': best_row['Algorithm'],
            'Best Threshold': best_row['Threshold'],
            'F1 Score': best_row['F1 Score'],
            'Precision': best_row['Precision'],
            'Recall': best_row['Recall']
        })
    
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df.to_csv('results/best_configurations.csv', index=False)
    
    print("Benchmark completed. Results saved to 'results/' directory.")

if __name__ == "__main__":
    main() 