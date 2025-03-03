#!/usr/bin/env python3
"""
Field Extraction Algorithm Benchmark

This script benchmarks different algorithms for extracting field identifiers from
regulatory planning documents. It compares rule-based, regex-based, and ML-based approaches
for accuracy, speed, and resource usage.
"""

import time
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import random
import string
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
import tempfile
import shutil
import logging
import traceback
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import spacy
from spacy.tokens import Doc
from spacy.training import Example
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create results directory
os.makedirs("results", exist_ok=True)

# Sample data generation
def generate_test_document(
    num_paragraphs: int = 50,
    fields_per_paragraph: float = 0.3,
    noise_level: float = 0.1
) -> Tuple[str, List[str]]:
    """
    Generate a test document with known field identifiers.
    
    Args:
        num_paragraphs: Number of paragraphs in the document
        fields_per_paragraph: Average number of fields per paragraph
        noise_level: Probability of adding noise/errors to fields
        
    Returns:
        Tuple of (document text, list of actual field identifiers)
    """
    # Common field prefixes in Norwegian regulatory documents
    prefixes = ['§', 'pkt.', 'kap.', 'art.', 'BRA', 'BYA', 'BFA', 'f_', 'o_', 
                'p_', 'b_', 'PBL', 'TEK', 'SAK', 'DOK', 'KPA']
    
    # Common Norwegian words for generating text
    norwegian_words = [
        'regulering', 'plan', 'bygning', 'areal', 'høyde', 'bredde', 'avstand',
        'grense', 'formål', 'utnyttelse', 'område', 'bebyggelse', 'bolig',
        'næring', 'industri', 'vei', 'parkering', 'friområde', 'lekeplass',
        'kommune', 'fylke', 'stat', 'eiendom', 'tomt', 'kart', 'bestemmelse'
    ]
    
    paragraphs = []
    actual_fields = []
    
    for i in range(num_paragraphs):
        # Decide if this paragraph contains fields
        if random.random() < fields_per_paragraph:
            # Generate 1-3 fields for this paragraph
            num_fields = random.randint(1, 3)
            
            paragraph_fields = []
            for _ in range(num_fields):
                prefix = random.choice(prefixes)
                number = f"{random.randint(1, 50)}"
                if random.random() < 0.5:
                    number += f".{random.randint(1, 20)}"
                if random.random() < 0.3:
                    number += f".{random.randint(1, 10)}"
                
                field = f"{prefix}{number}"
                
                # Add noise/errors to some fields
                if random.random() < noise_level:
                    error_type = random.choice(['typo', 'spacing', 'case'])
                    if error_type == 'typo':
                        # Replace a character
                        pos = random.randint(0, len(field) - 1)
                        chars = list(field)
                        chars[pos] = random.choice(string.ascii_letters + string.digits)
                        field = ''.join(chars)
                    elif error_type == 'spacing':
                        # Add or remove spacing
                        if ' ' in field:
                            field = field.replace(' ', '')
                        else:
                            pos = random.randint(1, len(field) - 1)
                            field = field[:pos] + ' ' + field[pos:]
                    elif error_type == 'case':
                        # Change case
                        field = field.swapcase()
                
                paragraph_fields.append(field)
                actual_fields.append(field)
            
            # Generate paragraph with fields
            paragraph = ""
            for field in paragraph_fields:
                # Add field with surrounding text
                words_before = ' '.join(random.choices(norwegian_words, k=random.randint(3, 10)))
                words_after = ' '.join(random.choices(norwegian_words, k=random.randint(5, 15)))
                paragraph += f"{words_before} {field} {words_after}. "
            
            paragraphs.append(paragraph)
        else:
            # Generate paragraph without fields
            num_sentences = random.randint(1, 5)
            paragraph = ""
            
            for _ in range(num_sentences):
                words = ' '.join(random.choices(norwegian_words, k=random.randint(5, 20)))
                paragraph += f"{words}. "
            
            paragraphs.append(paragraph)
    
    # Combine paragraphs into document
    document = '\n\n'.join(paragraphs)
    
    return document, actual_fields

def generate_test_dataset(
    num_documents: int = 100,
    min_paragraphs: int = 20,
    max_paragraphs: int = 100,
    fields_per_paragraph: float = 0.3,
    noise_level: float = 0.1
) -> List[Tuple[str, List[str]]]:
    """
    Generate a test dataset of documents with known field identifiers.
    
    Args:
        num_documents: Number of documents to generate
        min_paragraphs: Minimum number of paragraphs per document
        max_paragraphs: Maximum number of paragraphs per document
        fields_per_paragraph: Average number of fields per paragraph
        noise_level: Probability of adding noise/errors to fields
        
    Returns:
        List of (document text, list of actual field identifiers) tuples
    """
    dataset = []
    
    for _ in tqdm(range(num_documents), desc="Generating test documents"):
        num_paragraphs = random.randint(min_paragraphs, max_paragraphs)
        document, fields = generate_test_document(
            num_paragraphs=num_paragraphs,
            fields_per_paragraph=fields_per_paragraph,
            noise_level=noise_level
        )
        dataset.append((document, fields))
    
    return dataset

# Field extraction algorithms
def rule_based_extraction(text: str) -> List[str]:
    """
    Extract field identifiers using simple rule-based approach.
    
    Args:
        text: Document text
        
    Returns:
        List of extracted field identifiers
    """
    # Common field prefixes in Norwegian regulatory documents
    prefixes = ['§', 'pkt.', 'kap.', 'art.', 'BRA', 'BYA', 'BFA', 'f_', 'o_', 
                'p_', 'b_', 'PBL', 'TEK', 'SAK', 'DOK', 'KPA']
    
    # Split text into words
    words = re.findall(r'\S+', text)
    
    # Extract fields based on prefixes
    fields = []
    for word in words:
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix):
                # Check if the rest of the word contains numbers and dots
                rest = word[len(prefix):]
                if re.match(r'^[0-9.]+$', rest):
                    fields.append(word)
                    break
    
    return fields

def regex_based_extraction(text: str) -> List[str]:
    """
    Extract field identifiers using regex patterns.
    
    Args:
        text: Document text
        
    Returns:
        List of extracted field identifiers
    """
    # Define regex patterns for different field types
    patterns = [
        r'§\s*\d+(\.\d+)*',  # §1, §1.2, §1.2.3
        r'pkt\.\s*\d+(\.\d+)*',  # pkt.1, pkt.1.2
        r'kap\.\s*\d+(\.\d+)*',  # kap.1, kap.1.2
        r'art\.\s*\d+(\.\d+)*',  # art.1, art.1.2
        r'[BfopbPBLTEKSAKDOKKPA][A-Za-z]*\s*\d+(\.\d+)*',  # BRA1, BYA1.2, f_1, o_1.2
        r'[fopb]_\w+',  # f_GF1, o_VEG1
    ]
    
    # Extract fields using regex patterns
    fields = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        fields.extend(matches)
    
    # Remove duplicates and clean up
    fields = list(set(fields))
    fields = [field.strip() for field in fields]
    
    return fields

def spacy_based_extraction(text: str, nlp) -> List[str]:
    """
    Extract field identifiers using spaCy NER.
    
    Args:
        text: Document text
        nlp: Loaded spaCy model
        
    Returns:
        List of extracted field identifiers
    """
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract entities of relevant types
    fields = []
    for ent in doc.ents:
        if ent.label_ in ["FIELD_ID", "REGULATION"]:
            fields.append(ent.text)
    
    return fields

def transformer_based_extraction(text: str, extractor) -> List[str]:
    """
    Extract field identifiers using transformer-based NER.
    
    Args:
        text: Document text
        extractor: Loaded transformer NER pipeline
        
    Returns:
        List of extracted field identifiers
    """
    # Process text with transformer model
    results = extractor(text)
    
    # Extract entities of relevant types
    fields = []
    for result in results:
        if result['entity_group'] in ["FIELD_ID", "B-FIELD_ID", "I-FIELD_ID"]:
            fields.append(result['word'])
    
    return fields

# Benchmark functions
def measure_system_metrics() -> Dict[str, float]:
    """Measure current system metrics."""
    process = psutil.Process(os.getpid())
    
    return {
        "memory_mb": process.memory_info().rss / (1024 * 1024),  # Convert to MB
        "cpu_percent": process.cpu_percent(),
    }

def evaluate_extraction(
    extracted_fields: List[str],
    actual_fields: List[str]
) -> Dict[str, float]:
    """
    Evaluate extraction performance.
    
    Args:
        extracted_fields: List of extracted field identifiers
        actual_fields: List of actual field identifiers
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to sets for comparison
    extracted_set = set(extracted_fields)
    actual_set = set(actual_fields)
    
    # Calculate metrics
    true_positives = len(extracted_set.intersection(actual_set))
    false_positives = len(extracted_set - actual_set)
    false_negatives = len(actual_set - extracted_set)
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

def benchmark_algorithm(
    algorithm_name: str,
    algorithm_func: Callable,
    dataset: List[Tuple[str, List[str]]],
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark a field extraction algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        algorithm_func: Function implementing the algorithm
        dataset: List of (document text, actual fields) tuples
        **kwargs: Additional arguments for the algorithm function
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking {algorithm_name} with {len(dataset)} documents...")
    
    # Measure initial system metrics
    initial_metrics = measure_system_metrics()
    
    # Run the algorithm on each document and measure time
    start_time = time.time()
    
    all_actual_fields = []
    all_extracted_fields = []
    document_results = []
    
    for i, (document, actual_fields) in enumerate(tqdm(dataset, desc=f"Processing with {algorithm_name}")):
        doc_start_time = time.time()
        
        # Extract fields
        extracted_fields = algorithm_func(document, **kwargs)
        
        doc_end_time = time.time()
        doc_processing_time = doc_end_time - doc_start_time
        
        # Evaluate extraction
        evaluation = evaluate_extraction(extracted_fields, actual_fields)
        
        # Store results
        document_results.append({
            "document_id": i,
            "processing_time_s": doc_processing_time,
            "num_actual_fields": len(actual_fields),
            "num_extracted_fields": len(extracted_fields),
            **evaluation
        })
        
        # Accumulate fields for overall evaluation
        all_actual_fields.extend(actual_fields)
        all_extracted_fields.extend(extracted_fields)
    
    end_time = time.time()
    total_processing_time = end_time - start_time
    
    # Measure final system metrics
    final_metrics = measure_system_metrics()
    
    # Calculate overall metrics
    num_documents = len(dataset)
    avg_processing_time = total_processing_time / num_documents if num_documents > 0 else 0
    
    # Calculate overall evaluation
    overall_evaluation = evaluate_extraction(all_extracted_fields, all_actual_fields)
    
    # Calculate metric differences
    memory_usage = final_metrics["memory_mb"] - initial_metrics["memory_mb"]
    cpu_usage = final_metrics["cpu_percent"]
    
    # Calculate average metrics across documents
    avg_metrics = pd.DataFrame(document_results).mean().to_dict()
    
    # Combine results
    benchmark_results = {
        "algorithm": algorithm_name,
        "num_documents": num_documents,
        "total_processing_time_s": total_processing_time,
        "avg_processing_time_s": avg_processing_time,
        "memory_usage_mb": memory_usage,
        "cpu_percent": cpu_usage,
        "overall_precision": overall_evaluation["precision"],
        "overall_recall": overall_evaluation["recall"],
        "overall_f1": overall_evaluation["f1"],
        "avg_precision": avg_metrics["precision"],
        "avg_recall": avg_metrics["recall"],
        "avg_f1": avg_metrics["f1"],
        "document_results": document_results
    }
    
    return benchmark_results

def run_benchmarks(
    num_documents: int = 100,
    min_paragraphs: int = 20,
    max_paragraphs: int = 100,
    fields_per_paragraph: float = 0.3,
    noise_level: float = 0.1
) -> pd.DataFrame:
    """
    Run all field extraction algorithm benchmarks.
    
    Args:
        num_documents: Number of documents to generate
        min_paragraphs: Minimum number of paragraphs per document
        max_paragraphs: Maximum number of paragraphs per document
        fields_per_paragraph: Average number of fields per paragraph
        noise_level: Probability of adding noise/errors to fields
        
    Returns:
        DataFrame with benchmark results
    """
    # Generate test dataset
    dataset = generate_test_dataset(
        num_documents=num_documents,
        min_paragraphs=min_paragraphs,
        max_paragraphs=max_paragraphs,
        fields_per_paragraph=fields_per_paragraph,
        noise_level=noise_level
    )
    
    # Define algorithms to benchmark
    algorithms = [
        {
            "name": "rule_based",
            "func": rule_based_extraction,
            "params": {}
        },
        {
            "name": "regex_based",
            "func": regex_based_extraction,
            "params": {}
        }
    ]
    
    # Try to load spaCy model if available
    try:
        logger.info("Loading spaCy model...")
        nlp = spacy.load("nb_core_news_md")
        
        # Add spaCy-based algorithm
        algorithms.append({
            "name": "spacy_based",
            "func": spacy_based_extraction,
            "params": {"nlp": nlp}
        })
        
        logger.info("SpaCy model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load spaCy model: {e}")
    
    # Try to load transformer model if available
    try:
        logger.info("Loading transformer model...")
        tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base")
        model = AutoModelForTokenClassification.from_pretrained("NbAiLab/nb-bert-base")
        extractor = pipeline("ner", model=model, tokenizer=tokenizer)
        
        # Add transformer-based algorithm
        algorithms.append({
            "name": "transformer_based",
            "func": transformer_based_extraction,
            "params": {"extractor": extractor}
        })
        
        logger.info("Transformer model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load transformer model: {e}")
    
    # Run benchmarks
    all_results = []
    document_results = []
    
    for algorithm in algorithms:
        try:
            # Run benchmark
            result = benchmark_algorithm(
                algorithm_name=algorithm["name"],
                algorithm_func=algorithm["func"],
                dataset=dataset,
                **algorithm["params"]
            )
            
            # Extract document results
            for doc_result in result["document_results"]:
                doc_result["algorithm"] = algorithm["name"]
                document_results.append(doc_result)
            
            # Remove document results from main result to avoid duplication
            result_copy = result.copy()
            del result_copy["document_results"]
            
            # Add to results
            all_results.append(result_copy)
            
            # Log result
            logger.info(f"  {algorithm['name']} - F1: {result['overall_f1']:.4f}, Time: {result['avg_processing_time_s']:.4f}s")
            
        except Exception as e:
            logger.error(f"Error benchmarking {algorithm['name']}: {e}")
            traceback.print_exc()
    
    # Convert to DataFrames
    results_df = pd.DataFrame(all_results)
    document_results_df = pd.DataFrame(document_results)
    
    # Save raw results
    results_df.to_csv("results/field_extraction_benchmark_results.csv", index=False)
    document_results_df.to_csv("results/field_extraction_document_results.csv", index=False)
    
    return results_df, document_results_df

def visualize_results(results_df: pd.DataFrame, document_results_df: pd.DataFrame) -> None:
    """
    Create visualizations from benchmark results.
    
    Args:
        results_df: DataFrame with overall benchmark results
        document_results_df: DataFrame with per-document results
    """
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. F1 score comparison
    plt.figure(figsize=(10, 6))
    results_df = results_df.sort_values('overall_f1')
    plt.barh(results_df['algorithm'], results_df['overall_f1'])
    plt.xlabel('F1 Score')
    plt.ylabel('Algorithm')
    plt.title('F1 Score by Algorithm')
    plt.xlim(0, 1)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig("results/plots/f1_by_algorithm.png", dpi=300)
    plt.close()
    
    # 2. Precision-Recall comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['overall_recall'], results_df['overall_precision'], s=100)
    
    # Add labels for each point
    for i, row in results_df.iterrows():
        plt.annotate(
            row['algorithm'],
            (row['overall_recall'], row['overall_precision']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall by Algorithm')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/precision_recall.png", dpi=300)
    plt.close()
    
    # 3. Processing time comparison
    plt.figure(figsize=(10, 6))
    results_df = results_df.sort_values('avg_processing_time_s')
    plt.barh(results_df['algorithm'], results_df['avg_processing_time_s'])
    plt.xlabel('Average Processing Time (s)')
    plt.ylabel('Algorithm')
    plt.title('Processing Time by Algorithm')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig("results/plots/processing_time.png", dpi=300)
    plt.close()
    
    # 4. Memory usage comparison
    plt.figure(figsize=(10, 6))
    results_df = results_df.sort_values('memory_usage_mb')
    plt.barh(results_df['algorithm'], results_df['memory_usage_mb'])
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Algorithm')
    plt.title('Memory Usage by Algorithm')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig("results/plots/memory_usage.png", dpi=300)
    plt.close()
    
    # 5. F1 vs Processing Time
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['avg_processing_time_s'], results_df['overall_f1'], s=100)
    
    # Add labels for each point
    for i, row in results_df.iterrows():
        plt.annotate(
            row['algorithm'],
            (row['avg_processing_time_s'], row['overall_f1']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    plt.xlabel('Average Processing Time (s)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Processing Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/f1_vs_time.png", dpi=300)
    plt.close()
    
    # 6. Performance by document size
    plt.figure(figsize=(12, 8))
    
    # Group by algorithm and document size
    document_results_df['doc_size_bin'] = pd.qcut(document_results_df['num_actual_fields'], 4)
    size_performance = document_results_df.groupby(['algorithm', 'doc_size_bin'])['f1'].mean().reset_index()
    
    # Pivot for plotting
    size_performance_pivot = size_performance.pivot(index='doc_size_bin', columns='algorithm', values='f1')
    
    # Plot
    size_performance_pivot.plot(marker='o')
    plt.xlabel('Document Size (Number of Fields)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Document Size')
    plt.grid(True)
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig("results/plots/f1_by_document_size.png", dpi=300)
    plt.close()
    
    # 7. Performance by noise level
    if 'noise_level' in document_results_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Group by algorithm and noise level
        noise_performance = document_results_df.groupby(['algorithm', 'noise_level'])['f1'].mean().reset_index()
        
        # Pivot for plotting
        noise_performance_pivot = noise_performance.pivot(index='noise_level', columns='algorithm', values='f1')
        
        # Plot
        noise_performance_pivot.plot(marker='o')
        plt.xlabel('Noise Level')
        plt.ylabel('F1 Score')
        plt.title('F1 Score by Noise Level')
        plt.grid(True)
        plt.legend(title='Algorithm')
        plt.tight_layout()
        plt.savefig("results/plots/f1_by_noise_level.png", dpi=300)
        plt.close()
    
    # 8. Summary table
    summary = results_df[['algorithm', 'overall_precision', 'overall_recall', 'overall_f1', 'avg_processing_time_s', 'memory_usage_mb']]
    summary.columns = ['Algorithm', 'Precision', 'Recall', 'F1 Score', 'Avg Time (s)', 'Memory (MB)']
    summary = summary.sort_values('F1 Score', ascending=False)
    summary.to_csv("results/plots/summary_table.csv", index=False)
    
    # Create a visual summary table
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    
    # Format the data for display
    cell_text = []
    for i, row in summary.iterrows():
        cell_text.append([
            row['Algorithm'],
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1 Score']:.4f}",
            f"{row['Avg Time (s)']:.4f}",
            f"{row['Memory (MB)']:.1f}"
        ])
    
    table = plt.table(
        cellText=cell_text,
        colLabels=summary.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Field Extraction Algorithm Performance Summary', y=1.08)
    plt.tight_layout()
    plt.savefig("results/plots/summary_table.png", dpi=300)
    plt.close()

def main():
    """Main function to run all benchmarks and visualizations."""
    logger.info("Starting field extraction algorithm benchmark...")
    
    # Run benchmarks
    results_df, document_results_df = run_benchmarks(
        num_documents=50,  # Adjust based on your system's capabilities
        min_paragraphs=20,
        max_paragraphs=100,
        fields_per_paragraph=0.3,
        noise_level=0.1
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    visualize_results(results_df, document_results_df)
    
    logger.info("Benchmark completed. Results saved to 'results/' directory.")

if __name__ == "__main__":
    main() 