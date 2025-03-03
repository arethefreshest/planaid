#!/usr/bin/env python3
"""
Parallelization Benchmark

This script benchmarks different parallelization strategies for document processing:
- Multiprocessing
- Threading
- Async/Await
- Process Pools
- Thread Pools
- Task Partitioning

It measures performance metrics like throughput, CPU utilization, and scaling efficiency.
"""

import time
import os
import json
import asyncio
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue, Manager
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
import io
import sys
from functools import partial
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create results directory
os.makedirs("results", exist_ok=True)

# Document processing simulation
class Document:
    """Simulated document for processing."""
    
    def __init__(self, doc_id: str, size_kb: int, complexity: float = 1.0):
        """
        Initialize a document.
        
        Args:
            doc_id: Unique identifier for the document
            size_kb: Size of the document in KB
            complexity: Processing complexity multiplier (higher = more CPU intensive)
        """
        self.id = doc_id
        self.size_kb = size_kb
        self.complexity = complexity
        self.content = self._generate_content(size_kb)
        self.fields = self._generate_fields()
        self.metadata = {
            "source": random.choice(["plankart", "bestemmelser", "sosi"]),
            "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}",
            "timestamp": time.time()
        }
    
    def _generate_content(self, size_kb: int) -> str:
        """Generate random document content."""
        # Generate random text to reach the desired size
        target_size = size_kb * 1024
        chars_needed = target_size // 2  # Approximation for UTF-8 encoding
        return ''.join(random.choices(string.ascii_letters + string.digits + ' \n\t.,;:!?-_', k=chars_needed))
    
    def _generate_fields(self) -> List[str]:
        """Generate random field identifiers."""
        num_fields = random.randint(10, 50)
        prefixes = ['§', 'pkt.', 'kap.', 'art.', 'BRA', 'BYA', 'BFA', 'f_', 'o_', 'p_', 'b_']
        
        fields = []
        for _ in range(num_fields):
            prefix = random.choice(prefixes)
            number = f"{random.randint(1, 50)}"
            if random.random() < 0.5:
                number += f".{random.randint(1, 20)}"
            field = f"{prefix}{number}"
            fields.append(field)
        
        return fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "size_kb": self.size_kb,
            "complexity": self.complexity,
            "fields": self.fields,
            "metadata": self.metadata,
            # Exclude content to save memory
        }

# Processing functions
def cpu_intensive_task(complexity: float, duration: float = 0.1) -> None:
    """
    Simulate a CPU-intensive task.
    
    Args:
        complexity: Complexity multiplier (higher = more CPU intensive)
        duration: Base duration in seconds
    """
    start_time = time.time()
    target_time = start_time + (duration * complexity)
    
    # Perform CPU-intensive calculations
    while time.time() < target_time:
        # Matrix operations are CPU intensive
        size = int(20 * complexity)
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)

def io_intensive_task(size_kb: int, duration: float = 0.1) -> None:
    """
    Simulate an IO-intensive task.
    
    Args:
        size_kb: Size of data to write/read in KB
        duration: Base duration in seconds
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
        # Generate random data
        data = ''.join(random.choices(string.ascii_letters, k=size_kb * 1024))
        
        # Write data
        temp_file.write(data)
        temp_file.flush()
        
        # Simulate some IO delay
        time.sleep(duration)
        
        # Read data back
        temp_file.seek(0)
        _ = temp_file.read()

def process_document(doc: Document) -> Dict[str, Any]:
    """
    Process a document with simulated CPU and IO operations.
    
    Args:
        doc: Document to process
        
    Returns:
        Processed document data
    """
    # Simulate CPU-intensive processing (e.g., text analysis, field extraction)
    cpu_intensive_task(doc.complexity, 0.01 * doc.size_kb / 100)
    
    # Simulate IO-intensive processing (e.g., reading/writing files)
    io_intensive_task(doc.size_kb // 10, 0.005)
    
    # Extract fields (simulated)
    extracted_fields = doc.fields.copy()
    
    # Normalize fields (simulated)
    normalized_fields = [field.lower().replace(' ', '_') for field in extracted_fields]
    
    # Return processed data
    return {
        "id": doc.id,
        "size_kb": doc.size_kb,
        "extracted_fields": extracted_fields,
        "normalized_fields": normalized_fields,
        "processing_time": time.time(),
    }

# Parallelization strategies
def sequential_processing(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Process documents sequentially.
    
    Args:
        documents: List of documents to process
        
    Returns:
        List of processed document data
    """
    results = []
    for doc in tqdm(documents, desc="Sequential processing"):
        result = process_document(doc)
        results.append(result)
    return results

def thread_pool_processing(documents: List[Document], max_workers: int) -> List[Dict[str, Any]]:
    """
    Process documents using a thread pool.
    
    Args:
        documents: List of documents to process
        max_workers: Maximum number of worker threads
        
    Returns:
        List of processed document data
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_document, doc) for doc in documents]
        for future in tqdm(futures, desc=f"Thread pool ({max_workers} workers)"):
            results.append(future.result())
    return results

def process_pool_processing(documents: List[Document], max_workers: int) -> List[Dict[str, Any]]:
    """
    Process documents using a process pool.
    
    Args:
        documents: List of documents to process
        max_workers: Maximum number of worker processes
        
    Returns:
        List of processed document data
    """
    # Convert documents to dictionaries for serialization
    doc_dicts = [doc.to_dict() for doc in documents]
    
    # Define a wrapper function that recreates Document objects
    def process_doc_dict(doc_dict):
        doc = Document(
            doc_id=doc_dict["id"],
            size_kb=doc_dict["size_kb"],
            complexity=doc_dict["complexity"]
        )
        doc.fields = doc_dict["fields"]
        doc.metadata = doc_dict["metadata"]
        return process_document(doc)
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_doc_dict, doc_dict) for doc_dict in doc_dicts]
        for future in tqdm(futures, desc=f"Process pool ({max_workers} workers)"):
            results.append(future.result())
    return results

async def process_document_async(doc: Document) -> Dict[str, Any]:
    """
    Process a document asynchronously.
    
    Args:
        doc: Document to process
        
    Returns:
        Processed document data
    """
    # For CPU-bound tasks, run in a thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, process_document, doc)
    return result

async def async_processing(documents: List[Document], max_concurrency: int) -> List[Dict[str, Any]]:
    """
    Process documents using async/await with concurrency limit.
    
    Args:
        documents: List of documents to process
        max_concurrency: Maximum number of concurrent tasks
        
    Returns:
        List of processed document data
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_with_semaphore(doc):
        async with semaphore:
            return await process_document_async(doc)
    
    tasks = [process_with_semaphore(doc) for doc in documents]
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Async processing ({max_concurrency} concurrent)"):
        result = await task
        results.append(result)
    
    return results

def task_partitioning_processing(documents: List[Document], num_partitions: int, max_workers: int) -> List[Dict[str, Any]]:
    """
    Process documents using task partitioning with process pool.
    
    Args:
        documents: List of documents to process
        num_partitions: Number of partitions to split the documents into
        max_workers: Maximum number of worker processes
        
    Returns:
        List of processed document data
    """
    # Partition documents by size to balance workload
    documents.sort(key=lambda doc: doc.size_kb, reverse=True)
    
    partitions = [[] for _ in range(num_partitions)]
    for i, doc in enumerate(documents):
        partitions[i % num_partitions].append(doc)
    
    # Process each partition in a separate process
    def process_partition(partition):
        return [process_document(doc) for doc in partition]
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_partition, partition) for partition in partitions]
        for future in tqdm(futures, desc=f"Task partitioning ({num_partitions} partitions, {max_workers} workers)"):
            results.extend(future.result())
    
    return results

def multiprocessing_queue_processing(documents: List[Document], num_processes: int) -> List[Dict[str, Any]]:
    """
    Process documents using multiprocessing with queues.
    
    Args:
        documents: List of documents to process
        num_processes: Number of worker processes
        
    Returns:
        List of processed document data
    """
    # Convert documents to dictionaries for serialization
    doc_dicts = [doc.to_dict() for doc in documents]
    
    # Create queues
    task_queue = Queue()
    result_queue = Queue()
    
    # Define worker process function
    def worker(task_q, result_q):
        while True:
            try:
                # Get a task from the queue
                doc_dict = task_q.get()
                if doc_dict is None:  # Sentinel value to stop
                    break
                
                # Recreate document and process it
                doc = Document(
                    doc_id=doc_dict["id"],
                    size_kb=doc_dict["size_kb"],
                    complexity=doc_dict["complexity"]
                )
                doc.fields = doc_dict["fields"]
                doc.metadata = doc_dict["metadata"]
                
                # Process document and put result in result queue
                result = process_document(doc)
                result_q.put(result)
            except Exception as e:
                # Log error and continue
                print(f"Error in worker: {e}")
                traceback.print_exc()
    
    # Start worker processes
    processes = []
    for _ in range(num_processes):
        p = Process(target=worker, args=(task_queue, result_queue))
        p.daemon = True
        p.start()
        processes.append(p)
    
    # Put tasks in the queue
    for doc_dict in doc_dicts:
        task_queue.put(doc_dict)
    
    # Add sentinel values to stop workers
    for _ in range(num_processes):
        task_queue.put(None)
    
    # Collect results
    results = []
    with tqdm(total=len(documents), desc=f"Multiprocessing queue ({num_processes} processes)") as pbar:
        while len(results) < len(documents):
            result = result_queue.get()
            results.append(result)
            pbar.update(1)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    return results

# Benchmark functions
def measure_system_metrics() -> Dict[str, float]:
    """Measure current system metrics."""
    process = psutil.Process(os.getpid())
    
    return {
        "memory_mb": process.memory_info().rss / (1024 * 1024),  # Convert to MB
        "cpu_percent": process.cpu_percent(),
        "num_threads": process.num_threads(),
        "num_ctx_switches": sum(process.num_ctx_switches()),
    }

def run_benchmark(
    strategy_name: str,
    strategy_func: Callable,
    documents: List[Document],
    **kwargs
) -> Dict[str, Any]:
    """
    Run a benchmark for a specific parallelization strategy.
    
    Args:
        strategy_name: Name of the strategy
        strategy_func: Function implementing the strategy
        documents: List of documents to process
        **kwargs: Additional arguments for the strategy function
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking {strategy_name} with {len(documents)} documents...")
    
    # Measure initial system metrics
    initial_metrics = measure_system_metrics()
    
    # Run the strategy and measure time
    start_time = time.time()
    
    if strategy_name == "async":
        # Special handling for async strategy
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(strategy_func(documents, **kwargs))
        finally:
            loop.close()
    else:
        # Regular function call
        results = strategy_func(documents, **kwargs)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Measure final system metrics
    final_metrics = measure_system_metrics()
    
    # Calculate metrics
    num_documents = len(documents)
    total_size_mb = sum(doc.size_kb for doc in documents) / 1024  # Convert to MB
    throughput = num_documents / processing_time if processing_time > 0 else 0
    throughput_mb = total_size_mb / processing_time if processing_time > 0 else 0
    
    # Calculate metric differences
    memory_usage = final_metrics["memory_mb"] - initial_metrics["memory_mb"]
    cpu_usage = final_metrics["cpu_percent"]
    
    # Combine results
    benchmark_results = {
        "strategy": strategy_name,
        "num_documents": num_documents,
        "total_size_mb": total_size_mb,
        "processing_time_s": processing_time,
        "throughput_docs_per_sec": throughput,
        "throughput_mb_per_sec": throughput_mb,
        "memory_usage_mb": memory_usage,
        "cpu_percent": cpu_usage,
        "num_threads": final_metrics["num_threads"],
        "num_ctx_switches": final_metrics["num_ctx_switches"],
        "strategy_params": kwargs
    }
    
    return benchmark_results

def generate_test_documents(
    num_documents: int,
    size_range: Tuple[int, int] = (10, 500),
    complexity_range: Tuple[float, float] = (0.5, 2.0)
) -> List[Document]:
    """
    Generate test documents for benchmarking.
    
    Args:
        num_documents: Number of documents to generate
        size_range: Range of document sizes in KB (min, max)
        complexity_range: Range of processing complexity (min, max)
        
    Returns:
        List of generated documents
    """
    documents = []
    for i in tqdm(range(num_documents), desc="Generating test documents"):
        size_kb = random.randint(size_range[0], size_range[1])
        complexity = random.uniform(complexity_range[0], complexity_range[1])
        doc = Document(doc_id=f"doc_{i}", size_kb=size_kb, complexity=complexity)
        documents.append(doc)
    
    return documents

def run_benchmarks(
    num_documents: int = 100,
    document_sizes: Tuple[int, int] = (10, 500),
    worker_counts: List[int] = [1, 2, 4, 8, 16],
    repeat: int = 3
) -> pd.DataFrame:
    """
    Run all parallelization benchmarks.
    
    Args:
        num_documents: Number of documents to process in each benchmark
        document_sizes: Range of document sizes in KB (min, max)
        worker_counts: List of worker counts to test
        repeat: Number of times to repeat each benchmark
        
    Returns:
        DataFrame with benchmark results
    """
    # Generate test documents
    documents = generate_test_documents(
        num_documents=num_documents,
        size_range=document_sizes
    )
    
    # Define strategies to benchmark
    strategies = [
        {
            "name": "sequential",
            "func": sequential_processing,
            "params": {}
        },
    ]
    
    # Add thread pool strategies
    for worker_count in worker_counts:
        strategies.append({
            "name": "thread_pool",
            "func": thread_pool_processing,
            "params": {"max_workers": worker_count}
        })
    
    # Add process pool strategies
    for worker_count in worker_counts:
        strategies.append({
            "name": "process_pool",
            "func": process_pool_processing,
            "params": {"max_workers": worker_count}
        })
    
    # Add async strategies
    for worker_count in worker_counts:
        strategies.append({
            "name": "async",
            "func": async_processing,
            "params": {"max_concurrency": worker_count}
        })
    
    # Add task partitioning strategies
    for worker_count in worker_counts:
        strategies.append({
            "name": "task_partitioning",
            "func": task_partitioning_processing,
            "params": {"num_partitions": worker_count * 2, "max_workers": worker_count}
        })
    
    # Add multiprocessing queue strategies
    for worker_count in worker_counts:
        strategies.append({
            "name": "multiprocessing_queue",
            "func": multiprocessing_queue_processing,
            "params": {"num_processes": worker_count}
        })
    
    # Run benchmarks
    all_results = []
    
    for strategy in strategies:
        for r in range(repeat):
            logger.info(f"Running {strategy['name']} (repeat {r+1}/{repeat})...")
            
            try:
                # Create a copy of documents to ensure fair comparison
                docs_copy = documents.copy()
                
                # Run benchmark
                result = run_benchmark(
                    strategy_name=strategy["name"],
                    strategy_func=strategy["func"],
                    documents=docs_copy,
                    **strategy["params"]
                )
                
                # Add repeat number
                result["repeat"] = r + 1
                
                # Add to results
                all_results.append(result)
                
                # Log result
                logger.info(f"  Throughput: {result['throughput_docs_per_sec']:.2f} docs/sec")
                
            except Exception as e:
                logger.error(f"Error benchmarking {strategy['name']}: {e}")
                traceback.print_exc()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_df.to_csv("results/parallelization_benchmark_results.csv", index=False)
    
    return results_df

def visualize_results(results_df: pd.DataFrame) -> None:
    """
    Create visualizations from benchmark results.
    
    Args:
        results_df: DataFrame with benchmark results
    """
    os.makedirs("results/plots", exist_ok=True)
    
    # Calculate mean values across repeats
    mean_results = results_df.groupby(
        ['strategy', 'strategy_params']
    ).agg({
        'throughput_docs_per_sec': ['mean', 'std'],
        'throughput_mb_per_sec': ['mean', 'std'],
        'processing_time_s': ['mean', 'std'],
        'memory_usage_mb': ['mean', 'std'],
        'cpu_percent': ['mean', 'std']
    }).reset_index()
    
    # Flatten multi-level columns
    mean_results.columns = ['_'.join(col).strip('_') for col in mean_results.columns.values]
    
    # Extract worker count for each strategy
    def extract_worker_count(row):
        if row['strategy'] == 'sequential':
            return 1
        elif row['strategy'] == 'thread_pool':
            return row['strategy_params'].get('max_workers', 1)
        elif row['strategy'] == 'process_pool':
            return row['strategy_params'].get('max_workers', 1)
        elif row['strategy'] == 'async':
            return row['strategy_params'].get('max_concurrency', 1)
        elif row['strategy'] == 'task_partitioning':
            return row['strategy_params'].get('max_workers', 1)
        elif row['strategy'] == 'multiprocessing_queue':
            return row['strategy_params'].get('num_processes', 1)
        else:
            return 1
    
    mean_results['worker_count'] = mean_results.apply(extract_worker_count, axis=1)
    
    # 1. Throughput comparison by strategy
    plt.figure(figsize=(12, 8))
    
    # Group by strategy and get the best throughput for each
    best_throughput = mean_results.groupby('strategy')['throughput_docs_per_sec_mean'].max().reset_index()
    best_throughput = best_throughput.sort_values('throughput_docs_per_sec_mean')
    
    plt.barh(best_throughput['strategy'], best_throughput['throughput_docs_per_sec_mean'])
    plt.xlabel('Documents per Second')
    plt.ylabel('Strategy')
    plt.title('Best Throughput by Parallelization Strategy')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig("results/plots/best_throughput_by_strategy.png", dpi=300)
    plt.close()
    
    # 2. Scaling efficiency by worker count
    plt.figure(figsize=(12, 8))
    
    # Filter out sequential strategy
    parallel_results = mean_results[mean_results['strategy'] != 'sequential']
    
    # Get sequential throughput as baseline
    sequential_throughput = mean_results[mean_results['strategy'] == 'sequential']['throughput_docs_per_sec_mean'].values[0]
    
    # Calculate speedup relative to sequential
    parallel_results['speedup'] = parallel_results['throughput_docs_per_sec_mean'] / sequential_throughput
    
    # Calculate efficiency (speedup / worker_count)
    parallel_results['efficiency'] = parallel_results['speedup'] / parallel_results['worker_count']
    
    # Plot efficiency vs worker count for each strategy
    for strategy in parallel_results['strategy'].unique():
        strategy_results = parallel_results[parallel_results['strategy'] == strategy]
        plt.plot(strategy_results['worker_count'], strategy_results['efficiency'], 'o-', label=strategy)
    
    plt.xlabel('Worker Count')
    plt.ylabel('Efficiency (Speedup / Worker Count)')
    plt.title('Scaling Efficiency by Worker Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/scaling_efficiency.png", dpi=300)
    plt.close()
    
    # 3. Throughput vs worker count
    plt.figure(figsize=(12, 8))
    
    for strategy in parallel_results['strategy'].unique():
        strategy_results = parallel_results[parallel_results['strategy'] == strategy]
        plt.plot(strategy_results['worker_count'], strategy_results['throughput_docs_per_sec_mean'], 'o-', label=strategy)
    
    # Add sequential as horizontal line
    plt.axhline(y=sequential_throughput, color='r', linestyle='--', label='Sequential')
    
    plt.xlabel('Worker Count')
    plt.ylabel('Documents per Second')
    plt.title('Throughput vs Worker Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/throughput_vs_workers.png", dpi=300)
    plt.close()
    
    # 4. Memory usage vs throughput
    plt.figure(figsize=(12, 8))
    
    plt.scatter(
        mean_results['throughput_docs_per_sec_mean'],
        mean_results['memory_usage_mb_mean'],
        s=100,
        alpha=0.7
    )
    
    # Add labels for each point
    for i, row in mean_results.iterrows():
        label = f"{row['strategy']} ({row['worker_count']})"
        plt.annotate(
            label,
            (row['throughput_docs_per_sec_mean'], row['memory_usage_mb_mean']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    plt.xlabel('Documents per Second')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Throughput')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/memory_vs_throughput.png", dpi=300)
    plt.close()
    
    # 5. CPU usage vs throughput
    plt.figure(figsize=(12, 8))
    
    plt.scatter(
        mean_results['throughput_docs_per_sec_mean'],
        mean_results['cpu_percent_mean'],
        s=100,
        alpha=0.7
    )
    
    # Add labels for each point
    for i, row in mean_results.iterrows():
        label = f"{row['strategy']} ({row['worker_count']})"
        plt.annotate(
            label,
            (row['throughput_docs_per_sec_mean'], row['cpu_percent_mean']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    plt.xlabel('Documents per Second')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage vs Throughput')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/cpu_vs_throughput.png", dpi=300)
    plt.close()
    
    # 6. Summary table
    summary = mean_results.pivot_table(
        index='strategy',
        columns='worker_count',
        values='throughput_docs_per_sec_mean'
    )
    
    summary.to_csv("results/plots/throughput_summary.csv")
    
    # Create a visual summary table
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    
    # Format the data for display
    cell_text = []
    for strategy in summary.index:
        row = [strategy]
        for worker in summary.columns:
            value = summary.loc[strategy, worker]
            row.append(f"{value:.2f}")
        cell_text.append(row)
    
    col_labels = ['Strategy'] + [f"{w} Workers" for w in summary.columns]
    
    table = plt.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Throughput (docs/sec) by Strategy and Worker Count', y=1.08)
    plt.tight_layout()
    plt.savefig("results/plots/throughput_summary_table.png", dpi=300)
    plt.close()

def main():
    """Main function to run all benchmarks and visualizations."""
    logger.info("Starting parallelization benchmark...")
    
    # Run benchmarks
    results_df = run_benchmarks(
        num_documents=100,  # Adjust based on your system's capabilities
        document_sizes=(10, 500),
        worker_counts=[1, 2, 4, 8],  # Adjust based on your CPU cores
        repeat=3
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    visualize_results(results_df)
    
    logger.info("Benchmark completed. Results saved to 'results/' directory.")

if __name__ == "__main__":
    main() 