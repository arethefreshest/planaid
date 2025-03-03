#!/usr/bin/env python3
"""
Cache Strategy Benchmark

This script benchmarks different caching strategies for document processing:
- Redis
- In-memory (dictionary)
- File-based (SQLite)
- Filesystem (pickle)

It measures performance metrics like latency, throughput, and memory usage.
"""

import time
import os
import json
import pickle
import hashlib
import sqlite3
import redis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import random
import string
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
import tempfile
import shutil

# Create results directory
os.makedirs("results", exist_ok=True)

# Utility functions
def generate_random_document(size_kb: int) -> Dict[str, Any]:
    """Generate a random document with specified size in KB."""
    # Create a document with random field identifiers and text
    doc = {
        "id": f"doc_{random.randint(1, 10000)}",
        "timestamp": time.time(),
        "fields": [],
        "text": "",
        "metadata": {
            "source": random.choice(["plankart", "bestemmelser", "sosi"]),
            "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}",
            "processed_by": f"processor_{random.randint(1, 10)}"
        }
    }
    
    # Add random fields
    num_fields = random.randint(10, 50)
    prefixes = ['§', 'pkt.', 'kap.', 'art.', 'BRA', 'BYA', 'BFA', 'f_', 'o_', 'p_', 'b_']
    
    for _ in range(num_fields):
        prefix = random.choice(prefixes)
        number = f"{random.randint(1, 50)}"
        if random.random() < 0.5:
            number += f".{random.randint(1, 20)}"
        field = f"{prefix}{number}"
        doc["fields"].append(field)
    
    # Add random text to reach the desired size
    current_size = len(json.dumps(doc).encode('utf-8'))
    target_size = size_kb * 1024
    
    if current_size < target_size:
        # Generate random text to fill the remaining size
        chars_needed = (target_size - current_size) // 2  # Approximation for UTF-8 encoding
        random_text = ''.join(random.choices(string.ascii_letters + string.digits + ' \n\t.,;:!?-_', k=chars_needed))
        doc["text"] = random_text
    
    return doc

def get_document_key(doc: Dict[str, Any]) -> str:
    """Generate a unique key for a document."""
    doc_id = doc.get("id", "")
    source = doc.get("metadata", {}).get("source", "")
    version = doc.get("metadata", {}).get("version", "")
    key_str = f"{doc_id}:{source}:{version}"
    return hashlib.md5(key_str.encode()).hexdigest()

# Cache implementations
class BaseCache:
    """Base class for all cache implementations."""
    
    def __init__(self, name: str):
        self.name = name
        self.hit_count = 0
        self.miss_count = 0
        self.set_count = 0
        self.start_time = time.time()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an item from the cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set an item in the cache."""
        raise NotImplementedError
    
    def clear(self) -> None:
        """Clear the cache."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0
        elapsed_time = time.time() - self.start_time
        
        return {
            "name": self.name,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "sets": self.set_count,
            "hit_ratio": hit_ratio,
            "elapsed_time": elapsed_time,
            "operations_per_second": (total_requests + self.set_count) / elapsed_time if elapsed_time > 0 else 0
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.hit_count = 0
        self.miss_count = 0
        self.set_count = 0
        self.start_time = time.time()

class InMemoryCache(BaseCache):
    """Simple in-memory dictionary cache."""
    
    def __init__(self, max_size: Optional[int] = None):
        super().__init__("InMemory")
        self.cache: Dict[str, Tuple[Dict[str, Any], Optional[float]]] = {}
        self.max_size = max_size
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if expiry is None or expiry > time.time():
                    self.hit_count += 1
                    return value
                else:
                    # Expired
                    del self.cache[key]
            
            self.miss_count += 1
            return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        with self.lock:
            expiry = time.time() + ttl if ttl is not None else None
            
            # If max_size is reached, remove oldest item
            if self.max_size is not None and len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[key] = (value, expiry)
            self.set_count += 1
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()

class RedisCache(BaseCache):
    """Redis-based cache."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, password: Optional[str] = None):
        super().__init__("Redis")
        self.redis = redis.Redis(host=host, port=port, db=db, password=password)
        self.prefix = "doc:"
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        full_key = f"{self.prefix}{key}"
        result = self.redis.get(full_key)
        
        if result:
            self.hit_count += 1
            return json.loads(result)
        else:
            self.miss_count += 1
            return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        full_key = f"{self.prefix}{key}"
        self.redis.set(full_key, json.dumps(value), ex=ttl)
        self.set_count += 1
    
    def clear(self) -> None:
        # Clear only keys with our prefix
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, f"{self.prefix}*", 100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break

class SQLiteCache(BaseCache):
    """SQLite-based cache."""
    
    def __init__(self, db_path: str = ':memory:'):
        super().__init__("SQLite")
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.RLock()
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expiry REAL NULL
            )
            ''')
            self.conn.commit()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT value, expiry FROM cache WHERE key = ?",
                (key,)
            )
            result = cursor.fetchone()
            
            if result:
                value_str, expiry = result
                if expiry is None or expiry > time.time():
                    self.hit_count += 1
                    return json.loads(value_str)
                else:
                    # Expired, remove it
                    cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                    self.conn.commit()
            
            self.miss_count += 1
            return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        with self.lock:
            expiry = time.time() + ttl if ttl is not None else None
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO cache (key, value, expiry) VALUES (?, ?, ?)",
                (key, json.dumps(value), expiry)
            )
            self.conn.commit()
            self.set_count += 1
    
    def clear(self) -> None:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM cache")
            self.conn.commit()

class FileSystemCache(BaseCache):
    """File system based cache using pickle."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__("FileSystem")
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "doc_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock = threading.RLock()
        
        # Create metadata file to track expiry
        self.metadata_path = os.path.join(self.cache_dir, "metadata.json")
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _get_file_path(self, key: str) -> str:
        """Get the file path for a key."""
        return os.path.join(self.cache_dir, f"{key}.pickle")
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            file_path = self._get_file_path(key)
            
            # Check if file exists and not expired
            if os.path.exists(file_path) and key in self.metadata:
                expiry = self.metadata[key].get("expiry")
                if expiry is None or expiry > time.time():
                    try:
                        with open(file_path, 'rb') as f:
                            self.hit_count += 1
                            return pickle.load(f)
                    except (pickle.PickleError, EOFError):
                        # Corrupted file
                        os.remove(file_path)
                else:
                    # Expired
                    os.remove(file_path)
                    del self.metadata[key]
                    self._save_metadata()
            
            self.miss_count += 1
            return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        with self.lock:
            file_path = self._get_file_path(key)
            expiry = time.time() + ttl if ttl is not None else None
            
            # Save value to file
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            self.metadata[key] = {
                "created": time.time(),
                "expiry": expiry
            }
            self._save_metadata()
            self.set_count += 1
    
    def clear(self) -> None:
        with self.lock:
            # Remove all cache files
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Reset metadata
            self.metadata = {}
            self._save_metadata()

# Benchmark functions
def measure_memory_usage() -> float:
    """Measure current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def benchmark_cache(
    cache: BaseCache,
    num_docs: int,
    doc_sizes: List[int],
    read_write_ratio: float = 0.8,
    ttl: Optional[int] = None,
    threads: int = 1
) -> Dict[str, Any]:
    """
    Benchmark a cache implementation.
    
    Args:
        cache: The cache implementation to benchmark
        num_docs: Number of documents to use
        doc_sizes: List of document sizes in KB
        read_write_ratio: Ratio of reads to total operations
        ttl: Time-to-live for cache entries in seconds
        threads: Number of threads to use
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking {cache.name} cache with {num_docs} documents, {threads} threads...")
    
    # Generate documents
    print("Generating test documents...")
    documents = []
    for _ in tqdm(range(num_docs)):
        size = random.choice(doc_sizes)
        doc = generate_random_document(size)
        documents.append(doc)
    
    # Generate document keys
    doc_keys = [get_document_key(doc) for doc in documents]
    
    # Reset cache and stats
    cache.clear()
    cache.reset_stats()
    
    # Measure initial memory
    initial_memory = measure_memory_usage()
    
    # Populate cache with some documents
    print("Populating cache...")
    initial_docs = int(num_docs * 0.5)  # Populate with 50% of documents
    for i in tqdm(range(initial_docs)):
        key = doc_keys[i]
        doc = documents[i]
        cache.set(key, doc, ttl)
    
    # Measure memory after population
    populated_memory = measure_memory_usage()
    
    # Prepare operations
    num_operations = num_docs * 5  # Perform multiple operations per document
    num_reads = int(num_operations * read_write_ratio)
    num_writes = num_operations - num_reads
    
    read_keys = random.choices(doc_keys, k=num_reads)
    write_indices = random.choices(range(num_docs), k=num_writes)
    
    # Function for thread execution
    def perform_operations(thread_id, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            if i < num_reads:
                # Read operation
                key = read_keys[i]
                cache.get(key)
            else:
                # Write operation
                idx = write_indices[i - num_reads]
                key = doc_keys[idx]
                doc = documents[idx]
                cache.set(key, doc, ttl)
    
    # Measure operation time
    print("Performing operations...")
    start_time = time.time()
    
    if threads > 1:
        # Multi-threaded execution
        ops_per_thread = num_operations // threads
        futures = []
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for t in range(threads):
                start_idx = t * ops_per_thread
                end_idx = start_idx + ops_per_thread if t < threads - 1 else num_operations
                futures.append(executor.submit(perform_operations, t, start_idx, end_idx))
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
    else:
        # Single-threaded execution
        perform_operations(0, 0, num_operations)
    
    end_time = time.time()
    operation_time = end_time - start_time
    
    # Measure final memory
    final_memory = measure_memory_usage()
    
    # Get cache stats
    cache_stats = cache.get_stats()
    
    # Calculate metrics
    throughput = num_operations / operation_time if operation_time > 0 else 0
    latency_ms = (operation_time / num_operations) * 1000 if num_operations > 0 else 0
    memory_usage = final_memory - initial_memory
    
    # Combine results
    results = {
        "cache_name": cache.name,
        "num_documents": num_docs,
        "document_sizes_kb": doc_sizes,
        "read_write_ratio": read_write_ratio,
        "threads": threads,
        "ttl": ttl,
        "initial_memory_mb": initial_memory,
        "populated_memory_mb": populated_memory,
        "final_memory_mb": final_memory,
        "memory_usage_mb": memory_usage,
        "operation_time_s": operation_time,
        "throughput_ops_per_sec": throughput,
        "latency_ms": latency_ms,
        "hit_ratio": cache_stats["hit_ratio"],
        "hits": cache_stats["hits"],
        "misses": cache_stats["misses"],
        "sets": cache_stats["sets"]
    }
    
    return results

def run_benchmarks():
    """Run all cache benchmarks."""
    # Cache implementations to test
    caches = [
        InMemoryCache(),
        SQLiteCache(),
        FileSystemCache(),
    ]
    
    # Try to connect to Redis if available
    try:
        redis_cache = RedisCache(host='redis', port=6379)
        redis_cache.redis.ping()  # Test connection
        caches.append(redis_cache)
        print("Redis connection successful, including in benchmarks")
    except (redis.exceptions.ConnectionError, redis.exceptions.ResponseError):
        print("Redis not available, skipping Redis benchmarks")
    
    # Benchmark parameters
    doc_sizes = [10, 50, 100, 500]  # Document sizes in KB
    thread_counts = [1, 2, 4, 8]
    ttl_values = [None, 60, 300]  # None means no expiration
    read_write_ratios = [0.5, 0.8, 0.95]
    
    # Results storage
    all_results = []
    
    # Run benchmarks with different parameters
    for cache in caches:
        for threads in thread_counts:
            for ttl in ttl_values:
                for ratio in read_write_ratios:
                    # Adjust document count based on cache type to keep benchmark duration reasonable
                    if cache.name == "Redis":
                        num_docs = 1000
                    elif cache.name == "FileSystem":
                        num_docs = 500
                    else:
                        num_docs = 2000
                    
                    results = benchmark_cache(
                        cache=cache,
                        num_docs=num_docs,
                        doc_sizes=doc_sizes,
                        read_write_ratio=ratio,
                        ttl=ttl,
                        threads=threads
                    )
                    
                    all_results.append(results)
                    
                    # Clear cache after each benchmark
                    cache.clear()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_df.to_csv("results/cache_benchmark_results.csv", index=False)
    
    return results_df

def visualize_results(results_df):
    """Create visualizations from benchmark results."""
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. Throughput comparison
    plt.figure(figsize=(12, 8))
    throughput_by_cache = results_df.groupby('cache_name')['throughput_ops_per_sec'].mean().sort_values()
    throughput_by_cache.plot(kind='barh')
    plt.title('Average Throughput by Cache Type')
    plt.xlabel('Operations per Second')
    plt.ylabel('Cache Type')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig("results/plots/throughput_by_cache.png", dpi=300)
    plt.close()
    
    # 2. Latency comparison
    plt.figure(figsize=(12, 8))
    latency_by_cache = results_df.groupby('cache_name')['latency_ms'].mean().sort_values()
    latency_by_cache.plot(kind='barh')
    plt.title('Average Latency by Cache Type')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Cache Type')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig("results/plots/latency_by_cache.png", dpi=300)
    plt.close()
    
    # 3. Memory usage comparison
    plt.figure(figsize=(12, 8))
    memory_by_cache = results_df.groupby('cache_name')['memory_usage_mb'].mean().sort_values()
    memory_by_cache.plot(kind='barh')
    plt.title('Average Memory Usage by Cache Type')
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Cache Type')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig("results/plots/memory_by_cache.png", dpi=300)
    plt.close()
    
    # 4. Hit ratio comparison
    plt.figure(figsize=(12, 8))
    hit_ratio_by_cache = results_df.groupby('cache_name')['hit_ratio'].mean().sort_values()
    hit_ratio_by_cache.plot(kind='barh')
    plt.title('Average Hit Ratio by Cache Type')
    plt.xlabel('Hit Ratio')
    plt.ylabel('Cache Type')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig("results/plots/hit_ratio_by_cache.png", dpi=300)
    plt.close()
    
    # 5. Throughput vs Threads
    plt.figure(figsize=(12, 8))
    throughput_by_threads = results_df.groupby(['cache_name', 'threads'])['throughput_ops_per_sec'].mean().unstack()
    throughput_by_threads.plot(marker='o')
    plt.title('Throughput vs Thread Count')
    plt.xlabel('Thread Count')
    plt.ylabel('Operations per Second')
    plt.grid(True)
    plt.legend(title='Cache Type')
    plt.tight_layout()
    plt.savefig("results/plots/throughput_vs_threads.png", dpi=300)
    plt.close()
    
    # 6. Throughput vs Read/Write Ratio
    plt.figure(figsize=(12, 8))
    throughput_by_ratio = results_df.groupby(['cache_name', 'read_write_ratio'])['throughput_ops_per_sec'].mean().unstack()
    throughput_by_ratio.plot(marker='o')
    plt.title('Throughput vs Read/Write Ratio')
    plt.xlabel('Read/Write Ratio')
    plt.ylabel('Operations per Second')
    plt.grid(True)
    plt.legend(title='Cache Type')
    plt.tight_layout()
    plt.savefig("results/plots/throughput_vs_ratio.png", dpi=300)
    plt.close()
    
    # 7. Throughput with and without TTL
    plt.figure(figsize=(12, 8))
    # Create a new column for TTL category
    results_df['ttl_category'] = results_df['ttl'].apply(lambda x: 'No TTL' if x is None else f'TTL {x}s')
    throughput_by_ttl = results_df.groupby(['cache_name', 'ttl_category'])['throughput_ops_per_sec'].mean().unstack()
    throughput_by_ttl.plot(kind='bar')
    plt.title('Throughput with Different TTL Settings')
    plt.xlabel('Cache Type')
    plt.ylabel('Operations per Second')
    plt.grid(axis='y')
    plt.legend(title='TTL Setting')
    plt.tight_layout()
    plt.savefig("results/plots/throughput_by_ttl.png", dpi=300)
    plt.close()
    
    # 8. Summary table
    summary = results_df.groupby('cache_name').agg({
        'throughput_ops_per_sec': 'mean',
        'latency_ms': 'mean',
        'memory_usage_mb': 'mean',
        'hit_ratio': 'mean'
    }).sort_values('throughput_ops_per_sec', ascending=False)
    
    summary.columns = ['Avg Throughput (ops/s)', 'Avg Latency (ms)', 'Avg Memory Usage (MB)', 'Avg Hit Ratio']
    summary.to_csv("results/plots/summary_table.csv")
    
    # Create a visual summary table
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    table = plt.table(
        cellText=summary.round(2).values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Cache Performance Summary', y=1.08)
    plt.tight_layout()
    plt.savefig("results/plots/summary_table.png", dpi=300)
    plt.close()

def main():
    """Main function to run all benchmarks and visualizations."""
    print("Starting cache benchmark...")
    
    # Run benchmarks
    results_df = run_benchmarks()
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_results(results_df)
    
    print("Benchmark completed. Results saved to 'results/' directory.")

if __name__ == "__main__":
    main() 