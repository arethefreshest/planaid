"""
Data Structure Optimization Experiments

This script tests different data structures for field storage and lookup:
1. B-tree vs. Hash table
2. Compressed tries
3. Bitmap indices
4. Cache-friendly structures

The goal is to find optimal data structures for the project's needs.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import psutil
from memory_profiler import profile
from sortedcontainers import SortedDict, SortedSet
from bintrees import AVLTree
from bitarray import bitarray
from llist import dllist
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataStructureResult:
    """Stores results of a data structure test."""
    structure: str
    insertion_time: float
    lookup_time: float
    memory_usage: float
    cache_misses: int
    metrics: Dict[str, float]

class CompressedTrie:
    """Memory-efficient trie implementation."""
    class Node:
        def __init__(self):
            self.children = {}
            self.is_end = False
            self.value = None
            self.shared_prefix = ""
    
    def __init__(self):
        self.root = self.Node()
    
    def insert(self, key: str, value: Optional[any] = None):
        node = self.root
        
        while key:
            # Find longest common prefix with existing child
            longest_prefix = ""
            matching_child = None
            
            for child_key in node.children:
                common_prefix = ""
                for i in range(min(len(key), len(child_key))):
                    if key[i] != child_key[i]:
                        break
                    common_prefix += key[i]
                if len(common_prefix) > len(longest_prefix):
                    longest_prefix = common_prefix
                    matching_child = child_key
            
            if not longest_prefix:
                # No matching prefix, create new node
                new_node = self.Node()
                new_node.shared_prefix = key
                new_node.is_end = True
                new_node.value = value
                node.children[key] = new_node
                break
            
            if len(longest_prefix) < len(matching_child):
                # Split existing node
                old_node = node.children[matching_child]
                new_node = self.Node()
                new_node.shared_prefix = longest_prefix
                
                old_node.shared_prefix = matching_child[len(longest_prefix):]
                new_node.children[old_node.shared_prefix] = old_node
                
                node.children[longest_prefix] = new_node
                del node.children[matching_child]
                
                node = new_node
            else:
                node = node.children[matching_child]
            
            key = key[len(longest_prefix):]
    
    def search(self, key: str) -> Tuple[bool, Optional[any]]:
        node = self.root
        
        while key and node:
            found = False
            for child_key, child in node.children.items():
                if key.startswith(child.shared_prefix):
                    key = key[len(child.shared_prefix):]
                    node = child
                    found = True
                    break
            if not found:
                return False, None
        
        return node.is_end, node.value

class BitmapIndex:
    """Bitmap index for field attributes."""
    def __init__(self, num_fields: int):
        self.num_fields = num_fields
        self.prefix_map = {
            'o_': bitarray([0] * num_fields),
            'f_': bitarray([0] * num_fields),
            'none': bitarray([0] * num_fields)
        }
        self.has_number = bitarray([0] * num_fields)
    
    def add_field(self, index: int, field: str):
        if field.startswith('o_'):
            self.prefix_map['o_'][index] = 1
        elif field.startswith('f_'):
            self.prefix_map['f_'][index] = 1
        else:
            self.prefix_map['none'][index] = 1
        
        if any(c.isdigit() for c in field):
            self.has_number[index] = 1
    
    def query_prefix(self, prefix: str) -> bitarray:
        return self.prefix_map.get(prefix, bitarray([0] * self.num_fields))
    
    def query_with_number(self) -> bitarray:
        return self.has_number
    
    def combine_queries(self, *arrays: bitarray) -> bitarray:
        result = arrays[0].copy()
        for arr in arrays[1:]:
            result &= arr
        return result

class CacheFriendlyArray:
    """Cache-friendly array implementation."""
    def __init__(self, block_size: int = 64):
        self.block_size = block_size
        self.blocks = [[]]
        self.size = 0
    
    def append(self, item: any):
        if len(self.blocks[-1]) >= self.block_size:
            self.blocks.append([])
        self.blocks[-1].append(item)
        self.size += 1
    
    def get(self, index: int) -> any:
        block_index = index // self.block_size
        item_index = index % self.block_size
        return self.blocks[block_index][item_index]
    
    def __len__(self):
        return self.size

class DataStructureBenchmark:
    """Tests different data structure implementations."""
    
    def __init__(self, test_dir: Path):
        """Initialize benchmark with test directory."""
        self.test_dir = test_dir
        self.test_dir.mkdir(exist_ok=True)
        
        # Generate test data
        self.test_fields = [
            f"{prefix}{base}{num}" for prefix in ['', 'o_', 'f_']
            for base in ['BRA', 'SV', 'BE', 'GF']
            for num in range(1, 101)
        ]
        
        # Add special fields
        self.test_fields.extend([
            f"H{num}" for num in range(100, 400)
        ])
        self.test_fields.extend([
            f"#{num:02d} SNØ" for num in range(1, 21)
        ])
    
    @profile
    def test_btree(self) -> DataStructureResult:
        """Test B-tree implementation."""
        start_time = time.time()
        
        # Insert data
        tree = AVLTree()
        for i, field in enumerate(self.test_fields):
            tree[field] = i
        
        insertion_time = time.time() - start_time
        
        # Test lookups
        lookup_start = time.time()
        found = 0
        for field in self.test_fields[::10]:  # Test every 10th field
            if field in tree:
                found += 1
        
        lookup_time = time.time() - lookup_start
        
        # Get memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        return DataStructureResult(
            structure='btree',
            insertion_time=insertion_time,
            lookup_time=lookup_time,
            memory_usage=memory_usage,
            cache_misses=0,  # Would need hardware counters
            metrics={
                'size': len(tree),
                'found_ratio': found / (len(self.test_fields) // 10)
            }
        )
    
    @profile
    def test_compressed_trie(self) -> DataStructureResult:
        """Test compressed trie implementation."""
        start_time = time.time()
        
        # Insert data
        trie = CompressedTrie()
        for i, field in enumerate(self.test_fields):
            trie.insert(field, i)
        
        insertion_time = time.time() - start_time
        
        # Test lookups
        lookup_start = time.time()
        found = 0
        for field in self.test_fields[::10]:
            exists, _ = trie.search(field)
            if exists:
                found += 1
        
        lookup_time = time.time() - lookup_start
        
        # Get memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        return DataStructureResult(
            structure='compressed_trie',
            insertion_time=insertion_time,
            lookup_time=lookup_time,
            memory_usage=memory_usage,
            cache_misses=0,
            metrics={
                'found_ratio': found / (len(self.test_fields) // 10)
            }
        )
    
    @profile
    def test_bitmap_index(self) -> DataStructureResult:
        """Test bitmap index implementation."""
        start_time = time.time()
        
        # Create index
        index = BitmapIndex(len(self.test_fields))
        for i, field in enumerate(self.test_fields):
            index.add_field(i, field)
        
        insertion_time = time.time() - start_time
        
        # Test queries
        lookup_start = time.time()
        query_results = []
        
        # Test different queries
        o_fields = index.query_prefix('o_')
        f_fields = index.query_prefix('f_')
        with_numbers = index.query_with_number()
        
        # Combine queries
        o_with_numbers = index.combine_queries(o_fields, with_numbers)
        
        query_results.extend([
            o_fields.count(1),
            f_fields.count(1),
            with_numbers.count(1),
            o_with_numbers.count(1)
        ])
        
        lookup_time = time.time() - lookup_start
        
        # Get memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        return DataStructureResult(
            structure='bitmap_index',
            insertion_time=insertion_time,
            lookup_time=lookup_time,
            memory_usage=memory_usage,
            cache_misses=0,
            metrics={
                'avg_query_results': np.mean(query_results),
                'max_query_results': max(query_results)
            }
        )
    
    @profile
    def test_cache_friendly(self) -> DataStructureResult:
        """Test cache-friendly array implementation."""
        start_time = time.time()
        
        # Insert data
        array = CacheFriendlyArray(block_size=64)
        for field in self.test_fields:
            array.append(field)
        
        insertion_time = time.time() - start_time
        
        # Test lookups
        lookup_start = time.time()
        found = 0
        for i in range(0, len(array), 10):
            if array.get(i) in self.test_fields:
                found += 1
        
        lookup_time = time.time() - lookup_start
        
        # Get memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        return DataStructureResult(
            structure='cache_friendly_array',
            insertion_time=insertion_time,
            lookup_time=lookup_time,
            memory_usage=memory_usage,
            cache_misses=0,
            metrics={
                'size': len(array),
                'num_blocks': len(array.blocks),
                'found_ratio': found / (len(array) // 10)
            }
        )
    
    def run_experiments(self) -> pd.DataFrame:
        """Run all data structure experiments."""
        results = []
        
        # Test each data structure
        tests = [
            self.test_btree,
            self.test_compressed_trie,
            self.test_bitmap_index,
            self.test_cache_friendly
        ]
        
        for test_func in tqdm(tests, desc="Testing data structures"):
            try:
                result = test_func()
                results.append(result)
            except Exception as e:
                logger.error(f"Error testing {test_func.__name__}: {str(e)}")
        
        return self._analyze_results(results)
    
    def _analyze_results(self, results: List[DataStructureResult]) -> pd.DataFrame:
        """Analyze and visualize experiment results."""
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {
                'structure': result.structure,
                'insertion_time': result.insertion_time,
                'lookup_time': result.lookup_time,
                'memory_usage': result.memory_usage,
                'cache_misses': result.cache_misses,
                **result.metrics
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save results
        df.to_csv(self.test_dir / 'data_structure_results.csv', index=False)
        
        # Create visualizations
        self._plot_timing_comparison(df)
        self._plot_memory_comparison(df)
        
        return df
    
    def _plot_timing_comparison(self, df: pd.DataFrame):
        """Plot timing comparison."""
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df['insertion_time'], width, label='Insertion')
        plt.bar(x + width/2, df['lookup_time'], width, label='Lookup')
        
        plt.xlabel('Data Structure')
        plt.ylabel('Time (seconds)')
        plt.title('Operation Time by Data Structure')
        plt.xticks(x, df['structure'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.test_dir / 'timing_comparison.png')
        plt.close()
    
    def _plot_memory_comparison(self, df: pd.DataFrame):
        """Plot memory usage comparison."""
        plt.figure(figsize=(8, 6))
        
        plt.bar(df['structure'], df['memory_usage'])
        plt.xlabel('Data Structure')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage by Data Structure')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.test_dir / 'memory_comparison.png')
        plt.close()

def main():
    # Create test directory
    test_dir = Path("data_structure_test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = DataStructureBenchmark(test_dir)
    
    # Run experiments
    df = benchmark.run_experiments()
    
    # Print summary
    print("\nData Structure Benchmark Results:")
    print("\nTiming (seconds):")
    print(df[['structure', 'insertion_time', 'lookup_time']])
    
    print("\nMemory Usage (MB):")
    print(df[['structure', 'memory_usage']])
    
    print("\nResults saved to:", test_dir)

if __name__ == "__main__":
    main() 