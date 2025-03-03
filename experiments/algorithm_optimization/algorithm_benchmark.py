"""
Algorithm Optimization Experiments

This script implements and evaluates different algorithmic optimizations:
1. MinHash for fast document similarity
2. LSH for field matching
3. Trie-based field indexing
4. Suffix arrays for text search
5. Bloom filters for field existence

The goal is to find optimal algorithms and data structures for the project.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import mmh3
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AlgorithmResult:
    """Stores results of an algorithm test."""
    algorithm: str
    preprocessing_time: float
    query_time: float
    memory_usage: float
    accuracy: float
    metrics: Dict[str, float]

class TrieNode:
    """Node in a trie data structure."""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.data = None

class Trie:
    """Trie data structure for efficient prefix matching."""
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str, data: Optional[dict] = None):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.data = data
    
    def search(self, word: str) -> Tuple[bool, Optional[dict]]:
        node = self.root
        for char in word:
            if char not in node.children:
                return False, None
            node = node.children[char]
        return node.is_end, node.data

    def starts_with(self, prefix: str) -> List[Tuple[str, dict]]:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        results = []
        self._collect_words(node, prefix, results)
        return results
    
    def _collect_words(self, node: TrieNode, prefix: str, results: List[Tuple[str, dict]]):
        if node.is_end:
            results.append((prefix, node.data))
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, results)

class BloomFilter:
    """Bloom filter for probabilistic set membership testing."""
    def __init__(self, size: int, num_hash_functions: int):
        self.size = size
        self.num_hash_functions = num_hash_functions
        self.bit_array = np.zeros(size, dtype=bool)
    
    def add(self, item: str):
        for seed in range(self.num_hash_functions):
            index = mmh3.hash(item, seed) % self.size
            self.bit_array[index] = True
    
    def contains(self, item: str) -> bool:
        for seed in range(self.num_hash_functions):
            index = mmh3.hash(item, seed) % self.size
            if not self.bit_array[index]:
                return False
        return True

class SuffixArray:
    """Suffix array for efficient text search."""
    def __init__(self, text: str):
        self.text = text
        self.suffixes = sorted((text[i:], i) for i in range(len(text)))
    
    def search(self, pattern: str) -> List[int]:
        left = 0
        right = len(self.suffixes) - 1
        
        # Binary search for pattern
        while left <= right:
            mid = (left + right) // 2
            if self.suffixes[mid][0].startswith(pattern):
                # Found a match, collect all matches
                matches = []
                i = mid
                while i >= 0 and self.suffixes[i][0].startswith(pattern):
                    matches.append(self.suffixes[i][1])
                    i -= 1
                i = mid + 1
                while i < len(self.suffixes) and self.suffixes[i][0].startswith(pattern):
                    matches.append(self.suffixes[i][1])
                    i += 1
                return sorted(matches)
            elif pattern < self.suffixes[mid][0][:len(pattern)]:
                right = mid - 1
            else:
                left = mid + 1
        return []

class AlgorithmOptimizer:
    """Tests different algorithmic optimizations."""
    
    def __init__(self, test_dir: Path):
        """Initialize optimizer with test directory."""
        self.test_dir = test_dir
        self.test_dir.mkdir(exist_ok=True)
        
        # Test data
        self.test_fields = [
            "BRA1", "o_BRA2", "f_BRA3",
            "o_SV1", "o_SV2", "o_SV3",
            "f_BE1", "BE2", "BE3",
            "H210", "H220", "H320",
            "#01 SNØ", "#02 SNØ"
        ]
        
        self.test_documents = [
            "Document with BRA1 and o_SV1",
            "Another with o_BRA2 and f_BE1",
            "Document containing H210 and #01 SNØ",
            "Text with BE2 and o_SV2 fields"
        ]
    
    def test_minhash_lsh(self) -> AlgorithmResult:
        """Test MinHash and LSH for document similarity."""
        start_time = time.time()
        
        # Create MinHash objects
        num_perm = 128
        minhashes = []
        for doc in self.test_documents:
            m = MinHash(num_perm=num_perm)
            for word in doc.split():
                m.update(word.encode('utf-8'))
            minhashes.append(m)
        
        # Create LSH index
        lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
        for i, mh in enumerate(minhashes):
            lsh.insert(f"doc_{i}", mh)
        
        preprocessing_time = time.time() - start_time
        
        # Test queries
        query_start = time.time()
        results = []
        for i, mh in enumerate(minhashes):
            similar = lsh.query(mh)
            results.append(len(similar))
        
        query_time = time.time() - query_start
        
        return AlgorithmResult(
            algorithm='minhash_lsh',
            preprocessing_time=preprocessing_time,
            query_time=query_time,
            memory_usage=0,  # Would need to implement memory tracking
            accuracy=1.0,  # Would need ground truth for real accuracy
            metrics={
                'avg_similar_docs': np.mean(results),
                'max_similar_docs': max(results)
            }
        )
    
    def test_trie_indexing(self) -> AlgorithmResult:
        """Test Trie-based field indexing."""
        start_time = time.time()
        
        # Build trie
        trie = Trie()
        for field in self.test_fields:
            trie.insert(field, {'original': field})
        
        preprocessing_time = time.time() - start_time
        
        # Test queries
        query_start = time.time()
        results = []
        prefixes = ['o_', 'f_', 'BR', 'H']
        for prefix in prefixes:
            matches = trie.starts_with(prefix)
            results.append(len(matches))
        
        query_time = time.time() - query_start
        
        return AlgorithmResult(
            algorithm='trie_indexing',
            preprocessing_time=preprocessing_time,
            query_time=query_time,
            memory_usage=0,
            accuracy=1.0,
            metrics={
                'avg_matches': np.mean(results),
                'max_matches': max(results)
            }
        )
    
    def test_suffix_array(self) -> AlgorithmResult:
        """Test suffix array for text search."""
        start_time = time.time()
        
        # Build suffix array
        text = ' '.join(self.test_documents)
        sa = SuffixArray(text)
        
        preprocessing_time = time.time() - start_time
        
        # Test queries
        query_start = time.time()
        results = []
        for field in self.test_fields:
            matches = sa.search(field)
            results.append(len(matches))
        
        query_time = time.time() - query_start
        
        return AlgorithmResult(
            algorithm='suffix_array',
            preprocessing_time=preprocessing_time,
            query_time=query_time,
            memory_usage=0,
            accuracy=1.0,
            metrics={
                'avg_occurrences': np.mean(results),
                'max_occurrences': max(results)
            }
        )
    
    def test_bloom_filter(self) -> AlgorithmResult:
        """Test Bloom filter for field existence checking."""
        start_time = time.time()
        
        # Create Bloom filter
        bf = BloomFilter(size=1000, num_hash_functions=3)
        for field in self.test_fields:
            bf.add(field)
        
        preprocessing_time = time.time() - start_time
        
        # Test queries
        query_start = time.time()
        true_positives = 0
        test_queries = self.test_fields + ["NonExistentField1", "NonExistentField2"]
        for query in test_queries:
            if bf.contains(query):
                true_positives += 1
        
        query_time = time.time() - query_start
        
        return AlgorithmResult(
            algorithm='bloom_filter',
            preprocessing_time=preprocessing_time,
            query_time=query_time,
            memory_usage=0,
            accuracy=true_positives / len(self.test_fields),
            metrics={
                'true_positives': true_positives,
                'false_positives': sum(1 for q in test_queries[len(self.test_fields):] if bf.contains(q))
            }
        )
    
    def run_experiments(self) -> pd.DataFrame:
        """Run all algorithm experiments."""
        results = []
        
        # Test each algorithm
        algorithms = [
            self.test_minhash_lsh,
            self.test_trie_indexing,
            self.test_suffix_array,
            self.test_bloom_filter
        ]
        
        for algo_func in tqdm(algorithms, desc="Testing algorithms"):
            try:
                result = algo_func()
                results.append(result)
            except Exception as e:
                logger.error(f"Error testing {algo_func.__name__}: {str(e)}")
        
        return self._analyze_results(results)
    
    def _analyze_results(self, results: List[AlgorithmResult]) -> pd.DataFrame:
        """Analyze and visualize experiment results."""
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {
                'algorithm': result.algorithm,
                'preprocessing_time': result.preprocessing_time,
                'query_time': result.query_time,
                'memory_usage': result.memory_usage,
                'accuracy': result.accuracy,
                **result.metrics
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save results
        df.to_csv(self.test_dir / 'algorithm_results.csv', index=False)
        
        # Create visualizations
        self._plot_timing_comparison(df)
        self._plot_accuracy_comparison(df)
        
        return df
    
    def _plot_timing_comparison(self, df: pd.DataFrame):
        """Plot timing comparison."""
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df['preprocessing_time'], width, label='Preprocessing')
        plt.bar(x + width/2, df['query_time'], width, label='Query')
        
        plt.xlabel('Algorithm')
        plt.ylabel('Time (seconds)')
        plt.title('Algorithm Timing Comparison')
        plt.xticks(x, df['algorithm'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.test_dir / 'timing_comparison.png')
        plt.close()
    
    def _plot_accuracy_comparison(self, df: pd.DataFrame):
        """Plot accuracy comparison."""
        plt.figure(figsize=(8, 6))
        
        plt.bar(df['algorithm'], df['accuracy'])
        plt.xlabel('Algorithm')
        plt.ylabel('Accuracy')
        plt.title('Algorithm Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.test_dir / 'accuracy_comparison.png')
        plt.close()

def main():
    # Create test directory
    test_dir = Path("algorithm_test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize optimizer
    optimizer = AlgorithmOptimizer(test_dir)
    
    # Run experiments
    df = optimizer.run_experiments()
    
    # Print summary
    print("\nAlgorithm Optimization Results:")
    print("\nTiming (seconds):")
    print(df[['algorithm', 'preprocessing_time', 'query_time']])
    
    print("\nAccuracy:")
    print(df[['algorithm', 'accuracy']])
    
    print("\nResults saved to:", test_dir)

if __name__ == "__main__":
    main() 