"""
Performance Optimization Experiments

This script tests different performance optimization strategies:
1. Caching (Redis, in-memory)
2. Async processing
3. Parallel processing
4. Request batching
5. Response streaming

The goal is to find the most effective combination of optimizations for the
document processing pipeline.
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import aiohttp
import aiofiles
import redis
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import lru_cache
import json
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Stores results of a processing run."""
    strategy: str
    processing_time: float
    memory_usage: float
    success_rate: float
    metrics: Dict[str, float]

class CacheManager:
    """Manages different caching strategies."""
    
    def __init__(self):
        """Initialize cache manager with Redis and in-memory cache."""
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.memory_cache: Dict[str, Any] = {}
        
    def get_redis(self, key: str) -> Optional[str]:
        """Get value from Redis cache."""
        try:
            value = self.redis_client.get(key)
            return value.decode('utf-8') if value else None
        except redis.RedisError as e:
            logger.error(f"Redis error: {str(e)}")
            return None
            
    def set_redis(self, key: str, value: str, expire: int = 3600) -> bool:
        """Set value in Redis cache with expiration."""
        try:
            return self.redis_client.setex(key, expire, value)
        except redis.RedisError as e:
            logger.error(f"Redis error: {str(e)}")
            return False
            
    @lru_cache(maxsize=1000)
    def get_memory(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache."""
        return self.memory_cache.get(key)
        
    def set_memory(self, key: str, value: Any) -> None:
        """Set value in in-memory cache."""
        self.memory_cache[key] = value

class AsyncProcessor:
    """Handles async processing of documents."""
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize async processor with cache manager."""
        self.cache = cache_manager
        self.session = None
        
    async def __aenter__(self):
        """Set up async context."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context."""
        if self.session:
            await self.session.close()
            
    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file asynchronously."""
        # First check cache
        cache_key = f"file:{file_path}"
        
        # Try Redis first
        cached = self.cache.get_redis(cache_key)
        if cached:
            return json.loads(cached)
            
        # Try memory cache
        cached = self.cache.get_memory(cache_key)
        if cached:
            return cached
            
        # Process file
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            
        # Simulate processing
        await asyncio.sleep(0.1)  # Simulate some work
        result = {
            'file': file_path,
            'size': len(content),
            'processed': True
        }
        
        # Cache result
        self.cache.set_redis(cache_key, json.dumps(result))
        self.cache.set_memory(cache_key, result)
        
        return result
        
    async def process_batch(self, files: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files concurrently."""
        tasks = [self.process_file(f) for f in files]
        return await asyncio.gather(*tasks)

class ParallelProcessor:
    """Handles parallel processing using threads and processes."""
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize parallel processor."""
        self.cache = cache_manager
        self.num_threads = multiprocessing.cpu_count()
        
    def process_file_threaded(self, file_path: str) -> Dict[str, Any]:
        """Process a file using threads."""
        cache_key = f"file:{file_path}"
        
        # Check cache
        cached = self.cache.get_redis(cache_key)
        if cached:
            return json.loads(cached)
            
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Simulate processing
        time.sleep(0.1)  # Simulate some work
        result = {
            'file': file_path,
            'size': len(content),
            'processed': True
        }
        
        # Cache result
        self.cache.set_redis(cache_key, json.dumps(result))
        return result
        
    def process_file_multiprocess(self, file_path: str) -> Dict[str, Any]:
        """Process a file using multiple processes."""
        # Note: Can't use Redis cache in child processes
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Simulate processing
        time.sleep(0.1)  # Simulate some work
        return {
            'file': file_path,
            'size': len(content),
            'processed': True
        }
        
    def process_batch_threaded(self, files: List[str]) -> List[Dict[str, Any]]:
        """Process files using thread pool."""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            return list(executor.map(self.process_file_threaded, files))
            
    def process_batch_multiprocess(self, files: List[str]) -> List[Dict[str, Any]]:
        """Process files using process pool."""
        with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
            return list(executor.map(self.process_file_multiprocess, files))

class PerformanceTester:
    """Runs performance tests on different processing strategies."""
    
    def __init__(self, test_dir: Path):
        """Initialize performance tester."""
        self.test_dir = test_dir
        self.cache = CacheManager()
        self.async_processor = AsyncProcessor(self.cache)
        self.parallel_processor = ParallelProcessor(self.cache)
        
    async def test_async_processing(self, files: List[str]) -> ProcessingResult:
        """Test async processing with caching."""
        start_time = time.time()
        success_count = 0
        
        async with self.async_processor as processor:
            results = await processor.process_batch(files)
            success_count = sum(1 for r in results if r['processed'])
            
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            strategy='async',
            processing_time=processing_time,
            memory_usage=0,  # Would need to implement memory tracking
            success_rate=success_count / len(files),
            metrics={
                'batch_size': len(files),
                'cache_hits': sum(1 for r in results if r.get('from_cache', False))
            }
        )
        
    def test_threaded_processing(self, files: List[str]) -> ProcessingResult:
        """Test threaded processing with caching."""
        start_time = time.time()
        
        results = self.parallel_processor.process_batch_threaded(files)
        success_count = sum(1 for r in results if r['processed'])
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            strategy='threaded',
            processing_time=processing_time,
            memory_usage=0,  # Would need to implement memory tracking
            success_rate=success_count / len(files),
            metrics={
                'batch_size': len(files),
                'num_threads': self.parallel_processor.num_threads
            }
        )
        
    def test_multiprocess_processing(self, files: List[str]) -> ProcessingResult:
        """Test multiprocess processing."""
        start_time = time.time()
        
        results = self.parallel_processor.process_batch_multiprocess(files)
        success_count = sum(1 for r in results if r['processed'])
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            strategy='multiprocess',
            processing_time=processing_time,
            memory_usage=0,  # Would need to implement memory tracking
            success_rate=success_count / len(files),
            metrics={
                'batch_size': len(files),
                'num_processes': self.parallel_processor.num_threads
            }
        )
        
    async def run_all_tests(self, batch_sizes: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Run all performance tests with different batch sizes."""
        results = []
        
        # Generate test files if they don't exist
        test_files = self._ensure_test_files(max(batch_sizes))
        
        for batch_size in tqdm(batch_sizes, desc="Testing batch sizes"):
            files = test_files[:batch_size]
            
            # Test each strategy
            results.extend([
                await self.test_async_processing(files),
                self.test_threaded_processing(files),
                self.test_multiprocess_processing(files)
            ])
            
        return self._analyze_results(results)
        
    def _ensure_test_files(self, num_files: int) -> List[str]:
        """Create test files if they don't exist."""
        self.test_dir.mkdir(exist_ok=True)
        files = []
        
        for i in range(num_files):
            file_path = self.test_dir / f"test_{i}.txt"
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    f.write(f"Test content {i}\n" * 1000)  # 1000 lines
            files.append(str(file_path))
            
        return files
        
    def _analyze_results(self, results: List[ProcessingResult]) -> pd.DataFrame:
        """Analyze and visualize test results."""
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {
                'strategy': result.strategy,
                'processing_time': result.processing_time,
                'memory_usage': result.memory_usage,
                'success_rate': result.success_rate,
                **result.metrics
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save results
        df.to_csv(self.test_dir / 'performance_results.csv', index=False)
        
        # Create visualizations
        self._plot_processing_times(df)
        self._plot_success_rates(df)
        
        return df
        
    def _plot_processing_times(self, df: pd.DataFrame):
        """Plot processing times by strategy and batch size."""
        plt.figure(figsize=(10, 6))
        
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            plt.plot(
                strategy_df['batch_size'],
                strategy_df['processing_time'],
                marker='o',
                label=strategy
            )
            
        plt.title('Processing Time by Strategy and Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Processing Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.test_dir / 'processing_times.png')
        plt.close()
        
    def _plot_success_rates(self, df: pd.DataFrame):
        """Plot success rates by strategy."""
        plt.figure(figsize=(8, 6))
        
        df.groupby('strategy')['success_rate'].mean().plot(kind='bar')
        plt.title('Average Success Rate by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.test_dir / 'success_rates.png')
        plt.close()

async def main():
    # Create test directory
    test_dir = Path("performance_test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize tester
    tester = PerformanceTester(test_dir)
    
    # Run tests with different batch sizes
    df = await tester.run_all_tests(batch_sizes=[1, 5, 10, 20, 50])
    
    # Print summary
    print("\nPerformance Test Summary:")
    print("\nProcessing Time (seconds) by Strategy:")
    print(df.groupby('strategy')['processing_time'].agg(['mean', 'std']))
    
    print("\nSuccess Rate by Strategy:")
    print(df.groupby('strategy')['success_rate'].mean())
    
    print("\nResults saved to:", test_dir)

if __name__ == "__main__":
    asyncio.run(main()) 