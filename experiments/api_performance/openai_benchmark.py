"""
OpenAI/Azure API Performance Benchmark

This script tests different aspects of API usage:
1. Response times and latency
2. Batch processing efficiency
3. Rate limit handling
4. Cost optimization
5. Caching effectiveness

The goal is to find optimal settings for API usage in the project.
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
import json
import redis
from tqdm import tqdm
import numpy as np
from functools import lru_cache
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class APICallResult:
    """Stores results of an API call."""
    batch_size: int
    total_time: float
    time_per_request: float
    success: bool
    cached: bool
    cost: float  # Estimated cost in USD
    metrics: Dict[str, float]

class APIBenchmark:
    """Tests different API usage strategies."""
    
    def __init__(self, test_dir: Path):
        """Initialize benchmark with test directory."""
        self.test_dir = test_dir
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.api_version = os.getenv("OPENAI_API_VERSION")
        self.deployment = os.getenv("OPENAI_DEPLOYMENT_NAME")
        
        # Initialize Redis for caching
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        
        # Test prompts
        self.test_prompts = [
            "Extract field names from this text: BRA1, o_BRA2, f_BRA3",
            "Find regulation codes in: 1110 Boligbebyggelse, 1120 Fritidsbebyggelse",
            "Normalize these fields: BRA1, BRA-2, BRA_3",
            "Compare these fields: o_BRA1 vs BRA1",
            "Check consistency between: BRA1, o_BRA1, f_BRA1"
        ]
        
    async def test_single_call(self, prompt: str, use_cache: bool = True) -> APICallResult:
        """Test a single API call with optional caching."""
        start_time = time.time()
        cache_key = f"api_cache:{hash(prompt)}"
        cached = False
        
        if use_cache:
            # Try cache first
            cached_response = self.redis.get(cache_key)
            if cached_response:
                response = json.loads(cached_response)
                cached = True
                processing_time = time.time() - start_time
                return APICallResult(
                    batch_size=1,
                    total_time=processing_time,
                    time_per_request=processing_time,
                    success=True,
                    cached=True,
                    cost=0,  # No cost for cached responses
                    metrics={
                        'latency': processing_time,
                        'tokens': len(prompt.split())
                    }
                )
        
        # Make API call
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'api-key': self.api_key,
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 150
                }
                
                url = f"{self.api_base}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
                
                async with session.post(url, headers=headers, json=data) as response:
                    result = await response.json()
                    
                    processing_time = time.time() - start_time
                    success = response.status == 200
                    
                    if success and use_cache:
                        # Cache successful response
                        self.redis.setex(
                            cache_key,
                            3600,  # 1 hour expiration
                            json.dumps(result)
                        )
                    
                    # Calculate estimated cost
                    # Using Azure OpenAI pricing for gpt-4
                    prompt_tokens = len(prompt.split())
                    completion_tokens = len(result['choices'][0]['message']['content'].split())
                    cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
                    
                    return APICallResult(
                        batch_size=1,
                        total_time=processing_time,
                        time_per_request=processing_time,
                        success=success,
                        cached=cached,
                        cost=cost,
                        metrics={
                            'latency': processing_time,
                            'prompt_tokens': prompt_tokens,
                            'completion_tokens': completion_tokens,
                            'total_tokens': prompt_tokens + completion_tokens
                        }
                    )
                    
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            processing_time = time.time() - start_time
            return APICallResult(
                batch_size=1,
                total_time=processing_time,
                time_per_request=processing_time,
                success=False,
                cached=False,
                cost=0,
                metrics={
                    'error': str(e)
                }
            )
            
    async def test_batch_calls(self, batch_size: int, use_cache: bool = True) -> List[APICallResult]:
        """Test multiple API calls in a batch."""
        # Select prompts for this batch
        batch_prompts = self.test_prompts * (batch_size // len(self.test_prompts) + 1)
        batch_prompts = batch_prompts[:batch_size]
        
        # Make concurrent calls
        tasks = [self.test_single_call(prompt, use_cache) for prompt in batch_prompts]
        return await asyncio.gather(*tasks)
        
    async def run_experiments(self, batch_sizes: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Run all API performance experiments."""
        results = []
        
        # Test with and without caching
        for use_cache in [False, True]:
            for batch_size in tqdm(batch_sizes, desc=f"Testing batch sizes (cache={'on' if use_cache else 'off'})"):
                batch_results = await self.test_batch_calls(batch_size, use_cache)
                results.extend(batch_results)
                
                # Add delay between batches to avoid rate limits
                await asyncio.sleep(1)
                
        return self._analyze_results(results)
        
    def _analyze_results(self, results: List[APICallResult]) -> pd.DataFrame:
        """Analyze and visualize experiment results."""
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {
                'batch_size': result.batch_size,
                'total_time': result.total_time,
                'time_per_request': result.time_per_request,
                'success': result.success,
                'cached': result.cached,
                'cost': result.cost,
                **result.metrics
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save results
        df.to_csv(self.test_dir / 'api_performance_results.csv', index=False)
        
        # Create visualizations
        self._plot_timing_analysis(df)
        self._plot_cost_analysis(df)
        
        return df
        
    def _plot_timing_analysis(self, df: pd.DataFrame):
        """Plot timing analysis."""
        plt.figure(figsize=(12, 6))
        
        # Group by batch size and cached status
        grouped = df.groupby(['batch_size', 'cached'])['time_per_request'].mean().unstack()
        
        grouped.plot(marker='o')
        plt.title('Average Response Time by Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Request (seconds)')
        plt.legend(['Non-cached', 'Cached'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.test_dir / 'timing_analysis.png')
        plt.close()
        
    def _plot_cost_analysis(self, df: pd.DataFrame):
        """Plot cost analysis."""
        plt.figure(figsize=(10, 6))
        
        # Calculate cost per batch size
        cost_analysis = df.groupby('batch_size')['cost'].agg(['sum', 'mean'])
        
        # Plot total cost and average cost per request
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        cost_analysis['sum'].plot(ax=ax1, color='blue', marker='o', label='Total Cost')
        cost_analysis['mean'].plot(ax=ax2, color='red', marker='s', label='Cost per Request')
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Total Cost (USD)', color='blue')
        ax2.set_ylabel('Cost per Request (USD)', color='red')
        
        plt.title('Cost Analysis by Batch Size')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.test_dir / 'cost_analysis.png')
        plt.close()

async def main():
    # Create test directory
    test_dir = Path("api_test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = APIBenchmark(test_dir)
    
    # Run experiments
    df = await benchmark.run_experiments(batch_sizes=[1, 5, 10, 20, 50])
    
    # Print summary
    print("\nAPI Performance Summary:")
    print("\nResponse Times (seconds):")
    print(df.groupby(['batch_size', 'cached'])['time_per_request'].agg(['mean', 'std']))
    
    print("\nSuccess Rates:")
    print(df.groupby(['batch_size', 'cached'])['success'].mean())
    
    print("\nCost Analysis:")
    print(df.groupby('batch_size')['cost'].agg(['sum', 'mean']))
    
    print("\nResults saved to:", test_dir)

if __name__ == "__main__":
    asyncio.run(main()) 