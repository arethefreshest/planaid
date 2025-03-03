# Performance Optimization Experiments

This directory contains experiments to evaluate different performance optimization strategies for the document processing pipeline.

## Optimization Strategies

1. **Caching**
   - Redis caching
   - In-memory caching (LRU cache)
   - Cache invalidation strategies
   - Memory usage monitoring

2. **Async Processing**
   - Asynchronous file operations
   - Concurrent API requests
   - Request batching
   - Response streaming

3. **Parallel Processing**
   - Thread pools
   - Process pools
   - CPU utilization
   - Memory management

4. **Request Batching**
   - Different batch sizes
   - Batch processing strategies
   - Optimal batch size determination
   - Error handling in batches

5. **Response Streaming**
   - Streaming responses
   - Progressive loading
   - Memory efficiency
   - Network optimization

## Setup

1. Install Redis:
   ```bash
   # macOS
   brew install redis
   brew services start redis
   
   # Ubuntu
   sudo apt-get install redis-server
   sudo systemctl start redis
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments

Run the performance tests:
```bash
python performance_benchmark.py
```

## Test Scenarios

1. **Single File Processing**
   - Basic file operations
   - Caching effectiveness
   - Processing overhead

2. **Batch Processing**
   - Multiple files
   - Different batch sizes
   - Concurrent processing

3. **Mixed Workloads**
   - Combination of operations
   - Variable file sizes
   - Different processing strategies

## Results

The experiments generate:

1. **Performance Metrics**
   - Processing times
   - Memory usage
   - Success rates
   - Cache hit rates

2. **Visualizations**
   - Processing time plots
   - Memory usage graphs
   - Success rate comparisons
   - Batch size analysis

3. **CSV Data**
   - Detailed metrics
   - Raw timing data
   - Strategy comparisons

## Analysis

Results help determine:
- Most efficient processing strategy
- Optimal batch sizes
- Caching effectiveness
- Resource utilization

## Integration with Main Project

These optimizations can be applied to:

1. **Document Processing**
   ```
   Upload → Cache Check → Process → Store Result
   ```

2. **API Requests**
   ```
   Batch Requests → Parallel Process → Stream Response
   ```

3. **Resource Management**
   ```
   Monitor Usage → Adjust Strategy → Optimize Performance
   ```

## Dependencies

- Python 3.10+
- Redis server
- Async I/O libraries
- Data analysis tools

## Future Work (TODO)

Areas for further optimization:
1. Advanced caching strategies
2. Distributed processing
3. Load balancing
4. Network optimizations
5. Resource monitoring 