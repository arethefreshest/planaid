# Parallelization Strategies Experiment

This experiment evaluates different parallelization strategies for document processing to determine the most efficient approach for scaling the PlanAid system.

## Objectives

1. Compare different parallelization strategies:
   - Sequential processing (baseline)
   - Thread pools
   - Process pools
   - Async/await
   - Task partitioning
   - Multiprocessing with queues

2. Measure key performance metrics:
   - Throughput (documents per second)
   - CPU utilization
   - Memory usage
   - Scaling efficiency
   - Context switching overhead

3. Determine optimal parallelization strategy based on:
   - Document characteristics (size, complexity)
   - Available system resources
   - Processing task types (CPU-bound vs IO-bound)

## Experiment Setup

The experiment simulates document processing with configurable:
- Document sizes (10KB to 500KB)
- Processing complexity
- Worker counts (1, 2, 4, 8 cores)
- Repetitions for statistical significance

## Running the Experiment

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the benchmark:
   ```bash
   python parallel_processing_benchmark.py
   ```

3. View results in the `results/` directory:
   - CSV files with raw benchmark data
   - Visualizations in `results/plots/`

## Parallelization Strategies

### 1. Sequential Processing
- Single-threaded, processes documents one at a time
- Baseline for comparison
- Simple implementation, no concurrency issues

### 2. Thread Pool
- Uses Python's `ThreadPoolExecutor`
- Suitable for IO-bound tasks
- Shares memory space, lower overhead
- Limited by Global Interpreter Lock (GIL) for CPU-bound tasks

### 3. Process Pool
- Uses Python's `ProcessPoolExecutor`
- Suitable for CPU-bound tasks
- Bypasses GIL limitations
- Higher memory overhead due to process isolation

### 4. Async/Await
- Uses Python's `asyncio`
- Excellent for IO-bound tasks
- Low overhead cooperative multitasking
- Requires careful task design to avoid blocking the event loop

### 5. Task Partitioning
- Divides documents into balanced partitions
- Processes each partition in parallel
- Reduces synchronization overhead
- Can be combined with other strategies

### 6. Multiprocessing with Queues
- Uses dedicated worker processes with message queues
- Good for continuous processing pipelines
- Flexible work distribution
- Higher implementation complexity

## Metrics Collected

1. **Throughput**
   - Documents processed per second
   - MB processed per second

2. **Resource Utilization**
   - Memory usage (MB)
   - CPU utilization (%)
   - Thread count
   - Context switches

3. **Scaling Efficiency**
   - Speedup relative to sequential processing
   - Efficiency (speedup / worker count)
   - Scaling curve characteristics

## Analysis Approach

The experiment analyzes:
1. Absolute performance of each strategy
2. Scaling characteristics as worker count increases
3. Resource efficiency (performance per unit of resources)
4. Trade-offs between different strategies

## Expected Outcomes

1. Identification of the most efficient parallelization strategy for:
   - Different document sizes
   - Different processing complexities
   - Different system configurations

2. Guidelines for implementing parallelization in the PlanAid system:
   - When to use each strategy
   - How to configure worker counts
   - How to partition work effectively

3. Performance projections for scaled deployment

## Integration with PlanAid

Results from this experiment will inform:
1. Implementation of the document processing pipeline
2. Configuration of worker processes in production
3. Resource allocation for different processing tasks
4. Optimization of the overall system architecture

## Future Work

1. Test with real-world document distributions
2. Evaluate hybrid parallelization strategies
3. Investigate adaptive worker allocation based on system load
4. Benchmark with distributed processing across multiple machines 