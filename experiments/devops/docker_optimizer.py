"""
Docker Image Optimization Experiments

This script tests different Docker image optimization strategies:
1. Multi-stage builds
2. Layer optimization
3. Base image selection
4. Dependency management
5. Cache utilization

The goal is to find the most effective combination of optimizations for
reducing image size and build time while maintaining functionality.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
import yaml
import docker
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ImageBuildResult:
    """Stores results of a Docker image build."""
    strategy: str
    base_image: str
    build_time: float
    image_size: int
    layer_count: int
    cache_hits: int
    metrics: Dict[str, float]

class DockerOptimizer:
    """Tests different Docker image optimization strategies."""
    
    def __init__(self, test_dir: Path):
        """Initialize optimizer with test directory."""
        self.test_dir = test_dir
        self.client = docker.from_env()
        
        # Base images to test
        self.base_images = {
            'python-slim': 'python:3.10-slim',
            'python-alpine': 'python:3.10-alpine',
            'distroless': 'gcr.io/distroless/python3',
            'ubuntu': 'ubuntu:22.04'
        }
        
    def generate_dockerfile(self, strategy: str, base_image: str) -> Path:
        """Generate a Dockerfile for testing."""
        dockerfile_path = self.test_dir / f"Dockerfile.{strategy}"
        
        if strategy == 'basic':
            content = f"""
FROM {base_image}

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "app.py"]
"""
        elif strategy == 'multistage':
            content = f"""
FROM {base_image} AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM {base_image}
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY . .
CMD ["python", "app.py"]
"""
        elif strategy == 'layer_optimized':
            content = f"""
FROM {base_image}

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY app.py .
COPY config.yaml .
CMD ["python", "app.py"]
"""
        elif strategy == 'cache_optimized':
            content = f"""
FROM {base_image}

WORKDIR /app

# Copy only requirements first
COPY requirements.txt .

# Use build arguments for better cache control
ARG CACHEBUST=1
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .
CMD ["python", "app.py"]
"""
        
        with open(dockerfile_path, 'w') as f:
            f.write(content.strip())
            
        return dockerfile_path
        
    def generate_test_app(self):
        """Generate a test Python application."""
        # Create requirements.txt
        requirements = """
pandas>=2.2.0
numpy>=1.26.4
pyyaml>=6.0.1
fastapi>=0.109.2
uvicorn>=0.27.1
""".strip()
        
        with open(self.test_dir / 'requirements.txt', 'w') as f:
            f.write(requirements)
            
        # Create simple app
        app_code = """
from fastapi import FastAPI
import pandas as pd
import numpy as np
import yaml

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
""".strip()
        
        with open(self.test_dir / 'app.py', 'w') as f:
            f.write(app_code)
            
        # Create config file
        config = {
            'server': {
                'host': '0.0.0.0',
                'port': 8000
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        with open(self.test_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
            
    def build_image(self, dockerfile: Path, tag: str) -> ImageBuildResult:
        """Build Docker image and collect metrics."""
        start_time = time.time()
        
        try:
            # Build image
            image, logs = self.client.images.build(
                path=str(self.test_dir),
                dockerfile=str(dockerfile),
                tag=tag,
                rm=True
            )
            
            # Calculate build time
            build_time = time.time() - start_time
            
            # Get image details
            image_info = self.client.images.get(tag).attrs
            image_size = image_info['Size']
            layer_count = len(image_info['RootFS']['Layers'])
            
            # Count cache hits from build logs
            cache_hits = 0
            for log in logs:
                if isinstance(log, dict) and 'stream' in log:
                    if 'Using cache' in log['stream']:
                        cache_hits += 1
                        
            return ImageBuildResult(
                strategy=dockerfile.stem.split('.')[1],
                base_image=image_info['Config']['Image'],
                build_time=build_time,
                image_size=image_size,
                layer_count=layer_count,
                cache_hits=cache_hits,
                metrics={
                    'size_mb': image_size / (1024 * 1024),
                    'layers': layer_count,
                    'cache_hits': cache_hits
                }
            )
            
        except docker.errors.BuildError as e:
            logger.error(f"Build error: {str(e)}")
            raise
            
    def run_experiments(self) -> pd.DataFrame:
        """Run all Docker optimization experiments."""
        results = []
        
        # Generate test application
        self.generate_test_app()
        
        # Test each strategy with each base image
        strategies = ['basic', 'multistage', 'layer_optimized', 'cache_optimized']
        
        for strategy in tqdm(strategies, desc="Testing strategies"):
            for base_name, base_image in self.base_images.items():
                try:
                    # Generate Dockerfile
                    dockerfile = self.generate_dockerfile(strategy, base_image)
                    
                    # Build image
                    tag = f"test-{strategy}-{base_name}:latest"
                    result = self.build_image(dockerfile, tag)
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error testing {strategy} with {base_name}: {str(e)}")
                    
        return self._analyze_results(results)
        
    def _analyze_results(self, results: List[ImageBuildResult]) -> pd.DataFrame:
        """Analyze and visualize experiment results."""
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {
                'strategy': result.strategy,
                'base_image': result.base_image,
                'build_time': result.build_time,
                'image_size': result.image_size,
                'layer_count': result.layer_count,
                'cache_hits': result.cache_hits,
                **result.metrics
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save results
        df.to_csv(self.test_dir / 'docker_optimization_results.csv', index=False)
        
        # Create visualizations
        self._plot_image_sizes(df)
        self._plot_build_times(df)
        self._plot_layer_counts(df)
        
        return df
        
    def _plot_image_sizes(self, df: pd.DataFrame):
        """Plot image sizes by strategy and base image."""
        plt.figure(figsize=(12, 6))
        
        df.pivot(
            index='strategy',
            columns='base_image',
            values='size_mb'
        ).plot(kind='bar')
        
        plt.title('Image Size by Strategy and Base Image')
        plt.xlabel('Strategy')
        plt.ylabel('Size (MB)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.test_dir / 'image_sizes.png')
        plt.close()
        
    def _plot_build_times(self, df: pd.DataFrame):
        """Plot build times by strategy."""
        plt.figure(figsize=(10, 6))
        
        df.boxplot(column='build_time', by='strategy')
        plt.title('Build Time by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.test_dir / 'build_times.png')
        plt.close()
        
    def _plot_layer_counts(self, df: pd.DataFrame):
        """Plot layer counts by strategy and base image."""
        plt.figure(figsize=(12, 6))
        
        df.pivot(
            index='strategy',
            columns='base_image',
            values='layer_count'
        ).plot(kind='bar')
        
        plt.title('Layer Count by Strategy and Base Image')
        plt.xlabel('Strategy')
        plt.ylabel('Number of Layers')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.test_dir / 'layer_counts.png')
        plt.close()

def main():
    # Create test directory
    test_dir = Path("docker_test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize optimizer
    optimizer = DockerOptimizer(test_dir)
    
    # Run experiments
    df = optimizer.run_experiments()
    
    # Print summary
    print("\nDocker Optimization Results:")
    print("\nImage Sizes (MB):")
    print(df.groupby('strategy')['size_mb'].agg(['mean', 'min', 'max']))
    
    print("\nBuild Times (seconds):")
    print(df.groupby('strategy')['build_time'].agg(['mean', 'min', 'max']))
    
    print("\nLayer Counts:")
    print(df.groupby('strategy')['layer_count'].agg(['mean', 'min', 'max']))
    
    print("\nResults saved to:", test_dir)

if __name__ == "__main__":
    main() 