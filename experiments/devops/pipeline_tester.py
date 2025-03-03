"""
CI/CD Pipeline Testing Script

This script tests different CI/CD pipeline configurations:
1. Build strategies
2. Test configurations
3. Deployment approaches
4. Security scanning
5. Performance monitoring

The goal is to find the most effective pipeline configuration for the project.
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
from tqdm import tqdm
import shutil
import tempfile
import git

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Stores results of a pipeline run."""
    strategy: str
    total_time: float
    build_time: float
    test_time: float
    security_scan_time: float
    success: bool
    metrics: Dict[str, float]

class PipelineTester:
    """Tests different CI/CD pipeline configurations."""
    
    def __init__(self, test_dir: Path):
        """Initialize pipeline tester."""
        self.test_dir = test_dir
        self.repo_dir = test_dir / "test_repo"
        
    def setup_test_repo(self):
        """Set up a test repository with sample code."""
        if self.repo_dir.exists():
            shutil.rmtree(self.repo_dir)
            
        self.repo_dir.mkdir(parents=True)
        
        # Initialize git repo
        repo = git.Repo.init(self.repo_dir)
        
        # Create sample project structure
        self._create_sample_project()
        
        # Add files and commit
        repo.index.add("*")
        repo.index.commit("Initial commit")
        
    def _create_sample_project(self):
        """Create a sample project for testing."""
        # Create directory structure
        (self.repo_dir / "src").mkdir()
        (self.repo_dir / "tests").mkdir()
        (self.repo_dir / ".github/workflows").mkdir(parents=True)
        
        # Create sample Python files
        with open(self.repo_dir / "src/app.py", "w") as f:
            f.write("""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
""")
            
        with open(self.repo_dir / "tests/test_app.py", "w") as f:
            f.write("""
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}
""")
            
        # Create requirements.txt
        with open(self.repo_dir / "requirements.txt", "w") as f:
            f.write("""
fastapi>=0.109.2
uvicorn>=0.27.1
pytest>=7.4.0
pytest-cov>=4.1.0
bandit>=1.7.7
safety>=2.3.5
""")
            
    def generate_workflow(self, strategy: str) -> str:
        """Generate a GitHub Actions workflow file."""
        if strategy == "basic":
            return """
name: Basic CI/CD

on: [push]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest
"""
        elif strategy == "comprehensive":
            return """
name: Comprehensive CI/CD

on: [push]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Cache pip packages
      uses: actions/cache@v2
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests with coverage
      run: pytest --cov=src
    - name: Security scan
      run: |
        bandit -r src/
        safety check
"""
        elif strategy == "parallel":
            return """
name: Parallel CI/CD

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Security scan
      run: |
        bandit -r src/
        safety check
"""
        elif strategy == "matrix":
            return """
name: Matrix CI/CD

on: [push]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest
"""
            
    def test_pipeline(self, strategy: str) -> PipelineResult:
        """Test a pipeline configuration."""
        start_time = time.time()
        success = True
        
        try:
            # Write workflow file
            workflow_dir = self.repo_dir / ".github/workflows"
            workflow_file = workflow_dir / f"{strategy}.yml"
            workflow_content = self.generate_workflow(strategy)
            
            with open(workflow_file, "w") as f:
                f.write(workflow_content)
                
            # Simulate pipeline steps
            build_time = self._simulate_build()
            test_time = self._simulate_tests()
            security_time = self._simulate_security_scan()
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            success = False
            build_time = test_time = security_time = 0
            
        total_time = time.time() - start_time
        
        return PipelineResult(
            strategy=strategy,
            total_time=total_time,
            build_time=build_time,
            test_time=test_time,
            security_scan_time=security_time,
            success=success,
            metrics={
                'total_time': total_time,
                'build_time': build_time,
                'test_time': test_time,
                'security_time': security_time,
                'success': int(success)
            }
        )
        
    def _simulate_build(self) -> float:
        """Simulate build process."""
        start_time = time.time()
        
        # Install dependencies
        subprocess.run(
            ["pip", "install", "-r", str(self.repo_dir / "requirements.txt")],
            check=True,
            capture_output=True
        )
        
        return time.time() - start_time
        
    def _simulate_tests(self) -> float:
        """Simulate test execution."""
        start_time = time.time()
        
        # Run tests
        subprocess.run(
            ["pytest", str(self.repo_dir / "tests")],
            check=True,
            capture_output=True
        )
        
        return time.time() - start_time
        
    def _simulate_security_scan(self) -> float:
        """Simulate security scanning."""
        start_time = time.time()
        
        # Run security scans
        subprocess.run(
            ["bandit", "-r", str(self.repo_dir / "src")],
            check=True,
            capture_output=True
        )
        
        subprocess.run(
            ["safety", "check"],
            check=True,
            capture_output=True
        )
        
        return time.time() - start_time
        
    def run_experiments(self) -> pd.DataFrame:
        """Run all pipeline experiments."""
        results = []
        
        # Set up test repository
        self.setup_test_repo()
        
        # Test each strategy
        strategies = ['basic', 'comprehensive', 'parallel', 'matrix']
        
        for strategy in tqdm(strategies, desc="Testing pipeline strategies"):
            try:
                result = self.test_pipeline(strategy)
                results.append(result)
            except Exception as e:
                logger.error(f"Error testing strategy {strategy}: {str(e)}")
                
        return self._analyze_results(results)
        
    def _analyze_results(self, results: List[PipelineResult]) -> pd.DataFrame:
        """Analyze and visualize experiment results."""
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {
                'strategy': result.strategy,
                'total_time': result.total_time,
                'build_time': result.build_time,
                'test_time': result.test_time,
                'security_scan_time': result.security_scan_time,
                'success': result.success,
                **result.metrics
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save results
        df.to_csv(self.test_dir / 'pipeline_results.csv', index=False)
        
        # Create visualizations
        self._plot_timing_breakdown(df)
        self._plot_total_times(df)
        
        return df
        
    def _plot_timing_breakdown(self, df: pd.DataFrame):
        """Plot timing breakdown by strategy."""
        plt.figure(figsize=(12, 6))
        
        df[['build_time', 'test_time', 'security_scan_time']].plot(
            kind='bar',
            stacked=True
        )
        
        plt.title('Pipeline Time Breakdown by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Time (seconds)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.test_dir / 'timing_breakdown.png')
        plt.close()
        
    def _plot_total_times(self, df: pd.DataFrame):
        """Plot total pipeline times."""
        plt.figure(figsize=(10, 6))
        
        df.plot(kind='bar', x='strategy', y='total_time')
        plt.title('Total Pipeline Time by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.test_dir / 'total_times.png')
        plt.close()

def main():
    # Create test directory
    test_dir = Path("pipeline_test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize tester
    tester = PipelineTester(test_dir)
    
    # Run experiments
    df = tester.run_experiments()
    
    # Print summary
    print("\nPipeline Test Results:")
    print("\nTotal Times (seconds):")
    print(df.groupby('strategy')['total_time'].agg(['mean', 'min', 'max']))
    
    print("\nSuccess Rates:")
    print(df.groupby('strategy')['success'].mean())
    
    print("\nResults saved to:", test_dir)

if __name__ == "__main__":
    main() 