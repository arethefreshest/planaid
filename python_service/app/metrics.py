import time
import json
import os
import psutil
from datetime import datetime
from typing import Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

# Ensure metrics directory exists
os.makedirs("metrics", exist_ok=True)

class MetricsCollector:
    """Collects and saves metrics for processing operations."""
    
    def __init__(self, operation_name: str):
        """Initialize metrics collector.
        
        Args:
            operation_name: Name of the operation being measured
        """
        self.operation_name = operation_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.metrics = {
            "run_id": self.run_id,
            "operation": operation_name,
            "timestamp": datetime.now().isoformat(),
            "timings": {},
            "field_counts": {},
            "resource_usage": {},
            "document_info": {}
        }
    
    def add_document_info(self, doc_type: str, filename: str, size: Optional[int] = None):
        """Add information about a document being processed.
        
        Args:
            doc_type: Type of document (plankart, bestemmelser, sosi)
            filename: Name of the document file
            size: Size of the document in bytes (optional)
        """
        if "documents" not in self.metrics["document_info"]:
            self.metrics["document_info"]["documents"] = {}
        
        self.metrics["document_info"]["documents"][doc_type] = {
            "filename": filename,
            "size": size
        }
    
    def start_timer(self, step_name: str) -> float:
        """Start timing a processing step.
        
        Args:
            step_name: Name of the processing step
            
        Returns:
            Start time in seconds
        """
        return time.time()
    
    def stop_timer(self, step_name: str, start_time: float):
        """Stop timing a processing step and record the duration.
        
        Args:
            step_name: Name of the processing step
            start_time: Start time from start_timer()
        """
        duration = time.time() - start_time
        self.metrics["timings"][step_name] = duration
        return duration
    
    def record_field_count(self, field_type: str, fields: Set[str]):
        """Record the count of fields extracted.
        
        Args:
            field_type: Type of fields (plankart, bestemmelser, sosi, matching, etc.)
            fields: Set of fields extracted
        """
        self.metrics["field_counts"][field_type] = len(fields)
        
    def record_fields(self, field_type: str, fields: Set[str]):
        """Record the actual fields extracted.
        
        Args:
            field_type: Type of fields (plankart, bestemmelser, sosi, matching, etc.)
            fields: Set of fields extracted
        """
        if "fields" not in self.metrics:
            self.metrics["fields"] = {}
        
        self.metrics["fields"][field_type] = list(fields)
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize metrics collection and save to file.
        
        Returns:
            The collected metrics
        """
        # Record end metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.metrics["timings"]["total_processing"] = end_time - self.start_time
        self.metrics["resource_usage"]["memory_mb"] = end_memory - self.start_memory
        self.metrics["resource_usage"]["cpu_percent"] = psutil.Process().cpu_percent()
        
        # Save metrics to file
        try:
            metrics_file = f"metrics/{self.operation_name}_{self.run_id}.json"
            with open(metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
        
        return self.metrics
    
    def record_error(self, error: Exception) -> Dict[str, Any]:
        """Record an error in the metrics.
        
        Args:
            error: The exception that occurred
            
        Returns:
            The collected metrics
        """
        self.metrics["error"] = str(error)
        
        # Save metrics even in case of error
        try:
            metrics_file = f"metrics/{self.operation_name}_error_{self.run_id}.json"
            with open(metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Error metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
        
        return self.metrics 