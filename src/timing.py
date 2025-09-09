"""
Timing and latency measurement helper function for RIS jamming detection.
"""
import time
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class LatencyMeasurement:
    operation_name: str
    mean_ms: float
    median_ms: float
    std_ms: float
    p95_ms: float
    p99_ms: float
    n_samples: int

def measure_inference_latency(model, X_test: np.ndarray, n_runs: int = 100, warmup_runs: int = 10) -> LatencyMeasurement:
    """
    Measure model inference latency with statistical analysis.
    Automatically detects prediction method (predict_proba, decision_function, or predict).
    """
    logger.info(f"Measuring inference latency over {n_runs} runs with {warmup_runs} warmup runs")
    
    # Determine appropriate prediction method
    predict_method = None
    method_name = "unknown"
    
    if hasattr(model, 'predict_proba'):
        predict_method = model.predict_proba
        method_name = "predict_proba"
    elif hasattr(model, 'decision_function'):
        predict_method = model.decision_function
        method_name = "decision_function"
    elif hasattr(model, 'predict'):
        predict_method = model.predict
        method_name = "predict"
    else:
        raise ValueError("Model has no supported prediction method")
    
    logger.debug(f"Using prediction method: {method_name}")
    
    # Warmup runs before that to stabilise performance
    try:
        for _ in range(warmup_runs):
            _ = predict_method(X_test[:min(10, len(X_test))])
    except Exception as e:
        logger.warning(f"Warmup failed, continuing anyway: {e}")
    
    # timing 
    latencies = []
    failures = 0
    
    for i in range(n_runs):
        try:
            start_time = time.perf_counter()
            _ = predict_method(X_test)
            end_time = time.perf_counter()
            
            # Convert to milliseconds per sample
            latency_ms = ((end_time - start_time) / len(X_test)) * 1000.0
            latencies.append(latency_ms)
        except Exception as e:
            failures += 1
            logger.debug(f"Timing run {i} failed: {e}")
            if failures > n_runs // 4:  # Too many failures
                raise RuntimeError(f"Too many timing failures ({failures}/{i+1})")
    
    if not latencies:
        raise RuntimeError("All timing runs failed")
    
    latencies = np.array(latencies)
    
    measurement = LatencyMeasurement(
        operation_name='inference',
        mean_ms=float(np.mean(latencies)),
        median_ms=float(np.median(latencies)),
        std_ms=float(np.std(latencies)),
        p95_ms=float(np.percentile(latencies, 95)),
        p99_ms=float(np.percentile(latencies, 99)),
        n_samples=len(latencies)
    )
    
    logger.info(f"Inference latency: {measurement.median_ms:.3f} ms/sample "
               f"(mean: {measurement.mean_ms:.3f}, 95th percentile: {measurement.p95_ms:.3f})")
    
    if failures > 0:
        logger.warning(f"Had {failures} failed timing runs out of {n_runs}")
    
    return measurement

