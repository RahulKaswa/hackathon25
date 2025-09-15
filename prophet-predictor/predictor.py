"""
Enhanced Predictive Scaling Service

A mature, generic, and accurate microservice for predictive scaling that supports:
- Multiple prediction algorithms (Prophet, Linear Regression, Ensemble)
- Configurable metrics and thresholds
- Redis caching for performance
- Comprehensive error handling and logging
- Health checks and monitoring
- Model persistence and automatic retraining
"""

import logging
import time
import hashlib
from typing import Dict, Any, List, Optional
from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_config, AppConfig
from data_fetcher import DataFetcher, PrometheusClient
from models import ModelFactory, BasePredictorModel
from cache import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('predictor.log')
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('predictor_requests_total', 'Total prediction requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('predictor_request_duration_seconds', 'Request duration', ['endpoint'])
PREDICTION_GAUGE = Gauge('predicted_load', 'Predicted load value', ['metric'])
MODEL_TRAINING_COUNT = Counter('predictor_model_training_total', 'Model training count', ['metric_name', 'algorithm'])
CACHE_HIT_COUNT = Counter('predictor_cache_hits_total', 'Cache hits', ['type'])
ERROR_COUNT = Counter('predictor_errors_total', 'Total errors', ['type'])

# Global application state
app = Flask(__name__)
config: AppConfig = None
data_fetcher: DataFetcher = None
cache_manager: CacheManager = None
models: Dict[str, BasePredictorModel] = {}


def initialize_app():
    """Initialize the application with configuration and dependencies."""
    global config, data_fetcher, cache_manager, models
    
    try:
        # Load configuration
        config = load_config()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
        
        # Initialize Prometheus client
        prometheus_client = PrometheusClient(config.prometheus)
        
        # Initialize data fetcher
        data_fetcher = DataFetcher(prometheus_client)
        
        # Initialize cache manager
        cache_manager = CacheManager(config.cache)
        
        # Initialize models for each metric
        models = {}
        for metric_config in config.metrics:
            model_type = os.getenv(f"MODEL_TYPE_{metric_config.name.upper()}", "prophet")
            models[metric_config.name] = ModelFactory.create_model(
                model_type, metric_config, config.model
            )
            
            # Try to load cached model
            cached_model = cache_manager.get_model(f"{metric_config.name}_{model_type}")
            if cached_model:
                try:
                    loaded_model = BasePredictorModel.deserialize(
                        cached_model, metric_config, config.model
                    )
                    if loaded_model:
                        models[metric_config.name] = loaded_model
                        logger.info(f"Loaded cached model for {metric_config.name}")
                except Exception as e:
                    logger.warning(f"Failed to load cached model for {metric_config.name}: {e}")
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


def get_or_train_model(metric_name: str) -> Optional[BasePredictorModel]:
    """Get model for metric, training if necessary."""
    if metric_name not in models:
        logger.error(f"No model found for metric: {metric_name}")
        return None
    
    model = models[metric_name]
    metric_config = next((m for m in config.metrics if m.name == metric_name), None)
    
    if not metric_config:
        logger.error(f"No configuration found for metric: {metric_name}")
        return None
    
    try:
        # Check cache first
        cache_key = f"prediction_{metric_name}_{int(time.time() // 300)}"  # 5-minute buckets
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            CACHE_HIT_COUNT.labels(type="prediction").inc()
            return model
        
        # Fetch fresh data
        df, is_valid = data_fetcher.get_metric_data(metric_config, config.data_quality_min_points)
        
        if not is_valid or df.empty:
            logger.warning(f"Invalid data for metric {metric_name}")
            ERROR_COUNT.labels(type="invalid_data").inc()
            return None
        
        # Check if model needs retraining
        if model.should_retrain(df, config.model_retrain_interval):
            logger.info(f"Retraining model for {metric_name}")
            
            if model.train(df):
                MODEL_TRAINING_COUNT.labels(metric_name=metric_name, algorithm=type(model).__name__).inc()
                
                # Cache the trained model
                try:
                    model_data = model.serialize()
                    cache_manager.set_model(f"{metric_name}_{type(model).__name__.lower()}", model_data)
                except Exception as e:
                    logger.warning(f"Failed to cache model: {e}")
            else:
                logger.error(f"Failed to train model for {metric_name}")
                ERROR_COUNT.labels(type="training_failed").inc()
                return None
        
        return model
        
    except Exception as e:
        logger.error(f"Error getting/training model for {metric_name}: {e}")
        ERROR_COUNT.labels(type="model_error").inc()
        return None


@app.route('/predict')
@REQUEST_DURATION.labels(endpoint='predict').time()
def predict_endpoint():
    """Generate predictions for all configured metrics."""
    try:
        metric_name = request.args.get('metric')
        periods = request.args.get('periods', type=int)
        
        if metric_name:
            # Predict for specific metric
            result = predict_single_metric(metric_name, periods)
            if result:
                REQUEST_COUNT.labels(endpoint='predict', status='success').inc()
                return jsonify(result)
            else:
                REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
                return jsonify({"error": f"Failed to predict for metric: {metric_name}"}), 500
        else:
            # Predict for all metrics
            results = {}
            for metric_config in config.metrics:
                result = predict_single_metric(metric_config.name, periods)
                if result:
                    results[metric_config.name] = result
            
            if results:
                REQUEST_COUNT.labels(endpoint='predict', status='success').inc()
                return jsonify(results)
            else:
                REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
                return jsonify({"error": "No predictions generated"}), 500
                
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
        ERROR_COUNT.labels(type="endpoint_error").inc()
        return jsonify({"error": str(e)}), 500


def predict_single_metric(metric_name: str, periods: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Generate prediction for a single metric."""
    try:
        # Check cache first
        cache_key = f"prediction_{metric_name}_{periods or 'default'}_{int(time.time() // 300)}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            CACHE_HIT_COUNT.labels(type="prediction").inc()
            return cached_result
        
        model = get_or_train_model(metric_name)
        if not model:
            return None
        
        # Generate prediction
        result = model.predict(periods)
        if "error" in result:
            logger.error(f"Prediction error for {metric_name}: {result['error']}")
            return None
        
        # Add metadata
        result["metric_name"] = metric_name
        result["timestamp"] = time.time()
        result["cache_key"] = cache_key
        
        # Update Prometheus gauge
        if "next_value" in result:
            PREDICTION_GAUGE.labels(metric=metric_name).set(result["next_value"])
        
        # Cache the result
        cache_manager.set(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error predicting for metric {metric_name}: {e}")
        return None


@app.route('/metrics')
@REQUEST_DURATION.labels(endpoint='metrics').time()
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    try:
        # Generate predictions for Prometheus scraping
        metrics_output = []
        
        for metric_config in config.metrics:
            result = predict_single_metric(metric_config.name)
            if result and "next_value" in result:
                next_value = result["next_value"]
                
                # Enhanced prediction: if Prophet returns 0, use trend-based prediction
                if next_value == 0.0 and metric_config.name == "http_requests":
                    try:
                        # Get current 5-minute rate as a trend indicator
                        current_data = data_fetcher.fetch_data(metric_config.name, periods=5, step="1m")
                        if not current_data.empty and len(current_data) >= 2:
                            # Simple trend prediction: use average of recent values
                            recent_avg = current_data.tail(3)['value'].mean()
                            if recent_avg > 0.1:  # If there's significant traffic
                                next_value = recent_avg * 1.1  # Predict 10% increase
                                logger.info(f"Using trend-based prediction for {metric_config.name}: {next_value}")
                    except Exception as e:
                        logger.warning(f"Failed to calculate trend prediction: {e}")
                
                # Main prediction metric
                metrics_output.append(f'predicted_load{{metric="{metric_config.name}"}} {next_value}')
                
                # Additional metrics
                if "peak_value" in result:
                    metrics_output.append(f'predicted_peak{{metric="{metric_config.name}"}} {result["peak_value"]}')
                
                if "trend" in result:
                    trend_value = {"increasing": 1, "decreasing": -1, "stable": 0}.get(result["trend"], 0)
                    metrics_output.append(f'predicted_trend{{metric="{metric_config.name}"}} {trend_value}')
        
        # Add application metrics
        app_metrics = generate_latest().decode('utf-8')
        
        REQUEST_COUNT.labels(endpoint='metrics', status='success').inc()
        
        response = '\n'.join(metrics_output) + '\n' + app_metrics
        return response, 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}
        
    except Exception as e:
        logger.error(f"Error in metrics endpoint: {e}")
        REQUEST_COUNT.labels(endpoint='metrics', status='error').inc()
        ERROR_COUNT.labels(type="endpoint_error").inc()
        return str(e), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "models": {},
            "cache": cache_manager.get_cache_stats() if cache_manager else {"enabled": False},
            "config": {
                "metrics_count": len(config.metrics),
                "model_retrain_interval": config.model_retrain_interval
            }
        }
        
        # Check model health
        for metric_name, model in models.items():
            health_status["models"][metric_name] = {
                "trained": model.model is not None,
                "last_trained": model.last_trained,
                "algorithm": type(model).__name__
            }
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route('/config')
def config_endpoint():
    """Configuration endpoint for debugging."""
    try:
        if config.debug:
            config_data = {
                "prometheus": {
                    "url": config.prometheus.url,
                    "timeout": config.prometheus.timeout
                },
                "metrics": [
                    {
                        "name": m.name,
                        "query": m.query,
                        "range_minutes": m.range_minutes,
                        "prediction_periods": m.prediction_periods
                    } for m in config.metrics
                ],
                "cache": {
                    "enabled": config.cache.enabled,
                    "ttl_seconds": config.cache.ttl_seconds
                }
            }
            return jsonify(config_data)
        else:
            return jsonify({"error": "Debug mode not enabled"}), 403
            
    except Exception as e:
        logger.error(f"Error in config endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error(f"Unhandled error: {error}")
    ERROR_COUNT.labels(type="unhandled").inc()
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    initialize_app()
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug
    )
