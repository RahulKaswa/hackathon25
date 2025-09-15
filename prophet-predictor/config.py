"""Configuration management for the predictor service."""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml


@dataclass
class PrometheusConfig:
    """Prometheus connection configuration."""
    url: str
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5


@dataclass
class MetricConfig:
    """Configuration for a specific metric to predict."""
    name: str
    query: str
    step: str = "60s"
    range_minutes: int = 60
    prediction_periods: int = 5
    prediction_frequency: str = "min"
    threshold_min: float = 0.0
    threshold_max: Optional[float] = None


@dataclass
class ModelConfig:
    """Prophet model configuration."""
    seasonality_mode: str = "additive"
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    holidays_prior_scale: float = 10.0
    daily_seasonality: bool = True
    weekly_seasonality: bool = True
    yearly_seasonality: bool = False
    interval_width: float = 0.8


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    ttl_seconds: int = 300
    model_cache_ttl: int = 3600


@dataclass
class AppConfig:
    """Main application configuration."""
    prometheus: PrometheusConfig
    metrics: List[MetricConfig]
    model: ModelConfig
    cache: CacheConfig
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    data_quality_min_points: int = 10
    model_retrain_interval: int = 1800  # 30 minutes


def load_config() -> AppConfig:
    """Load configuration from environment variables and config files."""
    
    # Try to load from YAML file first
    config_file = os.getenv("CONFIG_FILE", "config.yaml")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {}
    
    # Prometheus configuration
    prometheus_config = PrometheusConfig(
        url=os.getenv("PROMETHEUS_URL", config_data.get("prometheus", {}).get("url", "http://prometheus:9090")),
        timeout=int(os.getenv("PROMETHEUS_TIMEOUT", config_data.get("prometheus", {}).get("timeout", 30))),
        retry_attempts=int(os.getenv("PROMETHEUS_RETRY_ATTEMPTS", config_data.get("prometheus", {}).get("retry_attempts", 3))),
        retry_delay=int(os.getenv("PROMETHEUS_RETRY_DELAY", config_data.get("prometheus", {}).get("retry_delay", 5)))
    )
    
    # Default metrics configuration
    default_metrics = [
        {
            "name": "http_requests",
            "query": "rate(http_requests_total[1m])",
            "step": "60s",
            "range_minutes": 60,
            "prediction_periods": 5,
            "prediction_frequency": "min"
        }
    ]
    
    metrics_config = []
    metrics_data = config_data.get("metrics", default_metrics)
    
    for metric_data in metrics_data:
        metrics_config.append(MetricConfig(
            name=metric_data.get("name"),
            query=metric_data.get("query"),
            step=metric_data.get("step", "60s"),
            range_minutes=metric_data.get("range_minutes", 60),
            prediction_periods=metric_data.get("prediction_periods", 5),
            prediction_frequency=metric_data.get("prediction_frequency", "min"),
            threshold_min=metric_data.get("threshold_min", 0.0),
            threshold_max=metric_data.get("threshold_max")
        ))
    
    # Model configuration
    model_data = config_data.get("model", {})
    model_config = ModelConfig(
        seasonality_mode=model_data.get("seasonality_mode", "additive"),
        changepoint_prior_scale=model_data.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=model_data.get("seasonality_prior_scale", 10.0),
        holidays_prior_scale=model_data.get("holidays_prior_scale", 10.0),
        daily_seasonality=model_data.get("daily_seasonality", True),
        weekly_seasonality=model_data.get("weekly_seasonality", True),
        yearly_seasonality=model_data.get("yearly_seasonality", False),
        interval_width=model_data.get("interval_width", 0.8)
    )
    
    # Cache configuration
    cache_data = config_data.get("cache", {})
    
    # Handle Redis port - Kubernetes sets REDIS_PORT as a URL, we need just the port number
    redis_port_env = os.getenv("REDIS_PORT", cache_data.get("redis_port", 6379))
    if isinstance(redis_port_env, str) and redis_port_env.startswith("tcp://"):
        # Extract port from URL like "tcp://10.100.180.168:6379"
        redis_port = int(redis_port_env.split(":")[-1])
    else:
        redis_port = int(redis_port_env)
    
    cache_config = CacheConfig(
        enabled=os.getenv("CACHE_ENABLED", str(cache_data.get("enabled", True))).lower() == "true",
        redis_host=os.getenv("REDIS_HOST", cache_data.get("redis_host", "redis")),
        redis_port=redis_port,
        redis_db=int(os.getenv("REDIS_DB", cache_data.get("redis_db", 0))),
        ttl_seconds=int(os.getenv("CACHE_TTL", cache_data.get("ttl_seconds", 300))),
        model_cache_ttl=int(os.getenv("MODEL_CACHE_TTL", cache_data.get("model_cache_ttl", 3600)))
    )
    
    return AppConfig(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        prometheus=prometheus_config,
        metrics=metrics_config,
        model=model_config,
        cache=cache_config,
        data_quality_min_points=int(os.getenv("DATA_QUALITY_MIN_POINTS", 10)),
        model_retrain_interval=int(os.getenv("MODEL_RETRAIN_INTERVAL", 1800))
    )
