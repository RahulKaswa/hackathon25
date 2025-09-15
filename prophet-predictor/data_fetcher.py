"""Data fetching and processing utilities."""
import logging
import time
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np

from config import PrometheusConfig, MetricConfig

logger = logging.getLogger(__name__)


class PrometheusClient:
    """Enhanced Prometheus client with retry logic and error handling."""
    
    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=config.retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def query_range(self, query: str, start: int, end: int, step: str) -> Optional[Dict]:
        """Execute a range query against Prometheus with error handling."""
        url = f"{self.config.url}/api/v1/query_range"
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step
        }
        
        try:
            logger.debug(f"Querying Prometheus: {query} from {start} to {end}")
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") != "success":
                logger.error(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
                return None
            
            return data.get("data", {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error querying Prometheus: {e}")
            return None


class DataProcessor:
    """Data processing and validation utilities."""
    
    @staticmethod
    def validate_and_clean_data(df: pd.DataFrame, min_points: int = 10) -> Tuple[pd.DataFrame, bool]:
        """Validate and clean the input data."""
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df, False
        
        # Check for minimum data points
        if len(df) < min_points:
            logger.warning(f"Insufficient data points: {len(df)} < {min_points}")
            return df, False
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ds'])
        
        # Sort by timestamp
        df = df.sort_values('ds')
        
        # Handle missing values
        if df['y'].isna().any():
            logger.warning("Found NaN values, forward filling")
            df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using IQR method
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df['y'] < lower_bound) | (df['y'] > upper_bound)
        if outliers_mask.any():
            logger.info(f"Removing {outliers_mask.sum()} outliers")
            df.loc[outliers_mask, 'y'] = np.nan
            df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')
        
        # Ensure non-negative values
        df['y'] = df['y'].clip(lower=0)
        
        return df, True
    
    @staticmethod
    def prometheus_to_dataframe(prometheus_data: Dict, metric_config: MetricConfig) -> pd.DataFrame:
        """Convert Prometheus query result to pandas DataFrame."""
        result = prometheus_data.get("result", [])
        
        if not result:
            logger.warning(f"No data found for metric {metric_config.name}")
            return pd.DataFrame(columns=["ds", "y"])
        
        # Handle multiple series - aggregate them
        all_values = []
        for series in result:
            values = series.get("values", [])
            all_values.extend(values)
        
        if not all_values:
            return pd.DataFrame(columns=["ds", "y"])
        
        # Convert to DataFrame
        df = pd.DataFrame(all_values, columns=["ds", "y"])
        
        # Convert timestamp to datetime
        df["ds"] = pd.to_datetime(df["ds"], unit="s")
        
        # Convert values to float
        df["y"] = pd.to_numeric(df["y"], errors='coerce')
        
        # If we have multiple series, aggregate by timestamp
        if len(result) > 1:
            df = df.groupby("ds")["y"].sum().reset_index()
        
        return df


class DataFetcher:
    """Main data fetching orchestrator."""
    
    def __init__(self, prometheus_client: PrometheusClient):
        self.prometheus_client = prometheus_client
        self.data_processor = DataProcessor()
    
    def get_metric_data(self, metric_config: MetricConfig, min_points: int = 10) -> Tuple[pd.DataFrame, bool]:
        """Fetch and process metric data from Prometheus."""
        end_time = int(time.time())
        start_time = end_time - (metric_config.range_minutes * 60)
        
        logger.info(f"Fetching data for metric: {metric_config.name}")
        
        # Query Prometheus
        prometheus_data = self.prometheus_client.query_range(
            query=metric_config.query,
            start=start_time,
            end=end_time,
            step=metric_config.step
        )
        
        if prometheus_data is None:
            logger.error(f"Failed to fetch data for metric: {metric_config.name}")
            return pd.DataFrame(columns=["ds", "y"]), False
        
        # Convert to DataFrame
        df = self.data_processor.prometheus_to_dataframe(prometheus_data, metric_config)
        
        # Validate and clean data
        df, is_valid = self.data_processor.validate_and_clean_data(df, min_points)
        
        if is_valid:
            logger.info(f"Successfully processed {len(df)} data points for metric: {metric_config.name}")
        else:
            logger.warning(f"Data validation failed for metric: {metric_config.name}")
        
        return df, is_valid
