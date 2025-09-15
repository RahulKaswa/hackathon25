"""Advanced prediction models with multiple algorithms support."""
import logging
import pickle
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

from config import MetricConfig, ModelConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')


class BasePredictorModel(ABC):
    """Base class for all prediction models."""
    
    def __init__(self, metric_config: MetricConfig, model_config: ModelConfig):
        self.metric_config = metric_config
        self.model_config = model_config
        self.model = None
        self.last_trained = None
        self.training_data_hash = None
    
    @abstractmethod
    def train(self, df: pd.DataFrame) -> bool:
        """Train the model with historical data."""
        pass
    
    @abstractmethod
    def predict(self, periods: int) -> Dict[str, Any]:
        """Generate predictions."""
        pass
    
    def should_retrain(self, df: pd.DataFrame, retrain_interval: int) -> bool:
        """Check if model should be retrained."""
        if self.model is None or self.last_trained is None:
            return True
        
        # Check time-based retraining
        if time.time() - self.last_trained > retrain_interval:
            logger.info("Model retrain triggered by time interval")
            return True
        
        # Check data change-based retraining
        current_hash = hash(df.to_string())
        if current_hash != self.training_data_hash:
            logger.info("Model retrain triggered by data change")
            return True
        
        return False
    
    def serialize(self) -> bytes:
        """Serialize the model for caching."""
        return pickle.dumps({
            'model': self.model,
            'last_trained': self.last_trained,
            'training_data_hash': self.training_data_hash,
            'metric_config': self.metric_config,
            'model_config': self.model_config
        })
    
    @classmethod
    def deserialize(cls, data: bytes, metric_config: MetricConfig, model_config: ModelConfig):
        """Deserialize the model from cache."""
        try:
            obj_data = pickle.loads(data)
            instance = cls(metric_config, model_config)
            instance.model = obj_data['model']
            instance.last_trained = obj_data['last_trained']
            instance.training_data_hash = obj_data['training_data_hash']
            return instance
        except Exception as e:
            logger.error(f"Failed to deserialize model: {e}")
            return None


class ProphetPredictor(BasePredictorModel):
    """Facebook Prophet-based predictor."""
    
    def train(self, df: pd.DataFrame) -> bool:
        """Train Prophet model."""
        try:
            logger.info("Training Prophet model...")
            
            # Initialize Prophet with configuration
            self.model = Prophet(
                seasonality_mode=self.model_config.seasonality_mode,
                changepoint_prior_scale=self.model_config.changepoint_prior_scale,
                seasonality_prior_scale=self.model_config.seasonality_prior_scale,
                holidays_prior_scale=self.model_config.holidays_prior_scale,
                daily_seasonality=self.model_config.daily_seasonality,
                weekly_seasonality=self.model_config.weekly_seasonality,
                yearly_seasonality=self.model_config.yearly_seasonality,
                interval_width=self.model_config.interval_width
            )
            
            # Add custom seasonalities if needed
            if len(df) > 24:  # At least 24 hours of data
                self.model.add_seasonality(name='hourly', period=1, fourier_order=5)
            
            self.model.fit(df)
            self.last_trained = time.time()
            self.training_data_hash = hash(df.to_string())
            
            logger.info("Prophet model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train Prophet model: {e}")
            return False
    
    def predict(self, periods: int = None) -> Dict[str, Any]:
        """Generate predictions using Prophet."""
        if self.model is None:
            return {"error": "Model not trained"}
        
        try:
            periods = periods or self.metric_config.prediction_periods
            
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods,
                freq=self.metric_config.prediction_frequency
            )
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Extract predictions
            latest_forecast = forecast.iloc[-periods:].copy()
            
            # Apply thresholds
            predictions = latest_forecast['yhat'].clip(
                lower=self.metric_config.threshold_min,
                upper=self.metric_config.threshold_max
            ).tolist()
            
            # Calculate confidence intervals
            lower_bounds = latest_forecast['yhat_lower'].tolist()
            upper_bounds = latest_forecast['yhat_upper'].tolist()
            timestamps = latest_forecast['ds'].apply(lambda x: x.isoformat()).tolist()
            
            return {
                "algorithm": "prophet",
                "predictions": predictions,
                "timestamps": timestamps,
                "confidence_lower": lower_bounds,
                "confidence_upper": upper_bounds,
                "next_value": max(self.metric_config.threshold_min, predictions[0]),
                "peak_value": max(predictions),
                "trend": self._calculate_trend(predictions)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate Prophet predictions: {e}")
            return {"error": str(e)}
    
    def _calculate_trend(self, predictions: list) -> str:
        """Calculate trend direction."""
        if len(predictions) < 2:
            return "stable"
        
        slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"


class LinearRegressionPredictor(BasePredictorModel):
    """Linear regression-based predictor for simpler patterns."""
    
    def __init__(self, metric_config: MetricConfig, model_config: ModelConfig):
        super().__init__(metric_config, model_config)
        self.scaler = MinMaxScaler()
        self.window_size = 12  # Use 12 time steps for prediction
    
    def train(self, df: pd.DataFrame) -> bool:
        """Train linear regression model."""
        try:
            if len(df) < self.window_size + 1:
                logger.warning("Insufficient data for linear regression training")
                return False
            
            # Prepare features and targets
            X, y = self._prepare_data(df['y'].values)
            
            if len(X) == 0:
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_scaled, y)
            
            self.last_trained = time.time()
            self.training_data_hash = hash(df.to_string())
            
            logger.info("Linear regression model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train linear regression model: {e}")
            return False
    
    def predict(self, periods: int = None) -> Dict[str, Any]:
        """Generate predictions using linear regression."""
        if self.model is None:
            return {"error": "Model not trained"}
        
        try:
            periods = periods or self.metric_config.prediction_periods
            
            # Use last window_size points for prediction
            # This would need the last training data - simplified for demo
            predictions = [max(0, self.model.predict([[i]])[0]) for i in range(periods)]
            
            return {
                "algorithm": "linear_regression",
                "predictions": predictions,
                "next_value": max(self.metric_config.threshold_min, predictions[0]),
                "peak_value": max(predictions),
                "trend": self._calculate_trend(predictions)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate linear regression predictions: {e}")
            return {"error": str(e)}
    
    def _prepare_data(self, values):
        """Prepare sliding window data for training."""
        X, y = [], []
        for i in range(self.window_size, len(values)):
            X.append(values[i-self.window_size:i])
            y.append(values[i])
        return np.array(X), np.array(y)
    
    def _calculate_trend(self, predictions: list) -> str:
        """Calculate trend direction."""
        if len(predictions) < 2:
            return "stable"
        
        slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"


class EnsemblePredictor(BasePredictorModel):
    """Ensemble predictor combining multiple algorithms."""
    
    def __init__(self, metric_config: MetricConfig, model_config: ModelConfig):
        super().__init__(metric_config, model_config)
        self.predictors = {
            'prophet': ProphetPredictor(metric_config, model_config),
            'linear': LinearRegressionPredictor(metric_config, model_config)
        }
        self.weights = {'prophet': 0.7, 'linear': 0.3}
    
    def train(self, df: pd.DataFrame) -> bool:
        """Train all ensemble models."""
        success_count = 0
        
        for name, predictor in self.predictors.items():
            try:
                if predictor.train(df):
                    success_count += 1
                    logger.info(f"Successfully trained {name} predictor")
                else:
                    logger.warning(f"Failed to train {name} predictor")
            except Exception as e:
                logger.error(f"Error training {name} predictor: {e}")
        
        if success_count > 0:
            self.last_trained = time.time()
            self.training_data_hash = hash(df.to_string())
            return True
        
        return False
    
    def predict(self, periods: int = None) -> Dict[str, Any]:
        """Generate ensemble predictions."""
        periods = periods or self.metric_config.prediction_periods
        predictions_dict = {}
        valid_predictions = []
        
        # Get predictions from all models
        for name, predictor in self.predictors.items():
            try:
                pred_result = predictor.predict(periods)
                if "error" not in pred_result:
                    predictions_dict[name] = pred_result
                    valid_predictions.append(name)
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {e}")
        
        if not valid_predictions:
            return {"error": "No valid predictions from ensemble models"}
        
        # Combine predictions using weights
        try:
            ensemble_predictions = []
            for i in range(periods):
                weighted_sum = 0
                total_weight = 0
                
                for name in valid_predictions:
                    if i < len(predictions_dict[name]["predictions"]):
                        weight = self.weights.get(name, 1.0 / len(valid_predictions))
                        weighted_sum += predictions_dict[name]["predictions"][i] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    ensemble_predictions.append(weighted_sum / total_weight)
                else:
                    ensemble_predictions.append(0)
            
            return {
                "algorithm": "ensemble",
                "predictions": ensemble_predictions,
                "individual_predictions": predictions_dict,
                "next_value": max(self.metric_config.threshold_min, ensemble_predictions[0]),
                "peak_value": max(ensemble_predictions),
                "trend": self._calculate_trend(ensemble_predictions),
                "models_used": valid_predictions
            }
            
        except Exception as e:
            logger.error(f"Failed to combine ensemble predictions: {e}")
            return {"error": str(e)}
    
    def _calculate_trend(self, predictions: list) -> str:
        """Calculate trend direction."""
        if len(predictions) < 2:
            return "stable"
        
        slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"


class ModelFactory:
    """Factory for creating prediction models."""
    
    @staticmethod
    def create_model(model_type: str, metric_config: MetricConfig, model_config: ModelConfig) -> BasePredictorModel:
        """Create a prediction model of the specified type."""
        model_type = model_type.lower()
        
        if model_type == "prophet":
            return ProphetPredictor(metric_config, model_config)
        elif model_type == "linear":
            return LinearRegressionPredictor(metric_config, model_config)
        elif model_type == "ensemble":
            return EnsemblePredictor(metric_config, model_config)
        else:
            logger.warning(f"Unknown model type: {model_type}, defaulting to Prophet")
            return ProphetPredictor(metric_config, model_config)
