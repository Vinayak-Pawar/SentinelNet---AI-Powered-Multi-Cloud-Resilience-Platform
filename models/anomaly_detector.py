#!/usr/bin/env python3
"""
SentinelNet Anomaly Detector
Statistical and ML-based anomaly detection for cloud service monitoring

Author: Vinayak Pawar
Version: 1.0
Compatible with: M1 Pro MacBook Pro (Apple Silicon)

This module provides:
- Statistical anomaly detection using Z-score and IQR methods
- Time series analysis for latency and error patterns
- Service-specific anomaly thresholds
- Confidence scoring and false positive reduction
- Integration with monitoring agents
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Local imports
try:
    from data.processor import MetricData
except ImportError:
    from ..data.processor import MetricData

# Configure logging
logger = logging.getLogger(__name__)

class AnomalySeverity(Enum):
    """Severity levels for detected anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectionMethod(Enum):
    """Anomaly detection methods available"""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    MOVING_AVERAGE = "moving_average"
    STATISTICAL = "statistical"

@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    severity: AnomalySeverity
    confidence: float  # 0.0 to 1.0
    method_used: DetectionMethod
    detected_value: float
    expected_range: Tuple[float, float]
    z_score: Optional[float] = None
    description: str = ""
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

@dataclass
class ServiceThresholds:
    """Service-specific anomaly detection thresholds"""
    service_name: str
    cloud_provider: str
    latency_warning_ms: float = 200.0
    latency_critical_ms: float = 1000.0
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.20  # 20%
    min_samples_for_baseline: int = 50
    baseline_window_hours: int = 24
    z_score_threshold: float = 3.0
    seasonal_adjustment: bool = True

@dataclass
class DetectionMetrics:
    """Metrics for evaluating anomaly detection performance"""
    total_samples: int = 0
    anomalies_detected: int = 0
    false_positives: int = 0
    true_positives: int = 0
    detection_accuracy: float = 0.0
    average_confidence: float = 0.0
    last_updated: datetime = None

class AnomalyDetector:
    """
    Multi-method anomaly detection system for SentinelNet
    Uses statistical methods and machine learning for robust detection
    """

    def __init__(self):
        """Initialize the anomaly detector"""
        # Default service thresholds
        self.service_thresholds = self._load_default_thresholds()

        # Model storage
        self.models_dir = "models/anomaly_models"
        os.makedirs(self.models_dir, exist_ok=True)

        # Detection metrics tracking
        self.detection_metrics: Dict[str, DetectionMetrics] = {}

        # Baseline data storage
        self.baseline_data: Dict[str, List[float]] = {}

        # Scalers for ML models
        self.scalers: Dict[str, StandardScaler] = {}

        logger.info("🔍 Anomaly Detector initialized")

    def _load_default_thresholds(self) -> Dict[str, ServiceThresholds]:
        """Load default thresholds for different services"""
        return {
            "bigquery": ServiceThresholds(
                service_name="BigQuery",
                cloud_provider="GCP",
                latency_warning_ms=500.0,
                latency_critical_ms=2000.0,
                error_rate_warning=0.02,
                error_rate_critical=0.10,
                z_score_threshold=2.5
            ),
            "vertex_ai": ServiceThresholds(
                service_name="Vertex AI",
                cloud_provider="GCP",
                latency_warning_ms=300.0,
                latency_critical_ms=1500.0,
                error_rate_warning=0.05,
                error_rate_critical=0.25,
                z_score_threshold=3.0
            ),
            "blob_storage": ServiceThresholds(
                service_name="Blob Storage",
                cloud_provider="Azure",
                latency_warning_ms=200.0,
                latency_critical_ms=1000.0,
                error_rate_warning=0.03,
                error_rate_critical=0.15,
                z_score_threshold=2.8
            ),
            "devops": ServiceThresholds(
                service_name="DevOps",
                cloud_provider="Azure",
                latency_warning_ms=1000.0,
                latency_critical_ms=5000.0,
                error_rate_warning=0.10,
                error_rate_critical=0.40,
                z_score_threshold=2.5
            )
        }

    def get_service_thresholds(self, service_name: str, cloud_provider: str) -> ServiceThresholds:
        """
        Get thresholds for a specific service

        Args:
            service_name: Name of the service
            cloud_provider: Cloud provider

        Returns:
            Service thresholds
        """
        key = f"{cloud_provider}_{service_name}".lower()
        return self.service_thresholds.get(key, ServiceThresholds(service_name, cloud_provider))

    async def detect_anomaly(self, metric_data: List[MetricData],
                           detection_method: DetectionMethod = DetectionMethod.STATISTICAL) -> Optional[AnomalyResult]:
        """
        Detect anomalies in metric data

        Args:
            metric_data: List of metric data points
            detection_method: Method to use for detection

        Returns:
            Anomaly result or None if no anomaly detected
        """
        if not metric_data:
            return None

        try:
            # Extract values and timestamps
            values = [m.value for m in metric_data]
            timestamps = [m.timestamp for m in metric_data]

            if len(values) < 10:  # Need minimum samples
                return None

            # Get service thresholds
            latest_metric = metric_data[-1]
            thresholds = self.get_service_thresholds(
                latest_metric.service_name,
                latest_metric.cloud_provider
            )

            # Choose detection method
            if detection_method == DetectionMethod.Z_SCORE:
                result = self._detect_z_score_anomaly(values, thresholds)
            elif detection_method == DetectionMethod.IQR:
                result = self._detect_iqr_anomaly(values, thresholds)
            elif detection_method == DetectionMethod.ISOLATION_FOREST:
                result = await self._detect_isolation_forest_anomaly(values, thresholds, latest_metric)
            elif detection_method == DetectionMethod.MOVING_AVERAGE:
                result = self._detect_moving_average_anomaly(values, thresholds)
            else:  # STATISTICAL - try multiple methods
                result = self._detect_statistical_anomaly(values, thresholds, latest_metric)

            if result and result.is_anomaly:
                result.timestamp = latest_metric.timestamp
                result.metadata = {
                    'service': latest_metric.service_name,
                    'cloud': latest_metric.cloud_provider,
                    'metric': latest_metric.metric_name,
                    'region': latest_metric.region,
                    'samples_used': len(values)
                }

                # Update detection metrics
                self._update_detection_metrics(latest_metric.service_name, result)

            return result

        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
            return None

    def _detect_z_score_anomaly(self, values: List[float], thresholds: ServiceThresholds) -> Optional[AnomalyResult]:
        """Detect anomalies using Z-score method"""
        try:
            # Calculate mean and standard deviation
            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val == 0:  # No variation
                return None

            # Check the latest value
            latest_value = values[-1]
            z_score = abs((latest_value - mean_val) / std_val)

            is_anomaly = z_score > thresholds.z_score_threshold

            if is_anomaly:
                # Determine severity based on z-score
                if z_score > 5.0:
                    severity = AnomalySeverity.CRITICAL
                    confidence = min(z_score / 6.0, 1.0)
                elif z_score > 4.0:
                    severity = AnomalySeverity.HIGH
                    confidence = min(z_score / 5.0, 1.0)
                elif z_score > 3.0:
                    severity = AnomalySeverity.MEDIUM
                    confidence = min(z_score / 4.0, 1.0)
                else:
                    severity = AnomalySeverity.LOW
                    confidence = min(z_score / 3.5, 1.0)

                expected_range = (mean_val - 2*std_val, mean_val + 2*std_val)

                return AnomalyResult(
                    is_anomaly=True,
                    severity=severity,
                    confidence=confidence,
                    method_used=DetectionMethod.Z_SCORE,
                    detected_value=latest_value,
                    expected_range=expected_range,
                    z_score=z_score,
                    description=f"Z-score: {z_score:.2f} (threshold: {thresholds.z_score_threshold})"
                )

            return None

        except Exception as e:
            logger.error(f"❌ Z-score detection failed: {e}")
            return None

    def _detect_iqr_anomaly(self, values: List[float], thresholds: ServiceThresholds) -> Optional[AnomalyResult]:
        """Detect anomalies using Interquartile Range method"""
        try:
            # Calculate IQR
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            if iqr == 0:  # No variation
                return None

            # Calculate bounds (1.5 * IQR rule)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            latest_value = values[-1]
            is_anomaly = latest_value < lower_bound or latest_value > upper_bound

            if is_anomaly:
                # Determine severity based on deviation
                if latest_value < lower_bound:
                    deviation = (lower_bound - latest_value) / iqr if iqr > 0 else float('inf')
                else:
                    deviation = (latest_value - upper_bound) / iqr if iqr > 0 else float('inf')

                if deviation > 3.0:
                    severity = AnomalySeverity.CRITICAL
                    confidence = min(deviation / 4.0, 1.0)
                elif deviation > 2.0:
                    severity = AnomalySeverity.HIGH
                    confidence = min(deviation / 3.0, 1.0)
                elif deviation > 1.0:
                    severity = AnomalySeverity.MEDIUM
                    confidence = min(deviation / 2.0, 1.0)
                else:
                    severity = AnomalySeverity.LOW
                    confidence = 0.6

                return AnomalyResult(
                    is_anomaly=True,
                    severity=severity,
                    confidence=confidence,
                    method_used=DetectionMethod.IQR,
                    detected_value=latest_value,
                    expected_range=(lower_bound, upper_bound),
                    description=f"IQR outlier: {latest_value:.2f} outside [{lower_bound:.2f}, {upper_bound:.2f}]"
                )

            return None

        except Exception as e:
            logger.error(f"❌ IQR detection failed: {e}")
            return None

    async def _detect_isolation_forest_anomaly(self, values: List[float],
                                             thresholds: ServiceThresholds,
                                             metric: MetricData) -> Optional[AnomalyResult]:
        """Detect anomalies using Isolation Forest"""
        try:
            model_key = f"{metric.cloud_provider}_{metric.service_name}_{metric.metric_name}"

            # Prepare data for ML model
            data = np.array(values).reshape(-1, 1)

            # Get or create scaler
            if model_key not in self.scalers:
                self.scalers[model_key] = StandardScaler()

            scaled_data = self.scalers[model_key].fit_transform(data)

            # Get or create model
            model_path = os.path.join(self.models_dir, f"{model_key}_isolation_forest.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
            else:
                # Create new model
                model = IsolationForest(
                    n_estimators=100,
                    contamination=0.1,  # Expected proportion of anomalies
                    random_state=42
                )
                model.fit(scaled_data[:-1])  # Train on all but latest
                joblib.dump(model, model_path)

            # Predict on latest value
            latest_scaled = scaled_data[-1].reshape(1, -1)
            prediction = model.predict(latest_scaled)

            is_anomaly = prediction[0] == -1  # -1 indicates anomaly

            if is_anomaly:
                # Calculate anomaly score
                scores = model.decision_function(latest_scaled)
                anomaly_score = -scores[0]  # Convert to positive score

                # Determine severity and confidence
                if anomaly_score > 0.6:
                    severity = AnomalySeverity.CRITICAL
                    confidence = min(anomaly_score * 1.2, 1.0)
                elif anomaly_score > 0.4:
                    severity = AnomalySeverity.HIGH
                    confidence = min(anomaly_score * 1.5, 1.0)
                elif anomaly_score > 0.2:
                    severity = AnomalySeverity.MEDIUM
                    confidence = anomaly_score * 2.0
                else:
                    severity = AnomalySeverity.LOW
                    confidence = anomaly_score * 2.5

                return AnomalyResult(
                    is_anomaly=True,
                    severity=severity,
                    confidence=min(confidence, 1.0),
                    method_used=DetectionMethod.ISOLATION_FOREST,
                    detected_value=values[-1],
                    expected_range=(np.mean(values[:-1]) - np.std(values[:-1]),
                                  np.mean(values[:-1]) + np.std(values[:-1])),
                    description=f"Isolation Forest anomaly score: {anomaly_score:.3f}"
                )

            return None

        except Exception as e:
            logger.error(f"❌ Isolation Forest detection failed: {e}")
            return None

    def _detect_moving_average_anomaly(self, values: List[float], thresholds: ServiceThresholds) -> Optional[AnomalyResult]:
        """Detect anomalies using moving average comparison"""
        try:
            if len(values) < 20:  # Need sufficient data
                return None

            # Calculate moving averages
            short_window = 5
            long_window = 20

            short_ma = np.convolve(values, np.ones(short_window)/short_window, mode='valid')[-1]
            long_ma = np.convolve(values, np.ones(long_window)/long_window, mode='valid')[-1]

            latest_value = values[-1]

            # Check for significant deviation from trend
            deviation = abs(latest_value - long_ma) / long_ma if long_ma > 0 else 0

            # Also check rate of change
            if len(values) >= short_window + 1:
                recent_trend = np.polyfit(range(short_window), values[-short_window:], 1)[0]
                rate_change = abs(recent_trend) / np.mean(values[-short_window:]) if np.mean(values[-short_window:]) > 0 else 0
            else:
                rate_change = 0

            # Determine if anomaly based on deviation and rate change
            is_anomaly = deviation > 0.5 or rate_change > 0.3  # 50% deviation or 30% rate change

            if is_anomaly:
                severity = AnomalySeverity.HIGH if deviation > 1.0 or rate_change > 0.5 else AnomalySeverity.MEDIUM
                confidence = min((deviation + rate_change) / 1.5, 1.0)

                return AnomalyResult(
                    is_anomaly=True,
                    severity=severity,
                    confidence=confidence,
                    method_used=DetectionMethod.MOVING_AVERAGE,
                    detected_value=latest_value,
                    expected_range=(long_ma * 0.7, long_ma * 1.3),
                    description=f"Moving average deviation: {deviation:.2%}, rate change: {rate_change:.2%}"
                )

            return None

        except Exception as e:
            logger.error(f"❌ Moving average detection failed: {e}")
            return None

    def _detect_statistical_anomaly(self, values: List[float], thresholds: ServiceThresholds,
                                  metric: MetricData) -> Optional[AnomalyResult]:
        """Use multiple statistical methods for robust anomaly detection"""
        try:
            results = []

            # Try Z-score method
            z_result = self._detect_z_score_anomaly(values, thresholds)
            if z_result:
                results.append(z_result)

            # Try IQR method
            iqr_result = self._detect_iqr_anomaly(values, thresholds)
            if iqr_result:
                results.append(iqr_result)

            # Try moving average
            ma_result = self._detect_moving_average_anomaly(values, thresholds)
            if ma_result:
                results.append(ma_result)

            if not results:
                return None

            # Combine results - require consensus for anomaly
            anomaly_count = sum(1 for r in results if r.is_anomaly)

            if anomaly_count >= 2:  # Majority vote
                # Use the result with highest confidence
                best_result = max(results, key=lambda r: r.confidence)

                # Adjust severity based on consensus
                if anomaly_count == len(results):  # All methods agree
                    best_result.severity = AnomalySeverity(min(
                        AnomalySeverity.CRITICAL.value,
                        best_result.severity.value + 1
                    ))

                best_result.confidence = min(best_result.confidence * 1.2, 1.0)  # Boost confidence
                best_result.method_used = DetectionMethod.STATISTICAL
                best_result.description = f"Consensus anomaly ({anomaly_count}/{len(results)} methods)"

                return best_result

            return None

        except Exception as e:
            logger.error(f"❌ Statistical anomaly detection failed: {e}")
            return None

    def _update_detection_metrics(self, service_name: str, result: AnomalyResult):
        """Update detection performance metrics"""
        if service_name not in self.detection_metrics:
            self.detection_metrics[service_name] = DetectionMetrics()

        metrics = self.detection_metrics[service_name]
        metrics.total_samples += 1
        metrics.anomalies_detected += 1
        metrics.average_confidence = (
            (metrics.average_confidence * (metrics.anomalies_detected - 1)) +
            result.confidence
        ) / metrics.anomalies_detected
        metrics.last_updated = datetime.now()

    def get_detection_metrics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get anomaly detection metrics

        Args:
            service_name: Specific service name or None for all

        Returns:
            Detection metrics
        """
        if service_name:
            return self.detection_metrics.get(service_name, DetectionMetrics()).__dict__
        else:
            return {name: metrics.__dict__ for name, metrics in self.detection_metrics.items()}

    def update_service_thresholds(self, service_name: str, cloud_provider: str,
                                new_thresholds: Dict[str, Any]):
        """
        Update thresholds for a specific service

        Args:
            service_name: Name of the service
            cloud_provider: Cloud provider
            new_thresholds: New threshold values
        """
        key = f"{cloud_provider}_{service_name}".lower()

        if key not in self.service_thresholds:
            self.service_thresholds[key] = ServiceThresholds(service_name, cloud_provider)

        thresholds = self.service_thresholds[key]

        # Update provided values
        for attr, value in new_thresholds.items():
            if hasattr(thresholds, attr):
                setattr(thresholds, attr, value)

        logger.info(f"✅ Updated thresholds for {service_name} ({cloud_provider})")

    async def analyze_service_health_trend(self, service_name: str, cloud_provider: str,
                                         metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze health trends for a service

        Args:
            service_name: Name of the service
            cloud_provider: Cloud provider
            metric_name: Name of the metric
            hours: Hours to analyze

        Returns:
            Trend analysis results
        """
        try:
            # This would integrate with data processor to get metrics
            # For now, return basic structure
            return {
                'service': service_name,
                'cloud': cloud_provider,
                'metric': metric_name,
                'trend': 'stable',  # increasing, decreasing, stable
                'volatility': 0.0,
                'baseline_mean': 0.0,
                'baseline_std': 0.0,
                'analysis_period_hours': hours,
                'data_points': 0,
                'last_analyzed': datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Trend analysis failed: {e}")
            return {}

# Global anomaly detector instance
_anomaly_detector_instance: Optional[AnomalyDetector] = None

def get_anomaly_detector() -> AnomalyDetector:
    """Get the global anomaly detector instance"""
    global _anomaly_detector_instance
    if _anomaly_detector_instance is None:
        _anomaly_detector_instance = AnomalyDetector()
    return _anomaly_detector_instance

# Convenience functions
async def detect_metric_anomaly(metric_data: List[MetricData],
                              method: DetectionMethod = DetectionMethod.STATISTICAL) -> Optional[AnomalyResult]:
    """Detect anomalies in metric data"""
    detector = get_anomaly_detector()
    return await detector.detect_anomaly(metric_data, method)

def get_service_thresholds(service_name: str, cloud_provider: str) -> ServiceThresholds:
    """Get thresholds for a service"""
    detector = get_anomaly_detector()
    return detector.get_service_thresholds(service_name, cloud_provider)

if __name__ == "__main__":
    # Test the anomaly detector
    async def test_anomaly_detector():
        detector = get_anomaly_detector()

        # Create test metric data with an anomaly
        base_time = datetime.now() - timedelta(hours=1)
        test_data = []

        # Normal data
        for i in range(50):
            value = 100 + np.random.normal(0, 10)  # Normal around 100ms
            test_data.append(MetricData(
                timestamp=base_time + timedelta(minutes=i),
                service_name="BigQuery",
                cloud_provider="GCP",
                metric_name="latency",
                value=value,
                unit="ms"
            ))

        # Add anomaly
        test_data.append(MetricData(
            timestamp=base_time + timedelta(minutes=51),
            service_name="BigQuery",
            cloud_provider="GCP",
            metric_name="latency",
            value=500.0,  # Anomalous value
            unit="ms"
        ))

        # Detect anomaly
        result = await detector.detect_anomaly(test_data)
        if result:
            print(f"✅ Anomaly detected: {result.is_anomaly}")
            print(f"Severity: {result.severity.value}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Method: {result.method_used.value}")
            print(f"Description: {result.description}")
        else:
            print("❌ No anomaly detected")

        # Get thresholds
        thresholds = detector.get_service_thresholds("BigQuery", "GCP")
        print(f"BigQuery latency threshold: {thresholds.latency_warning_ms}ms")

    asyncio.run(test_anomaly_detector())
