#!/usr/bin/env python3
"""
SentinelNet Data Processor
Handles data processing, storage, and analysis for SentinelNet

Author: Vinayak Pawar
Version: 1.0
Compatible with: M1 Pro MacBook Pro (Apple Silicon)

This module provides:
- Time series data processing for service metrics
- Cross-cloud correlation analysis
- Data persistence and retrieval
- Anomaly detection preprocessing
- Service dependency mapping
"""

import asyncio
import logging
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MetricData:
    """Data class for time series metrics"""
    timestamp: datetime
    service_name: str
    cloud_provider: str
    metric_name: str
    value: float
    unit: str
    region: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ServiceDependency:
    """Data class for service dependencies"""
    service_a: str
    service_b: str
    dependency_type: str  # "reads_from", "writes_to", "depends_on"
    cloud_provider: str
    region: Optional[str] = None
    criticality: str = "medium"  # "low", "medium", "high", "critical"
    last_updated: datetime = None

@dataclass
class CorrelationResult:
    """Data class for correlation analysis results"""
    service_pair: Tuple[str, str]
    correlation_coefficient: float
    time_window_minutes: int
    confidence_level: float
    incident_count: int
    analysis_timestamp: datetime

class DataProcessor:
    """
    Main data processing class for SentinelNet
    Handles all data operations and analysis
    """

    def __init__(self, db_path: str = "data/sentinelnet.db"):
        """
        Initialize the data processor

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety for SQLite
        self.db_lock = threading.Lock()

        # Initialize database
        self._init_database()

        # In-memory caches for performance
        self.metrics_cache: Dict[str, List[MetricData]] = {}
        self.dependencies_cache: Dict[str, List[ServiceDependency]] = {}
        self.cache_size_limit = 1000  # Max items per cache

        # Analysis parameters
        self.correlation_window_hours = 24
        self.anomaly_detection_window_minutes = 60

        logger.info(f"📊 Data Processor initialized with database: {db_path}")

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Metrics table for time series data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    service_name TEXT NOT NULL,
                    cloud_provider TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    region TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Service dependencies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS service_dependencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_a TEXT NOT NULL,
                    service_b TEXT NOT NULL,
                    dependency_type TEXT NOT NULL,
                    cloud_provider TEXT NOT NULL,
                    region TEXT,
                    criticality TEXT DEFAULT 'medium',
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Incidents table for historical incident data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT UNIQUE NOT NULL,
                    service_name TEXT NOT NULL,
                    cloud_provider TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolved_at TEXT,
                    description TEXT,
                    impact_analysis TEXT,
                    remediation_plan TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Correlation analysis results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS correlation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_a TEXT NOT NULL,
                    service_b TEXT NOT NULL,
                    correlation_coefficient REAL NOT NULL,
                    time_window_minutes INTEGER NOT NULL,
                    confidence_level REAL NOT NULL,
                    incident_count INTEGER NOT NULL,
                    analysis_timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_service ON metrics(service_name, cloud_provider)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_service ON incidents(service_name, cloud_provider)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dependencies_service_a ON service_dependencies(service_a)')

            conn.commit()
            conn.close()

        logger.info("✅ Database initialized successfully")

    async def store_metric(self, metric: MetricData) -> bool:
        """
        Store a metric data point

        Args:
            metric: Metric data to store

        Returns:
            bool: Success status
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO metrics (timestamp, service_name, cloud_provider,
                                       metric_name, value, unit, region, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.service_name,
                    metric.cloud_provider,
                    metric.metric_name,
                    metric.value,
                    metric.unit,
                    metric.region,
                    json.dumps(metric.metadata) if metric.metadata else None
                ))

                conn.commit()
                conn.close()

            # Update cache
            cache_key = f"{metric.cloud_provider}_{metric.service_name}_{metric.metric_name}"
            if cache_key not in self.metrics_cache:
                self.metrics_cache[cache_key] = []
            self.metrics_cache[cache_key].append(metric)

            # Maintain cache size
            if len(self.metrics_cache[cache_key]) > self.cache_size_limit:
                self.metrics_cache[cache_key] = self.metrics_cache[cache_key][-self.cache_size_limit:]

            return True

        except Exception as e:
            logger.error(f"❌ Failed to store metric: {e}")
            return False

    async def get_metrics(self, service_name: str, cloud_provider: str,
                         metric_name: str, hours: int = 24) -> List[MetricData]:
        """
        Retrieve metrics for a service

        Args:
            service_name: Name of the service
            cloud_provider: Cloud provider (GCP, Azure)
            metric_name: Name of the metric
            hours: Number of hours of historical data

        Returns:
            List of metric data points
        """
        try:
            # Check cache first
            cache_key = f"{cloud_provider}_{service_name}_{metric_name}"
            if cache_key in self.metrics_cache:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                cached_data = [m for m in self.metrics_cache[cache_key]
                             if m.timestamp > cutoff_time]
                if cached_data:
                    return cached_data

            # Query database
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

                cursor.execute('''
                    SELECT timestamp, service_name, cloud_provider, metric_name,
                           value, unit, region, metadata
                    FROM metrics
                    WHERE service_name = ? AND cloud_provider = ? AND metric_name = ?
                      AND timestamp > ?
                    ORDER BY timestamp DESC
                ''', (service_name, cloud_provider, metric_name, cutoff_time))

                rows = cursor.fetchall()
                conn.close()

            # Convert to MetricData objects
            metrics = []
            for row in rows:
                metadata = json.loads(row[7]) if row[7] else None
                metric = MetricData(
                    timestamp=datetime.fromisoformat(row[0]),
                    service_name=row[1],
                    cloud_provider=row[2],
                    metric_name=row[3],
                    value=row[4],
                    unit=row[5],
                    region=row[6],
                    metadata=metadata
                )
                metrics.append(metric)

            # Update cache
            self.metrics_cache[cache_key] = metrics[-self.cache_size_limit:]

            return metrics

        except Exception as e:
            logger.error(f"❌ Failed to retrieve metrics: {e}")
            return []

    async def store_service_dependency(self, dependency: ServiceDependency) -> bool:
        """
        Store a service dependency relationship

        Args:
            dependency: Service dependency to store

        Returns:
            bool: Success status
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO service_dependencies
                    (service_a, service_b, dependency_type, cloud_provider,
                     region, criticality, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    dependency.service_a,
                    dependency.service_b,
                    dependency.dependency_type,
                    dependency.cloud_provider,
                    dependency.region,
                    dependency.criticality,
                    datetime.now().isoformat()
                ))

                conn.commit()
                conn.close()

            # Update cache
            cache_key = dependency.service_a
            if cache_key not in self.dependencies_cache:
                self.dependencies_cache[cache_key] = []
            self.dependencies_cache[cache_key].append(dependency)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to store service dependency: {e}")
            return False

    async def get_service_dependencies(self, service_name: str) -> List[ServiceDependency]:
        """
        Get dependencies for a service

        Args:
            service_name: Name of the service

        Returns:
            List of service dependencies
        """
        try:
            # Check cache first
            if service_name in self.dependencies_cache:
                return self.dependencies_cache[service_name]

            # Query database
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT service_a, service_b, dependency_type, cloud_provider,
                           region, criticality, last_updated
                    FROM service_dependencies
                    WHERE service_a = ? OR service_b = ?
                ''', (service_name, service_name))

                rows = cursor.fetchall()
                conn.close()

            # Convert to ServiceDependency objects
            dependencies = []
            for row in rows:
                dependency = ServiceDependency(
                    service_a=row[0],
                    service_b=row[1],
                    dependency_type=row[2],
                    cloud_provider=row[3],
                    region=row[4],
                    criticality=row[5],
                    last_updated=datetime.fromisoformat(row[6]) if row[6] else None
                )
                dependencies.append(dependency)

            # Update cache
            self.dependencies_cache[service_name] = dependencies

            return dependencies

        except Exception as e:
            logger.error(f"❌ Failed to retrieve service dependencies: {e}")
            return []

    async def analyze_service_correlations(self, hours: int = 24) -> List[CorrelationResult]:
        """
        Analyze correlations between service metrics

        Args:
            hours: Number of hours to analyze

        Returns:
            List of correlation results
        """
        try:
            logger.info(f"🔍 Analyzing service correlations for last {hours} hours...")

            # Get all services with metrics in the time window
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

                cursor.execute('''
                    SELECT DISTINCT service_name, cloud_provider
                    FROM metrics
                    WHERE timestamp > ?
                    ORDER BY service_name
                ''', (cutoff_time,))

                services = cursor.fetchall()
                conn.close()

            correlation_results = []

            # Analyze correlations between all service pairs
            for i, service_a in enumerate(services):
                for service_b in services[i+1:]:
                    if service_a == service_b:
                        continue

                    # Get metrics for both services
                    metrics_a = await self.get_metrics(
                        service_a[0], service_a[1], "latency", hours
                    )
                    metrics_b = await self.get_metrics(
                        service_b[0], service_b[1], "latency", hours
                    )

                    if len(metrics_a) < 10 or len(metrics_b) < 10:
                        continue  # Not enough data for correlation

                    # Calculate correlation
                    values_a = [m.value for m in metrics_a]
                    values_b = [m.value for m in metrics_b]

                    # Align timestamps (simplified approach)
                    min_len = min(len(values_a), len(values_b))
                    correlation = np.corrcoef(values_a[-min_len:], values_b[-min_len:])[0, 1]

                    # Calculate confidence based on data points and correlation strength
                    confidence = min(abs(correlation) * (min_len / 100), 1.0)

                    # Check for incident co-occurrence
                    incidents_a = await self.get_incident_count(service_a[0], service_a[1], hours)
                    incidents_b = await self.get_incident_count(service_b[0], service_b[1], hours)

                    result = CorrelationResult(
                        service_pair=(f"{service_a[0]} ({service_a[1]})",
                                    f"{service_b[0]} ({service_b[1]})"),
                        correlation_coefficient=float(correlation),
                        time_window_minutes=hours * 60,
                        confidence_level=confidence,
                        incident_count=min(incidents_a, incidents_b),  # Co-occurring incidents
                        analysis_timestamp=datetime.now()
                    )

                    correlation_results.append(result)

                    # Store in database
                    await self._store_correlation_result(result)

            logger.info(f"✅ Correlation analysis complete. Found {len(correlation_results)} significant correlations")
            return correlation_results

        except Exception as e:
            logger.error(f"❌ Correlation analysis failed: {e}")
            return []

    async def get_incident_count(self, service_name: str, cloud_provider: str, hours: int) -> int:
        """
        Get incident count for a service in the given time window

        Args:
            service_name: Name of the service
            cloud_provider: Cloud provider
            hours: Time window in hours

        Returns:
            Number of incidents
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

                cursor.execute('''
                    SELECT COUNT(*) FROM incidents
                    WHERE service_name = ? AND cloud_provider = ?
                      AND detected_at > ?
                ''', (service_name, cloud_provider, cutoff_time))

                count = cursor.fetchone()[0]
                conn.close()

                return count

        except Exception as e:
            logger.error(f"❌ Failed to get incident count: {e}")
            return 0

    async def _store_correlation_result(self, result: CorrelationResult) -> bool:
        """Store correlation analysis result"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO correlation_results
                    (service_a, service_b, correlation_coefficient, time_window_minutes,
                     confidence_level, incident_count, analysis_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.service_pair[0],
                    result.service_pair[1],
                    result.correlation_coefficient,
                    result.time_window_minutes,
                    result.confidence_level,
                    result.incident_count,
                    result.analysis_timestamp.isoformat()
                ))

                conn.commit()
                conn.close()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to store correlation result: {e}")
            return False

    async def get_service_impact_analysis(self, service_name: str, cloud_provider: str) -> Dict[str, Any]:
        """
        Perform impact analysis for a service

        Args:
            service_name: Name of the affected service
            cloud_provider: Cloud provider

        Returns:
            Impact analysis results
        """
        try:
            # Get service dependencies
            dependencies = await self.get_service_dependencies(service_name)

            # Get recent metrics
            metrics = await self.get_metrics(service_name, cloud_provider, "latency", 24)

            # Get correlation data
            correlations = await self.analyze_service_correlations(24)

            # Analyze downstream impact
            downstream_services = [
                dep.service_b for dep in dependencies
                if dep.service_a == service_name
            ]

            # Calculate impact score based on dependencies and correlations
            impact_score = len(downstream_services) * 0.2  # Base impact from dependencies

            # Add correlation-based impact
            for corr in correlations:
                if service_name in str(corr.service_pair):
                    impact_score += abs(corr.correlation_coefficient) * corr.confidence_level * 0.3

            impact_score = min(impact_score, 1.0)  # Cap at 1.0

            # Determine impact level
            if impact_score > 0.7:
                impact_level = "critical"
            elif impact_score > 0.4:
                impact_level = "high"
            elif impact_score > 0.2:
                impact_level = "medium"
            else:
                impact_level = "low"

            return {
                'service': service_name,
                'cloud_provider': cloud_provider,
                'impact_score': impact_score,
                'impact_level': impact_level,
                'downstream_services': downstream_services,
                'dependency_count': len(dependencies),
                'metrics_available': len(metrics),
                'last_analyzed': datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Impact analysis failed: {e}")
            return {
                'service': service_name,
                'error': str(e),
                'impact_level': 'unknown'
            }

    async def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Clean up old data to manage database size

        Args:
            days_to_keep: Number of days of data to retain
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Clean up old metrics
                cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_date,))

                # Clean up old correlation results
                cursor.execute('DELETE FROM correlation_results WHERE analysis_timestamp < ?', (cutoff_date,))

                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()

            # Clear caches
            self.metrics_cache.clear()
            self.dependencies_cache.clear()

            logger.info(f"🧹 Cleaned up {deleted_count} old records")

        except Exception as e:
            logger.error(f"❌ Data cleanup failed: {e}")

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with database statistics
        """
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Get table sizes
                stats = {}
                tables = ['metrics', 'service_dependencies', 'incidents', 'correlation_results']

                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]

                # Get database file size
                db_size = Path(self.db_path).stat().st_size
                stats['database_size_mb'] = db_size / (1024 * 1024)

                conn.close()

                return stats

        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {}

# Global data processor instance
_data_processor_instance: Optional[DataProcessor] = None

def get_data_processor() -> DataProcessor:
    """Get the global data processor instance"""
    global _data_processor_instance
    if _data_processor_instance is None:
        _data_processor_instance = DataProcessor()
    return _data_processor_instance

# Convenience functions
async def store_metric(metric: MetricData) -> bool:
    """Store a metric data point"""
    processor = get_data_processor()
    return await processor.store_metric(metric)

async def get_service_metrics(service_name: str, cloud_provider: str,
                            metric_name: str, hours: int = 24) -> List[MetricData]:
    """Get metrics for a service"""
    processor = get_data_processor()
    return await processor.get_metrics(service_name, cloud_provider, metric_name, hours)

async def analyze_correlations(hours: int = 24) -> List[CorrelationResult]:
    """Analyze service correlations"""
    processor = get_data_processor()
    return await processor.analyze_service_correlations(hours)

if __name__ == "__main__":
    # Test the data processor
    async def test_data_processor():
        processor = get_data_processor()

        # Test storing a metric
        metric = MetricData(
            timestamp=datetime.now(),
            service_name="BigQuery",
            cloud_provider="GCP",
            metric_name="latency",
            value=145.5,
            unit="ms",
            region="us-east1"
        )

        success = await processor.store_metric(metric)
        print(f"Metric storage: {'✅ Success' if success else '❌ Failed'}")

        # Test retrieving metrics
        metrics = await processor.get_metrics("BigQuery", "GCP", "latency", 24)
        print(f"Retrieved {len(metrics)} metrics")

        # Get database stats
        stats = processor.get_database_stats()
        print("Database stats:", stats)

    asyncio.run(test_data_processor())
