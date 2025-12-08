#!/usr/bin/env python3
"""
SentinelNet GCP Service Monitor
Monitoring agents for Google Cloud Platform services

Author: Vinayak Pawar
Version: 1.0
Compatible with: M1 Pro MacBook Pro (Apple Silicon)

This module provides monitoring for:
- BigQuery: Dataset accessibility, query performance, storage metrics
- Vertex AI: Endpoint response times, model deployment status
- Cross-service health correlation
- Anomaly detection integration
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import os

# Google Cloud imports (optional - will work without them for demo)
try:
    from google.cloud import bigquery
    from google.cloud import aiplatform
    from google.api_core import exceptions as gcp_exceptions
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    bigquery = None
    aiplatform = None
    gcp_exceptions = None

# Local imports
try:
    from data.processor import DataProcessor, MetricData
    from models.anomaly_detector import AnomalyDetector, AnomalyResult
    from agents.orchestrator import AgentInfo, AgentStatus, ServiceStatus, ServiceHealth
except ImportError:
    from ..data.processor import DataProcessor, MetricData
    from ..models.anomaly_detector import AnomalyDetector, AnomalyResult
    from .orchestrator import AgentInfo, AgentStatus, ServiceStatus, ServiceHealth

# Configure logging
logger = logging.getLogger(__name__)

class GCPService(Enum):
    """GCP services monitored by this agent"""
    BIGQUERY = "bigquery"
    VERTEX_AI = "vertex_ai"

@dataclass
class GCPServiceConfig:
    """Configuration for GCP service monitoring"""
    service: GCPService
    project_id: str
    region: str = "us-central1"
    dataset_id: Optional[str] = None  # For BigQuery
    endpoint_id: Optional[str] = None  # For Vertex AI
    model_id: Optional[str] = None  # For Vertex AI
    check_interval_seconds: int = 60
    timeout_seconds: int = 30

@dataclass
class MonitoringResult:
    """Result of a service health check"""
    service: GCPService
    status: ServiceStatus
    latency_ms: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    timestamp: datetime = None

class GCPMonitorAgent:
    """
    GCP service monitoring agent for SentinelNet
    Monitors BigQuery and Vertex AI services with anomaly detection
    """

    def __init__(self, agent_id: str = "gcp_monitor_001",
                 project_id: Optional[str] = None):
        """
        Initialize the GCP monitoring agent

        Args:
            agent_id: Unique identifier for this agent
            project_id: GCP project ID to monitor
        """
        self.agent_id = agent_id
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')

        # Service configurations
        self.service_configs = self._load_service_configs()

        # Monitoring state
        self.last_checks: Dict[GCPService, datetime] = {}
        self.health_history: Dict[GCPService, List[MonitoringResult]] = {
            service: [] for service in GCPService
        }

        # Components
        self.data_processor = DataProcessor()
        self.anomaly_detector = AnomalyDetector()

        # Agent info for orchestrator
        self.agent_info = AgentInfo(
            agent_id=self.agent_id,
            agent_type="monitor",
            cloud_provider="GCP",
            services=["BigQuery", "Vertex AI"],
            status=AgentStatus.INITIALIZING,
            region="global"
        )

        # GCP clients (initialized when needed)
        self.bigquery_client = None
        self.vertexai_client = None

        logger.info(f"🔍 GCP Monitor Agent {agent_id} initialized for project: {self.project_id}")

    def _load_service_configs(self) -> Dict[GCPService, GCPServiceConfig]:
        """Load service monitoring configurations"""
        return {
            GCPService.BIGQUERY: GCPServiceConfig(
                service=GCPService.BIGQUERY,
                project_id=self.project_id or "demo-project",
                region="us-central1",
                dataset_id="sentinelnet_demo",
                check_interval_seconds=60
            ),
            GCPService.VERTEX_AI: GCPServiceConfig(
                service=GCPService.VERTEX_AI,
                project_id=self.project_id or "demo-project",
                region="us-central1",
                endpoint_id="demo-endpoint",
                model_id="demo-model",
                check_interval_seconds=120  # Less frequent for AI endpoints
            )
        }

    async def initialize_clients(self):
        """Initialize GCP client libraries"""
        try:
            if not GCP_AVAILABLE:
                logger.warning("⚠️ Google Cloud libraries not available - running in demo mode")
                return

            if self.project_id:
                # Initialize BigQuery client
                self.bigquery_client = bigquery.Client(project=self.project_id)

                # Initialize Vertex AI client
                aiplatform.init(project=self.project_id, location="us-central1")
                self.vertexai_client = aiplatform

                logger.info("✅ GCP clients initialized successfully")
            else:
                logger.warning("⚠️ No GCP project ID provided - using demo mode")

        except Exception as e:
            logger.error(f"❌ Failed to initialize GCP clients: {e}")
            logger.info("💡 Running in demo mode without real GCP access")

    async def start_monitoring(self):
        """Start the monitoring loop"""
        logger.info("🚀 Starting GCP service monitoring...")

        # Initialize clients
        await self.initialize_clients()

        # Update agent status
        self.agent_info.status = AgentStatus.HEALTHY

        # Start monitoring tasks
        tasks = []
        for service in GCPService:
            task = asyncio.create_task(self._monitor_service_loop(service))
            tasks.append(task)

        # Wait for all monitoring tasks
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _monitor_service_loop(self, service: GCPService):
        """Monitoring loop for a specific service"""
        config = self.service_configs[service]

        while True:
            try:
                # Check if it's time for monitoring
                now = datetime.now()
                last_check = self.last_checks.get(service)

                if last_check and (now - last_check).total_seconds() < config.check_interval_seconds:
                    await asyncio.sleep(10)  # Wait before checking again
                    continue

                # Perform health check
                start_time = time.time()
                result = await self._check_service_health(service)
                end_time = time.time()

                result.latency_ms = (end_time - start_time) * 1000
                result.timestamp = now

                # Store result
                self.last_checks[service] = now
                self.health_history[service].append(result)

                # Keep only recent history (last 100 checks)
                if len(self.health_history[service]) > 100:
                    self.health_history[service] = self.health_history[service][-100:]

                # Process metrics and detect anomalies
                await self._process_monitoring_result(result)

                # Log result
                status_emoji = "🟢" if result.status == ServiceStatus.HEALTHY else "🔴"
                logger.info(f"{status_emoji} {service.value}: {result.status.value} ({result.latency_ms:.1f}ms)")

                # Wait before next check
                await asyncio.sleep(config.check_interval_seconds)

            except Exception as e:
                logger.error(f"❌ Error monitoring {service.value}: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _check_service_health(self, service: GCPService) -> MonitoringResult:
        """Check health of a specific GCP service"""
        config = self.service_configs[service]

        try:
            if service == GCPService.BIGQUERY:
                return await self._check_bigquery_health(config)
            elif service == GCPService.VERTEX_AI:
                return await self._check_vertexai_health(config)
            else:
                return MonitoringResult(
                    service=service,
                    status=ServiceStatus.UNKNOWN,
                    latency_ms=0.0,
                    error_message=f"Unsupported service: {service}"
                )

        except Exception as e:
            logger.error(f"❌ Health check failed for {service.value}: {e}")
            return MonitoringResult(
                service=service,
                status=ServiceStatus.DOWN,
                latency_ms=0.0,
                error_message=str(e)
            )

    async def _check_bigquery_health(self, config: GCPServiceConfig) -> MonitoringResult:
        """Check BigQuery service health"""
        try:
            if not self.bigquery_client or not GCP_AVAILABLE:
                # Demo mode - simulate health check
                await asyncio.sleep(0.1)  # Simulate network delay

                # Simulate occasional issues for demo
                import random
                if random.random() < 0.1:  # 10% chance of issues
                    return MonitoringResult(
                        service=GCPService.BIGQUERY,
                        status=ServiceStatus.DEGRADED,
                        latency_ms=random.randint(500, 2000),
                        error_message="Simulated BigQuery performance degradation",
                        metrics={
                            "query_count": random.randint(10, 100),
                            "error_rate": random.uniform(0.05, 0.15),
                            "storage_bytes": random.randint(1000000, 10000000)
                        }
                    )
                else:
                    return MonitoringResult(
                        service=GCPService.BIGQUERY,
                        status=ServiceStatus.HEALTHY,
                        latency_ms=random.randint(50, 200),
                        metrics={
                            "query_count": random.randint(50, 200),
                            "error_rate": random.uniform(0.001, 0.01),
                            "storage_bytes": random.randint(5000000, 20000000)
                        }
                    )

            # Real BigQuery health check
            start_time = time.time()

            # Test basic connectivity with a simple query
            query = f"""
            SELECT COUNT(*) as table_count
            FROM `{config.project_id}.{config.dataset_id}.__TABLES__`
            """

            query_job = self.bigquery_client.query(query)
            results = query_job.result()

            for row in results:
                table_count = row.table_count
                break

            # Get dataset metadata
            dataset_ref = self.bigquery_client.dataset(config.dataset_id)
            dataset = self.bigquery_client.get_dataset(dataset_ref)

            end_time = time.time()
            latency = (end_time - start_time) * 1000

            return MonitoringResult(
                service=GCPService.BIGQUERY,
                status=ServiceStatus.HEALTHY,
                latency_ms=latency,
                metrics={
                    "table_count": table_count,
                    "dataset_size_mb": getattr(dataset, 'num_bytes', 0) / (1024 * 1024),
                    "last_modified": getattr(dataset, 'modified', datetime.now())
                }
            )

        except Exception as e:
            error_msg = str(e)
            status = ServiceStatus.DOWN

            # Determine if it's a temporary issue
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                status = ServiceStatus.DEGRADED

            return MonitoringResult(
                service=GCPService.BIGQUERY,
                status=status,
                latency_ms=0.0,
                error_message=error_msg
            )

    async def _check_vertexai_health(self, config: GCPServiceConfig) -> MonitoringResult:
        """Check Vertex AI service health"""
        try:
            if not self.vertexai_client or not GCP_AVAILABLE:
                # Demo mode - simulate health check
                await asyncio.sleep(0.2)  # Simulate network delay

                import random
                if random.random() < 0.05:  # 5% chance of issues (less frequent)
                    return MonitoringResult(
                        service=GCPService.VERTEX_AI,
                        status=ServiceStatus.DEGRADED,
                        latency_ms=random.randint(1000, 5000),
                        error_message="Simulated Vertex AI endpoint timeout",
                        metrics={
                            "active_endpoints": random.randint(1, 5),
                            "prediction_count": random.randint(100, 1000),
                            "error_rate": random.uniform(0.02, 0.10)
                        }
                    )
                else:
                    return MonitoringResult(
                        service=GCPService.VERTEX_AI,
                        status=ServiceStatus.HEALTHY,
                        latency_ms=random.randint(100, 500),
                        metrics={
                            "active_endpoints": random.randint(3, 8),
                            "prediction_count": random.randint(500, 2000),
                            "error_rate": random.uniform(0.001, 0.005)
                        }
                    )

            # Real Vertex AI health check
            start_time = time.time()

            # List endpoints to check service availability
            endpoints = aiplatform.Endpoint.list(
                project=config.project_id,
                location=config.region
            )

            # Get basic metrics (simplified)
            endpoint_count = len(endpoints)
            active_endpoints = sum(1 for ep in endpoints
                                 if hasattr(ep, 'display_name') and ep.display_name)

            end_time = time.time()
            latency = (end_time - start_time) * 1000

            return MonitoringResult(
                service=GCPService.VERTEX_AI,
                status=ServiceStatus.HEALTHY,
                latency_ms=latency,
                metrics={
                    "endpoint_count": endpoint_count,
                    "active_endpoints": active_endpoints,
                    "region": config.region
                }
            )

        except Exception as e:
            error_msg = str(e)
            status = ServiceStatus.DOWN

            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                status = ServiceStatus.DEGRADED

            return MonitoringResult(
                service=GCPService.VERTEX_AI,
                status=status,
                latency_ms=0.0,
                error_message=error_msg
            )

    async def _process_monitoring_result(self, result: MonitoringResult):
        """Process monitoring result and store metrics"""
        try:
            # Convert to MetricData
            metric_data = MetricData(
                timestamp=result.timestamp,
                service_name=result.service.value.replace('_', ' ').title(),
                cloud_provider="GCP",
                metric_name="latency",
                value=result.latency_ms,
                unit="ms",
                region=self.service_configs[result.service].region,
                metadata={
                    "status": result.status.value,
                    "error_message": result.error_message,
                    **(result.metrics or {})
                }
            )

            # Store metric
            success = await self.data_processor.store_metric(metric_data)
            if success:
                # Check for anomalies
                # Get recent metrics for anomaly detection
                recent_metrics = await self.data_processor.get_metrics(
                    metric_data.service_name,
                    metric_data.cloud_provider,
                    metric_data.metric_name,
                    hours=1
                )

                if len(recent_metrics) >= 10:
                    anomaly_result = await self.anomaly_detector.detect_anomaly(recent_metrics)

                    if anomaly_result and anomaly_result.is_anomaly:
                        logger.warning(f"🚨 Anomaly detected in {result.service.value}: {anomaly_result.description}")

                        # Report to orchestrator (would be implemented)
                        await self._report_anomaly_to_orchestrator(result.service, anomaly_result)

        except Exception as e:
            logger.error(f"❌ Failed to process monitoring result: {e}")

    async def _report_anomaly_to_orchestrator(self, service: GCPService, anomaly: AnomalyResult):
        """Report anomaly to the orchestrator"""
        # This would integrate with the orchestrator's incident detection
        # For now, just log it
        logger.warning(f"📢 Reporting anomaly for {service.value}: {anomaly.description}")

    def get_service_status(self, service: GCPService) -> Optional[MonitoringResult]:
        """Get the latest status for a service"""
        history = self.health_history.get(service, [])
        return history[-1] if history else None

    def get_service_health_history(self, service: GCPService,
                                 hours: int = 1) -> List[MonitoringResult]:
        """Get health history for a service"""
        history = self.health_history.get(service, [])
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [result for result in history
                if result.timestamp and result.timestamp > cutoff_time]

    async def force_health_check(self, service: GCPService) -> MonitoringResult:
        """Force an immediate health check for a service"""
        logger.info(f"🔍 Forcing health check for {service.value}")
        result = await self._check_service_health(service)
        result.timestamp = datetime.now()

        # Update history
        self.health_history[service].append(result)
        if len(self.health_history[service]) > 100:
            self.health_history[service] = self.health_history[service][-100:]

        # Process result
        await self._process_monitoring_result(result)

        return result

    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            'agent_id': self.agent_id,
            'status': self.agent_info.status.value,
            'services_monitored': len(GCPService),
            'last_checks': {
                service.value: self.last_checks.get(service).isoformat()
                if self.last_checks.get(service) else None
                for service in GCPService
            },
            'health_summary': {
                service.value: {
                    'latest_status': self.get_service_status(service).status.value
                    if self.get_service_status(service) else 'unknown',
                    'checks_in_last_hour': len(self.get_service_health_history(service, 1)),
                    'average_latency': sum(r.latency_ms for r in self.get_service_health_history(service, 1))
                                    / max(1, len(self.get_service_health_history(service, 1)))
                }
                for service in GCPService
            },
            'gcp_available': GCP_AVAILABLE,
            'project_id': self.project_id
        }

# Global GCP monitor instance
_gcp_monitor_instance: Optional[GCPMonitorAgent] = None

def get_gcp_monitor(agent_id: str = "gcp_monitor_001") -> GCPMonitorAgent:
    """Get the global GCP monitor instance"""
    global _gcp_monitor_instance
    if _gcp_monitor_instance is None:
        _gcp_monitor_instance = GCPMonitorAgent(agent_id=agent_id)
    return _gcp_monitor_instance

# Convenience functions
async def start_gcp_monitoring(agent_id: str = "gcp_monitor_001"):
    """Start GCP service monitoring"""
    monitor = get_gcp_monitor(agent_id)
    await monitor.start_monitoring()

async def check_bigquery_health() -> MonitoringResult:
    """Check BigQuery health"""
    monitor = get_gcp_monitor()
    return await monitor.force_health_check(GCPService.BIGQUERY)

async def check_vertexai_health() -> MonitoringResult:
    """Check Vertex AI health"""
    monitor = get_gcp_monitor()
    return await monitor.force_health_check(GCPService.VERTEX_AI)

if __name__ == "__main__":
    # Test the GCP monitor
    async def test_gcp_monitor():
        monitor = GCPMonitorAgent()

        print("Testing BigQuery health check...")
        bq_result = await monitor._check_bigquery_health(monitor.service_configs[GCPService.BIGQUERY])
        print(f"BigQuery: {bq_result.status.value} ({bq_result.latency_ms:.1f}ms)")

        print("Testing Vertex AI health check...")
        vai_result = await monitor._check_vertexai_health(monitor.service_configs[GCPService.VERTEX_AI])
        print(f"Vertex AI: {vai_result.status.value} ({vai_result.latency_ms:.1f}ms)")

        print("GCP Monitor test completed!")

    asyncio.run(test_gcp_monitor())
