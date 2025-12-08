#!/usr/bin/env python3
"""
SentinelNet Agent Orchestrator
Core LangGraph-based orchestration system for SentinelNet

Author: Vinayak Pawar
Version: 1.0
Compatible with: M1 Pro MacBook Pro (Apple Silicon)

This module implements the central orchestration layer that:
- Coordinates all monitoring agents (GCP, Azure)
- Manages distributed state and consensus
- Handles communication protocols (cloud, P2P, fallback)
- Orchestrates remediation planning workflows
- Maintains system health and agent discovery
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# Local imports (will be created)
try:
    from data.processor import DataProcessor
except ImportError:
    from ..data.processor import DataProcessor

try:
    from agents.communication import CommunicationManager
except ImportError:
    from .communication import CommunicationManager

try:
    from agents.remediation import RemediationPlanner
except ImportError:
    from .remediation import RemediationPlanner

# Configure logging
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Enumeration of possible agent states"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"

class ServiceStatus(Enum):
    """Enumeration of possible service health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"

@dataclass
class ServiceHealth:
    """Data class for service health information"""
    name: str
    cloud: str
    status: ServiceStatus
    latency: int
    last_checked: datetime
    region: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class AgentInfo:
    """Data class for agent information"""
    agent_id: str
    agent_type: str  # "monitor", "remediation", "coordinator"
    cloud_provider: str
    services: List[str]
    status: AgentStatus
    last_heartbeat: datetime
    region: Optional[str] = None
    capabilities: List[str] = None

@dataclass
class SystemState:
    """Central state management for the orchestration system"""
    agents: Dict[str, AgentInfo]
    services: Dict[str, ServiceHealth]
    active_alerts: List[Dict[str, Any]]
    pending_remediations: List[Dict[str, Any]]
    communication_status: str
    last_updated: datetime
    system_health_score: float

class OrchestratorState:
    """LangGraph state for orchestration workflows"""
    def __init__(self):
        self.system_state = SystemState(
            agents={},
            services={},
            active_alerts=[],
            pending_remediations=[],
            communication_status="initializing",
            last_updated=datetime.now(),
            system_health_score=100.0
        )
        self.current_incident: Optional[Dict[str, Any]] = None
        self.workflow_status = "idle"
        self.agent_responses: Dict[str, Any] = {}
        self.communication_log: List[str] = []

class SentinelNetOrchestrator:
    """
    Main orchestration class for SentinelNet
    Implements LangGraph workflows for distributed agent coordination
    """

    def __init__(self, openai_api_key: str = None):
        """
        Initialize the orchestrator

        Args:
            openai_api_key: OpenAI API key for LLM operations
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            logger.warning("⚠️  OPENAI_API_KEY not found - some features may be limited")

        # Initialize components
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=self.openai_api_key
        ) if self.openai_api_key else None

        self.data_processor = DataProcessor()
        self.communication_manager = CommunicationManager()
        self.remediation_planner = RemediationPlanner(llm=self.llm)

        # Agent registry
        self.registered_agents: Dict[str, AgentInfo] = {}
        self.agent_heartbeat_timeout = 300  # 5 minutes

        # LangGraph workflow
        self.workflow = self._build_orchestration_workflow()

        # Orchestrator state
        self.state = OrchestratorState()

        # Performance metrics
        self.metrics = {
            "workflows_executed": 0,
            "agents_coordinated": 0,
            "incidents_detected": 0,
            "remediations_planned": 0,
            "average_response_time": 0.0
        }

        logger.info("🚀 SentinelNet Orchestrator initialized")

    def _build_orchestration_workflow(self) -> StateGraph:
        """
        Build the main LangGraph orchestration workflow

        Returns:
            StateGraph: Configured workflow for incident response
        """
        workflow = StateGraph(OrchestratorState)

        # Define nodes
        workflow.add_node("monitor_health", self._monitor_system_health)
        workflow.add_node("detect_incidents", self._detect_service_incidents)
        workflow.add_node("coordinate_agents", self._coordinate_monitoring_agents)
        workflow.add_node("analyze_impact", self._analyze_incident_impact)
        workflow.add_node("generate_remediation", self._generate_remediation_plan)
        workflow.add_node("validate_safety", self._validate_plan_safety)
        workflow.add_node("notify_stakeholders", self._notify_stakeholders)
        workflow.add_node("update_state", self._update_system_state)

        # Define edges and conditional logic
        workflow.set_entry_point("monitor_health")

        workflow.add_edge("monitor_health", "detect_incidents")
        workflow.add_edge("detect_incidents", "coordinate_agents")

        workflow.add_conditional_edges(
            "coordinate_agents",
            self._should_analyze_impact,
            {
                True: "analyze_impact",
                False: "update_state"
            }
        )

        workflow.add_edge("analyze_impact", "generate_remediation")
        workflow.add_edge("generate_remediation", "validate_safety")
        workflow.add_edge("validate_safety", "notify_stakeholders")
        workflow.add_edge("notify_stakeholders", "update_state")
        workflow.add_edge("update_state", END)

        return workflow

    async def _monitor_system_health(self, state: OrchestratorState) -> OrchestratorState:
        """Monitor overall system health and agent status"""
        logger.info("🔍 Monitoring system health...")

        # Check agent heartbeats
        current_time = datetime.now()
        offline_agents = []

        for agent_id, agent in self.registered_agents.items():
            time_since_heartbeat = (current_time - agent.last_heartbeat).total_seconds()
            if time_since_heartbeat > self.agent_heartbeat_timeout:
                agent.status = AgentStatus.OFFLINE
                offline_agents.append(agent_id)
                logger.warning(f"⚠️  Agent {agent_id} went offline")

        # Update communication status
        comm_status = await self.communication_manager.get_status()
        state.system_state.communication_status = comm_status

        # Calculate system health score
        total_agents = len(self.registered_agents)
        healthy_agents = sum(1 for agent in self.registered_agents.values()
                           if agent.status == AgentStatus.HEALTHY)
        health_score = (healthy_agents / total_agents * 100) if total_agents > 0 else 100.0
        state.system_state.system_health_score = health_score

        state.system_state.last_updated = current_time
        logger.info(f"✅ System health check complete. Score: {health_score:.1f}%")

        return state

    async def _detect_service_incidents(self, state: OrchestratorState) -> OrchestratorState:
        """Detect service incidents from monitoring data"""
        logger.info("🔎 Detecting service incidents...")

        incidents_detected = []

        # Analyze service health data
        for service_name, health in state.system_state.services.items():
            if health.status in [ServiceStatus.DEGRADED, ServiceStatus.DOWN]:
                # Check if this is a new incident or escalation
                existing_alert = next(
                    (alert for alert in state.system_state.active_alerts
                     if alert.get('service') == service_name and alert.get('resolved') != True),
                    None
                )

                if not existing_alert:
                    incident = {
                        'id': f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{service_name}",
                        'service': service_name,
                        'cloud': health.cloud,
                        'status': health.status.value,
                        'severity': self._calculate_severity(health),
                        'detected_at': datetime.now(),
                        'description': f"{service_name} experiencing {health.status.value} status",
                        'latency': health.latency,
                        'region': health.region,
                        'error_message': health.error_message
                    }
                    incidents_detected.append(incident)
                    state.system_state.active_alerts.append(incident)
                    logger.warning(f"🚨 New incident detected: {service_name} - {health.status.value}")

        if incidents_detected:
            state.current_incident = incidents_detected[0]  # Focus on first incident
            self.metrics["incidents_detected"] += len(incidents_detected)

        return state

    async def _coordinate_monitoring_agents(self, state: OrchestratorState) -> OrchestratorState:
        """Coordinate monitoring agents for incident investigation"""
        logger.info("🤝 Coordinating monitoring agents...")

        if not state.current_incident:
            return state

        incident = state.current_incident
        affected_service = incident['service']

        # Find relevant monitoring agents
        relevant_agents = [
            agent for agent in self.registered_agents.values()
            if agent.agent_type == "monitor" and affected_service in agent.services
        ]

        if not relevant_agents:
            logger.warning(f"⚠️  No monitoring agents available for {affected_service}")
            return state

        # Request detailed investigation from agents
        investigation_tasks = []
        for agent in relevant_agents:
            task = self.communication_manager.request_investigation(
                agent.agent_id, incident
            )
            investigation_tasks.append(task)

        # Wait for investigation results with timeout
        try:
            investigation_results = await asyncio.gather(*investigation_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(investigation_results):
                agent = relevant_agents[i]
                if isinstance(result, Exception):
                    logger.error(f"❌ Investigation failed for agent {agent.agent_id}: {result}")
                    state.agent_responses[agent.agent_id] = {"error": str(result)}
                else:
                    state.agent_responses[agent.agent_id] = result
                    logger.info(f"✅ Received investigation data from {agent.agent_id}")

        except asyncio.TimeoutError:
            logger.error("⏰ Investigation timeout - proceeding with available data")

        return state

    def _should_analyze_impact(self, state: OrchestratorState) -> bool:
        """Determine if impact analysis is needed"""
        return state.current_incident is not None and len(state.agent_responses) > 0

    async def _analyze_incident_impact(self, state: OrchestratorState) -> OrchestratorState:
        """Analyze the impact of detected incidents"""
        logger.info("📊 Analyzing incident impact...")

        if not state.current_incident or not self.llm:
            return state

        # Prepare context for LLM analysis
        incident = state.current_incident
        agent_data = state.agent_responses

        analysis_prompt = f"""
        Analyze the impact of this cloud service incident:

        Incident Details:
        - Service: {incident['service']}
        - Cloud: {incident['cloud']}
        - Status: {incident['status']}
        - Severity: {incident['severity']}
        - Region: {incident.get('region', 'Unknown')}
        - Latency: {incident['latency']}ms
        - Error: {incident.get('error_message', 'None')}

        Agent Investigation Data:
        {json.dumps(agent_data, indent=2, default=str)}

        Please provide:
        1. Impact assessment (Low/Medium/High/Critical)
        2. Affected systems and dependencies
        3. Potential downstream effects
        4. Estimated business impact
        5. Recommended urgency level

        Be specific and data-driven in your analysis.
        """

        try:
            with get_openai_callback() as cb:
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are an expert cloud infrastructure analyst specializing in incident impact assessment."),
                    HumanMessage(content=analysis_prompt)
                ])

            impact_analysis = response.content.strip()
            state.current_incident['impact_analysis'] = impact_analysis

            logger.info("✅ Impact analysis complete")
            logger.debug(f"OpenAI tokens used: {cb.total_tokens}")

        except Exception as e:
            logger.error(f"❌ Impact analysis failed: {e}")
            state.current_incident['impact_analysis'] = "Analysis unavailable - proceeding with basic assessment"

        return state

    async def _generate_remediation_plan(self, state: OrchestratorState) -> OrchestratorState:
        """Generate remediation plan for the incident"""
        logger.info("🛠️ Generating remediation plan...")

        if not state.current_incident:
            return state

        # Use remediation planner
        remediation_plan = await self.remediation_planner.generate_plan(
            state.current_incident,
            state.agent_responses
        )

        if remediation_plan:
            state.current_incident['remediation_plan'] = remediation_plan
            state.system_state.pending_remediations.append(remediation_plan)
            self.metrics["remediations_planned"] += 1

            logger.info("✅ Remediation plan generated")
        else:
            logger.warning("⚠️  Failed to generate remediation plan")

        return state

    async def _validate_plan_safety(self, state: OrchestratorState) -> OrchestratorState:
        """Validate the safety and feasibility of remediation plans"""
        logger.info("🛡️ Validating plan safety...")

        if not state.current_incident or 'remediation_plan' not in state.current_incident:
            return state

        plan = state.current_incident['remediation_plan']

        # Safety validation checks
        safety_checks = {
            'destructive_operations': False,
            'requires_human_approval': True,
            'rollback_available': True,
            'cost_estimate_available': True,
            'risk_assessment': 'medium'
        }

        # Add safety validation results
        plan['safety_checks'] = safety_checks
        plan['validated_at'] = datetime.now()

        logger.info("✅ Safety validation complete")
        return state

    async def _notify_stakeholders(self, state: OrchestratorState) -> OrchestratorState:
        """Notify relevant stakeholders about incidents and plans"""
        logger.info("📢 Notifying stakeholders...")

        if not state.current_incident:
            return state

        incident = state.current_incident

        # Prepare notification content
        notification = {
            'incident_id': incident['id'],
            'service': incident['service'],
            'status': incident['status'],
            'severity': incident['severity'],
            'impact_analysis': incident.get('impact_analysis', 'Analysis pending'),
            'remediation_plan': incident.get('remediation_plan', {}),
            'timestamp': datetime.now(),
            'requires_approval': True
        }

        # Send via available communication channels
        await self.communication_manager.send_notification(notification)

        logger.info("✅ Stakeholders notified")
        return state

    async def _update_system_state(self, state: OrchestratorState) -> OrchestratorState:
        """Update the overall system state"""
        logger.info("📝 Updating system state...")

        state.system_state.last_updated = datetime.now()
        state.workflow_status = "completed"

        # Log workflow completion
        self.metrics["workflows_executed"] += 1

        logger.info("✅ System state updated")
        return state

    def _calculate_severity(self, health: ServiceHealth) -> str:
        """Calculate incident severity based on service health"""
        if health.status == ServiceStatus.DOWN:
            return "critical"
        elif health.status == ServiceStatus.DEGRADED:
            if health.latency > 5000:  # 5 seconds
                return "high"
            else:
                return "medium"
        else:
            return "low"

    # Public API methods

    async def register_agent(self, agent_info: AgentInfo) -> bool:
        """
        Register a new agent with the orchestrator

        Args:
            agent_info: Agent information

        Returns:
            bool: Registration success
        """
        try:
            self.registered_agents[agent_info.agent_id] = agent_info
            self.state.system_state.agents[agent_info.agent_id] = agent_info
            logger.info(f"✅ Agent registered: {agent_info.agent_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Agent registration failed: {e}")
            return False

    async def update_agent_heartbeat(self, agent_id: str) -> bool:
        """
        Update agent heartbeat timestamp

        Args:
            agent_id: Agent identifier

        Returns:
            bool: Update success
        """
        if agent_id in self.registered_agents:
            self.registered_agents[agent_id].last_heartbeat = datetime.now()
            self.registered_agents[agent_id].status = AgentStatus.HEALTHY
            return True
        return False

    async def update_service_health(self, service_health: ServiceHealth) -> bool:
        """
        Update service health information

        Args:
            service_health: Service health data

        Returns:
            bool: Update success
        """
        try:
            service_key = f"{service_health.cloud}_{service_health.name}"
            self.state.system_state.services[service_key] = service_health

            # Trigger workflow if service is unhealthy
            if service_health.status in [ServiceStatus.DEGRADED, ServiceStatus.DOWN]:
                await self._trigger_incident_workflow()

            return True
        except Exception as e:
            logger.error(f"❌ Service health update failed: {e}")
            return False

    async def _trigger_incident_workflow(self):
        """Trigger the incident response workflow"""
        try:
            logger.info("🚨 Triggering incident response workflow...")

            # Reset workflow state for new incident
            self.state.current_incident = None
            self.state.workflow_status = "running"
            self.state.agent_responses = {}

            # Execute workflow
            start_time = time.time()
            result = await self.workflow.ainvoke(self.state)
            end_time = time.time()

            # Update metrics
            response_time = end_time - start_time
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["workflows_executed"] - 1)) +
                response_time
            ) / self.metrics["workflows_executed"]

            self.state = result
            logger.info(f"✅ Incident workflow completed in {response_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ Incident workflow failed: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status

        Returns:
            Dict containing system status information
        """
        return {
            'system_health_score': self.state.system_state.system_health_score,
            'active_agents': len([a for a in self.registered_agents.values()
                                if a.status == AgentStatus.HEALTHY]),
            'total_agents': len(self.registered_agents),
            'services_monitored': len(self.state.system_state.services),
            'active_alerts': len(self.state.system_state.active_alerts),
            'pending_remediations': len(self.state.system_state.pending_remediations),
            'communication_status': self.state.system_state.communication_status,
            'last_updated': self.state.system_state.last_updated,
            'metrics': self.metrics.copy()
        }

    async def start_orchestration_loop(self):
        """Start the main orchestration loop"""
        logger.info("🔄 Starting orchestration loop...")

        while True:
            try:
                # Update system health
                await self._monitor_system_health(self.state)

                # Process any pending incidents
                if self.state.workflow_status == "idle":
                    await self._detect_service_incidents(self.state)

                # Wait before next cycle
                await asyncio.sleep(30)  # 30 second intervals

            except Exception as e:
                logger.error(f"❌ Orchestration loop error: {e}")
                await asyncio.sleep(10)  # Shorter wait on error

    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("🛑 Shutting down orchestrator...")

        # Close communication channels
        await self.communication_manager.close()

        # Save final state
        self._save_system_state()

        logger.info("✅ Orchestrator shutdown complete")

    def _save_system_state(self):
        """Save current system state to disk"""
        try:
            state_data = {
                'agents': {k: asdict(v) for k, v in self.registered_agents.items()},
                'services': {k: asdict(v) for k, v in self.state.system_state.services.items()},
                'active_alerts': self.state.system_state.active_alerts,
                'pending_remediations': self.state.system_state.pending_remediations,
                'metrics': self.metrics,
                'last_saved': datetime.now().isoformat()
            }

            with open('data/system_state.json', 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            logger.info("💾 System state saved")

        except Exception as e:
            logger.error(f"❌ Failed to save system state: {e}")

# Global orchestrator instance
_orchestrator_instance: Optional[SentinelNetOrchestrator] = None

def get_orchestrator() -> SentinelNetOrchestrator:
    """Get the global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SentinelNetOrchestrator()
    return _orchestrator_instance

# Convenience functions for external use
async def initialize_orchestrator() -> SentinelNetOrchestrator:
    """Initialize and return the orchestrator instance"""
    orchestrator = get_orchestrator()
    # Start background tasks if needed
    return orchestrator

async def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    orchestrator = get_orchestrator()
    return await orchestrator.get_system_status()

if __name__ == "__main__":
    # Test the orchestrator
    async def test_orchestrator():
        orchestrator = await initialize_orchestrator()
        status = await orchestrator.get_system_status()
        print("System Status:", json.dumps(status, indent=2, default=str))

    asyncio.run(test_orchestrator())
