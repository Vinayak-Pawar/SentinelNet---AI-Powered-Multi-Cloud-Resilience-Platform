#!/usr/bin/env python3
"""
SentinelNet Remediation Planner
AI-powered remediation planning and validation

Author: Vinayak Pawar
Version: 1.0
Compatible with: M1 Pro MacBook Pro (Apple Silicon)

This module provides:
- AI-generated remediation plans using LangGraph and ChatGPT
- Safety validation for remediation actions
- Risk assessment and cost-benefit analysis
- Human-readable playbook generation
- Plan execution coordination (planning only, no actual execution)
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# LangChain and OpenAI imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# Configure logging
logger = logging.getLogger(__name__)

class RemediationPriority(Enum):
    """Priority levels for remediation actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RemediationRisk(Enum):
    """Risk levels for remediation actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RemediationAction:
    """Individual remediation action"""
    action_id: str
    description: str
    priority: RemediationPriority
    estimated_time_minutes: int
    risk_level: RemediationRisk
    automated: bool  # Whether this can be automated
    prerequisites: List[str]  # Actions that must be completed first
    rollback_plan: Optional[str] = None
    cost_estimate: Optional[Dict[str, Any]] = None
    validation_steps: List[str] = None

@dataclass
class RemediationPlan:
    """Complete remediation plan for an incident"""
    plan_id: str
    incident_id: str
    incident_summary: str
    impact_assessment: str
    actions: List[RemediationAction]
    total_estimated_time: int  # minutes
    overall_risk_level: RemediationRisk
    cost_benefit_analysis: Dict[str, Any]
    generated_at: datetime
    generated_by: str  # "ai" or "human"
    safety_validations: List[str]
    human_approval_required: bool = True
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

@dataclass
class SafetyValidation:
    """Safety validation results"""
    validation_id: str
    plan_id: str
    checks_passed: List[str]
    checks_failed: List[str]
    risk_assessment: str
    recommendations: List[str]
    validated_at: datetime
    validated_by: str

class RemediationPlanner:
    """
    AI-powered remediation planner for SentinelNet
    Generates safe, actionable remediation plans
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None, openai_api_key: str = None):
        """
        Initialize the remediation planner

        Args:
            llm: Pre-configured ChatOpenAI instance
            openai_api_key: OpenAI API key (if not using pre-configured LLM)
        """
        self.openai_api_key = openai_api_key
        if llm:
            self.llm = llm
        elif openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.1,
                api_key=openai_api_key
            )
        else:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm = ChatOpenAI(
                    model="gpt-4-turbo-preview",
                    temperature=0.1,
                    api_key=api_key
                )
            else:
                self.llm = None
                logger.warning("⚠️ No OpenAI API key provided - AI planning features will be limited")

        # Remediation templates and knowledge base
        self.remediation_templates = self._load_remediation_templates()
        self.safety_rules = self._load_safety_rules()

        # Plan history and validation cache
        self.plan_history: List[RemediationPlan] = []
        self.validation_cache: Dict[str, SafetyValidation] = {}

        logger.info("🛠️ Remediation Planner initialized")

    def _load_remediation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load remediation templates for different service types"""
        return {
            "bigquery": {
                "regional_failure": {
                    "description": "BigQuery dataset replication to secondary region",
                    "actions": [
                        "Enable BigQuery dataset replication to {secondary_region}",
                        "Update application connection strings",
                        "Validate data consistency across regions",
                        "Switch DNS/load balancer to secondary region"
                    ],
                    "estimated_time": 45,
                    "risk_level": "medium"
                },
                "quota_exceeded": {
                    "description": "BigQuery quota limit exceeded",
                    "actions": [
                        "Increase BigQuery quota limits",
                        "Implement query optimization",
                        "Set up usage monitoring and alerts"
                    ],
                    "estimated_time": 30,
                    "risk_level": "low"
                }
            },
            "vertex_ai": {
                "endpoint_down": {
                    "description": "Vertex AI prediction endpoint unavailable",
                    "actions": [
                        "Deploy model to backup region",
                        "Update endpoint configurations",
                        "Test model predictions in backup region",
                        "Update DNS/load balancer routing"
                    ],
                    "estimated_time": 60,
                    "risk_level": "medium"
                }
            },
            "blob_storage": {
                "high_latency": {
                    "description": "Azure Blob Storage high latency",
                    "actions": [
                        "Enable geo-redundant storage (RA-GRS)",
                        "Configure CDN for static content",
                        "Implement client-side caching",
                        "Monitor replication status"
                    ],
                    "estimated_time": 30,
                    "risk_level": "low"
                }
            },
            "devops": {
                "pipeline_failure": {
                    "description": "Azure DevOps pipeline failures",
                    "actions": [
                        "Scale up agent pool capacity",
                        "Implement pipeline retry logic",
                        "Review and optimize pipeline configurations",
                        "Set up parallel pipeline execution"
                    ],
                    "estimated_time": 45,
                    "risk_level": "medium"
                }
            }
        }

    def _load_safety_rules(self) -> List[str]:
        """Load safety rules for remediation validation"""
        return [
            "Never execute destructive operations without explicit human approval",
            "Always validate resource availability before recommending failovers",
            "Include rollback plans for all automated actions",
            "Estimate costs and require approval for high-cost operations",
            "Check for data consistency requirements before recommending switches",
            "Validate network connectivity before recommending regional failovers",
            "Include monitoring and verification steps for all actions",
            "Never recommend actions that could cause cascading failures",
            "Always consider business impact and downtime tolerance",
            "Include human validation checkpoints for complex multi-step actions"
        ]

    async def generate_plan(self, incident: Dict[str, Any],
                          agent_responses: Dict[str, Any] = None) -> Optional[RemediationPlan]:
        """
        Generate a remediation plan for an incident

        Args:
            incident: Incident information
            agent_responses: Investigation data from monitoring agents

        Returns:
            Generated remediation plan or None if generation fails
        """
        try:
            logger.info(f"🛠️ Generating remediation plan for incident: {incident.get('id')}")

            if not self.llm:
                logger.warning("⚠️ No LLM available - using template-based planning")
                return await self._generate_template_based_plan(incident, agent_responses)

            # Generate plan using AI
            plan_data = await self._generate_ai_plan(incident, agent_responses)
            if not plan_data:
                logger.warning("⚠️ AI plan generation failed - falling back to template")
                return await self._generate_template_based_plan(incident, agent_responses)

            # Create RemediationPlan object
            plan = RemediationPlan(
                plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                incident_id=incident.get('id', 'unknown'),
                incident_summary=incident.get('description', ''),
                impact_assessment=plan_data.get('impact_assessment', ''),
                actions=plan_data.get('actions', []),
                total_estimated_time=plan_data.get('total_time', 0),
                overall_risk_level=RemediationRisk(plan_data.get('risk_level', 'medium')),
                cost_benefit_analysis=plan_data.get('cost_analysis', {}),
                generated_at=datetime.now(),
                generated_by="ai",
                safety_validations=plan_data.get('safety_validations', []),
                human_approval_required=True
            )

            # Add to history
            self.plan_history.append(plan)
            if len(self.plan_history) > 100:  # Keep last 100 plans
                self.plan_history = self.plan_history[-100:]

            logger.info(f"✅ Generated remediation plan: {plan.plan_id}")
            return plan

        except Exception as e:
            logger.error(f"❌ Failed to generate remediation plan: {e}")
            return None

    async def _generate_ai_plan(self, incident: Dict[str, Any],
                               agent_responses: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Generate remediation plan using AI"""
        try:
            # Prepare context for AI
            service_name = incident.get('service', 'unknown')
            cloud_provider = incident.get('cloud', 'unknown')
            incident_status = incident.get('status', 'unknown')
            severity = incident.get('severity', 'medium')

            agent_data = json.dumps(agent_responses or {}, indent=2, default=str)

            prompt = f"""
You are an expert cloud infrastructure remediation specialist. Generate a detailed, safe remediation plan for the following incident:

INCIDENT DETAILS:
- Service: {service_name}
- Cloud Provider: {cloud_provider}
- Status: {incident_status}
- Severity: {severity}
- Description: {incident.get('description', 'N/A')}
- Region: {incident.get('region', 'N/A')}
- Latency: {incident.get('latency', 'N/A')}ms

AGENT INVESTIGATION DATA:
{agent_data}

SAFETY REQUIREMENTS:
- All plans must be human-validated before execution
- Include rollback procedures for all actions
- Estimate costs and business impact
- Never recommend destructive operations
- Prioritize data safety and consistency

Generate a JSON response with this structure:
{{
  "impact_assessment": "Brief assessment of business impact",
  "actions": [
    {{
      "action_id": "unique_id",
      "description": "Detailed action description",
      "priority": "low|medium|high|critical",
      "estimated_time_minutes": 30,
      "risk_level": "low|medium|high|critical",
      "automated": false,
      "prerequisites": ["action_ids"],
      "rollback_plan": "How to undo this action",
      "cost_estimate": {{"currency": "USD", "amount": 50, "description": "Cost breakdown"}},
      "validation_steps": ["Step 1", "Step 2"]
    }}
  ],
  "total_time": 60,
  "risk_level": "medium",
  "cost_analysis": {{
    "estimated_cost": 150,
    "estimated_downtime_savings": 3600,
    "roi_hours": 24
  }},
  "safety_validations": ["Safety check 1", "Safety check 2"]
}}

Be specific, actionable, and prioritize safety. Focus on the affected service's remediation patterns.
"""

            with get_openai_callback() as cb:
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are a senior cloud infrastructure engineer specializing in incident response and disaster recovery. Always prioritize safety, data integrity, and business continuity."),
                    HumanMessage(content=prompt)
                ])

            # Parse JSON response
            plan_text = response.content.strip()
            if plan_text.startswith('```json'):
                plan_text = plan_text[7:]
            if plan_text.endswith('```'):
                plan_text = plan_text[:-3]

            plan_data = json.loads(plan_text)

            logger.debug(f"OpenAI tokens used: {cb.total_tokens}")
            return plan_data

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse AI response as JSON: {e}")
            logger.debug(f"Raw response: {response.content if 'response' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"❌ AI plan generation failed: {e}")
            return None

    async def _generate_template_based_plan(self, incident: Dict[str, Any],
                                          agent_responses: Dict[str, Any] = None) -> Optional[RemediationPlan]:
        """Generate remediation plan using templates (fallback when AI is unavailable)"""
        try:
            service_name = incident.get('service', '').lower()
            cloud_provider = incident.get('cloud', '').lower()

            # Find matching template
            service_templates = self.remediation_templates.get(service_name, {})

            # Simple incident type detection
            incident_type = "regional_failure"  # Default
            if "latency" in incident.get('description', '').lower():
                incident_type = "high_latency"
            elif "quota" in incident.get('description', '').lower():
                incident_type = "quota_exceeded"
            elif "endpoint" in incident.get('description', '').lower():
                incident_type = "endpoint_down"

            template = service_templates.get(incident_type)
            if not template:
                logger.warning(f"⚠️ No template found for {service_name}/{incident_type}")
                return None

            # Create actions from template
            actions = []
            for i, action_desc in enumerate(template['actions']):
                action = RemediationAction(
                    action_id=f"action_{i+1}",
                    description=action_desc,
                    priority=RemediationPriority.MEDIUM,
                    estimated_time_minutes=10,
                    risk_level=RemediationRisk(template['risk_level']),
                    automated=False,
                    prerequisites=[],
                    rollback_plan="Revert configuration changes",
                    validation_steps=["Verify service health", "Check data consistency"]
                )
                actions.append(action)

            # Create plan
            plan = RemediationPlan(
                plan_id=f"template_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                incident_id=incident.get('id', 'unknown'),
                incident_summary=incident.get('description', ''),
                impact_assessment="Impact assessment requires AI analysis",
                actions=actions,
                total_estimated_time=template['estimated_time'],
                overall_risk_level=RemediationRisk(template['risk_level']),
                cost_benefit_analysis={"estimated_cost": 100, "method": "template_estimate"},
                generated_at=datetime.now(),
                generated_by="template",
                safety_validations=["Template-based safety checks applied"],
                human_approval_required=True
            )

            logger.info(f"✅ Generated template-based remediation plan: {plan.plan_id}")
            return plan

        except Exception as e:
            logger.error(f"❌ Template-based plan generation failed: {e}")
            return None

    async def validate_plan_safety(self, plan: RemediationPlan) -> SafetyValidation:
        """
        Validate the safety of a remediation plan

        Args:
            plan: Plan to validate

        Returns:
            Safety validation results
        """
        try:
            logger.info(f"🛡️ Validating safety of plan: {plan.plan_id}")

            # Check cache first
            if plan.plan_id in self.validation_cache:
                cached = self.validation_cache[plan.plan_id]
                if (datetime.now() - cached.validated_at).total_seconds() < 3600:  # 1 hour cache
                    return cached

            checks_passed = []
            checks_failed = []
            recommendations = []

            # Safety validation checks
            for action in plan.actions:
                # Check for destructive operations
                destructive_keywords = ['delete', 'drop', 'remove', 'destroy', 'terminate']
                action_text = action.description.lower()
                if any(keyword in action_text for keyword in destructive_keywords):
                    if not action.rollback_plan:
                        checks_failed.append(f"Action {action.action_id}: Destructive operation without rollback plan")
                        recommendations.append(f"Add rollback plan for action {action.action_id}")
                    else:
                        checks_passed.append(f"Action {action.action_id}: Destructive operation has rollback plan")

                # Check for automation safety
                if action.automated and action.risk_level in [RemediationRisk.HIGH, RemediationRisk.CRITICAL]:
                    checks_failed.append(f"Action {action.action_id}: High-risk automated action")
                    recommendations.append(f"Require human approval for action {action.action_id}")

                # Check prerequisites
                for prereq in action.prerequisites:
                    if not any(a.action_id == prereq for a in plan.actions):
                        checks_failed.append(f"Action {action.action_id}: Invalid prerequisite {prereq}")

                # Check validation steps
                if not action.validation_steps:
                    checks_failed.append(f"Action {action.action_id}: Missing validation steps")
                    recommendations.append(f"Add validation steps for action {action_id}")

            # Overall risk assessment
            high_risk_actions = sum(1 for a in plan.actions if a.risk_level in [RemediationRisk.HIGH, RemediationRisk.CRITICAL])
            risk_assessment = "low"
            if high_risk_actions > len(plan.actions) / 2:
                risk_assessment = "high"
            elif high_risk_actions > 0:
                risk_assessment = "medium"

            # Create validation result
            validation = SafetyValidation(
                validation_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                plan_id=plan.plan_id,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                validated_at=datetime.now(),
                validated_by="system"
            )

            # Cache result
            self.validation_cache[plan.plan_id] = validation

            logger.info(f"✅ Safety validation complete for plan {plan.plan_id}")
            return validation

        except Exception as e:
            logger.error(f"❌ Safety validation failed: {e}")
            return SafetyValidation(
                validation_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                plan_id=plan.plan_id,
                checks_passed=[],
                checks_failed=["Validation system error"],
                risk_assessment="unknown",
                recommendations=["Manual safety review required"],
                validated_at=datetime.now(),
                validated_by="error"
            )

    def get_plan_history(self, limit: int = 10) -> List[RemediationPlan]:
        """
        Get recent remediation plans

        Args:
            limit: Maximum number of plans to return

        Returns:
            List of recent plans
        """
        return self.plan_history[-limit:] if self.plan_history else []

    def get_plan_by_id(self, plan_id: str) -> Optional[RemediationPlan]:
        """
        Get a specific remediation plan by ID

        Args:
            plan_id: Plan ID to retrieve

        Returns:
            Remediation plan or None if not found
        """
        for plan in self.plan_history:
            if plan.plan_id == plan_id:
                return plan
        return None

# Global remediation planner instance
_remediation_planner_instance: Optional[RemediationPlanner] = None

def get_remediation_planner(llm: Optional[ChatOpenAI] = None) -> RemediationPlanner:
    """Get the global remediation planner instance"""
    global _remediation_planner_instance
    if _remediation_planner_instance is None:
        _remediation_planner_instance = RemediationPlanner(llm=llm)
    return _remediation_planner_instance

# Convenience functions
async def generate_remediation_plan(incident: Dict[str, Any],
                                  agent_responses: Dict[str, Any] = None) -> Optional[RemediationPlan]:
    """Generate a remediation plan for an incident"""
    planner = get_remediation_planner()
    return await planner.generate_plan(incident, agent_responses)

async def validate_plan_safety(plan: RemediationPlan) -> SafetyValidation:
    """Validate the safety of a remediation plan"""
    planner = get_remediation_planner()
    return await planner.validate_plan_safety(plan)

if __name__ == "__main__":
    # Test the remediation planner
    async def test_remediation_planner():
        planner = RemediationPlanner()

        # Test incident
        incident = {
            'id': 'test_incident_001',
            'service': 'BigQuery',
            'cloud': 'GCP',
            'status': 'degraded',
            'severity': 'high',
            'description': 'BigQuery experiencing high latency in us-east1',
            'region': 'us-east1',
            'latency': 500
        }

        # Generate plan
        plan = await planner.generate_plan(incident)
        if plan:
            print(f"Generated plan: {plan.plan_id}")
            print(f"Actions: {len(plan.actions)}")
            print(f"Total time: {plan.total_estimated_time} minutes")
            print(f"Risk level: {plan.overall_risk_level.value}")

            # Validate safety
            validation = await planner.validate_plan_safety(plan)
            print(f"Safety checks passed: {len(validation.checks_passed)}")
            print(f"Safety checks failed: {len(validation.checks_failed)}")
        else:
            print("❌ Plan generation failed")

    asyncio.run(test_remediation_planner())
