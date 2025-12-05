#!/usr/bin/env python3
"""
SentinelNet API
FastAPI backend for SentinelNet services

Author: Vinayak Pawar
Version: 1.0
Compatible with: M1 Pro MacBook Pro (Apple Silicon)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SentinelNet API",
    description="AI-Powered Multi-Cloud Resilience Platform API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],  # Streamlit and React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ServiceStatus(BaseModel):
    name: str
    cloud: str
    status: str  # healthy, warning, error
    latency: int
    last_checked: datetime
    region: Optional[str] = None

class Alert(BaseModel):
    id: str
    level: str  # info, warning, error
    service: str
    message: str
    timestamp: datetime

class RemediationPlan(BaseModel):
    id: str
    incident_id: str
    actions: List[dict]
    estimated_time: str
    risk_level: str
    generated_at: datetime
    approved: bool = False

# Mock data
def get_mock_services():
    """Get mock service data for demo"""
    return [
        ServiceStatus(
            name="BigQuery",
            cloud="GCP",
            status="healthy",
            latency=45,
            last_checked=datetime.now(),
            region="us-east1"
        ),
        ServiceStatus(
            name="Vertex AI",
            cloud="GCP",
            status="healthy",
            latency=67,
            last_checked=datetime.now(),
            region="us-central1"
        ),
        ServiceStatus(
            name="Blob Storage",
            cloud="Azure",
            status="warning",
            latency=234,
            last_checked=datetime.now(),
            region="East US"
        ),
        ServiceStatus(
            name="DevOps",
            cloud="Azure",
            status="healthy",
            latency=89,
            last_checked=datetime.now(),
            region="Global"
        )
    ]

def get_mock_alerts():
    """Get mock alerts for demo"""
    return [
        Alert(
            id="alert_001",
            level="warning",
            service="Blob Storage",
            message="High latency detected (234ms > 200ms threshold)",
            timestamp=datetime.now()
        ),
        Alert(
            id="alert_002",
            level="info",
            service="System",
            message="All monitoring agents health check passed",
            timestamp=datetime.now()
        )
    ]

# API Routes
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "SentinelNet API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

@app.get("/health/services", response_model=List[ServiceStatus])
async def get_services():
    """Get all monitored services status"""
    return get_mock_services()

@app.get("/health/services/{service_name}", response_model=ServiceStatus)
async def get_service(service_name: str):
    """Get specific service status"""
    services = get_mock_services()
    for service in services:
        if service.name.lower() == service_name.lower():
            return service
    raise HTTPException(status_code=404, detail=f"Service {service_name} not found")

@app.post("/health/check")
async def manual_health_check():
    """Trigger manual health check for all services"""
    # In real implementation, this would trigger actual checks
    return {
        "message": "Health check initiated",
        "services_checked": 4,
        "timestamp": datetime.now()
    }

@app.get("/alerts", response_model=List[Alert])
async def get_alerts():
    """Get active alerts"""
    return get_mock_alerts()

@app.post("/remediation/plan")
async def generate_remediation_plan(incident_description: str):
    """Generate AI-powered remediation plan"""
    # Mock remediation plan generation
    plan = RemediationPlan(
        id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        incident_id="incident_001",
        actions=[
            {
                "priority": "high",
                "action": "Switch to geo-redundant storage",
                "estimated_time": "5 minutes",
                "risk": "low",
                "automated": True
            },
            {
                "priority": "medium",
                "action": "Scale up compute resources",
                "estimated_time": "10 minutes",
                "risk": "medium",
                "automated": False
            }
        ],
        estimated_time="15 minutes",
        risk_level="low",
        generated_at=datetime.now(),
        approved=False
    )

    return {
        "message": "Remediation plan generated",
        "plan": plan,
        "ai_insights": "Based on historical patterns, this appears to be regional network congestion. Geo-redundant failover should resolve within 5 minutes."
    }

@app.get("/remediation/{plan_id}")
async def get_remediation_plan(plan_id: str):
    """Get details of a specific remediation plan"""
    # Mock plan retrieval
    return {
        "id": plan_id,
        "status": "generated",
        "actions": [
            {
                "step": 1,
                "action": "Enable geo-redundant storage",
                "status": "pending",
                "requires_approval": True
            }
        ]
    }

@app.post("/remediation/{plan_id}/approve")
async def approve_remediation_plan(plan_id: str):
    """Approve a remediation plan for execution"""
    # In real implementation, this would trigger safety checks
    return {
        "message": f"Remediation plan {plan_id} approved",
        "approved_by": "human_operator",
        "timestamp": datetime.now(),
        "safety_checks": "passed",
        "execution_ready": True
    }

@app.get("/dashboard/metrics")
async def get_dashboard_metrics():
    """Get metrics for dashboard display"""
    return {
        "services_healthy": 3,
        "services_warning": 1,
        "services_error": 0,
        "active_alerts": 2,
        "uptime_percentage": 99.9,
        "average_latency": 108.75,
        "last_updated": datetime.now()
    }

@app.get("/system/status")
async def get_system_status():
    """Get overall system status"""
    return {
        "agents_active": 4,
        "services_monitored": 4,
        "communication_status": "healthy",
        "last_health_check": datetime.now(),
        "system_load": "normal",
        "memory_usage": "45%",
        "disk_usage": "23%"
    }

def run_api():
    """Run the FastAPI server"""
    port = int(os.getenv("API_PORT", 8000))
    logger.info(f"Starting SentinelNet API on port {port}")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_api()
