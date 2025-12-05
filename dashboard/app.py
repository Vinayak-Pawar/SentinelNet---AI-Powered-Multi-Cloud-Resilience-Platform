#!/usr/bin/env python3
"""
SentinelNet Dashboard
Interactive dashboard for monitoring cloud services and viewing remediation plans

Author: Vinayak Pawar
Version: 1.0
Compatible with: M1 Pro MacBook Pro (Apple Silicon)
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import random
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="SentinelNet - Cloud Resilience Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock data for demo purposes
def generate_mock_service_data():
    """Generate mock service health data for demonstration"""
    services = [
        {"name": "BigQuery", "cloud": "GCP", "status": "healthy", "latency": 45},
        {"name": "Vertex AI", "cloud": "GCP", "status": "healthy", "latency": 67},
        {"name": "Blob Storage", "cloud": "Azure", "status": "warning", "latency": 234},
        {"name": "DevOps", "cloud": "Azure", "status": "healthy", "latency": 89}
    ]

    # Randomly simulate some issues for demo
    if random.random() < 0.3:  # 30% chance of issues
        affected = random.choice(services)
        affected["status"] = "error" if random.random() < 0.5 else "warning"
        affected["latency"] = random.randint(500, 2000)

    return services

def run_dashboard():
    """Main dashboard application"""
    st.title("🛡️ SentinelNet - Cloud Resilience Dashboard")
    st.markdown("*AI-Powered Multi-Cloud Outage Detection & Remediation*")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")

        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()

        st.divider()

        st.subheader("🎯 Quick Actions")
        if st.button("🚨 Simulate Outage"):
            st.session_state.simulate_outage = True

        if st.button("📋 Generate Remediation Plan"):
            st.session_state.show_remediation = True

        st.divider()

        st.subheader("⚙️ Settings")
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        demo_mode = st.checkbox("Demo Mode", value=True)

        st.divider()

        st.markdown("### 📊 System Status")
        st.metric("Active Agents", "4/4", "🟢")
        st.metric("Services Monitored", "4", "🟢")
        st.metric("Uptime", "99.9%", "🟢")

    # Main content
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌐 Service Health Overview")
        services = generate_mock_service_data()

        # Service status cards
        for service in services:
            status_color = {
                "healthy": "🟢",
                "warning": "🟡",
                "error": "🔴"
            }

            with st.container():
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    st.write(f"{status_color[service['status']]} {service['name']} ({service['cloud']})")
                with col_b:
                    st.metric("Latency", f"{service['latency']}ms")
                with col_c:
                    st.write(service['status'].title())

    with col2:
        st.subheader("📈 Real-time Metrics")

        # Mock metrics chart
        hours = list(range(24))
        healthy_count = [4 - random.randint(0, 1) for _ in hours]

        chart_data = pd.DataFrame({
            'Hour': hours,
            'Healthy Services': healthy_count
        })

        st.line_chart(chart_data.set_index('Hour'))

        # Current stats
        st.metric("Current Health Score", "95%", "🟢")

    with col3:
        st.subheader("🚨 Active Alerts")

        # Mock alerts
        alerts = [
            {"level": "warning", "service": "Blob Storage", "message": "High latency detected", "time": "2 min ago"},
            {"level": "info", "service": "System", "message": "Agent health check passed", "time": "5 min ago"}
        ]

        for alert in alerts:
            with st.container():
                if alert["level"] == "warning":
                    st.warning(f"⚠️ {alert['service']}: {alert['message']}")
                elif alert["level"] == "error":
                    st.error(f"❌ {alert['service']}: {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['service']}: {alert['message']}")
                st.caption(alert['time'])

    st.divider()

    # Remediation Planning Section
    if st.session_state.get('show_remediation', False):
        st.header("🛠️ AI-Generated Remediation Plan")

        with st.container():
            st.subheader("📋 Incident Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Affected Services", "1", "🔴")
                st.metric("Estimated Impact", "Medium", "🟡")

            with col2:
                st.metric("Recovery Time", "~15 min", "🟢")
                st.metric("Risk Level", "Low", "🟢")

            st.markdown("**Detected Issue:** High latency on Azure Blob Storage")
            st.markdown("**Root Cause:** Potential regional network congestion")

            st.subheader("🎯 Recommended Actions")

            actions = [
                {
                    "priority": "High",
                    "action": "Switch to geo-redundant storage (RA-GRS)",
                    "estimated_time": "5 minutes",
                    "risk": "Low",
                    "automated": True
                },
                {
                    "priority": "Medium",
                    "action": "Scale up compute resources in secondary region",
                    "estimated_time": "10 minutes",
                    "risk": "Medium",
                    "automated": False
                },
                {
                    "priority": "Low",
                    "action": "Update traffic routing policies",
                    "estimated_time": "2 minutes",
                    "risk": "Low",
                    "automated": True
                }
            ]

            for action in actions:
                with st.expander(f"{action['priority']} Priority: {action['action']}", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Time", action['estimated_time'])
                    with col2:
                        st.metric("Risk", action['risk'])
                    with col3:
                        st.write("🤖 Auto" if action['automated'] else "👤 Manual")
                    with col4:
                        if st.button("Execute", key=f"execute_{action['action'][:20]}"):
                            st.success("✅ Action executed successfully!")

        # Close remediation view
        if st.button("❌ Close Remediation Plan"):
            st.session_state.show_remediation = False
            st.rerun()

    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🛡️ SentinelNet v1.0 - MVP")
    with col2:
        st.caption("Built on M1 Pro MacBook Pro")
    with col3:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    run_dashboard()
