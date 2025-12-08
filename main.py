#!/usr/bin/env python3
"""
SentinelNet - AI-Powered Multi-Cloud Resilience Platform
Main entry point for the SentinelNet application

Author: Vinayak Pawar
Version: 1.0
Compatible with: M1 Pro MacBook Pro (Apple Silicon)
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sentinelnet.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    """Main application entry point"""
    logger.info("🚀 Starting SentinelNet...")

    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("⚠️  OPENAI_API_KEY not found in environment variables")
        logger.warning("Please edit the .env file with your API keys")

    # Check if we're in demo mode
    demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'

    if demo_mode:
        logger.info("🎯 Running in demo mode - using mock data and emulators")

    try:
        # Import core components
        from agents.orchestrator import SentinelNetOrchestrator
        from agents.gcp_monitor import get_gcp_monitor
        from agents.communication import initialize_communication
        from dashboard.app import run_dashboard

        # Initialize communication layer
        logger.info("📡 Initializing communication layer...")
        comm_manager = await initialize_communication()

        # Initialize orchestrator
        logger.info("🎯 Initializing SentinelNet orchestrator...")
        orchestrator = SentinelNetOrchestrator()

        # Initialize GCP monitor
        logger.info("🔍 Initializing GCP monitor...")
        gcp_monitor = get_gcp_monitor()

        # Register GCP monitor with orchestrator
        agent_info = await orchestrator.register_agent(gcp_monitor.agent_info)

        # Start monitoring in background
        if demo_mode:
            logger.info("🎮 Starting monitoring in demo mode...")
            # Start GCP monitoring in background
            monitor_task = asyncio.create_task(gcp_monitor.start_monitoring())
        else:
            logger.info("🔗 Starting real GCP monitoring...")
            monitor_task = asyncio.create_task(gcp_monitor.start_monitoring())

        # Start dashboard
        logger.info("📊 Starting dashboard...")
        run_dashboard()

    except ImportError as e:
        logger.error(f"❌ Failed to import required modules: {e}")
        logger.info("💡 Make sure all requirements are installed: uv pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down SentinelNet...")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

def check_environment():
    """Check if the environment is properly configured"""
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("⚠️  Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("Please edit the .env file with your actual API keys")
        return False

    # Check for optional cloud credentials
    gcp_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if gcp_creds and not Path(gcp_creds).exists():
        print(f"⚠️  GCP credentials file not found: {gcp_creds}")

    print("✅ Environment configuration looks good!")
    return True

if __name__ == "__main__":
    print("🚀 SentinelNet - AI-Powered Multi-Cloud Resilience Platform")
    print("=" * 60)

    # Check environment before starting
    if not check_environment():
        print("\n💡 Tip: Run ./setup_and_run.sh to configure your environment")
        sys.exit(1)

    # Run the main application
    asyncio.run(main())
