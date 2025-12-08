# SentinelNet: AI-Powered Multi-Cloud Resilience Platform

## 🚀 Vision: Intelligent Action on Enterprise Monitoring

**Datadog alerts you when BigQuery goes down. SentinelNet tells you exactly what to do and does it for you.**

**The Problem:** Teams spend hours investigating alerts and manually executing remediation. During major outages, this costs millions in downtime.

**The Solution:** SentinelNet connects to your existing Datadog monitoring and adds an intelligent action layer powered by LangGraph agents.

**Example Flow:**
```
Datadog Alert: "BigQuery US-east-1 High Latency"
     ↓
SentinelNet AI Agent: Analyzes alert + infrastructure context
     ↓
Intelligent Plan: "Switch to BigQuery US-west-2, update DNS, validate data consistency"
     ↓
Human Approval: SRE reviews cost/risk assessment
     ↓
Automated Execution: Terraform apply + monitoring + rollback ready
```

**Unlike pure monitoring tools, SentinelNet transforms alerts into automated, intelligent remediation.**

## ✨ Key Features

- **Datadog Integration**: Leverages existing enterprise monitoring infrastructure
- **Intelligent Action Engine**: Transforms Datadog alerts into executable remediation plans
- **Multi-Cloud Reconfiguration**: AI agents generate Terraform/ARM templates for failover
- **Impact Analysis**: Understands application dependencies and cascading effects
- **Safe Automation**: Human-in-the-loop validation with rollback capabilities
- **Cost Intelligence**: Automated cost-benefit analysis for reconfiguration decisions
- **Enterprise Dashboard**: Rich UI for alert triage and remediation approval

## 🔍 Datadog Integration: Monitoring + Intelligent Action

| Component | Datadog (Monitoring) | SentinelNet (Action Intelligence) |
|-----------|---------------------|----------------------------------|
| **Data Collection** | ✅ Metrics, logs, traces | ❌ (Uses Datadog data) |
| **Alert Generation** | ✅ Threshold-based alerts | ❌ (Processes Datadog alerts) |
| **Visualization** | ✅ Rich dashboards | ✅ Action-focused triage UI |
| **Root Cause Analysis** | ⚠️ Manual correlation | ✅ AI-powered dependency analysis |
| **Remediation Planning** | ❌ Alert-only | ✅ Multi-step recovery strategies |
| **Automated Execution** | ❌ Manual processes | ✅ Human-approved automation |
| **Cost Analysis** | ❌ Basic cost monitoring | ✅ Reconfiguration cost-benefit |

**SentinelNet enhances Datadog by adding the "What should we do?" and "Do it safely" layers that traditional monitoring lacks.**

## 🔄 Automated Reconfiguration Examples

### BigQuery Regional Failover
```
Outage Detected: BigQuery US-east-1 unavailable
↓
AI Analysis: Identifies 15 applications using this region
↓
Safe Reconfiguration Plan:
  • Generate BigQuery dataset replication commands
  • Create Terraform configuration for region switching
  • Validate data consistency and replication lag
  • Estimate costs: $50 downtime cost vs $200 cross-region transfer
↓
Human Approval: SRE reviews risk assessment and approves
↓
Execution: Automated Terraform apply with monitoring
↓
Verification: Confirm applications working in new region
↓
Rollback: Automated revert if issues detected
```

### Safe Reconfiguration Boundaries
- **What We Automate**: Dataset replication, endpoint switching, configuration updates
- **What Requires Human Approval**: Cost analysis, business logic validation, final execution
- **What We Never Touch**: Production databases, customer data, billing configurations
- **Safety Measures**: Comprehensive validation, cost estimation, rollback planning

### Vertex AI Endpoint Migration
```
Outage Detected: Vertex AI prediction endpoint down
↓
AI Analysis: Maps affected ML services and dependencies
↓
Reconfiguration Plan: Deploy model to backup region
↓
Cost-Benefit Analysis: Compare latency vs. availability
↓
Human Approval: ML engineer validates model compatibility
↓
Execution: Automated endpoint switching with traffic routing
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Datadog      │    │   SentinelNet   │    │   Enterprise    │
│   Monitoring    │───►│   AI Agents     │───►│   Systems       │
│                 │    │                 │    │                 │
│ • Real-time     │    │ • LangGraph     │    │ • Terraform     │
│ • Alerts        │    │ • Impact        │    │ • Kubernetes    │
│ • Metrics       │    │ • Reconfig      │    │ • Cloud APIs    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Action        │
                    │   Dashboard     │
                    │                 │
                    │ • Alert Triage  │
                    │ • Plan Review   │
                    │ • Execution     │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Hardware**: M1 Pro MacBook Pro (Apple Silicon)
- **OS**: macOS with all development tools installed
- **Python**: 3.9+ with UV package manager
- **Cloud Accounts**: GCP Always Free tier + Azure free services

### One-Click Setup (Recommended)
```bash
# Make script executable and run (only needed once)
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### Manual Setup
```bash
# Create environment
uv venv sentinelnet_env
source sentinelnet_env/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the application
python main.py
```

## 📁 Project Structure

```
sentinelnet/
├── Rule.txt              # Development rules and guidelines
├── Plan.txt              # Complete project roadmap
├── setup_and_run.sh      # One-click setup script
├── requirements.txt      # Python dependencies
├── main.py              # Application entry point
├── agents/              # AI agent implementations
│   ├── orchestrator.py   # LangGraph coordination
│   ├── gcp_monitor.py    # GCP service monitoring
│   ├── azure_monitor.py  # Azure service monitoring
│   └── remediation.py    # Remediation planning
├── dashboard/           # Streamlit dashboard
│   └── app.py           # Main dashboard application
├── api/                 # FastAPI backend
│   └── main.py          # API endpoints
├── data/                # Data storage and processing
├── models/              # ML models and anomaly detection
├── docs/                # Documentation
└── logs/                # Application logs
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Datadog Integration
DATADOG_API_KEY=your-datadog-api-key
DATADOG_APP_KEY=your-datadog-app-key
DATADOG_SITE=datadoghq.com  # or datadoghq.eu

# Cloud Platforms (for reconfiguration)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json

AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id
AZURE_SUBSCRIPTION_ID=your-subscription-id

# AI Services
OPENAI_API_KEY=your-openai-api-key

# Application
DEBUG=true
DASHBOARD_PORT=8501
DATABASE_URL=postgresql://localhost/sentinelnet
```

### Datadog Setup
1. **Create Datadog Account**: Sign up at [datadoghq.com](https://datadoghq.com)
2. **Get API Keys**: Navigate to Organization Settings → API Keys
3. **Configure Webhooks**: Set up webhooks to send alerts to SentinelNet
4. **Install Integrations**: Configure GCP and Azure integrations in Datadog

### Cloud Setup (for Reconfiguration Actions)

#### GCP Always Free Tier
- BigQuery: 1TB queries/month, 10GB storage
- Vertex AI: Limited predictions for AI analysis
- Cloud Storage: 5GB storage
- Compute Engine: f1-micro instances for testing

#### Azure Free Services
- Blob Storage: 5GB hot storage
- Functions: 1M executions/month
- DevOps: 5 users, unlimited repos
- App Service: Basic tier for testing

## 🎯 Development Phases

### Phase 1: Integration & Intelligence (Weeks 1-8)
- ✅ Datadog API integration for alert consumption
- ✅ LangGraph agent workflows for alert processing and correlation
- ✅ Impact analysis engine for dependency mapping
- ✅ AI-generated remediation plans with safety validation
- ✅ Human approval dashboard for plan review and execution
- ✅ One-click setup script with Datadog webhook configuration

### Phase 2: Enterprise Automation (Months 3-4)
- **Multi-Cloud Reconfiguration**:
  - BigQuery cross-region failover automation
  - Vertex AI endpoint migration with model validation
  - Azure Blob Storage geo-redundancy activation
  - DevOps pipeline region switching
- **Advanced Intelligence**:
  - Cost-benefit analysis for reconfiguration decisions
  - Predictive failure analysis based on historical data
  - Automated rollback and recovery validation
  - Multi-step remediation orchestration
- Enterprise integrations (Slack, PagerDuty, ServiceNow)

## 🛠️ Technology Stack

### Core Technologies
- **Datadog API**: Integration with enterprise monitoring platform
- **LangGraph**: Agent orchestration for intelligent remediation workflows
- **FastAPI**: Backend API for agent coordination and Datadog webhooks
- **Terraform**: Infrastructure automation for safe reconfiguration
- **Streamlit**: Interactive dashboard for alert triage and plan approval
- **PostgreSQL**: Store remediation plans, execution history, and cost analysis

### Action Intelligence Engine
- **Alert Processing**: Parse Datadog webhooks and correlate related alerts
- **Impact Analysis**: Graph-based dependency mapping and blast radius calculation
- **Plan Generation**: AI agents create multi-step remediation strategies
- **Safety Validation**: Automated risk assessment, cost estimation, and rollback planning
- **Execution Control**: Human-in-the-loop approval with automated execution

### Cloud & Infrastructure
- **Google Cloud**: BigQuery, Vertex AI, Cloud Monitoring
- **Azure**: Blob Storage, DevOps, Monitor
- **Docker**: Containerized deployment
- **Kubernetes**: Orchestration (local with Kind)

### AI/ML Stack
- **OpenAI GPT**: Remediation plan generation
- **Scikit-learn**: Statistical anomaly detection
- **Pandas/NumPy**: Data processing

## 🔒 Security & Safety

### Liability Protection
- **Demo-Only**: Never executes actions in production
- **Human Validation**: All remediation plans require approval
- **Test Credentials**: Only uses sandbox/test accounts
- **Safety Validation**: Automated risk assessment

### Security Features
- Encrypted credential storage
- Least privilege access
- Audit logging
- Rate limiting protection

## 📊 Monitoring & Metrics

### Performance Benchmarks
- Health checks: <5 seconds response time
- Memory usage: <500MB per agent
- False positives: <5% for anomaly detection

### Key Metrics
- Service availability detection accuracy
- Remediation plan generation time
- Cross-cloud correlation latency

## 🧪 Testing & Validation

### Testing Strategy
- Unit tests for individual components
- Integration tests for agent communication
- End-to-end tests with simulated outages
- Performance benchmarking

### Local Development
- Azurite for Azure Storage emulation
- BigQuery public datasets for testing
- Mock APIs for comprehensive testing

## 🚦 API Endpoints

### Datadog Integration
```
POST /webhooks/datadog     # Receive Datadog alerts and trigger analysis
GET /alerts/{alert_id}     # Get detailed alert information
POST /alerts/{alert_id}/acknowledge  # Acknowledge alert processing
```

### AI Analysis Engine
```
POST /analysis/impact      # Analyze alert impact and dependencies
POST /analysis/plan        # Generate remediation plan for alert
GET /analysis/{plan_id}    # Get detailed remediation plan
POST /analysis/{plan_id}/validate  # Validate plan safety and costs
```

### Execution Control
```
POST /execute/{plan_id}    # Execute approved remediation plan
POST /execute/{execution_id}/rollback  # Rollback failed execution
GET /execute/{execution_id}/status     # Get execution status
```

### Dashboard & Monitoring
```
GET /dashboard/alerts      # Get active alerts with AI analysis
GET /dashboard/plans       # Get pending remediation plans
GET /dashboard/executions  # Get execution history and costs
GET /dashboard/metrics     # Get system performance metrics
```

## 🤝 Contributing

### Development Guidelines
1. Follow the rules in `Rule.txt`
2. Follow the plan in `Plan.txt`
3. Use UV for environment management
4. Add comprehensive comments and documentation
5. Test thoroughly before committing

### Code Quality
- Black for code formatting
- isort for import sorting
- flake8 for linting
- pytest for testing

## 📈 Roadmap

### Short Term (MVP)
- Complete distributed monitoring system
- Implement basic LangGraph workflows
- Build offline-capable dashboard
- Create comprehensive documentation

### Medium Term (Phase 2)
- Add predictive capabilities
- Implement enterprise features
- Expand to additional cloud providers
- Enhance security and compliance

### Long Term (Future)
- Multi-tenant SaaS platform
- Integration with enterprise tools
- Advanced AI capabilities
- Global monitoring network

## 📞 Support

### Getting Help
1. Check the troubleshooting section in docs/
2. Review the setup script logs
3. Check the FAQ in documentation
4. Open an issue with detailed error logs

### Documentation
- `docs/setup.md`: Detailed setup instructions
- `docs/api.md`: API documentation
- `docs/troubleshooting.md`: Common issues and solutions
- `docs/architecture.md`: System architecture details

## ⚖️ Disclaimer

**Educational/Portfolio Project**: This project is designed for educational purposes and portfolio demonstration. It implements safety measures to prevent unintended actions, but should never be used in production environments without enterprise-grade security reviews and liability assessments.

**No Warranty**: The software is provided "as is" without warranty of any kind. Users are responsible for their own security and compliance requirements.

## 📄 License

This project is for educational and portfolio purposes. See individual component licenses for distribution terms.

---

**Built with ❤️ on M1 Pro MacBook Pro**

*Demonstrating advanced AI, distributed systems, and cloud resilience*
