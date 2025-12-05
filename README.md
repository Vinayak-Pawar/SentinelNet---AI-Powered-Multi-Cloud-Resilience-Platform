# SentinelNet: AI-Powered Multi-Cloud Resilience Platform

## 🚀 Overview

SentinelNet is an innovative distributed AI system that monitors cloud services and generates intelligent remediation plans during outages. Unlike traditional monitoring tools that fail when the cloud goes down, SentinelNet uses a hybrid communication architecture to maintain resilience.

## ✨ Key Features

- **Distributed Monitoring**: AI agents across multiple clouds monitoring service health
- **Intelligent Remediation**: LangGraph-powered agents generate human-validated recovery plans
- **Hybrid Communication**: Works during cloud outages using P2P networks
- **Multi-Cloud Support**: Azure + GCP with enterprise-grade security
- **Offline Dashboard**: PWA that works without internet connectivity

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GCP Agents    │    │   Azure Agents  │    │ Coordination   │
│                 │    │                 │    │    Layer       │
│ • BigQuery      │◄──►│ • Blob Storage  │◄──►│ • LangGraph    │
│ • Vertex AI     │    │ • DevOps        │    │ • Consensus    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  User Dashboard │
                    │    (Offline)    │
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
# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json

# Azure
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Application
DEBUG=true
DASHBOARD_PORT=8501
```

### Cloud Setup

#### GCP Always Free Tier
- BigQuery: 1TB queries/month, 10GB storage
- Vertex AI: Limited predictions
- Cloud Storage: 5GB storage

#### Azure Free Services
- Blob Storage: 5GB hot storage
- Functions: 1M executions/month
- DevOps: 5 users, unlimited repos

## 🎯 Development Phases

### Phase 1: MVP (Weeks 1-8)
- ✅ Distributed monitoring agents
- ✅ Anomaly detection and correlation
- ✅ AI-generated remediation plans
- ✅ Offline-capable dashboard
- ✅ One-click setup script

### Phase 2: Enterprise Features (Months 3-4)
- Human-approved automated execution
- ML-based outage prediction
- Multi-region support
- Enterprise alerting

## 🛠️ Technology Stack

### Core Technologies
- **LangGraph**: Agent orchestration and workflows
- **FastAPI**: High-performance API backend
- **Streamlit**: Interactive dashboard
- **WebRTC**: P2P communication during outages

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

### Health Monitoring
```
GET /health/services    # Get all service statuses
GET /health/{service}   # Get specific service health
POST /health/check      # Manual health check
```

### Remediation Planning
```
POST /remediation/plan     # Generate remediation plan
GET /remediation/{plan_id} # Get plan details
POST /remediation/execute  # Execute approved plan (with confirmation)
```

### Dashboard Data
```
GET /dashboard/metrics     # Get dashboard metrics
GET /dashboard/alerts      # Get active alerts
GET /dashboard/incidents   # Get incident history
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
