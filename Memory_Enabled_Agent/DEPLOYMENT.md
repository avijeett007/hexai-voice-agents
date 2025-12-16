# Hexai LiveKit Agent - Production Deployment Guide

## 📋 Overview

This is the enhanced NHS Virtual Assistant LiveKit Agent with support for:
- ✅ NHS Voice Portal (doctor/patient rooms)
- ✅ Hexai Experience App (hexai-experience- rooms)
- ✅ Conversation analytics tracking
- ✅ Dynamic knowledge base loading
- ✅ Vector-based memory storage (Qdrant)
- ✅ Cost-efficient AI with Cerebras fallback

**Last Updated:** December 15, 2024  
**Version:** 2.0 (Enhanced with Experience App support)

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- LiveKit server running and accessible
- Qdrant vector database instance
- OpenAI API key
- Cartesia API key (for TTS)
- Deepgram API key (for STT)
- NHS Embedding Engine Portal API running

### 1. Environment Setup

```bash
# Copy the environment template
cp .env.production.example .env

# Edit .env and fill in all required values
nano .env
```

### 2. Build and Run with Docker Compose

```bash
# Build the Docker image
docker-compose build

# Start the agent
docker-compose up -d

# View logs
docker-compose logs -f nhs-agent

# Stop the agent
docker-compose down
```

### 3. Verify Deployment

```bash
# Check if container is running
docker ps | grep hexai-livekit-agent

# Check health status
docker inspect hexai-livekit-agent | grep -A 10 Health

# View recent logs
docker-compose logs --tail=100 nhs-agent
```

---

## 🔧 Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LIVEKIT_API_KEY` | LiveKit API key | `APIxxxxxxxxxxxx` |
| `LIVEKIT_API_SECRET` | LiveKit API secret | `secretxxxxxxxxx` |
| `LIVEKIT_URL` | LiveKit server URL | `wss://livekit.hexai.care` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-xxxxxxxxxxxxxxxx` |
| `CARTESIA_API_KEY` | Cartesia TTS API key | `xxxxxxxxxxxxxxxx` |
| `DEEPGRAM_API_KEY` | Deepgram STT API key | `xxxxxxxxxxxxxxxx` |
| `QDRANT_HOST` | Qdrant server host | `qdrant.hexai.care` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `QDRANT_API_KEY` | Qdrant API key | `xxxxxxxxxxxxxxxx` |
| `QDRANT_TLS` | Enable TLS for Qdrant | `true` |
| `NHS_API_BASE` | Embedding Engine API URL | `https://api.hexai.care/api` |
| `NHS_API_KEY` | Embedding Engine API key | `xxxxxxxxxxxxxxxx` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CEREBRAS_API_KEY` | Cerebras API key (cost-efficient) | Falls back to OpenAI |
| `MODEL_NAME` | OpenAI model name | `gpt-4o-mini` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## 📦 Docker Build Options

### Option 1: Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Option 2: Docker Build Manually

```bash
# Build the image
docker build -t hexai-livekit-agent:latest -f Dockerfile .

# Run the container
docker run -d \
  --name hexai-livekit-agent \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -p 8080:8080 \
  --restart unless-stopped \
  hexai-livekit-agent:latest
```

---

## 🔍 Monitoring and Logs

### View Live Logs

```bash
# All logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific time range
docker-compose logs --since 30m
```

### Log Files

Logs are stored in:
- Container: `/app/logs/`
- Host: `./logs/` (mounted volume)

### Health Checks

The container includes automatic health checks:
- **Interval:** 30 seconds
- **Timeout:** 10 seconds
- **Retries:** 3
- **Start Period:** 30 seconds

Check health status:
```bash
docker inspect hexai-livekit-agent --format='{{.State.Health.Status}}'
```

---

## 🎯 Features

### 1. Multi-Platform Support

**NHS Voice Portal:**
- Room pattern: `nhs-{random8}-doctor` or `nhs-{random8}-patient`
- Full medical knowledge base access
- Memory persistence per user

**Hexai Experience App:**
- Room pattern: `hexai-experience-{random8}-{userType}`
- Platform-neutral greeting
- Analytics tracking enabled
- Dynamic knowledge base loading

### 2. Analytics Tracking

For Experience App users, the agent automatically:
- ✅ Generates AI-powered conversation summaries
- ✅ Calculates satisfaction scores (1.0-5.0)
- ✅ Sends analytics to Embedding Engine API
- ✅ Tracks conversation duration and engagement

### 3. Dynamic Knowledge Base

- Fetches organization-specific knowledge bases from API
- Caches knowledge base mappings for performance
- Falls back to default knowledge base if API unavailable

### 4. Cost Optimization

- Uses Cerebras for knowledge base selection (90% cost reduction)
- Falls back to OpenAI if Cerebras unavailable
- Efficient vector search with Qdrant

---

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs for errors
docker-compose logs nhs-agent

# Verify environment variables
docker-compose config

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Agent Not Connecting to LiveKit

1. Verify LiveKit credentials in `.env`
2. Check LiveKit server is accessible
3. Verify network connectivity:
   ```bash
   docker exec hexai-livekit-agent ping livekit.hexai.care
   ```

### Qdrant Connection Issues

1. Verify Qdrant credentials
2. Check TLS settings (`QDRANT_TLS=true`)
3. Test connection:
   ```bash
   docker exec hexai-livekit-agent python -c "from qdrant_client import QdrantClient; print('OK')"
   ```

### High Memory Usage

Adjust resource limits in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 2G  # Reduce if needed
```

---

## 🔄 Updates and Maintenance

### Update Agent Code

```bash
# Pull latest changes
git pull origin develop

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Backup Data

```bash
# Backup data directory
tar -czf hexai-agent-data-$(date +%Y%m%d).tar.gz ./data

# Backup logs
tar -czf hexai-agent-logs-$(date +%Y%m%d).tar.gz ./logs
```

---

## 📊 Resource Requirements

**Minimum:**
- CPU: 1 core
- RAM: 2GB
- Disk: 5GB

**Recommended:**
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB

---

## 🆘 Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Review this documentation
3. Contact: Hexai Development Team

---

**Production Ready! 🚀**

