# Hexai LiveKit Agent - Enhanced Multi-Platform Voice Assistant

**Version 2.0** - Production Ready with Experience App Support

---

## 📋 Overview

This is the production-ready LiveKit agent for the Hexai platform, supporting:

### Supported Platforms

1. **NHS Voice Portal** (`nhs-{random8}-doctor` / `nhs-{random8}-patient`)
   - Full NHS-branded experience
   - Medical records access
   - Consent management
   - Memory persistence

2. **Hexai Experience App** (`hexai-experience-{random8}-{userType}`)
   - Platform-neutral branding
   - Organization-specific knowledge bases
   - Conversation analytics tracking
   - Professional medical assistant experience

3. **Demo & Intro Modes**
   - Demo rooms for testing
   - Intro rooms for onboarding

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- LiveKit server running
- Qdrant vector database
- OpenAI, Cartesia, and Deepgram API keys
- NHS Embedding Engine Portal API

### Installation

```bash
# 1. Copy environment template
cp .env.production.example .env

# 2. Edit .env with your credentials
nano .env

# 3. Start the agent
./start-production.sh
```

Or manually:

```bash
docker-compose up -d
```

---

## 🎯 Key Features

### 1. Multi-Platform Support
- Automatic platform detection based on room name
- Platform-specific greetings and behavior
- Unified codebase for all platforms

### 2. Conversation Analytics (Experience App Only)
- AI-powered conversation summaries
- Satisfaction score calculation (1.0-5.0)
- Automatic submission to Embedding Engine API
- Duration and engagement tracking

### 3. Dynamic Knowledge Base Loading
- Organization-specific knowledge bases
- Fetched from Embedding Engine API
- Cached for performance
- Fallback to default knowledge base

### 4. Cost Optimization
- Cerebras for knowledge base selection (90% cost reduction)
- OpenAI fallback for reliability
- Efficient vector search with Qdrant

### 5. Memory Persistence
- User-specific memory collections
- Vector-based storage in Qdrant
- Context-aware conversations

---

## 📦 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LiveKit Server                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Hexai LiveKit Agent                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Room         │  │ Knowledge    │  │ Analytics    │     │
│  │ Detection    │  │ Base Loader  │  │ Tracker      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Memory       │  │ AI Models    │  │ Voice        │     │
│  │ Manager      │  │ (GPT/Cerebras)│  │ Processing   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
         ↓                  ↓                    ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Qdrant     │  │  Embedding   │  │  OpenAI      │
│   Vector DB  │  │  Engine API  │  │  Cartesia    │
│              │  │              │  │  Deepgram    │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 🔧 Configuration

### Required Environment Variables

See `.env.production.example` for complete list.

**Critical Variables:**
- `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL`
- `OPENAI_API_KEY`
- `CARTESIA_API_KEY`, `DEEPGRAM_API_KEY`
- `QDRANT_HOST`, `QDRANT_API_KEY`
- `NHS_API_BASE`, `NHS_API_KEY`

**Optional:**
- `CEREBRAS_API_KEY` (for cost optimization)
- `MODEL_NAME` (default: gpt-4o-mini)

---

## 📊 Analytics System

For Experience App users, the agent automatically tracks:

### Metrics Collected
- **Conversation Summary:** AI-generated summary of the conversation
- **Satisfaction Score:** 1.0-5.0 based on engagement metrics
- **Duration:** Total conversation time
- **Engagement:** Message count, balance, medical terminology usage

### Satisfaction Score Algorithm

**Positive Factors:**
- Longer conversations (+0.5 for >2 min)
- Multiple exchanges (+0.3 for 3+, +0.2 for 5+)
- Balanced conversation (+0.3)
- Longer messages (+0.2)
- Medical terminology (+0.2)

**Negative Factors:**
- Very short (<30s: -0.5)
- Repetitive questions (-0.3)
- Error indicators (-0.5)

---

## 🐛 Troubleshooting

### Common Issues

**Agent won't start:**
```bash
# Check logs
docker-compose logs -f

# Rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**Can't connect to LiveKit:**
- Verify `LIVEKIT_URL` is correct
- Check firewall settings
- Ensure LiveKit server is running

**Qdrant connection issues:**
- Verify `QDRANT_TLS=true` for cloud instances
- Check API key is correct
- Test connectivity: `ping <QDRANT_HOST>`

---

## 📚 Documentation

- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Complete deployment guide
- **[CHANGELOG.md](./CHANGELOG.md)** - Version history and changes
- **[.env.production.example](./.env.production.example)** - Environment variables

---

## 🔄 Updates

```bash
# Pull latest code
git pull origin develop

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

---

## 📝 Files

- `nhs_agents.py` - Main agent code (2699 lines)
- `Dockerfile` - Production Docker image
- `docker-compose.yml` - Docker Compose configuration
- `requirements-docker.txt` - Python dependencies (pinned versions)
- `start-production.sh` - Interactive startup script

---

## 🆘 Support

For issues or questions, check:
1. Logs: `docker-compose logs -f`
2. [DEPLOYMENT.md](./DEPLOYMENT.md) troubleshooting section
3. Contact Hexai Development Team

---

**Production Ready! 🚀**

