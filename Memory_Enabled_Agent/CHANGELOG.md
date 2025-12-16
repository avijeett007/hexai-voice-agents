# Hexai LiveKit Agent - Changelog

## Version 2.0 - Enhanced Multi-Platform Support (December 15, 2024)

### 🎉 New Features

#### 1. **Hexai Experience App Support**
- ✅ Added support for Experience App rooms (`hexai-experience-{random8}-{userType}`)
- ✅ Platform-neutral greeting for Experience App users
- ✅ Automatic user type detection (`experience`)
- ✅ Treats Experience App users as medical professionals (doctor-level access)

#### 2. **Conversation Analytics Tracking**
- ✅ AI-powered conversation summary generation using GPT-4o-mini
- ✅ Automatic satisfaction score calculation (1.0-5.0 scale)
- ✅ Analytics sent to Embedding Engine API after conversation ends
- ✅ Tracks conversation duration, engagement, and quality metrics

**Analytics Metrics:**
- Conversation summary (AI-generated)
- Satisfaction score (heuristic algorithm)
- Duration tracking
- User engagement metrics
- Session metadata

#### 3. **Dynamic Knowledge Base Loading**
- ✅ Fetches organization-specific knowledge bases from API
- ✅ Uses `customer_id` (organizationId) for knowledge base mapping
- ✅ Caches knowledge base mappings for performance
- ✅ Falls back to default knowledge base if API unavailable
- ✅ Supports unlimited knowledge bases per organization

#### 4. **Enhanced Room Validation**
- ✅ Added `EXPERIENCE_PREFIX` constant for Experience App detection
- ✅ Updated `validate_room_name()` to handle Experience App rooms
- ✅ Maintains backward compatibility with NHS Voice Portal rooms

#### 5. **Improved Metadata Handling**
- ✅ Fixed critical bug: Changed `customer_id: userId` to `customer_id: metadata?.organizationId || userId`
- ✅ Prevents knowledge base fetching failures for Experience App users
- ✅ Proper organization-level knowledge base mapping

### 🔧 Technical Improvements

#### API Integration
- ✅ Added `ANALYTICS_ENDPOINT` constant
- ✅ Implemented `send_conversation_analytics()` function
- ✅ Implemented `calculate_satisfaction_score()` function
- ✅ Implemented `generate_conversation_summary()` function

#### Code Organization
- ✅ Added `create_experience_welcome()` function for platform-specific greetings
- ✅ Enhanced error handling for analytics submission
- ✅ Improved logging for debugging

#### Docker & Deployment
- ✅ Updated Dockerfile to use `requirements-docker.txt` for pinned versions
- ✅ Enhanced docker-compose.yml with health checks and resource limits
- ✅ Created `.env.production.example` with all required variables
- ✅ Added production deployment guide (DEPLOYMENT.md)
- ✅ Created startup script (`start-production.sh`)

### 📊 Satisfaction Score Algorithm

The satisfaction score is calculated based on:

**Positive Factors:**
- Conversation length (longer = more engaged)
- Multiple exchanges (3+ exchanges = +0.3, 5+ = +0.2)
- Balanced conversation (assistant/user ratio 0.8-1.5 = +0.3)
- Longer messages (avg >20 chars = +0.2)
- Medical terminology usage (+0.2)

**Negative Factors:**
- Very short conversations (<30s = -0.5)
- Repetitive questions (-0.3)
- Error indicators (-0.5)

**Score Range:** 1.0 - 5.0

### 🔄 Backward Compatibility

All existing NHS Voice Portal functionality is preserved:
- ✅ Doctor rooms (`nhs-{random8}-doctor`)
- ✅ Patient rooms (`nhs-{random8}-patient`)
- ✅ Demo rooms (`nhs-{random8}-hexaidemo`)
- ✅ Intro rooms (`nhs-{random8}-intro`)
- ✅ Memory persistence
- ✅ Medical records access
- ✅ Consent management

### 📝 Configuration Changes

**New Environment Variables:**
- `CARTESIA_API_KEY` - Required for TTS
- `DEEPGRAM_API_KEY` - Required for STT
- `CEREBRAS_API_KEY` - Optional for cost optimization
- `NHS_API_BASE` - Backend API URL
- `NHS_API_KEY` - Backend API authentication

**Updated Endpoints:**
- Added: `ANALYTICS_ENDPOINT = ${NHS_API_BASE}/analytics/session`

### 🐛 Bug Fixes

1. **Critical:** Fixed knowledge base fetching for Experience App users
   - Changed `customer_id: userId` to `customer_id: metadata?.organizationId || userId`
   - Prevents 404 errors when fetching organization knowledge bases

2. **Enhancement:** Improved error handling for analytics submission
   - Non-blocking: Analytics failures don't crash the agent
   - Detailed logging for debugging

### 📦 Dependencies

**Added:**
- `cerebras_cloud_sdk>=0.1.0` - Cost-efficient AI for knowledge base selection

**Updated:**
- All LiveKit plugins to latest stable versions
- OpenAI SDK to 1.12.0+

### 🚀 Deployment

**Docker Compose:**
```bash
# Quick start
./start-production.sh

# Or manually
docker-compose up -d
```

**Environment Setup:**
```bash
cp .env.production.example .env
# Edit .env with your credentials
```

### 📚 Documentation

**New Files:**
- `DEPLOYMENT.md` - Comprehensive production deployment guide
- `.env.production.example` - Environment variable template
- `start-production.sh` - Interactive startup script
- `CHANGELOG.md` - This file

**Updated Files:**
- `Dockerfile` - Uses requirements-docker.txt
- `docker-compose.yml` - Enhanced with health checks and resource limits

---

## Version 1.0 - Initial Release

### Features
- NHS Voice Portal support (doctor/patient rooms)
- Vector-based memory storage (Qdrant)
- Medical knowledge base integration
- API integration for patient/doctor data
- Consent management
- Cerebras integration for cost optimization

---

**For detailed deployment instructions, see [DEPLOYMENT.md](./DEPLOYMENT.md)**

