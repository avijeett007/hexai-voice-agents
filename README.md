# Hexai AI Agent

This repository contains the AI agent components for the Hexai healthcare platform.

## Structure

```
hexai_ai_agent/
├── Memory_Enabled_Agent/    # Legacy agent (reference only, no longer actively developed)
└── [New agent to be developed with modern architecture]
```

## Memory_Enabled_Agent (Legacy)

The `Memory_Enabled_Agent` folder contains the original AI agent implementation with memory capabilities. This serves as a reference implementation and will not be modified going forward.

### Features of Legacy Agent
- Memory-enabled conversations
- LiveKit integration for voice
- AutoGen framework
- Supabase integration
- RAG capabilities with LlamaIndex

## New Agent (Coming Soon)

A new agent will be developed from scratch with:
- Modern architecture based on new requirements
- Improved performance and scalability
- Enhanced memory management
- Better integration with the Hexai ecosystem

## Development

### Prerequisites
- Python 3.10+
- Docker (optional)
- Virtual environment recommended

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (for legacy agent reference)
cd Memory_Enabled_Agent
pip install -r requirements.txt
```

## Contributing

This repository is part of the Hexai monorepo. Please refer to the main repository documentation for contribution guidelines.

## License

Proprietary - All rights reserved

