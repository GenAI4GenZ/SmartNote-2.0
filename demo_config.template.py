# Configuration Template for Release Note Dashboard
# Copy this file to config.py and fill in your actual values

# API Keys
GITHUB_TOKEN = "your-github-token-here"
OPENAI_API_KEY = "your-openai-api-key-here"

# List of repositories to choose from
REPO_LIST = [
    "WerWolv/ImHex",
    "janhq/jan",
    "rustdesk/rustdesk",
    "GyulyVGC/sniffnet",
    "Stirling-Tools/Stirling-PDF",
    "marticliment/UniGetUI",
    "zulip/zulip",
    "akfamily/akshare",
    "bentoml/bentoml",
    "toss/es-toolkit",
    "google/flatbuffers",
    "sisong/HDiffPatch",
    "langchain-ai/langchain",
    "twpayne/chezmoi",
    "continuedev/continue",
    "fullstorydev/grpcurl",
    # Add more repositories as needed
]

# NOTE: For production, consider using environment variables or a secure vault
# instead of hardcoding sensitive information like API keys.
