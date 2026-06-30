#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== APEX Environment Setup (uv) ==="

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create venv with Python 3.10
echo "Creating virtual environment with Python 3.10..."
uv venv "$VENV_DIR" --python 3.10

# Use pfss for uv cache to avoid overlay disk space issues
export UV_CACHE_DIR="$PROJECT_DIR/.uv_cache"

# Install PyTorch with CUDA 12.4 (compatible with driver >=12.4)
echo "Installing PyTorch (cu124)..."
uv pip install --python "$VENV_DIR/bin/python" \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
echo "Installing dependencies..."
uv pip install --python "$VENV_DIR/bin/python" -r "$PROJECT_DIR/requirements.txt"

# Download spaCy model
echo "Downloading spaCy English model..."
uv run --python "$VENV_DIR/bin/python" python -m spacy download en_core_web_sm

# Create .env from example if it doesn't exist
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "Created .env from .env.example — fill in your API keys."
else
    echo ".env already exists, skipping."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
