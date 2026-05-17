#!/bin/bash
# CE Properties Wizard runner script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."

    # Check if Python is available
    PYTHON_CMD=""
    for cmd in python3 python py; do
        if command -v $cmd &> /dev/null; then
            PYTHON_CMD=$cmd
            break
        fi
    done

    if [ -z "$PYTHON_CMD" ]; then
        echo "Python is not installed. Please install Python first."
        exit 1
    fi

    # Install uv
    echo "Downloading and installing uv..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        # Add uv to PATH for current session
        export PATH="$HOME/.local/bin:$PATH"

        # Verify installation
        if ! command -v uv &> /dev/null; then
            echo "uv installation failed. Please install manually from https://docs.astral.sh/uv/"
            exit 1
        fi

        echo "uv installed successfully!"
    else
        echo "Failed to install uv automatically."
        echo "Please install manually from https://docs.astral.sh/uv/"
        exit 1
    fi
fi

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    uv sync --all-extras
fi

# Check for --format parameter
if [[ "$1" == "--format" ]]; then
    shift

    if [[ "$1" == "--check" ]]; then
        echo "Checking code formatting..."
        uv run black --check --diff .
        echo "Checking code with ruff..."
        uv run ruff check .
        echo "Running pytype..."
        uv run pytype .
        echo "All formatting checks passed!"
    else
        echo "Formatting code with black..."
        uv run black .
        echo "Formatting code with ruff..."
        uv run ruff check --fix .
        echo "Running pytype..."
        uv run pytype .
        echo "Code formatting complete!"
    fi
    exit 0
fi

# Run the wizard with all arguments passed through
uv run ce-props-wizard "$@"
