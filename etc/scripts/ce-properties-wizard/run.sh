#!/bin/bash
# CE Properties Wizard runner script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install it first:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    poetry install
fi

# Check for --format parameter
if [[ "$1" == "--format" ]]; then
    shift
    
    if [[ "$1" == "--check" ]]; then
        echo "Checking code formatting..."
        poetry run black --check --diff .
        echo "Checking code with ruff..."
        poetry run ruff check .
        echo "Running pytype..."
        poetry run pytype .
        echo "All formatting checks passed!"
    else
        echo "Formatting code with black..."
        poetry run black .
        echo "Formatting code with ruff..."
        poetry run ruff check --fix .
        echo "Running pytype..."
        poetry run pytype .
        echo "Code formatting complete!"
    fi
    exit 0
fi

# Run the wizard with all arguments passed through
poetry run ce-props-wizard "$@"