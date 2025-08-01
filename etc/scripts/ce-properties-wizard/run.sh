#!/bin/bash
# CE Properties Wizard runner script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Installing Poetry..."
    
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
    
    # Install Poetry
    echo "Downloading and installing Poetry..."
    if curl -sSL https://install.python-poetry.org | $PYTHON_CMD -; then
        # Add Poetry to PATH for current session
        export PATH="$HOME/.local/bin:$PATH"
        
        # Verify installation
        if ! command -v poetry &> /dev/null; then
            echo "Poetry installation failed. Please install manually from https://python-poetry.org/docs/#installation"
            exit 1
        fi
        
        echo "Poetry installed successfully!"
    else
        echo "Failed to install Poetry automatically."
        echo "Please install manually from https://python-poetry.org/docs/#installation"
        exit 1
    fi
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