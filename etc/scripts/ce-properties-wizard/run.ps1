# CE Properties Wizard runner script for Windows PowerShell

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Installing uv..." -ForegroundColor Yellow

    try {
        # Download and install uv
        Write-Host "Downloading uv installer..." -ForegroundColor Green
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

        # Verify installation
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Host "uv installation failed. Please install manually from https://docs.astral.sh/uv/" -ForegroundColor Red
            exit 1
        }

        Write-Host "uv installed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "Failed to install uv automatically: $_" -ForegroundColor Red
        Write-Host "Please install manually from https://docs.astral.sh/uv/" -ForegroundColor Yellow
        exit 1
    }
}

# Install dependencies if needed
if (-not (Test-Path ".venv")) {
    Write-Host "Setting up virtual environment..." -ForegroundColor Green
    uv sync --all-extras
}

# Run the wizard with all arguments passed through
if ($Arguments) {
    uv run ce-props-wizard @Arguments
} else {
    uv run ce-props-wizard
}
