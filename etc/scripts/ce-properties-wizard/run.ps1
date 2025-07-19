# CE Properties Wizard runner script for Windows PowerShell

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check if poetry is installed
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "Poetry is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "  Invoke-RestMethod -Uri https://install.python-poetry.org | python -" -ForegroundColor Yellow
    Write-Host "  Or visit: https://python-poetry.org/docs/#installation" -ForegroundColor Yellow
    exit 1
}

# Install dependencies if needed
if (-not (Test-Path ".venv")) {
    Write-Host "Setting up virtual environment..." -ForegroundColor Green
    poetry install
}

# Run the wizard with all arguments passed through
if ($Arguments) {
    poetry run ce-props-wizard @Arguments
} else {
    poetry run ce-props-wizard
}