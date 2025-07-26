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
    Write-Host "Poetry is not installed. Installing Poetry..." -ForegroundColor Yellow
    
    # Check if Python is available
    $pythonCmd = $null
    foreach ($cmd in @("python", "python3", "py")) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            $pythonCmd = $cmd
            break
        }
    }
    
    if (-not $pythonCmd) {
        Write-Host "Python is not installed. Please install Python first." -ForegroundColor Red
        exit 1
    }
    
    try {
        # Download and install Poetry
        Write-Host "Downloading Poetry installer..." -ForegroundColor Green
        $poetryInstaller = Invoke-RestMethod -Uri https://install.python-poetry.org
        $poetryInstaller | & $pythonCmd -
        
        # Update PATH for current session
        $env:Path = "$env:APPDATA\Python\Scripts;$env:Path"
        
        # Verify installation
        if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
            Write-Host "Poetry installation failed. Please install manually from https://python-poetry.org/docs/#installation" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Poetry installed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "Failed to install Poetry automatically: $_" -ForegroundColor Red
        Write-Host "Please install manually from https://python-poetry.org/docs/#installation" -ForegroundColor Yellow
        exit 1
    }
}

# Install dependencies if needed
if (-not (Test-Path ".venv")) {
    Write-Host "Setting up virtual environment..." -ForegroundColor Green
    # On Windows, use --only main to skip dev dependencies and avoid pytype build issues
    poetry install --only main
    Write-Host "Note: Development dependencies skipped on Windows (pytype doesn't build on Windows)" -ForegroundColor Yellow
}

# Check if we're running under Git Bash (which can cause issues with Poetry)
$isGitBash = $false
if ($env:SHELL -match "bash" -or $env:MSYSTEM) {
    $isGitBash = $true
    Write-Host "Warning: Git Bash detected. This may cause issues with Poetry." -ForegroundColor Yellow

    # Find the virtual environment
    $venvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        # Check Poetry's cache location
        $poetryVenvs = "$env:LOCALAPPDATA\pypoetry\Cache\virtualenvs"
        $venvDir = Get-ChildItem $poetryVenvs -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "ce-properties-wizard*" } | Select-Object -First 1
        if ($venvDir) {
            $venvPython = Join-Path $venvDir.FullName "Scripts\python.exe"
        }
    }
    
    if (Test-Path $venvPython) {
        Write-Host "Using Python at: $venvPython" -ForegroundColor Green
        # Set UTF-8 encoding for Python to handle Unicode characters
        $env:PYTHONIOENCODING = "utf-8"
        if ($Arguments) {
            & $venvPython -m ce_properties_wizard.main @Arguments
        } else {
            & $venvPython -m ce_properties_wizard.main
        }
    } else {
        Write-Host "Could not find Python executable in virtual environment" -ForegroundColor Red
        Write-Host "This might be due to Git Bash compatibility issues with Poetry on Windows" -ForegroundColor Yellow
        Write-Host "Please run this script in a native PowerShell window instead" -ForegroundColor Yellow
        exit 1
    }
} else {
	# Run the wizard with all arguments passed through
	if ($Arguments) {
		poetry run ce-props-wizard @Arguments
	} else {
		poetry run ce-props-wizard
	}
}
