# setup_env.ps1 — creates a conda env and installs all dependencies
# Run from inside khana_classifier/ with:
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1

$ENV_NAME = "khana"
$PYTHON_VERSION = "3.11"

Write-Host "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..." -ForegroundColor Cyan
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

Write-Host "Activating environment..." -ForegroundColor Cyan
conda activate $ENV_NAME

Write-Host "Installing packages..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "Done! Verify CUDA is working:" -ForegroundColor Green
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

Write-Host ""
Write-Host "To activate this environment in future sessions:" -ForegroundColor Yellow
Write-Host "  conda activate $ENV_NAME" -ForegroundColor Yellow
