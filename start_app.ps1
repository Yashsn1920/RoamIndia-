Set-Location $PSScriptRoot
Write-Host ""
Write-Host "Starting RoamIndia..." -ForegroundColor Green
Write-Host "Open in browser: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""
streamlit run app.py
