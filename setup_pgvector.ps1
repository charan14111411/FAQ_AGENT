# setup_pgvector.ps1
# This script installs pgvector for PostgreSQL 18 on Windows.
# Run this script in an Administrator PowerShell window!

$ErrorActionPreference = "Stop"

$url = "https://github.com/andreiramani/pgvector_pgsql_windows/releases/download/0.8.2_18.0.2/vector.v0.8.2-pg18.zip"
$zipPath = Join-Path $env:TEMP "vector.v0.8.2-pg18.zip"
$extractPath = Join-Path $env:TEMP "vector_extract"
$pgDir = "C:\Program Files\PostgreSQL\18"

# Check for admin privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "This script MUST be run as an Administrator to copy files into C:\Program Files!"
    Write-Host "Please open PowerShell as Administrator and run:" -ForegroundColor Yellow
    Write-Host "powershell -ExecutionPolicy Bypass -File .\setup_pgvector.ps1" -ForegroundColor Yellow
    Exit
}

# 1. Download (if not already present locally)
$localZip = Join-Path $PSScriptRoot "vector.v0.8.2-pg18.zip"
if (Test-Path $localZip) {
    Write-Host "Found local zip at $localZip. Skipping download." -ForegroundColor Green
    Copy-Item -Path $localZip -Destination $zipPath -Force
} else {
    Write-Host "1. Downloading pgvector v0.8.2 for PostgreSQL 18..." -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $url -OutFile $zipPath
    } catch {
        Write-Host "`nCould not download automatically due to a network or DNS issue." -ForegroundColor Red
        Write-Host "Please download the zip file manually in your web browser using this link:" -ForegroundColor Yellow
        Write-Host "  $url" -ForegroundColor White
        Write-Host "Then, place the downloaded file 'vector.v0.8.2-pg18.zip' in this folder ($PSScriptRoot) and run this script again." -ForegroundColor Yellow
        Exit
    }
}

Write-Host "2. Extracting files..." -ForegroundColor Cyan
if (Test-Path $extractPath) { Remove-Item -Recurse -Force $extractPath }
Expand-Archive -Path $zipPath -DestinationPath $extractPath

Write-Host "3. Copying files to $pgDir..." -ForegroundColor Cyan
if (-not (Test-Path $pgDir)) {
    throw "PostgreSQL 18 directory not found at $pgDir. Please ensure PostgreSQL 18 is installed."
}

# Copy DLL
$dllSource = Join-Path $extractPath "lib\vector.dll"
$dllDest = Join-Path $pgDir "lib\"
Copy-Item -Path $dllSource -Destination $dllDest -Force
Write-Host "Copied vector.dll successfully." -ForegroundColor Green

# Copy extensions share files
$shareSource = Join-Path $extractPath "share\extension\*"
$shareDest = Join-Path $pgDir "share\extension\"
Copy-Item -Path $shareSource -Destination $shareDest -Force
Write-Host "Copied extension files successfully." -ForegroundColor Green

Write-Host "`npgvector installation completed successfully!" -ForegroundColor Green
Write-Host "Now you can run database migrations." -ForegroundColor Green
