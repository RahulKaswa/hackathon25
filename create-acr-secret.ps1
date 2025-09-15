# Quick ACR Secret Creation Script
# Run this if you need to recreate the ACR secret

Write-Host "Creating ACR secret for hackathon25..." -ForegroundColor Green

# Get fresh token
Write-Host "Getting ACR token..." -ForegroundColor Blue
$tokenInfo = az acr login --name hackathon25 --expose-token --output json | ConvertFrom-Json

if (-not $tokenInfo) {
    Write-Error "Failed to get ACR token. Please check Azure login and ACR access."
    exit 1
}

$accessToken = $tokenInfo.accessToken
$loginServer = $tokenInfo.loginServer

Write-Host "Login Server: $loginServer" -ForegroundColor Blue

# Delete existing secret if it exists
Write-Host "Removing existing secret..." -ForegroundColor Blue
kubectl delete secret acr-secret --ignore-not-found=true

# Create new secret
Write-Host "Creating new ACR secret..." -ForegroundColor Blue
kubectl create secret docker-registry acr-secret `
  --docker-server=$loginServer `
  --docker-username="00000000-0000-0000-0000-000000000000" `
  --docker-password=$accessToken

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ ACR secret created successfully!" -ForegroundColor Green
    Write-Host "You can now deploy applications that pull from $loginServer" -ForegroundColor Green
} else {
    Write-Error "Failed to create ACR secret."
    exit 1
}

# Verify secret exists
kubectl get secret acr-secret