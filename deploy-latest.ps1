# PowerShell script to redeploy with latest Docker images

Write-Host "🚀 Redeploying with latest Docker images..." -ForegroundColor Green

# Check if kubectl is available
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Write-Host "❌ kubectl not found. Please install kubectl." -ForegroundColor Red
    exit 1
}

# Check if cluster is accessible
try {
    kubectl cluster-info | Out-Null
    Write-Host "✅ Kubernetes cluster is accessible" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot connect to Kubernetes cluster. Please start minikube." -ForegroundColor Red
    Write-Host "Run: minikube start" -ForegroundColor Yellow
    exit 1
}

# Restart app deployment to pull latest image
Write-Host "📱 Restarting sample-app deployment..." -ForegroundColor Yellow
kubectl rollout restart deployment/sample-app

# Restart prophet predictor deployment to pull latest image  
Write-Host "🔮 Restarting prophet-predictor deployment..." -ForegroundColor Yellow
kubectl rollout restart deployment/prophet-predictor

# Wait for rollouts to complete
Write-Host "⏳ Waiting for deployments to complete..." -ForegroundColor Yellow
kubectl rollout status deployment/sample-app --timeout=300s
kubectl rollout status deployment/prophet-predictor --timeout=300s

Write-Host "✅ Deployment complete! Latest images are now running." -ForegroundColor Green

