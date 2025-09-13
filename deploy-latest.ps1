# PowerShell script to redeploy with latest Docker images

Write-Host "🚀 Redeploying with latest Docker images..." -ForegroundColor Green

# Restart app deployment to pull latest image
Write-Host "📱 Restarting sample-app deployment..." -ForegroundColor Yellow
kubectl rollout restart deployment/sample-app

# Restart prophet predictor deployment to pull latest image  
Write-Host "🔮 Restarting prophet-predictor deployment..." -ForegroundColor Yellow
kubectl rollout restart deployment/prophet-predictor

# Wait for rollouts to complete
Write-Host "⏳ Waiting for deployments to complete..." -ForegroundColor Yellow
kubectl rollout status deployment/sample-app
kubectl rollout status deployment/prophet-predictor

Write-Host "✅ Deployment complete! Latest images are now running." -ForegroundColor Green

# Show current pod status
Write-Host "📊 Current pod status:" -ForegroundColor Cyan
kubectl get pods -l app=sample-app
kubectl get pods -l app=prophet-predictor