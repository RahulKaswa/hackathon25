#!/bin/bash
# Script to redeploy with latest Docker images

echo "🚀 Redeploying with latest Docker images..."

# Restart app deployment to pull latest image
echo "📱 Restarting sample-app deployment..."
kubectl rollout restart deployment/sample-app

# Restart prophet predictor deployment to pull latest image  
echo "🔮 Restarting prophet-predictor deployment..."
kubectl rollout restart deployment/prophet-predictor

# Wait for rollouts to complete
echo "⏳ Waiting for deployments to complete..."
kubectl rollout status deployment/sample-app
kubectl rollout status deployment/prophet-predictor

echo "✅ Deployment complete! Latest images are now running."

# Show current pod status
echo "📊 Current pod status:"
kubectl get pods -l app=sample-app
kubectl get pods -l app=prophet-predictor