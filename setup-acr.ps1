# ACR Authentication Setup Script
# This script sets up Azure Container Registry authentication for Kubernetes

param(
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId,
    
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,
    
    [Parameter(Mandatory=$true)]
    [string]$AcrName = "hackathon25",
    
    [Parameter()]
    [string]$KubernetesNamespace = "default",
    
    [Parameter()]
    [switch]$TestMode
)

Write-Host "Setting up ACR authentication for Kubernetes..." -ForegroundColor Green

# Login to Azure (if not already logged in)
try {
    $context = az account show --query "id" -o tsv 2>$null
    if (-not $context) {
        Write-Host "Please login to Azure..." -ForegroundColor Yellow
        az login
    }
} catch {
    Write-Host "Please login to Azure..." -ForegroundColor Yellow
    az login
}

# Set the subscription
Write-Host "Setting subscription to: $SubscriptionId" -ForegroundColor Blue
az account set --subscription $SubscriptionId

# Get ACR login server
$acrLoginServer = az acr show --name $AcrName --resource-group $ResourceGroupName --query "loginServer" -o tsv
if (-not $acrLoginServer) {
    Write-Error "Failed to get ACR login server. Please check ACR name and resource group."
    exit 1
}

Write-Host "ACR Login Server: $acrLoginServer" -ForegroundColor Blue

# Create service principal for ACR access
Write-Host "Creating service principal for ACR access..." -ForegroundColor Blue
$servicePrincipal = az ad sp create-for-rbac --name "acr-service-principal-$AcrName" --scopes "/subscriptions/$SubscriptionId/resourceGroups/$ResourceGroupName/providers/Microsoft.ContainerRegistry/registries/$AcrName" --role acrpull --query "{clientId:appId, clientSecret:password}" -o json | ConvertFrom-Json

if (-not $servicePrincipal) {
    Write-Error "Failed to create service principal."
    exit 1
}

$clientId = $servicePrincipal.clientId
$clientSecret = $servicePrincipal.clientSecret

Write-Host "Service Principal created with Client ID: $clientId" -ForegroundColor Blue

# Create Docker config JSON
$dockerConfig = @{
    auths = @{
        $acrLoginServer = @{
            username = $clientId
            password = $clientSecret
            auth = [System.Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes("${clientId}:${clientSecret}"))
        }
    }
} | ConvertTo-Json -Depth 3

$dockerConfigBase64 = [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($dockerConfig))

# Create or update the Kubernetes secret
Write-Host "Creating Kubernetes secret for ACR authentication..." -ForegroundColor Blue
kubectl delete secret acr-secret --namespace $KubernetesNamespace --ignore-not-found=true

Write-Host "Creating new secret..." -ForegroundColor Blue
kubectl create secret docker-registry acr-secret --namespace $KubernetesNamespace --docker-server=$acrLoginServer --docker-username=$clientId --docker-password=$clientSecret

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create Kubernetes secret. Error code: $LASTEXITCODE"
    Write-Host "Troubleshooting tips:" -ForegroundColor Yellow
    Write-Host "1. Check if kubectl is connected: kubectl get nodes" -ForegroundColor White
    Write-Host "2. Check if namespace exists: kubectl get namespace $KubernetesNamespace" -ForegroundColor White
    Write-Host "3. Check permissions: kubectl auth can-i create secrets" -ForegroundColor White
    exit 1
}

Write-Host "Secret created successfully!" -ForegroundColor Green

# Apply the service account
Write-Host "Creating service account..." -ForegroundColor Blue
if (-not (Test-Path ".\k8s\acr-serviceaccount.yaml")) {
    Write-Error "Service account file not found: .\k8s\acr-serviceaccount.yaml"
    exit 1
}

kubectl apply -f ".\k8s\acr-serviceaccount.yaml" --namespace $KubernetesNamespace

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create service account. Error code: $LASTEXITCODE"
    Write-Host "Troubleshooting tips:" -ForegroundColor Yellow
    Write-Host "1. Check file exists: Test-Path .\k8s\acr-serviceaccount.yaml" -ForegroundColor White
    Write-Host "2. Check YAML syntax: kubectl apply -f .\k8s\acr-serviceaccount.yaml --dry-run=client" -ForegroundColor White
    Write-Host "3. Check permissions: kubectl auth can-i create serviceaccounts" -ForegroundColor White
    exit 1
}

Write-Host "Service account created successfully!" -ForegroundColor Green

Write-Host "ACR authentication setup completed successfully!" -ForegroundColor Green
Write-Host "You can now deploy your applications with ACR image pulling." -ForegroundColor Green

# Display next steps
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Apply your deployments: kubectl apply -f app-deployment.yaml" -ForegroundColor White
Write-Host "2. Apply prophet deployment: kubectl apply -f prophet-deployment.yaml" -ForegroundColor White
Write-Host "3. Check pod status: kubectl get pods" -ForegroundColor White