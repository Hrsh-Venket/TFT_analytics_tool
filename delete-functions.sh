#!/bin/bash

# TFT Analytics Cloud Functions Deletion Script
# Deletes all Cloud Functions to turn OFF autoscaling (cost control)

set -e  # Exit on any error

echo "🛑 Deleting TFT Analytics Cloud Functions..."

# Configuration
REGION="us-central1"
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: No Google Cloud project set. Run 'gcloud config set project YOUR_PROJECT_ID'"
    exit 1
fi

echo "📋 Project: $PROJECT_ID"
echo "📍 Region: $REGION"
echo ""

# Delete all functions
FUNCTIONS=("api-stats" "api-clusters" "api-query" "api-cluster-details")

for func in "${FUNCTIONS[@]}"; do
    echo "🗑️  Deleting $func..."
    gcloud functions delete $func --region=$REGION --quiet || echo "⚠️  $func not found or already deleted"
done

echo ""
echo "✅ All functions deleted successfully!"
echo ""
echo "💰 Autoscaling is now OFF - no Cloud Functions costs will be incurred"
echo ""
echo "🔄 To turn ON autoscaling (redeploy all functions):"
echo "   ./deploy-functions.sh"