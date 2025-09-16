#!/bin/bash

# TFT Analytics Cloud Functions Deployment Script
# Deploys all Cloud Functions for the TFT analytics API

set -e  # Exit on any error

echo "🚀 Deploying TFT Analytics Cloud Functions..."

# Configuration
REGION="us-central1"  # Always Free region
RUNTIME="python311"
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: No Google Cloud project set. Run 'gcloud config set project YOUR_PROJECT_ID'"
    exit 1
fi

echo "📋 Project: $PROJECT_ID"
echo "📍 Region: $REGION"
echo ""

# Function 1: API Stats
echo "🔧 Deploying get-stats function..."
gcloud functions deploy api-stats \
    --source=functions/get-stats \
    --runtime=$RUNTIME \
    --trigger-http \
    --allow-unauthenticated \
    --region=$REGION \
    --memory=512MB \
    --timeout=30s \
    --max-instances=10 \
    --no-gen2 \
    --entry-point=get_stats

echo "✅ api-stats deployed"

# Function 2: Get Clusters
echo "🔧 Deploying get-clusters function..."
gcloud functions deploy api-clusters \
    --source=functions/get-clusters \
    --runtime=$RUNTIME \
    --trigger-http \
    --allow-unauthenticated \
    --region=$REGION \
    --memory=1GB \
    --timeout=60s \
    --max-instances=10 \
    --no-gen2 \
    --entry-point=get_clusters

echo "✅ api-clusters deployed"

# Function 3: Execute Query
echo "🔧 Deploying execute-query function..."
gcloud functions deploy api-query \
    --source=functions/execute-query \
    --runtime=$RUNTIME \
    --trigger-http \
    --allow-unauthenticated \
    --region=$REGION \
    --memory=1GB \
    --timeout=60s \
    --max-instances=10 \
    --no-gen2 \
    --entry-point=execute_query

echo "✅ api-query deployed"

# Function 4: Cluster Details
echo "🔧 Deploying cluster-details function..."
gcloud functions deploy api-cluster-details \
    --source=functions/cluster-details \
    --runtime=$RUNTIME \
    --trigger-http \
    --allow-unauthenticated \
    --region=$REGION \
    --memory=512MB \
    --timeout=30s \
    --max-instances=5 \
    --no-gen2 \
    --entry-point=cluster_details

echo "✅ api-cluster-details deployed"

echo ""
echo "🎉 All functions deployed successfully!"
echo ""
echo "📋 Function URLs:"
echo "   Stats:          https://$REGION-$PROJECT_ID.cloudfunctions.net/api-stats"
echo "   Clusters:       https://$REGION-$PROJECT_ID.cloudfunctions.net/api-clusters"
echo "   Query:          https://$REGION-$PROJECT_ID.cloudfunctions.net/api-query"
echo "   Cluster Details: https://$REGION-$PROJECT_ID.cloudfunctions.net/api-cluster-details"
echo ""
echo "🔄 To turn OFF autoscaling (delete all functions):"
echo "   ./delete-functions.sh"
echo ""
echo "📊 To monitor functions:"
echo "   gcloud functions logs read --limit=50"