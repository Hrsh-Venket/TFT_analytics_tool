#!/bin/bash

# TFT Analytics Cloud Functions Deployment Script
# Deploys all Cloud Functions for the TFT analytics API
# Uses copy-on-deploy strategy for shared modules

set -e  # Exit on any error

echo "Deploying TFT Analytics Cloud Functions..."

# Configuration
REGION="us-central1"  # Always Free region
RUNTIME="python311"
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}

if [ -z "$PROJECT_ID" ]; then
    echo "Error: No Google Cloud project set. Run 'gcloud config set project YOUR_PROJECT_ID'"
    exit 1
fi

echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Step 1: Copy shared modules to cloud function directories
echo "Copying shared modules to cloud function directories..."

# Copy to 03-Querying/cloud_function
cp 03-Querying/querying.py 03-Querying/cloud_function/
cp 03-Querying/bigquery_operations.py 03-Querying/cloud_function/
cp 01-Name_Mapping/name_mapper.py 03-Querying/cloud_function/
mkdir -p 03-Querying/cloud_function/Latest_Mappings
mkdir -p 03-Querying/cloud_function/Default_Mappings
cp 01-Name_Mapping/Latest_Mappings/*.csv 03-Querying/cloud_function/Latest_Mappings/
cp 01-Name_Mapping/Default_Mappings/*.csv 03-Querying/cloud_function/Default_Mappings/

# Copy to 04-Clustering/cloud_function
cp 04-Clustering/clustering.py 04-Clustering/cloud_function/
cp 03-Querying/querying.py 04-Clustering/cloud_function/
cp 03-Querying/bigquery_operations.py 04-Clustering/cloud_function/
cp 01-Name_Mapping/name_mapper.py 04-Clustering/cloud_function/
mkdir -p 04-Clustering/cloud_function/Latest_Mappings
mkdir -p 04-Clustering/cloud_function/Default_Mappings
cp 01-Name_Mapping/Latest_Mappings/*.csv 04-Clustering/cloud_function/Latest_Mappings/
cp 01-Name_Mapping/Default_Mappings/*.csv 04-Clustering/cloud_function/Default_Mappings/

echo "Shared modules copied."
echo ""

# Function 1: Execute Query
echo "Deploying execute-query function..."
gcloud functions deploy api-query \
    --source=03-Querying/cloud_function \
    --runtime=$RUNTIME \
    --trigger-http \
    --allow-unauthenticated \
    --region=$REGION \
    --memory=1GB \
    --timeout=60s \
    --max-instances=10 \
    --no-gen2 \
    --entry-point=execute_query

echo "api-query deployed"

# Function 2: Get Clusters
echo "Deploying get-clusters function..."
gcloud functions deploy api-clusters \
    --source=04-Clustering/cloud_function \
    --runtime=$RUNTIME \
    --trigger-http \
    --allow-unauthenticated \
    --region=$REGION \
    --memory=1GB \
    --timeout=60s \
    --max-instances=10 \
    --no-gen2 \
    --entry-point=get_clusters

echo "api-clusters deployed"

# Step 2: Cleanup copied files (optional - keep for debugging)
echo ""
echo "Cleaning up copied shared modules..."
rm -f 03-Querying/cloud_function/querying.py
rm -f 03-Querying/cloud_function/bigquery_operations.py
rm -f 03-Querying/cloud_function/name_mapper.py
rm -rf 03-Querying/cloud_function/Latest_Mappings
rm -rf 03-Querying/cloud_function/Default_Mappings

rm -f 04-Clustering/cloud_function/clustering.py
rm -f 04-Clustering/cloud_function/querying.py
rm -f 04-Clustering/cloud_function/bigquery_operations.py
rm -f 04-Clustering/cloud_function/name_mapper.py
rm -rf 04-Clustering/cloud_function/Latest_Mappings
rm -rf 04-Clustering/cloud_function/Default_Mappings

echo "Cleanup complete."

echo ""
echo "All functions deployed successfully!"
echo ""
echo "Function URLs:"
echo "   Query:    https://$REGION-$PROJECT_ID.cloudfunctions.net/api-query"
echo "   Clusters: https://$REGION-$PROJECT_ID.cloudfunctions.net/api-clusters"
echo ""
echo "To turn OFF autoscaling (delete all functions):"
echo "   ./delete-functions.sh"
echo ""
echo "To monitor functions:"
echo "   gcloud functions logs read --limit=50"