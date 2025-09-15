#!/bin/bash

# TFT Analytics API - Authentication Setup Script
# This script creates a service account with minimal BigQuery permissions

set -e  # Exit on any error

echo "🔐 Setting up authentication for TFT Analytics API..."

# Get current project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: No GCP project configured. Run 'gcloud config set project YOUR_PROJECT_ID'"
    exit 1
fi

echo "📋 Project ID: $PROJECT_ID"

# Define service account details
SERVICE_ACCOUNT_NAME="tft-analytics-api"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="/opt/tft-analytics/credentials.json"
KEY_DIR="/opt/tft-analytics"

echo "👤 Creating service account: $SERVICE_ACCOUNT_NAME"

# Check if service account already exists
if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &>/dev/null; then
    echo "ℹ️  Service account already exists, skipping creation"
else
    # Create the service account
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --description="Service account for TFT Analytics API BigQuery access" \
        --display-name="TFT Analytics API"
    echo "✅ Service account created"
fi

echo "🔑 Granting BigQuery permissions..."

# Grant minimal required BigQuery permissions
# dataViewer: Read access to BigQuery datasets
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/bigquery.dataViewer" \
    --quiet

# jobUser: Ability to run BigQuery jobs (queries)
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/bigquery.jobUser" \
    --quiet

echo "✅ Permissions granted"

echo "📁 Setting up credentials directory..."

# Create secure directory for credentials
sudo mkdir -p "$KEY_DIR"
sudo chmod 755 "$KEY_DIR"

echo "🔐 Generating service account key..."

# Generate and download the service account key
# This creates a JSON file with the private key for authentication
gcloud iam service-accounts keys create "$KEY_FILE" \
    --iam-account="$SERVICE_ACCOUNT_EMAIL"

# Secure the key file
sudo chmod 600 "$KEY_FILE"
echo "✅ Key file created at: $KEY_FILE"

echo ""
echo "🎉 Authentication setup complete!"
echo "📋 Summary:"
echo "   Service Account: $SERVICE_ACCOUNT_EMAIL"
echo "   Key File: $KEY_FILE"
echo "   Permissions: BigQuery dataViewer + jobUser"
echo ""
echo "🚀 Next step: Run the deployment script to update systemd service"