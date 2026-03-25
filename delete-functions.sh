#!/bin/bash
set -e

REGION="asia-southeast1"

echo "=== Deleting TFT Analytics Cloud Functions ==="
echo "This will stop all cloud function costs."
echo ""

for func in api-stats api-clusters api-query api-cluster-details; do
    echo "Deleting $func..."
    gcloud functions delete "$func" --region="$REGION" --quiet || echo "  $func not found, skipping"
done

echo ""
echo "=== All functions deleted ==="
