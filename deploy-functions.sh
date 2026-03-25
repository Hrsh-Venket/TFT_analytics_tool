#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CF_DIR="$PROJECT_DIR/cloud_functions"
UTILS="$CF_DIR/utils.py"
REGION="asia-southeast1"
RUNTIME="python311"

echo "=== Deploying TFT Analytics Cloud Functions ==="
echo "Project dir: $PROJECT_DIR"
echo "Region: $REGION"
echo ""

# --- api-stats ---
echo "1/4 Deploying api-stats..."
cp "$UTILS" "$CF_DIR/stats/utils.py"
gcloud functions deploy api-stats \
    --source="$CF_DIR/stats" \
    --entry-point=get_stats \
    --runtime="$RUNTIME" \
    --trigger-http \
    --allow-unauthenticated \
    --region="$REGION" \
    --memory=256MB \
    --timeout=60s
rm "$CF_DIR/stats/utils.py"
echo ""

# --- api-clusters ---
echo "2/4 Deploying api-clusters..."
cp "$UTILS" "$CF_DIR/clusters/utils.py"
gcloud functions deploy api-clusters \
    --source="$CF_DIR/clusters" \
    --entry-point=get_clusters \
    --runtime="$RUNTIME" \
    --trigger-http \
    --allow-unauthenticated \
    --region="$REGION" \
    --memory=256MB \
    --timeout=60s
rm "$CF_DIR/clusters/utils.py"
echo ""

# --- api-query ---
echo "3/4 Deploying api-query..."
cp "$UTILS" "$CF_DIR/query/utils.py"
cp "$PROJECT_DIR/tft_analytics/query.py" "$CF_DIR/query/querying.py"
gcloud functions deploy api-query \
    --source="$CF_DIR/query" \
    --entry-point=execute_query \
    --runtime="$RUNTIME" \
    --trigger-http \
    --allow-unauthenticated \
    --region="$REGION" \
    --memory=256MB \
    --timeout=120s
rm "$CF_DIR/query/utils.py" "$CF_DIR/query/querying.py"
echo ""

# --- api-cluster-details ---
echo "4/4 Deploying api-cluster-details..."
cp "$UTILS" "$CF_DIR/cluster_details/utils.py"
gcloud functions deploy api-cluster-details \
    --source="$CF_DIR/cluster_details" \
    --entry-point=get_cluster_details \
    --runtime="$RUNTIME" \
    --trigger-http \
    --allow-unauthenticated \
    --region="$REGION" \
    --memory=256MB \
    --timeout=60s
rm "$CF_DIR/cluster_details/utils.py"
echo ""

echo "=== All functions deployed ==="
echo "Endpoints:"
echo "  api-stats:            https://$REGION-$(gcloud config get-value project).cloudfunctions.net/api-stats"
echo "  api-clusters:         https://$REGION-$(gcloud config get-value project).cloudfunctions.net/api-clusters"
echo "  api-query:            https://$REGION-$(gcloud config get-value project).cloudfunctions.net/api-query"
echo "  api-cluster-details:  https://$REGION-$(gcloud config get-value project).cloudfunctions.net/api-cluster-details"
