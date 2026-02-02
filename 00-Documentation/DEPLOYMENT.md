---
noteId: "ac260ac0923d11f08ffe3fbc0e264f39"
tags: []

---

# TFT Analytics Cloud Functions Deployment Guide

## Architecture Overview

Your TFT Analytics tool has been migrated to a serverless architecture:

```
Firebase Web App (Frontend)
    ↓ (HTTPS requests)
Cloud Functions (Python 3.11, auto-scaling)
├── api-stats (GET /api/stats)
├── api-clusters (GET /api/clusters)
├── api-query (POST /api/query)
└── api-cluster-details (GET /api/clusters/{id})
    ↓ (BigQuery client)
BigQuery (10GB storage + 1TB queries/month)
├── Raw TFT match data
├── Clustering results
└── Query analytics
```

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **Google Cloud SDK** installed on your VM
3. **Firebase CLI** installed: `npm install -g firebase-tools`
4. **Existing BigQuery dataset** with TFT data

## Deployment Steps

### 1. VM Setup

```bash
# On your VM, navigate to project directory
cd /path/to/TFT_analytics_tool

# Set your Google Cloud project
gcloud config set project YOUR-PROJECT-ID

# Enable required APIs
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable bigquery.googleapis.com

# Make deployment script executable
chmod +x deploy-functions.sh delete-functions.sh
```

### 2. Deploy Cloud Functions

```bash
# Deploy all functions at once
./deploy-functions.sh
```

This will deploy 4 functions:
- `api-stats` - Database statistics
- `api-clusters` - Cluster listings
- `api-query` - Query execution
- `api-cluster-details` - Cluster details

### 3. Configure Firebase Frontend

1. **Update API endpoint** in `firebase/public/js/api.js`:
   ```javascript
   this.PROJECT_ID = 'your-actual-project-id'; // Replace this line
   ```

2. **Initialize Firebase project**:
   ```bash
   cd firebase
   firebase login
   firebase init hosting
   # Select existing project or create new one
   # Use 'public' as public directory
   # Configure as single-page app: Yes
   ```

3. **Deploy Frontend**:
   ```bash
   firebase deploy --only hosting
   ```

## Cost Control Features

### Turn OFF Autoscaling (Zero Costs)
```bash
./delete-functions.sh
```
This deletes all Cloud Functions, stopping all traffic and costs.

### Turn ON Autoscaling
```bash
./deploy-functions.sh
```
Redeploys all functions with autoscaling enabled.

### Monitor Costs
```bash
# View function invocations
gcloud logging read "resource.type=cloud_function" --limit=100

# Check BigQuery usage
bq show --format=json $PROJECT:tft_analytics
```

## Testing

### Local Testing (without BigQuery)
```bash
python test-simple.py
```

### Live Function Testing
Once deployed, test your functions:

```bash
# Test stats endpoint
curl https://us-central1-YOUR-PROJECT.cloudfunctions.net/api-stats

# Test clusters endpoint
curl https://us-central1-YOUR-PROJECT.cloudfunctions.net/api-clusters
```

## Function URLs

After deployment, your functions will be available at:
- **Stats**: `https://us-central1-YOUR-PROJECT.cloudfunctions.net/api-stats`
- **Clusters**: `https://us-central1-YOUR-PROJECT.cloudfunctions.net/api-clusters`
- **Query**: `https://us-central1-YOUR-PROJECT.cloudfunctions.net/api-query`
- **Details**: `https://us-central1-YOUR-PROJECT.cloudfunctions.net/api-cluster-details`

## Troubleshooting

### Common Issues

1. **"BigQuery dependencies not available"**
   - Functions will install dependencies automatically during deployment
   - This error only affects local testing

2. **"Permission denied" on deployment**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Functions timing out**
   - Check BigQuery dataset exists and has data
   - Verify service account has BigQuery access

4. **CORS errors in frontend**
   - All functions include proper CORS headers
   - Check PROJECT_ID is set correctly in api.js

### Logs and Monitoring

```bash
# View function logs
gcloud functions logs read api-stats --limit=50

# View all function logs
gcloud functions logs read --limit=100

# Monitor in real-time
gcloud functions logs tail api-stats
```

## Data Pipeline

Your VM cron jobs continue to work with BigQuery:

```bash
# Existing cron jobs work unchanged:
# Data collection every 4 hours
0 */4 * * * cd /path/to/tft_analytics && python3 data_collection.py

# Clustering analysis daily at 2 AM
0 2 * * * cd /path/to/tft_analytics && python3 clustering.py
```

## Cost Estimation

- **0-2M requests/month**: $0 (completely free)
- **10M requests/month**: ~$3.20
- **Typical usage**: $0-1/month

With manual on/off controls, you can eliminate costs entirely when not using the system.

## Next Steps

1. ✅ Deploy Cloud Functions: `./deploy-functions.sh`
2. ✅ Update PROJECT_ID in `firebase/public/js/api.js`
3. ✅ Deploy Firebase: `firebase deploy --only hosting`
4. ✅ Test web interface functionality
5. ✅ Verify VM cron jobs continue working with BigQuery
6. ✅ Set up monitoring and alerting if needed