#!/bin/bash

# TFT Analytics Cloud Functions Test Script
# Comprehensive testing of all deployed endpoints

BASE_URL="https://us-central1-tft-analytics-tool.cloudfunctions.net"

echo "🧪 Testing TFT Analytics Cloud Functions"
echo "========================================"
echo ""

test_endpoint() {
    local name="$1"
    local url="$2"
    local method="$3"
    local data="$4"

    echo "Testing $name..."
    echo "URL: $url"

    if [ "$method" = "POST" ]; then
        echo "Method: POST"
        echo "Data: $data"
        response=$(curl -s -w "HTTPSTATUS:%{http_code}" -X POST "$url" \
                  -H "Content-Type: application/json" \
                  -d "$data" \
                  --max-time 30)
    else
        echo "Method: GET"
        response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$url" --max-time 30)
    fi

    # Extract HTTP status and body
    http_code=$(echo $response | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    body=$(echo $response | sed -e 's/HTTPSTATUS:.*//g')

    echo -n "Result: "
    if [ "$http_code" = "200" ]; then
        echo "✅ SUCCESS (HTTP $http_code)"
        # Pretty print JSON if it's valid JSON
        if echo "$body" | python3 -m json.tool >/dev/null 2>&1; then
            echo "Response (formatted):"
            echo "$body" | python3 -m json.tool | head -20
            if [ $(echo "$body" | wc -c) -gt 1000 ]; then
                echo "... (response truncated)"
            fi
        else
            echo "Response (raw):"
            echo "${body:0:500}"
            if [ ${#body} -gt 500 ]; then
                echo "... (response truncated)"
            fi
        fi
    else
        echo "❌ FAILED (HTTP $http_code)"
        echo "Error response:"
        echo "$body"
    fi
    echo ""
    echo "----------------------------------------"
    echo ""
}

# Test all endpoints
echo "Starting endpoint tests..."
echo ""

test_endpoint "Stats API" "$BASE_URL/api-stats" "GET"

test_endpoint "Clusters API" "$BASE_URL/api-clusters" "GET"

test_endpoint "Basic Query API" "$BASE_URL/api-query" "POST" '{"query": "TFTQuery().get_stats()"}'

test_endpoint "Unit Query API" "$BASE_URL/api-query" "POST" '{"query": "TFTQuery().add_unit(\"Jinx\").get_stats()"}'

test_endpoint "Participants Query API" "$BASE_URL/api-query" "POST" '{"query": "TFTQuery().add_unit(\"Jinx\").get_participants()[:3]"}'

test_endpoint "Cluster Details API" "$BASE_URL/api-cluster-details?id=1" "GET"

test_endpoint "Cluster Details API (different ID)" "$BASE_URL/api-cluster-details?id=4" "GET"

echo "🏁 Testing complete!"
echo ""
echo "📊 Next steps if any tests failed:"
echo "   1. Check function logs: gcloud functions logs read FUNCTION_NAME --limit=10"
echo "   2. Check BigQuery data: bq query 'SELECT COUNT(*) FROM tft_analytics.match_participants'"
echo "   3. Test BigQuery connection: python3 -c 'from google.cloud import bigquery; print(bigquery.Client().project)'"