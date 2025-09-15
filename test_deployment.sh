#!/bin/bash

# Test script for TFT Analytics API deployment
echo "🧪 Testing TFT Analytics API Deployment"
echo "========================================"

# Function to test endpoint
test_endpoint() {
    local url=$1
    local description=$2
    echo -n "Testing $description... "

    response=$(curl -s -w "%{http_code}" "$url" -o /tmp/response.txt)
    http_code=${response: -3}

    if [ "$http_code" = "200" ]; then
        echo "✅ SUCCESS (HTTP $http_code)"
        if [ "$3" = "show" ]; then
            echo "Response: $(cat /tmp/response.txt | jq . 2>/dev/null || cat /tmp/response.txt)"
        fi
    else
        echo "❌ FAILED (HTTP $http_code)"
        echo "Response: $(cat /tmp/response.txt)"
    fi
    echo ""
}

# Get server IP
SERVER_IP=$(curl -s ifconfig.me)
echo "Server IP: $SERVER_IP"
echo ""

# Test service status
echo "📊 Service Status:"
echo "- Nginx: $(sudo systemctl is-active nginx 2>/dev/null || echo 'unknown')"
echo "- TFT API: $(sudo systemctl is-active tft-api 2>/dev/null || echo 'unknown')"
echo ""

# Test endpoints
echo "🌐 Testing API Endpoints:"
test_endpoint "http://localhost/api/health" "Health Check (local)" show
test_endpoint "http://$SERVER_IP/api/health" "Health Check (external)" show
test_endpoint "http://localhost/api/stats" "Statistics (local)"
test_endpoint "http://$SERVER_IP/api/stats" "Statistics (external)"

# Test authentication specifically
echo "🔐 Authentication Test:"
echo "Checking BigQuery connection in API response..."
health_response=$(curl -s "http://localhost/api/health")
if echo "$health_response" | grep -q '"bigquery":{"status":"up"'; then
    echo "✅ BigQuery authentication working"
elif echo "$health_response" | grep -q '"bigquery":{"status":"down"'; then
    echo "❌ BigQuery authentication failed"
    echo "Debug info:"
    echo "$health_response" | jq .services.bigquery 2>/dev/null || echo "$health_response"
else
    echo "⚠️  Unexpected response format"
    echo "$health_response"
fi

echo ""
echo "📋 Debug Commands:"
echo "- View API logs: sudo journalctl -u tft-api.service -f"
echo "- Check service status: sudo systemctl status tft-api.service"
echo "- Test manual start: cd $(pwd) && source venv/bin/activate && python api_server.py"
echo "- Check credentials: ls -la /opt/tft-analytics/credentials.json"

# Clean up
rm -f /tmp/response.txt

echo ""
echo "🎯 Test complete!"