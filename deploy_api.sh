#!/bin/bash
# TFT Analytics API Server Deployment Script for GCP VM
# Run this script on your GCP VM to deploy the API server

set -e

echo "=== TFT Analytics API Server Deployment ==="
echo "Deploying to GCP VM..."

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip if not already installed
echo "Installing Python dependencies..."
sudo apt install -y python3 python3-pip python3-venv nginx

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements_api.txt

# Install additional dependencies for your existing system
pip install google-cloud-bigquery pandas numpy scikit-learn

# Check if authentication is set up
CREDENTIALS_FILE="/opt/tft-analytics/credentials.json"
if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo ""
    echo "🔐 Authentication Setup Required"
    echo "Service account credentials not found at: $CREDENTIALS_FILE"
    echo ""
    echo "Please run the authentication setup first:"
    echo "  chmod +x setup_auth.sh"
    echo "  ./setup_auth.sh"
    echo ""
    echo "Then re-run this deployment script."
    exit 1
fi

echo "✅ Service account credentials found"

echo "Testing API server..."
# Run a quick test to make sure imports work
python3 -c "
import sys
try:
    from querying import TFTQuery
    from clustering import TFTClusteringEngine
    from api_server import app
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Create systemd service file with authentication
echo "Creating systemd service with BigQuery authentication..."
sudo tee /etc/systemd/system/tft-api.service > /dev/null <<EOF
[Unit]
Description=TFT Analytics API Server
After=network.target

[Service]
Type=forking
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
Environment=TFT_TEST_MODE=false
Environment=ENVIRONMENT=production
Environment=GOOGLE_APPLICATION_CREDENTIALS=$CREDENTIALS_FILE
ExecStart=$(pwd)/venv/bin/gunicorn -c gunicorn_config.py api_server:app
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable tft-api.service

# Configure nginx reverse proxy
echo "Configuring nginx..."
sudo tee /etc/nginx/sites-available/tft-api > /dev/null <<EOF
server {
    listen 80;
    server_name _;  # Replace with your domain

    # API routes
    location /api/ {
        proxy_pass http://localhost:8080/api/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization, X-API-Key' always;
        
        # Handle preflight requests
        if (\$request_method = 'OPTIONS') {
            return 204;
        }
    }

    # Health check
    location /health {
        proxy_pass http://localhost:8080/api/health;
        proxy_set_header Host \$host;
    }
    
    # Basic info page
    location / {
        return 200 "TFT Analytics API Server - Visit /api/info for documentation";
        add_header Content-Type text/plain;
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/tft-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default  # Remove default site
sudo nginx -t  # Test configuration
sudo systemctl enable nginx
sudo systemctl restart nginx

# Configure firewall (using iptables as fallback for ufw)
echo "Configuring firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 22/tcp   # SSH
    sudo ufw allow 80/tcp   # HTTP
    sudo ufw allow 443/tcp  # HTTPS (for future SSL setup)
    sudo ufw --force enable
else
    echo "UFW not available, using iptables..."
    # Allow SSH (port 22)
    sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
    # Allow HTTP (port 80)
    sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
    # Allow HTTPS (port 443)
    sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
    # Allow established connections
    sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    # Allow loopback
    sudo iptables -A INPUT -i lo -j ACCEPT
    # Save iptables rules
    sudo apt install -y iptables-persistent
    sudo netfilter-persistent save
fi

# Start the API service
echo "Starting TFT Analytics API service..."
sudo systemctl start tft-api.service

# Check service status
echo "Checking service status..."
sleep 3
sudo systemctl status tft-api.service --no-pager

echo ""
echo "=== Deployment Complete ==="
echo "✅ TFT Analytics API Server deployed successfully!"
echo ""
echo "Service Status:"
echo "- API Server: http://$(curl -s ifconfig.me)/api/health"
echo "- Nginx: $(sudo systemctl is-active nginx)"
echo "- TFT API: $(sudo systemctl is-active tft-api)"
echo ""
echo "Authentication:"
echo "- Service Account: $(basename "$CREDENTIALS_FILE")"
echo "- Credentials: $CREDENTIALS_FILE"
echo ""
echo "Commands:"
echo "- Check logs: sudo journalctl -u tft-api.service -f"
echo "- Restart API: sudo systemctl restart tft-api.service"
echo "- Stop API: sudo systemctl stop tft-api.service"
echo ""
echo "Next Steps:"
echo "1. Test API: curl http://$(curl -s ifconfig.me)/api/health"
echo "2. Test BigQuery: curl http://$(curl -s ifconfig.me)/api/stats"
echo "3. Configure your domain/SSL if needed"
echo "4. Update Firebase webapp with your API URL"
echo "5. Monitor logs for any issues"
echo ""
echo "Your API is now ready for Firebase integration!"