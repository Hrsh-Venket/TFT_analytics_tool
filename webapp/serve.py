#!/usr/bin/env python3
"""
Simple HTTP server for local TFT Analytics frontend testing.
Run with: python serve.py
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=Path(__file__).parent, **kwargs)

    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 TFT Analytics Dashboard running at http://localhost:{PORT}")
        print(f"📁 Serving from: {Path(__file__).parent}")
        print("\n🎯 Ready to test integration with Cloud Functions!")
        print("   ├── Dashboard will load cluster data automatically")
        print("   ├── Query builder connects to api-query endpoint")
        print("   └── All APIs point to us-central1-tft-analytics-tool.cloudfunctions.net")
        print(f"\n🌐 Opening browser...")

        try:
            webbrowser.open(f'http://localhost:{PORT}')
        except:
            pass

        print("Press Ctrl+C to stop the server")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")