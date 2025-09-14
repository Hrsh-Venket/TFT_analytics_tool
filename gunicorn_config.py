#!/usr/bin/env python3
"""
Gunicorn Configuration for TFT Analytics API Server

Production-ready configuration for serving the Flask API on GCP VM.
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"
backlog = 2048

# Worker processes
workers = min(multiprocessing.cpu_count() * 2 + 1, 8)  # Cap at 8 workers
worker_class = "gevent"
worker_connections = 1000
timeout = 120
keepalive = 2

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "tft-analytics-api"

# Server mechanics
preload_app = True
max_requests = 1000
max_requests_jitter = 100

# SSL (if using HTTPS directly)
# keyfile = "/path/to/ssl/key.pem"
# certfile = "/path/to/ssl/cert.pem"

def when_ready(server):
    server.log.info("TFT Analytics API Server is ready")

def worker_init(worker):
    worker.log.info(f"Worker {worker.pid} initialized")

def on_exit(server):
    server.log.info("TFT Analytics API Server shutting down")