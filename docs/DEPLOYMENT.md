# Deployment Guide

This guide provides comprehensive instructions for deploying the Complete the Look system in production environments.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Database Setup](#database-setup)
- [Application Deployment](#application-deployment)
- [Monitoring and Logging](#monitoring-and-logging)
- [Scaling](#scaling)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## Overview

The Complete the Look system can be deployed in various environments:
- **Development**: Local machine for testing and development
- **Staging**: Pre-production environment for testing
- **Production**: Live environment serving real users

## Prerequisites

### System Requirements

- **CPU**: 8+ cores (16+ recommended for production)
- **RAM**: 16GB+ (32GB+ recommended for production)
- **Storage**: 100GB+ SSD storage
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Network**: Stable internet connection for API calls

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows
- **Python**: 3.11 or higher
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (for multi-service deployment)

### External Services

- **Neo4j Database**: Self-hosted or cloud service (Neo4j AuraDB)
- **Pinecone**: Vector database service
- **Azure OpenAI**: For GPT-4 attribute extraction
- **Cloud Storage**: For image storage (AWS S3, Google Cloud Storage, etc.)

## Environment Setup

### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/ashleyashok/fashion-knowledge-graph.git
cd fashion-knowledge-graph

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 2. Environment Configuration

```bash
# Copy environment template
cp env.template .env

# Edit environment variables
nano .env
```

**Required Environment Variables:**

```env
# Neo4j Database
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password

# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_HOST_IMAGE=your_image_index_host
PINECONE_HOST_STYLE=your_style_index_host

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Application Configuration
LOG_LEVEL=INFO
SIMILARITY_THRESHOLD=0.75
MAX_RECOMMENDATIONS=10

# Optional: Cloud Storage
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET=your_s3_bucket_name
```

## Database Setup

### 1. Neo4j Database

#### Option A: Self-Hosted Neo4j

```bash
# Install Neo4j (Ubuntu/Debian)
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j

# Start Neo4j service
sudo systemctl start neo4j
sudo systemctl enable neo4j

# Set initial password
cypher-shell -u neo4j -p neo4j
# In the shell: ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'your_new_password'
```

#### Option B: Neo4j AuraDB (Cloud)

1. Create account at [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/)
2. Create a new database instance
3. Note the connection URI, username, and password
4. Update your `.env` file with the credentials

### 2. Pinecone Vector Database

```bash
# Install Pinecone client
pip install pinecone-client

# Run setup script
python scripts/setup_pinecone.py
```

**Manual Setup:**

```python
import pinecone

# Initialize Pinecone
pinecone.init(api_key='your_api_key')

# Create indexes
pinecone.create_index(
    name='catalog-clothes',
    dimension=512,
    metric='cosine',
    pod_type='p1.x1'
)

pinecone.create_index(
    name='catalog-style-description',
    dimension=384,
    metric='cosine',
    pod_type='p1.x1'
)
```

## Application Deployment

### 1. Local Development Deployment

```bash
# Process catalog data
python src/engine/process_catalog.py

# Process social media images (optional)
python src/engine/process_social_media_images.py

# Start Streamlit application
streamlit run app/main.py
```

### 2. Docker Deployment

#### Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Start application
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Create docker-compose.yml

```yaml
version: '3.8'

services:
  complete-the-look:
    build: .
    ports:
      - "8501:8501"
    environment:
      - NEO4J_URI=${NEO4J_URI}
      - NEO4J_USERNAME=${NEO4J_USERNAME}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_HOST_IMAGE=${PINECONE_HOST_IMAGE}
      - PINECONE_HOST_STYLE=${PINECONE_HOST_STYLE}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
    volumes:
      - ./output:/app/output
      - ./temp_images:/app/temp_images
    depends_on:
      - neo4j

  neo4j:
    image: neo4j:5.15
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/your_password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

volumes:
  neo4j_data:
  neo4j_logs:
```

#### Deploy with Docker

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f complete-the-look

# Stop services
docker-compose down
```

### 3. Cloud Deployment

#### AWS Deployment

**EC2 Instance Setup:**

```bash
# Launch EC2 instance (t3.xlarge or larger)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and deploy
git clone https://github.com/ashleyashok/fashion-knowledge-graph.git
cd fashion-knowledge-graph
docker-compose up -d
```

**AWS ECS Deployment:**

```yaml
# task-definition.json
{
  "family": "complete-the-look",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "complete-the-look",
      "image": "your-account/complete-the-look:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "NEO4J_URI", "value": "bolt://your-neo4j-host:7687"},
        {"name": "PINECONE_API_KEY", "value": "your_pinecone_key"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/complete-the-look",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Platform

**Cloud Run Deployment:**

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/complete-the-look

# Deploy to Cloud Run
gcloud run deploy complete-the-look \
  --image gcr.io/PROJECT_ID/complete-the-look \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars NEO4J_URI=bolt://your-neo4j-host:7687
```

### 4. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: complete-the-look
spec:
  replicas: 3
  selector:
    matchLabels:
      app: complete-the-look
  template:
    metadata:
      labels:
        app: complete-the-look
    spec:
      containers:
      - name: complete-the-look
        image: your-registry/complete-the-look:latest
        ports:
        - containerPort: 8501
        env:
        - name: NEO4J_URI
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: neo4j-uri
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: pinecone-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: complete-the-look-service
spec:
  selector:
    app: complete-the-look
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

## Monitoring and Logging

### 1. Application Logging

```python
# Configure structured logging
import logging
from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add structured logging
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    level="INFO"
)

# Add file logging
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="30 days",
    compression="zip",
    level="DEBUG"
)
```

### 2. Health Checks

```python
# health_check.py
import requests
from loguru import logger

def check_neo4j_health():
    """Check Neo4j database health."""
    try:
        response = requests.get(f"{NEO4J_URI}/db/data/", auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        return False

def check_pinecone_health():
    """Check Pinecone service health."""
    try:
        import pinecone
        pinecone.init(api_key=PINECONE_API_KEY)
        indexes = pinecone.list_indexes()
        return len(indexes) > 0
    except Exception as e:
        logger.error(f"Pinecone health check failed: {e}")
        return False
```

### 3. Metrics Collection

```python
# metrics.py
import time
from functools import wraps
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total requests', ['endpoint'])
REQUEST_DURATION = Histogram('app_request_duration_seconds', 'Request duration', ['endpoint'])
RECOMMENDATION_COUNT = Counter('app_recommendations_total', 'Total recommendations generated')

def track_metrics(endpoint):
    """Decorator to track request metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                REQUEST_COUNT.labels(endpoint=endpoint).inc()
                return result
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
        return wrapper
    return decorator
```

## Scaling

### 1. Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  complete-the-look:
    build: .
    ports:
      - "8501-8510:8501"
    deploy:
      replicas: 5
    environment:
      - NEO4J_URI=${NEO4J_URI}
    volumes:
      - ./output:/app/output:ro  # Read-only for multiple instances

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - complete-the-look
```

### 2. Load Balancing

```nginx
# nginx.conf
upstream complete_the_look {
    server complete-the-look:8501;
    server complete-the-look:8502;
    server complete-the-look:8503;
    server complete-the-look:8504;
    server complete-the-look:8505;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://complete_the_look;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Database Scaling

**Neo4j Clustering:**

```yaml
# neo4j-cluster.yml
version: '3.8'

services:
  neo4j-core1:
    image: neo4j:5.15
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_dbms_mode=CORE
      - NEO4J_causal__clustering_initial__discovery__members=neo4j-core1:5000,neo4j-core2:5000,neo4j-core3:5000
    ports:
      - "7474:7474"
      - "7687:7687"

  neo4j-core2:
    image: neo4j:5.15
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_dbms_mode=CORE
      - NEO4J_causal__clustering_initial__discovery__members=neo4j-core1:5000,neo4j-core2:5000,neo4j-core3:5000

  neo4j-core3:
    image: neo4j:5.15
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_dbms_mode=CORE
      - NEO4J_causal__clustering_initial__discovery__members=neo4j-core1:5000,neo4j-core2:5000,neo4j-core3:5000
```

## Security

### 1. Network Security

```bash
# Configure firewall (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8501/tcp  # Streamlit
sudo ufw enable
```

### 2. SSL/TLS Configuration

```python
# ssl_config.py
import ssl
from streamlit.web.server import Server

# Configure SSL
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

# Update Streamlit server configuration
Server.ssl_context = ssl_context
```

### 3. Authentication

```python
# auth.py
import streamlit as st
import hashlib
import os

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        return True
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

```bash
# Check Neo4j connection
cypher-shell -u neo4j -p password -a bolt://localhost:7687

# Check Pinecone connection
python -c "import pinecone; pinecone.init(api_key='your_key'); print(pinecone.list_indexes())"
```

#### 2. Memory Issues

```bash
# Monitor memory usage
htop
free -h

# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Install CUDA drivers if needed
sudo apt-get install nvidia-driver-470
```

### Performance Optimization

#### 1. Model Caching

```python
# cache_models.py
import joblib
from functools import lru_cache

@lru_cache(maxsize=1)
def load_segmentation_model():
    """Cache segmentation model in memory."""
    return SegmentationModel()

@lru_cache(maxsize=1)
def load_embedding_model():
    """Cache embedding model in memory."""
    return EmbeddingModel()
```

#### 2. Database Optimization

```cypher
// Create indexes for better performance
CREATE INDEX product_id_index FOR (p:Product) ON (p.product_id);
CREATE INDEX product_type_index FOR (p:Product) ON (p.type);
CREATE INDEX relationship_weight_index FOR ()-[r:WORN_WITH]-() ON (r.weight);
```

#### 3. Batch Processing

```python
# batch_processor.py
def process_catalog_batch(catalog_df, batch_size=100):
    """Process catalog in batches to manage memory."""
    for i in range(0, len(catalog_df), batch_size):
        batch = catalog_df[i:i + batch_size]
        process_batch(batch)
        gc.collect()  # Force garbage collection
```

### Monitoring Commands

```bash
# Check application status
docker-compose ps

# View application logs
docker-compose logs -f complete-the-look

# Monitor resource usage
docker stats

# Check database performance
cypher-shell -u neo4j -p password -a bolt://localhost:7687 \
  -c "CALL dbms.listQueries() YIELD query, elapsedTimeMillis ORDER BY elapsedTimeMillis DESC LIMIT 10"
```

---

For additional support, please refer to the [Contributing Guidelines](../CONTRIBUTING.md) or contact the maintainers.
