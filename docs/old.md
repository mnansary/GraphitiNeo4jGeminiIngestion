# KnowledgeGraphNeo4j

# Production Deployment Guide: Graphiti Ingestion Service

## 1. Overview

This guide provides a complete, step-by-step walkthrough for setting up, configuring, and deploying the Graphiti Ingestion Service in a production-like environment on a Linux server.

The goal is to create a robust, persistent service that runs in the background, automatically starts on boot, and can be reliably managed. We will use `conda` for Python environment management and `systemd` for service orchestration.

### Final Architecture

The final setup consists of several independent services that communicate over the network. This guide focuses on deploying the central **Ingestion Service**.

```+---------------------------+       +-------------------------+
|                           |-----> |  Neo4j Database         |
|  Graphiti Ingestion       |       |  (Docker Container)     |
|  Service (FastAPI App)    |       +-------------------------+
|  (systemd Service)        |
|                           |       +-------------------------+
|                           |-----> |  vLLM LLM Server (Gemma)|
|                           |       |  (Docker Container)     |
+---------------------------+       +-------------------------+
                                    |
                                    +-------------------------+
                                    |  Triton Embedder Server |
                                    |  (Docker Container)     |
                                    +-------------------------+
```

---

## 2. Table of Contents

- [Prerequisites](#3-prerequisites)
- [Installation and Local Setup](#4-installation-and-local-setup)
  - [Step 2.1: Clone the Repository](#step-21-clone-the-repository)
  - [Step 2.2: Set Up the Conda Environment](#step-22-set-up-the-conda-environment)
  - [Step 2.3: Install Python Dependencies](#step-23-install-python-dependencies)
  - [Step 2.4: Configure the Application (.env)](#step-24-configure-the-application-env)
- [Running for Development and Testing](#5-running-for-development-and-testing)
  - [Step 3.1: Start the Server Manually](#step-31-start-the-server-manually)
  - [Step 3.2: Test the API Endpoint](#step-32-test-the-api-endpoint)
- [Production Deployment with `systemd`](#6-production-deployment-with-systemd)
  - [Step 4.1: Understand the `systemd` Service File](#step-41-understand-the-systemd-service-file)
  - [Step 4.2: Edit the Service File](#step-42-edit-the-service-file)
  - [Step 4.3: Deploy the Service](#step-43-deploy-the-service)
- [Managing the Production Service](#7-managing-the-production-service)
  - [Checking Status](#checking-status)
  - [Viewing Logs](#viewing-logs)
  - [Stopping, Starting, and Restarting](#stopping-starting-and-restarting)
- [Troubleshooting Common Issues](#8-troubleshooting-common-issues)

---

## 3. Prerequisites

Before you begin, ensure your Linux server meets the following requirements:

- **Hardware**: An NVIDIA GPU with sufficient VRAM for your LLM and embedder models (e.g., 24GB+ recommended).
- **Software**:
  - A modern Linux distribution (e.g., Ubuntu 20.04+).
  - `git` installed.
  - `Docker` and `docker-compose` (or `docker compose`) installed.
  - `Miniconda` or `Anaconda` installed.
- **Running Services**: You must have the three backing services already running and accessible from the host machine:
  1.  **Neo4j Server**: Running in Docker as per the project's setup guide.
  2.  **vLLM Server**: Running your Gemma model in a Docker container with the OpenAI-compatible endpoint exposed.
  3.  **Triton Server**: Running your Jina embedder models in a Docker container.

---

## 4. Installation and Local Setup

This section covers cloning the code and preparing the Python environment.

### Step 2.1: Clone the Repository

```bash
# Clone the project into your home directory or another desired location
git clone https://github.com/your-username/KnowledgeGraphNeo4j.git
cd KnowledgeGraphNeo4j
```

### Step 2.2: Set Up the Conda Environment

We will create a dedicated, isolated Python environment for this service.

```bash
# Create a new Conda environment named 'graphiti-ingestion' with Python 3.11
conda create --name graphiti-ingestion python=3.11 -y

# Activate the newly created environment
conda activate graphiti-ingestion
```

### Step 2.3: Install Python Dependencies

The `setup.py` file is configured to read the `requirements.txt` file. We can install the project in "editable" mode, which is the recommended approach.

```bash
# From the project root directory (e.g., ~/KnowledgeGraphNeo4j)
pip install -e .
```
This command installs all required packages and makes your project's code available on the Python path without needing to reinstall after every change.

### Step 2.4: Configure the Application (`.env`)

Configuration is managed via an environment file.

```bash
# Create your personal configuration file from the example
cp .env.example .env
```

**Now, you must edit the `.env` file** with the correct credentials and URLs for your setup.

```ini
# .env

# --- Application Settings ---
# Recommended log level for production is "INFO"
LOG_LEVEL="INFO"

# --- Neo4j Connection ---
# Use bolt+ssc:// for self-signed certificates.
# Replace 'localhost' if your DB is on another machine.
NEO4J_URI="bolt+ssc://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your-secure-neo4j-password-from-secrets-file"

# --- vLLM (Gemma LLM) Connection ---
# The URL to your vLLM container's OpenAI-compatible endpoint.
VLLM_BASE_URL="http://localhost:5000/v1"
VLLM_API_KEY="your-secret-key-if-you-configured-one"
VLLM_MODEL_NAME="RedHatAI/gemma-3-27b-it-FP8-dynamic"

# --- Triton (Jina Embedder) Connection ---
# The URL to your Triton container.
TRITON_URL="http://localhost:4000"
```

---

## 5. Running for Development and Testing

Before deploying as a system service, it's crucial to run the application manually to ensure everything is configured correctly.

### Step 3.1: Start the Server Manually

With your `conda activate graphiti-ingestion` environment active, run the following command from your project root:

```bash
uvicorn main:app --host 0.0.0.0 --port 6000 --reload
```
You should see Uvicorn start up, followed by your application's log messages, including "Application starting up..." and "Background worker started."

### Step 3.2: Test the API Endpoint

Open a **new terminal** (do not close the server terminal) and send a test request using `curl`.

```bash
curl -X POST "http://localhost:6000/episodes/" \
-H "Content-Type: application/json" \
-d '{
  "content": "The Statue of Liberty was a gift to the United States from the people of France in 1886.",
  "type": "text",
  "description": "Manual API test"
}'
```

- **In the `curl` terminal:** You should get back a JSON response with a `job_id`.
- **In the server terminal:** You should see logs indicating the request was received, the job was submitted, and the worker started processing it.

If this works, you are ready for production deployment.

---

## 6. Production Deployment with `systemd`

`systemd` is the standard service manager on modern Linux systems. It will ensure your application is always running.

### Step 4.1: Understand the `systemd` Service File

The `graphiti-ingestion.service` file in your project is a template that tells `systemd` how to run your application. It defines:
- **`[Unit]`**: Metadata about the service.
- **`[Service]`**: The core commands, user, working directory, and restart policies.
- **`[Install]`**: How the service integrates with the system's boot process.

### Step 4.2: Edit the Service File

Before deploying, you **must** customize the `graphiti-ingestion.service` file with the absolute paths for your environment.

```ini
# graphiti-ingestion.service

[Unit]
Description=Graphiti Ingestion Service
After=network.target

[Service]
# ❗️ EDIT THIS: The Linux user the service should run as.
User=vpa

# ❗️ EDIT THIS: The absolute path to your project's root directory.
WorkingDirectory=/home/vpa/KnowledgeGraphNeo4j

# ❗️ EDIT THIS: The absolute path to the Python executable in your Conda environment.
# Find this by running: conda activate graphiti-ingestion && which python
ExecStart=/home/vpa/miniconda3/envs/graphiti-ingestion/bin/python -m uvicorn main:app --host 0.0.0.0 --port 6000

# This line automatically loads your .env file.
EnvironmentFile=/home/vpa/KnowledgeGraphNeo4j/.env

# Configuration for automatic restarts and logging.
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Step 4.3: Deploy the Service

Run the following commands to copy, enable, and start your service.

```bash
# 1. Copy the customized service file to the systemd directory
sudo cp graphiti-ingestion.service /etc/systemd/system/

# 2. Reload the systemd daemon to find the new service file
sudo systemctl daemon-reload

# 3. Enable the service to start automatically on system boot
sudo systemctl enable graphiti-ingestion.service

# 4. Start the service immediately
sudo systemctl start graphiti-ingestion.service
```

---

## 7. Managing the Production Service

Once deployed, use these `systemctl` commands to manage your application.

### Checking Status

To see if the service is active, running, and view the latest logs:
```bash
sudo systemctl status graphiti-ingestion.service
```
✅ **Good Output:** You should see `Active: active (running)` in green.

### Viewing Logs

`systemd` redirects all application output to the system journal.
```bash
# View the last 50 log lines
sudo journalctl -u graphiti-ingestion.service -n 50

# Follow the logs in real-time (like `tail -f`)
sudo journalctl -u graphiti-ingestion.service -f
```

### Stopping, Starting, and Restarting

```bash
# Stop the service
sudo systemctl stop graphiti-ingestion.service

# Start the service
sudo systemctl start graphiti-ingestion.service

# Restart the service (e.g., after updating code or the .env file)
sudo systemctl restart graphiti-ingestion.service
```
**Note**: If you change the `.service` file itself, you must run `sudo systemctl daemon-reload` before restarting.

---

## 8. Troubleshooting Common Issues

- **Problem:** Service fails to start (`Active: failed`).
  - **Solution:** Check the logs immediately with `sudo journalctl -u graphiti-ingestion.service`. The most common cause is an incorrect path in the `ExecStart` or `WorkingDirectory` lines of your `.service` file.

- **Problem:** Application logs show connection errors.
  - **Solution:** Ensure all your Docker services (Neo4j, vLLM, Triton) are running and that the URLs in your `.env` file are correct and accessible from the server.

- **Problem:** `Permission denied` errors in the logs.
  - **Solution:** Ensure the user specified in the `.service` file (`User=vpa`) has read/write permissions for the project directory.

- **Problem:** Changes to `.env` file are not reflected.
  - **Solution:** You must restart the service for it to reload the environment file: `sudo systemctl restart graphiti-ingestion.service`.

---
