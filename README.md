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
# Set up Neo4j with Docker

**Action Required:** Throughout this guide, replace the placeholder `YOUR_SERVER_IP` with your machine’s actual Local Area Network (LAN) IP address (e.g., `192.168.1.101`).

---

### Step 1: Find Your Server's LAN IP Address (Prerequisite)

First, identify the local IP address of the server that will host the Docker container. This IP will be used in the TLS certificate and the Docker port bindings.

```bash
# On Linux
hostname -I | awk '{print $1}'
# Or
ip addr show

# On macOS
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'
```
**Why this is important:** This IP is the address other machines on your network will use to connect to the database. It must be correct for the security settings to work.

#### For localhost actually these are not needed


---

### Step 2: Create Project Structure and Set Permissions

This step organizes all your configuration and data files and ensures the Neo4j process inside the container can access them without permission errors.

```bash
# Create the complete directory structure
mkdir -p neo4j-local-secure/{conf,data,logs,plugins,certificates,secrets}
cd neo4j-local-secure

# Set ownership to match the container's internal neo4j user (UID 101, GID 101)
sudo chown -R 101:101 data logs conf plugins certificates
```
**Why this is important:** This prevents "Permission Denied" errors when the non-root `neo4j` user in the container tries to write to the data or log directories.

---

### Step 3: Secure the Admin Password via Docker Secrets

We will generate a strong password and store it in a file. Docker Secrets will manage this file, keeping the password out of your configuration and environment variables.

```bash
# 1. Generate a strong, random password for the 'neo4j' user
openssl rand -base64 32 > secrets/neo4j_password

# 2. Create the final secret file in the format Neo4j expects: username/<password>
echo "neo4j/$(cat secrets/neo4j_password)" > secrets/neo4j_auth

# 3. (Optional) Display the password once to save it in your password manager
echo "Your secure password is: $(cat secrets/neo4j_password)"
```
**Why this is important:** This is a secure method for handling credentials, vastly superior to hardcoding them or using plain environment variables.

---
#### Alternative
**IMPORTANT**: If you want to do it with root:The problem is a combination of file permissions in the root (`/`) directory and a common issue with how `sudo` works with output redirection (`>`).

Here is the explanation and the corrected set of commands for only this step.

### The Problem

1.  **Directory Ownership:** When you ran `sudo mkdir -p ...`, the `secrets` directory was created and is owned by the `root` user.
2.  **Redirection (`>`) Failure:** The command `sudo openssl ... > ...` fails because the shell (bash) tries to set up the file redirection *before* it runs the `sudo` command. Your regular user (`your non root user`) does not have permission to write to the `root`-owned `secrets` directory, so the operation fails before `openssl` even runs.

### The Solution

We need to pipe (`|`) the output of `openssl` to a command that can be run with `sudo` and can write to a file. The `tee` command is perfect for this.

Here are the corrected commands to create your password secrets in `/neo4j-local-secure/secrets`.

```bash
# Ensure you are in the correct directory
cd /neo4j-local-secure

# --- CORRECTED COMMANDS START HERE ---

# 1. Generate the password and pipe it to `tee`, which uses sudo to write the file.
openssl rand -base64 32 | sudo tee secrets/neo4j_password > /dev/null

# 2. Create the final auth secret file using the same technique.
echo "neo4j/$(sudo cat secrets/neo4j_password)" | sudo tee secrets/neo4j_auth > /dev/null

# 3. (Optional) Verify the files were created and are owned by root.
sudo ls -l secrets

# 4. (Optional) Display the password so you can save it.
echo "Your secure password is: $(sudo cat secrets/neo4j_password)"
```

#### Command Explanation

*   **`openssl rand -base64 32`**: This part is the same; it generates the random password string.
*   **`|`**: This is the "pipe" operator. It sends the output of the command on its left as the input to the command on its right.
*   **`sudo tee secrets/neo4j_password`**:
    *   `tee` is a command that reads input and writes it to both the screen and a file.
    *   By running `sudo tee`, the `tee` process itself has root privileges and therefore has permission to write the file `secrets/neo4j_password`.
*   **`> /dev/null`**: We add this because `tee` also outputs to the screen by default. Piping the screen output to `/dev/null` (a special file that discards everything written to it) keeps your terminal clean.

---

### Step 4: Generate a TLS Certificate for Your IP Address (with SAN)

Modern browsers and clients require the IP address to be in the **Subject Alternative Name (SAN)** field of a certificate. This command creates a certificate that is valid for your specific server IP.

```bash
# Set your server's IP address as an environment variable
export SERVER_IP="YOUR_SERVER_IP"

# Generate the key and certificate with the IP in both the CN and SAN fields
# This single command works on most modern systems (OpenSSL ≥ 1.1.1)
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -subj "/CN=${SERVER_IP}" \
  -addext "subjectAltName = IP:${SERVER_IP}" \
  -keyout certificates/private.key \
  -out certificates/public.crt
```
**Why this is important:** Including the IP in the SAN is the modern standard and prevents hostname mismatch errors from clients, ensuring a secure and valid TLS connection.

#### For localhost

```bash
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -subj "/CN=localhost" \
  -addext "subjectAltName = DNS:localhost" \
  -keyout certificates/private.key \
  -out certificates/public.crt
```

* give access: ```sudo chown -R 101:101 certificates```

---

### Step 5: Configure Neo4j with Modern 5.x Settings

Create the file `conf/neo4j.conf`. This configuration enforces encrypted-only connections and advertises the correct IP address to connecting clients.

**Remember to replace `YOUR_SERVER_IP` in this file.**

```ini
# conf/neo4j.conf (Corrected for Neo4j 5.x)

# ================= Network =================
# Listen on all interfaces inside the container
server.default_listen_address=0.0.0.0

# Advertise the correct LAN IP to clients. CRITICAL for drivers and clustering.
# For a LAN setup, uncomment and set this. For a pure localhost setup, leave it commented.
# server.default_advertised_address=YOUR_SERVER_IP

# ================= Connectors =================
# Disable the insecure HTTP connector entirely.
server.http.enabled=false

# --- HTTPS Connector ---
server.https.enabled=true
server.https.listen_address=:7473
# TLS configuration using SSL policy
dbms.ssl.policy.https.enabled=true
dbms.ssl.policy.https.base_directory=/certificates
dbms.ssl.policy.https.private_key=private.key
dbms.ssl.policy.https.public_certificate=public.crt

# --- Bolt Connector ---
server.bolt.enabled=true
server.bolt.listen_address=:7687
# Enforce encrypted connections for Bolt.
server.bolt.tls_level=REQUIRED
# Use the same SSL policy for Bolt
dbms.ssl.policy.bolt.enabled=true
dbms.ssl.policy.bolt.base_directory=/certificates
dbms.ssl.policy.bolt.private_key=private.key
dbms.ssl.policy.bolt.public_certificate=public.crt

# ================= Security =================
# Keep strict validation enabled. This is a critical security feature.
server.config.strict_validation.enabled=true

# ================= Memory (Tune for your machine) =================
server.memory.heap.initial_size=4G
server.memory.heap.max_size=4G
server.memory.pagecache.size=8G
```

---

### Step 6: Create the Docker Compose File

Create the `docker-compose.yml` file. This is the heart of the deployment, binding the service specifically to your LAN IP and using the robust `cypher-shell` healthcheck.

**Remember to replace `YOUR_SERVER_IP` in this file.**

```yaml

services:
  neo4j:
    image: neo4j:5-enterprise          # Tracks the latest 5.x version
    container_name: neo4j_local_secure
    restart: unless-stopped
    user: "101:101"                    # Run as a non-root user

    # Security: Bind host ports specifically to your LAN IP, not 0.0.0.0 (all interfaces).
    ports:
      - "YOUR_SERVER_IP:7473:7473"     # Secure HTTPS
      - "YOUR_SERVER_IP:7687:7687"     # Secure Bolt

    volumes:
      - ./data:/data
      - ./logs:/logs
      - ./conf:/conf
      - ./plugins:/plugins
      - ./certificates:/certificates

    environment:
      - NEO4J_AUTH_FILE=/run/secrets/neo4j_auth_secret
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes

    secrets:
      - source: neo4j_auth_secret
        target: neo4j_auth_secret

    healthcheck:
      # Robust check using cypher-shell to query the database directly.
      # 'bolt+ssc' scheme trusts the self-signed certificate for the healthcheck.
      test: ["CMD-SHELL", "PASS=$$(cut -d/ -f2 /run/secrets/neo4j_auth_secret) && cypher-shell -a bolt+ssc://localhost:7687 -u neo4j -p \"$$PASS\" 'RETURN 1' >/dev/null 2>&1 || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 10

secrets:
  neo4j_auth_secret:
    file: ./secrets/neo4j_auth
```

#### For localhost:

```bash
    # Security: Bind host ports specifically to localhost (127.0.0.1).
    ports:
      - "127.0.0.1:7473:7473"     # Accessible only from https://localhost:7473
      - "127.0.0.1:7687:7687"     # Accessible only from bolt+ssc://localhost:7687
```

---

### Step 7: Launch, Verify, and Connect

You are now ready to start the service and connect from another machine on your local network.

```bash
# 1. Start the service in the background
docker compose up -d

# 2. Check the status. Wait for the STATUS to become 'running (healthy)'
docker compose ps

# 3. (Optional) Follow the logs on first startup
docker logs -f neo4j_local_secure
```

**How to Connect:**

*   **Neo4j Browser:** Navigate to `https://YOUR_SERVER_IP:7473`
    *   *You will see a security warning. This is expected. Click "Advanced" and proceed.*
*   **Drivers & Tools:** Use a secure connection string. The `+ssc` scheme is a convenient shortcut that trusts self-signed certificates without needing a custom trust store.
    *   `bolt+ssc://YOUR_SERVER_IP:7687`
*   **`cypher-shell` from your host machine:**
    ```bash
    cypher-shell -a bolt+ssc://YOUR_SERVER_IP:7687 -u neo4j -p "$(cat secrets/neo4j_password)"
    ```


#### For Localhost: 
**Local Port Forwarding** (or an "SSH Tunnel").

you bound the ports to `127.0.0.1` on the server, you cannot connect to `https://YOUR_SERVER_IP:7473`. 

The solution is to tell SSH to create a secure tunnel from your local machine, through the SSH connection, directly to the `localhost` port on the remote server.

### The Concept

You will forward a port on your **local machine** (the one you are typing on) to the Neo4j port on the **remote server's localhost**.

*   Traffic going into `localhost:7473` on **your laptop**
*   ...travels securely through the SSH connection...
*   ...and comes out on `localhost:7473` on the **remote server**, where Neo4j is listening.

### The Solution: The `-L` Flag in SSH

You need to forward two ports: `7473` for the browser (HTTPS) and `7687` for the Bolt driver. You can do this in a single SSH command by using the `-L` flag twice.

#### Step 1: Disconnect and Reconnect with Port Forwarding

First, if you are currently connected to your server, `exit` that SSH session.

Now, reconnect using the following command. This is the **only command you need to change**.

```bash
# General Syntax: ssh -L <local_port>:<destination_host>:<destination_port> user@server
#
# We will forward:
# - Our local port 7473 to the remote server's localhost:7473
# - Our local port 7687 to the remote server's localhost:7687

ssh -L 7473:localhost:7473 -L 7687:localhost:7687 vpa@172.22.11.241
```

**Command Breakdown:**

*   **`ssh user@the.server.ip.addreess`**: Your standard SSH login command.
*   **`-L 7473:localhost:7473`**:
    *   `-L`: Specifies **L**ocal port forwarding.
    *   `7473`: The port to open on **your local machine**.
    *   `localhost`: The destination host *from the remote server's perspective*. We want to connect to `localhost` on the server.
    *   `7473`: The destination port on the remote server.
*   **`-L 7687:localhost:7687`**: Does the same thing for the Bolt port.

#### Step 2: Keep the SSH Connection Open

As long as this SSH terminal window is open, the secure tunnels are active. If you close this window, the tunnels will close.

#### Step 3: Connect Using `localhost` on Your Local Machine

Now, on your local machine (your laptop), you can access Neo4j as if it were running locally.

*   **Open Your Web Browser (on your laptop):**
    Navigate to: `https://localhost:7473`
    *   Your browser will send the request to your local port 7473.
    *   SSH will intercept it, send it through the tunnel, and deliver it to Neo4j on the server.
    *   You will still see the security warning for the self-signed certificate, which is expected.

*   **Use `cypher-shell` or a Driver (from your laptop):**
    Use the connection string: `bolt+ssc://localhost:7687`
    ```bash
    # You would run this in a NEW terminal window on your local machine,
    # NOT in the SSH window.
    cypher-shell -a bolt+ssc://localhost:7687 -u neo4j -p "YOUR_SECURE_PASSWORD"
    ```

This is the standard and most secure way to manage services that are intentionally not exposed to the network. You maintain a very high level of security on the server while still having full access for management from your trusted machine.

---

### Step 8: Harden with a Firewall (Recommended)

For true defense-in-depth, configure a firewall on the host machine to only allow traffic to the Neo4j ports from your trusted local network.

**UFW (Ubuntu) Example:**

```bash
# Replace with your LAN subnet (e.g., 192.168.1.0/24)
export LAN_SUBNET="YOUR_LAN_SUBNET"

# Allow incoming connections from the LAN to the Neo4j ports
sudo ufw allow from ${LAN_SUBNET} to any port 7473 proto tcp
sudo ufw allow from ${LAN_SUBNET} to any port 7687 proto tcp

# Ensure UFW is enabled and check the status
sudo ufw enable
sudo ufw status
```
**Why this is important:** Even if Docker is bound to a specific IP, the firewall acts as a second, powerful layer of defense against unwanted network access.


# Connecting to Your Secure Neo4j Docker Instance

This guide provides detailed instructions on how to connect to and verify your secure Neo4j 5.x Docker container. It covers all common scenarios, including connecting from the host machine, a remote machine on the same network, and securely through an SSH tunnel.

## Prerequisites

Before you begin, ensure you have the following information and tools:

1.  **Server IP Address**: The LAN IP address of the machine hosting the Docker container (e.g., `192.168.1.101`). This will be referred to as `YOUR_SERVER_IP`.
2.  **Neo4j Password**: The secure password you generated. You can retrieve it from the host machine at any time by running:
    ```bash
    # Run this from your neo4j-local-secure directory
    sudo cat secrets/neo4j_password
    ```
3.  **`cypher-shell`**: The official Neo4j command-line interface. This must be installed on any machine you wish to connect from.

---

## Connection Methods

The correct connection method depends on how you configured the `ports` section in your `docker-compose.yml` file.

### Scenario 1: Ports are Bound to Your Server's LAN IP

This is the standard setup for making Neo4j available to other machines on your local network. Your `docker-compose.yml` `ports` section looks like this:

```yaml
    ports:
      - "YOUR_SERVER_IP:7473:7473" # Secure HTTPS
      - "YOUR_SERVER_IP:7687:7687" # Secure Bolt
```

#### ► Method 1: Connecting from the Host Machine Terminal

This is the most direct way to interact with the database.

1.  **Open a terminal** on the machine running the Docker container.
2.  **Navigate** to your `neo4j-local-secure` project directory.
3.  **Run the `cypher-shell` command**:

    ```bash
    cypher-shell \
      -a "bolt+ssc://YOUR_SERVER_IP:7687" \
      -u neo4j \
      -p "$(sudo cat secrets/neo4j_password)"
    ```
    > **Note**: The `+ssc` in the address tells the client to trust the server's **S**elf-**S**igned **C**ertificate, which is exactly what we need for this setup.

4.  **Verify the connection** as described in the [Verification](#verification) section below.

#### ► Method 2: Connecting from a Remote Machine Terminal

You can connect from any other computer on the same LAN that has `cypher-shell` installed.

1.  **Open a terminal** on your remote machine (e.g., your laptop).
2.  **Run the `cypher-shell` command**. You will be prompted to enter your password securely.

    ```bash
    cypher-shell -a "bolt+ssc://YOUR_SERVER_IP:7687" -u neo4j
    ```
3.  **Enter your password** when prompted.
4.  **Verify the connection** as described in the [Verification](#verification) section.

#### ► Method 3: Connecting via Web Browser

You can access the Neo4j Browser from any machine on the same LAN.

1.  **Open your web browser** (Chrome, Firefox, etc.).
2.  **Navigate to the following URL**:
    `https://YOUR_SERVER_IP:7473`
3.  **Handle the Security Warning**: Your browser will show a privacy or security warning because the certificate is self-signed. This is expected.
    *   Click **"Advanced"**.
    *   Click **"Proceed to YOUR_SERVER_IP (unsafe)"** or "Accept the Risk and Continue".
4.  **Log in**: Use `neo4j` as the username and your secure password. The connection URI should default to `bolt://YOUR_SERVER_IP:7687`.
5.  Click **Connect**.

---

### Scenario 2: Ports are Bound to `localhost` (127.0.0.1)

This is a high-security setup where the database is not exposed to the network at all. Remote access is only possible via a secure SSH tunnel. Your `docker-compose.yml` `ports` section looks like this:

```yaml
    ports:
      - "127.0.0.1:7473:7473"
      - "127.0.0.1:7687:7687"
```

#### ► Method 1: Connecting from the Host Machine (Terminal & Browser)

Connecting from the host is simple because you are already on `localhost`.

*   **Terminal**:
    ```bash
    cypher-shell \
      -a "bolt+ssc://localhost:7687" \
      -u neo4j \
      -p "$(sudo cat secrets/neo4j_password)"
    ```
*   **Browser**:
    1.  Navigate to `https://localhost:7473`.
    2.  Bypass the security warning as explained above.
    3.  Log in with your credentials.

#### ► Method 2: Connecting from a Remote Machine (via SSH Tunnel)

This is the standard, secure way to manage a service that is not exposed to the network.

1.  **Establish the SSH Tunnel**: From your remote machine's terminal, run the following command to connect to your server. This command forwards your local ports `7473` and `7687` through the SSH connection to the server's `localhost`.

    ```bash
    # Syntax: ssh -L <local_port>:<destination_host>:<destination_port> user@server
    ssh -L 7473:localhost:7473 -L 7687:localhost:7687 your_user@YOUR_SERVER_IP
    ```
    **Important**: You must keep this SSH terminal window open. The tunnel remains active only as long as this connection is open.

2.  **Connect in a *New* Terminal or Browser**: With the tunnel active, open a **new** terminal window or your browser on your remote machine and connect to `localhost` as if Neo4j were running locally.

    *   **New Terminal**:
        ```bash
        # Connect to your local port, which SSH will forward to the server
        cypher-shell -a "bolt+ssc://localhost:7687" -u neo4j
        ```
    *   **Browser**:
        1.  Navigate to `https://localhost:7473`.
        2.  Bypass the security warning.
        3.  Log in. Your traffic will be securely tunneled to the server.

---

## Verification

A successful connection can be confirmed in two ways.

### In `cypher-shell`

Upon successful connection, your terminal prompt will change to:
`neo4j@bolt+ssc://...> `

Run this basic command to confirm the database is responsive:

```cypher
SHOW DATABASES;
```

A successful query will return a table of the available databases:

```
+--------------------------------------------------------------------------------------------+
| name     | type       | aliases | access      | address               | role      | writer |
+--------------------------------------------------------------------------------------------+
| "neo4j"  | "standard" | []      | "read-write"| "localhost:7687"      | "primary" | TRUE   |
| "system" | "system"   | []      | "read-write"| "localhost:7687"      | "primary" | TRUE   |
+--------------------------------------------------------------------------------------------+
```

To exit the shell, type `:exit` and press Enter.

### In Neo4j Browser

After logging in, you will see the main Neo4j Browser interface. The top-left corner will show that you are connected to the database, and you can type Cypher queries into the editor at the top of the screen.


---

# Managing Graph Data in Your Secure Neo4j Docker Instance

This guide provides essential instructions for managing the data within your Neo4j database. It covers how to perform logical exports of your entire graph into portable formats and how to completely clear the database of all nodes and relationships for a clean reset.

These operations are performed while the database is running and do not require you to stop the container.

## Table of Contents
- [Prerequisite: Installing the APOC Plugin](#prerequisite-installing-the-apoc-plugin)
- [Exporting All Graph Data](#exporting-all-graph-data)
- [Deleting All Graph Data](#deleting-all-graph-data)

---

## Prerequisite: Installing the APOC Plugin

The most powerful and flexible way to manage data is with the **APOC ("Awesome Procedures On Cypher")** library. This is a one-time setup.

1.  **Download APOC**: Go to the [APOC Releases page](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases) and find the release that **exactly matches your Neo4j version**. For example, if you are using Neo4j `5.26.10`, download the APOC `5.26.10` JAR file.
    *   Under the "Assets" section, download the file named `apoc-x.x.x-core.jar`.

2.  **Place the Plugin**: Move the downloaded `.jar` file into the `./plugins` directory of your project on the host machine.
    ```bash
    # Example command
    mv ~/Downloads/apoc-5.26.10-core.jar ./plugins/
    ```

3.  **Configure Neo4j to Allow APOC**: Edit your `conf/neo4j.conf` file and add the following line. This gives APOC the necessary permissions to perform operations like writing files.

    ```ini
    # Add this line to the end of your conf/neo4j.conf
    dbms.security.procedures.unrestricted=apoc.*
    ```

4.  **Restart the Container**: To load the plugin and apply the new configuration, restart your Docker service from your project directory.
    ```bash
    docker compose down && docker compose up -d
    ```

---

## Exporting All Graph Data

These procedures export your data into files saved on the host machine, making them easy to access, share, or use for migration.

#### Setup: Create a Shared Directory for Exports

To easily retrieve exported files, we'll map a local directory into the container.

1.  **Create an `exports` directory** on your host:
    ```bash
    mkdir ./exports
    ```
2.  **Add the volume to `docker-compose.yml`**:
    ```yaml
    # in your docker-compose.yml
    services:
      neo4j:
        # ... other settings
        volumes:
          - ./data:/data
          - ./logs:/logs
          - ./conf:/conf
          - ./plugins:/plugins
          - ./certificates:/certificates
          - ./exports:/exports   # <--- ADD THIS LINE
    ```
3.  **Restart the container** if you just added the volume: `docker compose up -d --force-recreate`.

#### ► Export Option 1: Cypher Script
This method creates a single `.cypher` file containing all the `CREATE` statements needed to perfectly rebuild your graph. It is ideal for backups and migrations to other Neo4j instances.

Connect to your database via `cypher-shell` or the Neo4j Browser and run:
```cypher
/*
  This procedure will create a 'all-data.cypher' file
  inside the ./exports directory on your host machine.
*/
CALL apoc.export.cypher.all('all-data.cypher', {
  format: 'cypher-shell',
  useOptimizations: {type: 'UNWIND', unwindBatchSize: 100}
});
```

#### ► Export Option 2: GraphML
GraphML is a standard XML-based format that can be imported into other graph visualization and analysis tools, such as Gephi.

```cypher
/*
  This procedure will create a 'all-data.graphml' file
  inside the ./exports directory on your host machine.
*/
CALL apoc.export.graphml.all('all-data.graphml', {});
```

---

## Deleting All Graph Data

This action removes **all nodes and relationships** from your `neo4j` database. It does **not** remove your user accounts, and it leaves your schema (indexes and constraints) intact.

#### ► Method 1: Simple Delete (For Small to Medium Graphs)
This command is easy to remember and effective for databases that are not excessively large.

```cypher
MATCH (n) DETACH DELETE n;
```
> **Warning**: On very large graphs (millions of nodes/edges), this single transaction can consume a large amount of memory and may fail. For large datasets, use the batched method below.

#### ► Method 2: Batched Delete (For Very Large Graphs)
This is the recommended, robust method for clearing any size of graph. It uses an APOC procedure to delete nodes in smaller, manageable batches, preventing memory issues.

```cypher
/*
  This procedure finds all nodes and runs DETACH DELETE on them in
  batches of 50,000 until the database is empty.
*/
CALL apoc.periodic.iterate(
  'MATCH (n) RETURN n',
  'DETACH DELETE n',
  {batchSize: 50000}
)
YIELD batches, total;
```

#### Verification
After running a delete command, you can confirm the database is empty with the following query. The expected result is `0`.
```cypher
MATCH (n) RETURN count(n);
```