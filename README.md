# KnowledgeGraphNeo4j
### Step 1: Set up Neo4j with Docker

#### Official Neo4j Docker Image

First, here is the link to the official Neo4j Docker Hub page. You can find more details and documentation about the image here.

*   **Link:** [Official Neo4j on Docker Hub](https://hub.docker.com/_/neo4j)

#### Neo4j Docker Space Requirement

The size of the Neo4j Docker image can vary slightly between versions.

*   The latest Neo4j 5 image is typically around **400-500 MB** to download (compressed size). Once uncompressed and running, the container itself has a small footprint, but the space you need will primarily depend on the amount of data you plan to store. For starting out, a few gigabytes of available disk space for the data volume (`$HOME/neo4j/data` in the command below) is more than sufficient.

#### Docker Pull Command

Open your terminal or command prompt and use the following command to download the latest official Neo4j image to your system.

```bash
docker pull neo4j
```

#### Docker Launch Command

This command will start a Neo4j container with your specified password `neo4jbcc`. It also sets up persistent storage, so your data will be saved even if you stop or restart the container.

```bash
docker run \
    --name neo4j-bcc-container \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    --env NEO4J_AUTH=neo4j/neo4jbcc \
    neo4j:latest
```

**Command Breakdown:**

*   `--name neo4j-bcc-container`: Gives your container a memorable name.
*   `-p 7474:7474 -p 7687:7687`: Publishes the necessary ports. `7474` is for the Neo4j Browser (web interface), and `7687` is for the Bolt driver, which your Python script will use to connect.
*   `-d`: Runs the container in detached mode (in the background).
*   `-v $HOME/neo4j/data:/data`: Creates a persistent volume for your database files. This links the `/data` folder inside the container to a `neo4j/data` folder in your home directory. **This is crucial for not losing your data.**
*   `--env NEO4J_AUTH=neo4j/neo4jbcc`: Sets the authentication. The username will be `neo4j` and the password will be `neo4jbcc`.
*   `neo4j:latest`: Specifies that you want to run the latest version of the Neo4j image you just pulled.

After running this command, you can check if the container is running with `docker ps`. You should be able to access the Neo4j Browser by navigating to `http://localhost:7474` in your web browser. You will use the username `neo4j` and password `neo4jbcc` to log in.