# Production Deployment Guide — FAQ Chatbot (Qdrant Version)

Follow this step-by-step guide to deploy the FastAPI FAQ Chatbot backend and connect it to your production **Qdrant** instance.

---

## Step 1: Install Server-Level Prerequisites

Log in to your production server terminal and install Python, PostgreSQL, and Docker (if needed):

```bash
# 1. Update the server package repository
sudo apt update && sudo apt upgrade -y

# 2. Install Python 3.10+ and virtual environment utilities
sudo apt install python3 python3-venv -y

# 2b. Install uv (fast Python package manager — replaces pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Make uv available in the current shell (or re-login)
source $HOME/.local/bin/env

# 3. Install PostgreSQL database server (for transactional users/sessions data)
sudo apt install postgresql postgresql-contrib -y

# 4. (Optional) Install Docker — ONLY if Qdrant is not already running on the server
# sudo apt install docker.io -y
# sudo systemctl enable --now docker
```

---

## Step 2: Spin Up Qdrant (Skip if Qdrant is already running)

If a Qdrant service is **already running** on your production infrastructure, **skip this step**. 

If you need to start a fresh instance, run the persistent container:

```bash
docker run -d \
  --name qdrant \
  --restart always \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

---

## Step 3: Upload Project Files

Copy your project code to the target directory on your server (e.g., `/var/www/faq_chatbot/`).

### Keep these files/directories:
* `app/` (all agents, routes, data, and database logic)
* `migrations/` (SQL migration files for Postgres tables)
* `main.py`
* `run_migrations.py`
* `requirements.txt`
* `requirements.lock` (pinned dependency versions for reproducible installs)
* `.env`

### Exclude these (do not copy):
* `.venv/`
* `__pycache__/`
* `logs/`

---

## Step 4: Set Up Python Virtual Environment

Navigate to the project directory on your server and initialize the Python environment:

```bash
# 1. Create a clean virtual environment with uv
uv venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install all required libraries from the pinned lockfile
uv pip install -r requirements.lock
```

> **Note:** Installs use `requirements.lock` (exact pinned versions) for
> reproducible deployments. After changing `requirements.txt`, regenerate the
> lock with: `uv pip compile requirements.txt --universal -o requirements.lock`

---

## Step 5: Configure the `.env` File

Open the `.env` file on the server and update it. Point the chatbot to your **existing Qdrant service**:

* **Database Connection**: Set `DATABASE_URL` and `CHECKPOINT_DB_URL` to point to your PostgreSQL server.
* **Qdrant Connection**:
  * Set `QDRANT_URL` to your production Qdrant IP/domain (e.g., `QDRANT_URL=http://192.168.1.50:6333` or `http://localhost:6333`).
  * If your production Qdrant requires an API key, set `QDRANT_API_KEY=your-production-api-key`. Otherwise, leave it empty.
  * Set `QDRANT_COLLECTION=faq_embeddings`.
* **LLM Provider**: Configure the `LLM_PROVIDER` (e.g., `groq`, `openai`, or `gemini`) and set the corresponding API keys.
* **Urls**: Update `FRONTEND_URL` and `BACKEND_URL` with your public production domains.

---

## Step 6: Run Database Migrations (PostgreSQL)

Run the PostgreSQL database migration script to construct all the relational tables (`users`, `sessions`, `messages`, `logs`, etc.):

```bash
python run_migrations.py
```

---

## Step 7: Verify and Run the Application

Start the FastAPI server using Uvicorn. On the first startup, the application lifespan will:
1. Connect to PostgreSQL and initialize the LangGraph checkpointer tables.
2. Connect to your active Qdrant database, create the `faq_embeddings` collection, and automatically seed the FAQ questions and answers.
3. Download the semantic embedding model (if configured for online download).

To launch the server with multiple parallel workers for production traffic:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Step 8: Docker-Based Deployment (Recommended Alternative)

Instead of setting up Python virtual environments and Systemd services manually, you can use the newly created Docker configurations to deploy the application in a self-contained container.

### 1. Ensure Prerequisites on the Production Server:
- **Docker** and **Docker Compose** installed.
- Your production `.env` file populated with production database, Qdrant URLs, and API keys.
- Your Google Cloud credentials file `varsapradaya-credentials.json` placed in the project root directory.

### 2. Build and Start the Application in Background Mode:
Run the following command on your production server:
```bash
docker compose up -d --build
```
> **Detached Mode (`-d`)**: This starts the container in the background. You can safely close your terminal or laptop, and the chatbot will continue running.

### 3. Monitor Production Logs:
To check if the database migrations applied correctly and the server is running:
```bash
docker compose logs -f
```

### 4. Manage Container Lifespan:
- **Stop the service**: `docker compose down`
- **Restart the service**: `docker compose restart`
- **Check service status**: `docker compose ps`

---

## Step 9: Manage Server Lifespan with Systemd

To keep the application running continuously in the background and ensure it automatically restarts if the server reboots or crashes, create a systemd service:

1. Create the service file:
   ```bash
   sudo nano /etc/systemd/system/faq-chatbot.service
   ```

2. Paste the following configuration (replace path placeholders with your actual directory path):
   ```ini
   [Unit]
   Description=FastAPI FAQ Chatbot Service
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/var/www/faq_chatbot
   ExecStart=/var/www/faq_chatbot/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 4
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the background service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable faq-chatbot
   sudo systemctl start faq-chatbot
   ```

---

## Step 10: Configure Nginx as a Reverse Proxy with SSL (HTTPS)

Serve the chatbot backend securely over HTTPS:

1. Install Nginx and Certbot:
   ```bash
   sudo apt install nginx certbot python3-certbot-nginx -y
   ```

2. Open Nginx configuration:
   ```bash
   sudo nano /etc/nginx/sites-available/faq_chatbot
   ```

3. Add the server proxy block:
   ```nginx
   server {
       server_name yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

4. Enable the site and restart Nginx:
   ```bash
   sudo ln -s /etc/nginx/sites-available/faq_chatbot /etc/nginx/sites-enabled/
   ```

5. Test Nginx and restart:
   ```bash
   sudo nginx -t
   sudo systemctl restart nginx
   ```

6. Request and install an SSL certificate:
   ```bash
   sudo certbot --nginx -d yourdomain.com
   ```
