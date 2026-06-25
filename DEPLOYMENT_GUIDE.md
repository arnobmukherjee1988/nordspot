# SE3 Electricity Price Forecast — Complete Deployment Guide

This guide walks you through replicating the full production deployment of the SE3 day-ahead electricity price forecasting system on AWS from scratch. It covers every account, every website, every click, and every terminal command in the exact order they were executed.

**What you will have at the end:**
- A live Streamlit dashboard at your own domain (e.g. `https://se3.your-domain.com`)
- Daily automated model retraining via AWS SSM State Manager
- Model artifact persistence on AWS S3 (survives container restarts)
- CloudWatch monitoring (memory, disk, pipeline failure) with email alerts
- HTTPS with zero open server ports via Cloudflare Tunnel

**Estimated time:** 3–4 hours for a first-time deployment.

---

## Prerequisites

| Requirement | Details |
|---|---|
| A computer with macOS or Linux | Commands marked `[Mac terminal]` run on your local machine |
| An AWS account | Free tier is sufficient for the first year |
| A domain name | Any registrar; we will move DNS to Cloudflare |
| A Cloudflare account | Free tier is sufficient |
| An ENTSO-E API key | Free registration at transparency.entsoe.eu |
| Git | `brew install git` on macOS |

---

## Repository Structure

```
SE3_prod/
├── Dockerfile                  # Python 3.12-slim app image
├── docker-compose.yml          # Orchestrates ClickHouse + Streamlit app
├── entrypoint.sh               # Downloads models from S3 on cold start
├── requirements.txt            # All Python dependencies
├── clickhouse-config.xml       # Memory cap for 2 GB server
├── .env.example                # Template — copy to .env and fill in
├── .gitignore
├── .dockerignore
├── db/
│   ├── __init__.py
│   └── schema.py               # TimeDB series IDs and schema init
├── pipeline/
│   ├── __init__.py
│   ├── fetch_prices.py         # ENTSO-E REST API → TimeDB
│   ├── fetch_weather.py        # Open-Meteo REST API → TimeDB
│   ├── features.py             # Feature matrix builder
│   └── store.py                # TimeDB write helper
├── ml/
│   ├── __init__.py
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation utilities
│   ├── run_eval.py             # Standalone evaluation runner
│   └── models/
│       ├── __init__.py
│       ├── lgbm.py             # LightGBM quantile regression + conformal calibration
│       └── lear.py             # LEAR (24 per-hour LassoCV models)
├── dashboard/
│   └── app.py                  # 6-tab Streamlit dashboard
└── model/                      # Generated at runtime — not committed to git
    ├── lgbm_q05.pkl
    ├── lgbm_q50.pkl
    ├── lgbm_q95.pkl
    ├── lgbm_conformal.pkl
    ├── lear_h00.pkl … lear_h23.pkl
    ├── metrics.json
    ├── trained_at.json
    └── MODEL_LOG.md
```

---

## Part 1 — External API Accounts

### 1.1 ENTSO-E API Key

The pipeline fetches SE3 electricity prices from the ENTSO-E Transparency Platform.

1. Go to **https://transparency.entsoe.eu**
2. Click **Login** (top right) → **Register**
3. Fill in name, email, organisation. Accept terms. Submit.
4. Confirm your email via the link sent to your inbox.
5. Log in → click your name (top right) → **My Account Settings**
6. Scroll down to **Web API Security Token** → click **Generate a new token**
7. Copy the token. This is your `ENTSOE_API_KEY`.

> The SE3 bidding zone EIC code is `10Y1001A1001A46L` — already hardcoded in `pipeline/fetch_prices.py`.

---

## Part 2 — AWS Account and Services

All AWS steps use the **AWS Management Console** at https://console.aws.amazon.com.

> **Region for everything:** `eu-north-1` (Stockholm). Select it in the top-right region dropdown before creating any resource.

### 2.1 Create an AWS account (skip if you already have one)

1. Go to **https://aws.amazon.com** → click **Create an AWS Account**
2. Enter email, choose account name → click **Verify email address**
3. Enter verification code sent to your email
4. Set a root password (use a strong one — this is the most privileged credential)
5. Choose **Personal** account type, fill in billing details (a credit card is required even for free tier)
6. Verify phone number via call or SMS
7. Choose **Basic support** (free) → complete registration
8. Log in to the console at https://console.aws.amazon.com

> **Security:** Enable MFA on your root account immediately. IAM → Security recommendations → Add MFA.

---

### 2.2 Create an S3 Bucket (model artifact storage)

After each training run, `ml/train.py` uploads the trained model files to S3. When the Docker container restarts, `entrypoint.sh` downloads them back.

1. In the AWS Console, go to **S3** (search in the top bar)
2. Click **Create bucket**
3. **Bucket name:** choose a globally unique name, e.g. `se3-model-artifacts-YOURNAME`
   > Bucket names are global across all AWS accounts — add your name/initials to make it unique
4. **AWS Region:** `eu-north-1`
5. **Block all public access:** leave enabled (default) — this bucket is private
6. **Bucket versioning:** Disable (we overwrite files on each training run)
7. Leave all other settings at defaults
8. Click **Create bucket**
9. **Note your bucket name** — you will need it in `.env` as `S3_BUCKET`

---

### 2.3 Create an IAM User (least-privilege credentials for the server)

Never use your root account credentials on the server. Create a dedicated IAM user with only the permissions needed.

1. Go to **IAM** → **Users** → **Create user**
2. **User name:** `se3-pipeline`
3. Click **Next**
4. Select **Attach policies directly**
5. Click **Create policy** (opens a new tab):
   - Click the **JSON** tab
   - Paste the following policy:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:PutObject",
           "s3:GetObject",
           "s3:DeleteObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::YOUR-BUCKET-NAME",
           "arn:aws:s3:::YOUR-BUCKET-NAME/*"
         ]
       },
       {
         "Effect": "Allow",
         "Action": "cloudwatch:PutMetricData",
         "Resource": "*"
       }
     ]
   }
   ```
   > Replace `YOUR-BUCKET-NAME` with your actual bucket name.
   > Note: `cloudwatch:PutMetricData` requires `Resource: *` — CloudWatch does not support resource-level ARN restrictions for this action. This is correct, not an over-grant.
   - Click **Next** → **Policy name:** `se3-pipeline-policy` → **Create policy**
6. Return to the user creation tab → refresh the policy list → search for `se3-pipeline-policy` → select it
7. Click **Next** → **Create user**
8. Click the new user → **Security credentials** tab → **Create access key**
9. **Use case:** Command Line Interface (CLI)
10. Click through → **Create access key**
11. **Download the CSV** or copy both values now — the secret key is shown only once:
    - Access key ID: `AKIAXXXXXXXXXXXXXXXX`
    - Secret access key: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

### 2.4 Create a Lightsail Instance (the server)

1. Go to **https://lightsail.aws.amazon.com**
2. Click **Create instance**
3. **Instance location:** `eu-north-1` (Stockholm)
4. **Platform:** Linux/Unix
5. **Blueprint:** OS Only → **Ubuntu 22.04 LTS**
6. **Instance plan:** $10/month (2 GB RAM, 1 vCPU, 60 GB SSD)
   > The 2 GB plan is the minimum — LightGBM training requires ~1.2 GB RAM. The 1 GB plan will OOM-kill.
7. **Key pair:** Create a new key pair or use the default. If creating new:
   - Click **Create new** → name it `se3-key`
   - Download the `.pem` file immediately — it cannot be re-downloaded
   - Save it as `~/.ssh/se3-key.pem` on your Mac
8. **Instance name:** `se3-forecast`
9. Click **Create instance**. Wait ~60 seconds for the server to boot (status turns green).

**Attach a static IP:**
1. In the Lightsail console → **Networking** tab → **Create static IP**
2. **Attach to:** `se3-forecast`
3. **Static IP name:** `se3-ip`
4. Click **Create**
5. Note the assigned IP address — this is your server's permanent IP

**Open required ports:**
In Lightsail, the firewall is configured per-instance:
1. Click the instance → **Networking** tab → **IPv4 Firewall**
2. Verify SSH (port 22) is already open
3. Everything else should remain closed — the dashboard is served via Cloudflare Tunnel (no inbound HTTP/HTTPS ports needed)

---

### 2.5 Connect to the Server via SSH

[Mac terminal]
```bash
# Fix file permissions — SSH refuses keys that are world-readable
chmod 400 ~/.ssh/se3-key.pem

# Connect (replace with your actual static IP)
ssh -i ~/.ssh/se3-key.pem ubuntu@YOUR_SERVER_IP
```

You should see the Ubuntu welcome message. All commands in **[Server SSH terminal]** sections below run here.

---

### 2.6 Initial Server Setup

[Server SSH terminal]
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker, Docker Compose plugin, Git, and unzip
sudo apt install -y docker.io docker-compose-plugin git unzip

# Add ubuntu to the docker group (avoids needing sudo before every docker command)
sudo usermod -aG docker ubuntu

# Log out and back in for the group change to take effect
exit
```

[Mac terminal]
```bash
ssh -i ~/.ssh/se3-key.pem ubuntu@YOUR_SERVER_IP
```

---

### 2.7 Install AWS CLI v2 on the Server

The `run_pipeline.sh` script uses `aws cloudwatch put-metric-data`. The AWS CLI must be installed on the server.

[Server SSH terminal]
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install -y unzip
unzip awscliv2.zip
sudo ./aws/install

# Verify
aws --version
# Expected output: aws-cli/2.x.x Python/3.x.x Linux/...

# Clean up
rm -rf awscliv2.zip aws/
```

### 2.8 Configure AWS Credentials on the Server

[Server SSH terminal]
```bash
mkdir -p /home/ubuntu/.aws

cat > /home/ubuntu/.aws/credentials << 'EOF'
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
EOF

cat > /home/ubuntu/.aws/config << 'EOF'
[default]
region = eu-north-1
output = json
EOF

# Verify — should show the se3-pipeline user
aws sts get-caller-identity
```

---

### 2.9 Deploy the Project Code

[Mac terminal]
```bash
# Sync the project to the server — exclude secrets and cache files
rsync -av \
  --exclude '.env' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'model/*.pkl' \
  --exclude 'model/*.json' \
  --exclude '.weather_cache*' \
  /path/to/SE3_prod/ \
  ubuntu@YOUR_SERVER_IP:/home/ubuntu/SE3_prod/
```

**Create the `.env` file on the server:**

[Server SSH terminal]
```bash
cat > /home/ubuntu/SE3_prod/.env << 'EOF'
ENTSOE_API_KEY=your_entsoe_api_key_here
TIMEDB_CH_URL=http://se3user:se3password@localhost:8123/se3db
SE3_LAT=59.33
SE3_LON=18.07
SE3_TRAIN_START=2020-01-01
MODEL_DIR=model
S3_BUCKET=your-bucket-name-here
EOF
```

> `S3_BUCKET` is the bucket name you created in step 2.2. It is read by `ml/train.py` to upload model files after training, and by `entrypoint.sh` to download them on container startup.

**Create the pipeline script:**

[Server SSH terminal]
```bash
cat > /home/ubuntu/SE3_prod/run_pipeline.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/SE3_prod
docker compose exec -T app python -m pipeline.fetch_prices && \
docker compose exec -T app python -m pipeline.fetch_weather && \
docker compose exec -T app python -m ml.train
STATUS=$?
/usr/local/bin/aws cloudwatch put-metric-data \
  --namespace SE3/Pipeline \
  --metric-name PipelineFailure \
  --value $STATUS \
  --region eu-north-1
exit $STATUS
EOF

chmod +x /home/ubuntu/SE3_prod/run_pipeline.sh
```

> The explicit `/usr/local/bin/aws` path is important — when SSM runs the script, it may have a minimal PATH that does not include the AWS CLI location.

---

### 2.10 Start the Docker Stack

[Server SSH terminal]
```bash
cd /home/ubuntu/SE3_prod

# Build the app image and start both containers
docker compose up -d --build
```

This will:
1. Pull the ClickHouse 24.3 image (~600 MB — takes a few minutes on first run)
2. Build the Python app image from `Dockerfile`
3. Start both containers
4. `entrypoint.sh` runs: if `model/` is empty and `S3_BUCKET` is set, downloads models from S3; otherwise proceeds directly
5. Starts the Streamlit dashboard on port 8501

**Verify containers are running:**
```bash
docker compose ps
# Both containers should show status "running" or "healthy"

# Follow app logs (Ctrl+C to stop)
docker compose logs -f app
```

**Run the pipeline manually for the first time:**
```bash
/home/ubuntu/SE3_prod/run_pipeline.sh
```

This fetches ~6 years of prices and weather data from ENTSO-E and Open-Meteo, trains both models (LightGBM + LEAR), and uploads model files to S3. It takes 15–30 minutes on first run.

---

### 2.11 Set Up AWS SSM for Daily Automated Retraining

SSM State Manager sends the `run_pipeline.sh` command to the server every day without requiring SSH access.

**Install the SSM Agent:**

[Server SSH terminal]
```bash
sudo snap install amazon-ssm-agent --classic
sudo systemctl enable amazon-ssm-agent
sudo systemctl start amazon-ssm-agent
sudo systemctl status amazon-ssm-agent
# Should show: active (running)
```

The agent uses the AWS credentials in `/home/ubuntu/.aws/credentials` to register with AWS SSM. After ~30 seconds, verify the instance appears in the console:

In the AWS Console → **Systems Manager** → **Fleet Manager** → **Managed nodes**. Your server should appear with status "Online".

**Create the State Manager Association:**

1. In the AWS Console → **Systems Manager** → **State Manager** → **Create association**
2. **Name:** `se3-daily-pipeline`
3. **Document:** search for and select `AWS-RunShellScript`
4. **Parameters → Commands:**
   ```
   /home/ubuntu/SE3_prod/run_pipeline.sh
   ```
5. **Targets:** Select **Choose instances manually** → select your server
6. **Specify schedule:**
   - Select **CRON/Rate expression**
   - Enter: `cron(0 3 * * ? *)`
   - This runs at 03:00 UTC daily (05:00 Stockholm summer time / 04:00 winter)
   - Timing note: ENTSO-E publishes D-1 final prices by ~12:00 CET, so 03:00 UTC the next day is well after data is available
7. Click **Create association**

To test the association immediately (without waiting for the scheduled time):
[Mac terminal]
```bash
# Get your association ID from the console, then:
aws ssm start-associations-once \
  --association-ids YOUR-ASSOCIATION-ID \
  --region eu-north-1
```

---

### 2.12 Set Up CloudWatch Monitoring

#### Install the CloudWatch Agent

[Server SSH terminal]
```bash
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb
```

**Create the agent configuration:**

[Server SSH terminal]
```bash
sudo mkdir -p /opt/aws/amazon-cloudwatch-agent/etc

sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json > /dev/null << 'EOF'
{
  "metrics": {
    "namespace": "SE3/Server",
    "metrics_collected": {
      "mem": {
        "measurement": ["mem_used_percent"],
        "metrics_collection_interval": 60
      },
      "disk": {
        "measurement": ["disk_used_percent"],
        "resources": ["/"],
        "metrics_collection_interval": 60
      }
    }
  }
}
EOF
```

The CloudWatch Agent runs as root — give it access to the AWS credentials:

```bash
sudo mkdir -p /root/.aws
sudo cp /home/ubuntu/.aws/credentials /root/.aws/credentials
sudo cp /home/ubuntu/.aws/config /root/.aws/config
```

**Start the agent:**

```bash
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m onPremise \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
  -s

sudo systemctl enable amazon-cloudwatch-agent
sudo systemctl status amazon-cloudwatch-agent
# Should show: active (running)
```

After ~2 minutes, verify metrics appear in the console:
AWS Console → **CloudWatch** → **Metrics** → **All metrics** → **Custom namespaces** → **SE3/Server** → you should see `mem_used_percent` and `disk_used_percent`.

#### Create an SNS Topic for Email Alerts

1. AWS Console → **SNS** → **Topics** → **Create topic**
2. **Type:** Standard
3. **Name:** `se3-alerts`
4. Click **Create topic**
5. Click into the topic → **Create subscription**
6. **Protocol:** Email
7. **Endpoint:** your email address
8. Click **Create subscription**
9. Check your email for a message from AWS with subject "AWS Notification - Subscription Confirmation"
10. Click **Confirm subscription** in that email
11. Status in the console changes from "PendingConfirmation" to "Confirmed"
12. **Copy the topic ARN** — you need it in the next step. Format: `arn:aws:sns:eu-north-1:ACCOUNT_ID:se3-alerts`

#### Create CloudWatch Alarms

Create three alarms. For each: AWS Console → **CloudWatch** → **Alarms** → **Create alarm**

**Alarm 1 — High Memory:**
1. Click **Select metric** → **Custom namespaces** → **SE3/Server** → find `mem_used_percent` → **Select metric**
2. **Period:** 1 minute
3. **Statistic:** Average
4. **Threshold:** Greater than **80**
5. Click **Next**
6. **Send a notification to:** select your SNS topic `se3-alerts`
7. Click **Next** → **Alarm name:** `se3-high-memory` → **Create alarm**

**Alarm 2 — High Disk:**
Same steps, but select `disk_used_percent` → threshold Greater than **80** → alarm name `se3-high-disk`

**Alarm 3 — Pipeline Failure:**
1. Click **Select metric** → **Custom namespaces** → **SE3/Pipeline** → `PipelineFailure`
   > Note: this metric only appears after `run_pipeline.sh` has been run at least once. Run it manually first (step 2.10 above) if needed.
2. **Period:** 1 day (86400 seconds)
3. **Statistic:** Maximum
4. **Threshold:** Greater than or equal to **1**
5. **Missing data treatment:** Treat missing data as **OK** (the pipeline runs once a day, so missing data between runs is expected)
6. **Next** → select SNS topic → **Alarm name:** `se3-pipeline-failure` → **Create alarm**

---

## Part 3 — Cloudflare Tunnel (Public HTTPS Access)

### 3.1 Create a Cloudflare Account and Add Your Domain

1. Go to **https://dash.cloudflare.com** → **Sign up**
2. Enter email and password → click **Create Account**
3. Click **Add a site** → enter your domain name → click **Add site**
4. Select the **Free** plan → click **Continue**
5. Cloudflare will scan your existing DNS records. Click **Continue**
6. Cloudflare displays its nameservers (e.g. `ada.ns.cloudflare.com`, `bob.ns.cloudflare.com`)
7. Log into your domain registrar → find DNS/Nameserver settings → replace the existing nameservers with Cloudflare's two nameservers
8. Save in your registrar. DNS propagation takes 5–60 minutes.
9. Back in Cloudflare, click **Check nameservers**. Once active, the site shows "Active" with a green checkmark.

### 3.2 Create the Tunnel

1. In Cloudflare dashboard → **Zero Trust** (left sidebar, may require enabling)
2. If prompted to set up Zero Trust: enter a team name (e.g. `your-name-team`) → choose Free plan → complete setup
3. Go to **Networks** → **Tunnels** → **Create a tunnel**
4. **Connector type:** Cloudflared → click **Next**
5. **Tunnel name:** `se3-tunnel` → click **Save tunnel**
6. Cloudflare displays installation commands. Copy the command — it includes a long token unique to your tunnel.

### 3.3 Install cloudflared on the Server

[Server SSH terminal]
```bash
curl -L --output cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
rm cloudflared.deb

# Install and start the tunnel service using the token from the Cloudflare dashboard
# Replace <TOKEN> with the full token string Cloudflare provided
sudo cloudflared service install <TOKEN>

sudo systemctl enable cloudflared
sudo systemctl start cloudflared
sudo systemctl status cloudflared
# Should show: active (running)
```

### 3.4 Configure the Public Hostname

Back in the Cloudflare dashboard (the browser tab from step 3.2):

1. Click **Next** to proceed past the connector installation screen
2. On the **Route tunnel** page:
   - **Subdomain:** `se3`
   - **Domain:** your domain (e.g. `your-domain.com`)
   - **Type:** HTTP
   - **URL:** `localhost:8501`
3. Click **Save tunnel**

Your tunnel config on the server is at `/etc/cloudflared/config.yml`:
```yaml
tunnel: <YOUR-TUNNEL-UUID>
credentials-file: /etc/cloudflared/<YOUR-TUNNEL-UUID>.json
ingress:
  - hostname: se3.your-domain.com
    service: http://localhost:8501
  - service: http_status:404
```

### 3.5 Verify the Dashboard is Live

Open `https://se3.your-domain.com` in a browser. You should see the Streamlit dashboard served over HTTPS with a valid Cloudflare certificate.

If the page doesn't load:
```bash
# Check cloudflared logs
sudo journalctl -u cloudflared -f

# Check that the Streamlit container is running
docker compose -f /home/ubuntu/SE3_prod/docker-compose.yml ps
```

---

## Part 4 — Verification Checklist

Run through every item to confirm the deployment is complete.

### 4.1 Containers

[Server SSH terminal]
```bash
docker compose -f /home/ubuntu/SE3_prod/docker-compose.yml ps
```
Expected: both `se3_clickhouse` and `se3_app` show `running` or `healthy`.

### 4.2 Pipeline

[Server SSH terminal]
```bash
/home/ubuntu/SE3_prod/run_pipeline.sh
```
Expected: no errors. The script should print fetch counts, training metrics, and end with `[OK] All models trained`.

### 4.3 S3 Upload

[Mac terminal]
```bash
aws s3 ls s3://YOUR-BUCKET-NAME/model/ --region eu-north-1
```
Expected: a list of `.pkl` and `.json` files including `lgbm_q50.pkl`, `lear_h00.pkl`, `metrics.json`, `trained_at.json`.

### 4.4 CloudWatch Metric

[Mac terminal]
```bash
aws cloudwatch get-metric-statistics \
  --namespace SE3/Pipeline \
  --metric-name PipelineFailure \
  --start-time $(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 3600 \
  --statistics Sum \
  --region eu-north-1
```
Expected: one datapoint with `Sum: 0.0` (success). A value of `1.0` means the pipeline failed — check the run_pipeline.sh logs.

> On Linux (not macOS): replace `date -u -v-1H` with `date -u -d '1 hour ago'`

### 4.5 Dashboard

Open `https://se3.your-domain.com` in a browser. The Forecast and Model Performance tabs should show data.

### 4.6 SSM Scheduled Run

To trigger a manual test run of the SSM association:
[Mac terminal]
```bash
aws ssm start-associations-once \
  --association-ids YOUR-ASSOCIATION-ID \
  --region eu-north-1
```
Check the result in AWS Console → Systems Manager → State Manager → click your association → **Execution history** tab.

---

## Part 5 — Environment Variables Reference

Copy `.env.example` to `.env` on the server and populate all values.

| Variable | Required | Description | Example |
|---|---|---|---|
| `ENTSOE_API_KEY` | Yes | API key from transparency.entsoe.eu | `abc123def456...` |
| `TIMEDB_CH_URL` | Yes | ClickHouse connection string (Docker internal hostname) | `http://se3user:se3password@clickhouse:8123/se3db` |
| `SE3_LAT` | No | Latitude for weather fetching | `59.33` (Stockholm) |
| `SE3_LON` | No | Longitude for weather fetching | `18.07` (Stockholm) |
| `SE3_TRAIN_START` | No | Earliest date for historical data fetch | `2020-01-01` |
| `MODEL_DIR` | No | Directory for model artifacts | `model` |
| `S3_BUCKET` | No | S3 bucket name for model persistence | `se3-model-artifacts-yourname` |
| `WEATHER_CACHE_PATH` | No | SQLite cache path for Open-Meteo | `/app/data/.weather_cache.sqlite` |

> If `S3_BUCKET` is not set, model files are only saved locally inside `model/` (no S3 upload). The pipeline still works but models won't survive container removal.

---

## Part 6 — Architecture Reference

```
Internet User
     │ HTTPS
     ▼
Cloudflare Edge (DDoS protection, TLS termination, CDN)
     │
     ▼ (outbound-only encrypted tunnel)
cloudflared daemon (running on server)
     │
     ▼
localhost:8501
     │
     ▼
Docker: se3_app container (Python 3.12-slim)
  └── Streamlit dashboard (dashboard/app.py)
  └── Pipeline modules (pipeline/, ml/)
     │
     ▼
Docker: se3_clickhouse container (ClickHouse 24.3)
  └── Named volume: clickhouse_data (persistent)
  └── Named volume: weather_cache (persistent)

AWS SSM State Manager
  cron(0 3 * * ? *)  →  run_pipeline.sh  →  docker compose exec app
                                          →  fetch_prices + fetch_weather + train
                                          →  aws cloudwatch put-metric-data

AWS S3 (se3-model-artifacts-*)
  ← upload: ml/train.py via boto3 after training
  → download: entrypoint.sh at container startup (if model/ is empty)

AWS CloudWatch
  ← SE3/Server metrics (mem, disk) from CloudWatch Agent every 60s
  ← SE3/Pipeline PipelineFailure from run_pipeline.sh daily
  → Alarms: se3-high-memory, se3-high-disk, se3-pipeline-failure

AWS SNS (se3-alerts)
  ← CloudWatch alarms
  → Email to you
```

---

## Part 7 — Common Operations

### Redeploy code changes

[Mac terminal]
```bash
rsync -av --exclude '.env' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.git' --exclude 'model/*.pkl' --exclude 'model/*.json' \
  /path/to/SE3_prod/ ubuntu@YOUR_SERVER_IP:/home/ubuntu/SE3_prod/
```

[Server SSH terminal]
```bash
cd /home/ubuntu/SE3_prod
docker compose up -d --build
```

### View live logs

[Server SSH terminal]
```bash
docker compose logs -f app          # Streamlit + pipeline output
docker compose logs -f clickhouse   # Database logs
sudo journalctl -u cloudflared -f   # Tunnel logs
```

### Manual pipeline run

[Server SSH terminal]
```bash
/home/ubuntu/SE3_prod/run_pipeline.sh
```

### Force retrain (skip the 7-day freshness cache)

[Server SSH terminal]
```bash
cd /home/ubuntu/SE3_prod
docker compose exec -T app python -m ml.train --force
```

### Restart everything after a server reboot

Docker containers are set to `restart: unless-stopped` — they start automatically on server reboot. The SSM agent and Cloudflare Tunnel are `systemctl enable`d and start on boot. No manual action needed after a server reboot.

### View model performance

[Server SSH terminal]
```bash
cat /home/ubuntu/SE3_prod/model/metrics.json
cat /home/ubuntu/SE3_prod/model/MODEL_LOG.md
```

### Check S3 model artifacts

[Mac terminal]
```bash
aws s3 ls s3://YOUR-BUCKET-NAME/model/ --region eu-north-1 --human-readable
```

---

## Part 8 — Cost Estimate

| Service | Plan | Monthly cost |
|---|---|---|
| AWS Lightsail | 2 GB / 1 vCPU | ~$10 |
| AWS S3 | <1 GB model files | ~$0.02 |
| AWS CloudWatch | Custom metrics + 3 alarms | ~$1 |
| AWS SNS | <10 emails/month | Free tier |
| AWS SSM | State Manager (self-managed) | Free |
| Cloudflare | Free plan | $0 |
| **Total** | | **~$11/month** |

---

## Part 9 — Security Notes

- The `.env` file is excluded from git (`.gitignore`) and from rsync (`--exclude '.env'`). Never commit it.
- The AWS credentials in `/home/ubuntu/.aws/credentials` must never appear in code or logs. The `ENTSOE_API_KEY` has the same requirement.
- The IAM user `se3-pipeline` has the minimum permissions needed: S3 read/write on one specific bucket, and `cloudwatch:PutMetricData`. It cannot delete EC2 instances, access other buckets, or perform any other AWS action.
- The Cloudflare Tunnel creates zero open inbound ports. The Lightsail firewall should have only port 22 (SSH) open.
- ClickHouse listens on `0.0.0.0:8123` inside Docker. For a tighter deployment, bind it to `127.0.0.1:8123:8123` in `docker-compose.yml` so it is only reachable from within the server.
- Rotate the IAM access keys every 90 days. To rotate: IAM → Users → se3-pipeline → Security credentials → Create access key → update `/home/ubuntu/.aws/credentials` → deactivate old key → delete old key.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker compose up` OOM-kills | Not enough RAM | Use the 2 GB Lightsail plan; confirm `clickhouse-config.xml` is present |
| `ENTSOE_API_KEY` error on pipeline run | `.env` missing or wrong key | Check `/home/ubuntu/SE3_prod/.env` |
| `aws: command not found` in run_pipeline.sh | AWS CLI not in PATH when run by SSM | Use `/usr/local/bin/aws` explicitly in the script |
| `AccessDenied` on `cloudwatch:PutMetricData` | IAM policy missing | Add `cloudwatch:PutMetricData` to the IAM user policy (see step 2.3) |
| CloudWatch metric `PipelineFailure` missing | Pipeline never run, or metric namespace not yet created | Run `run_pipeline.sh` manually once |
| Cloudflare Tunnel shows disconnected | `cloudflared` service stopped | `sudo systemctl restart cloudflared` |
| Dashboard loads but shows no data | Pipeline never run / ClickHouse empty | Run the pipeline manually |
| SSM shows managed node offline | SSM agent stopped or credentials wrong | `sudo systemctl restart amazon-ssm-agent`; check `aws sts get-caller-identity` |

---

*Guide accurate as of June 2026. SE3 EIC code: `10Y1001A1001A46L`. Open-Meteo archive endpoint: `https://archive-api.open-meteo.com/v1/archive`.*
