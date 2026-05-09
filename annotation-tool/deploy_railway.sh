#!/usr/bin/env bash
#
# Deploy annotation-tool to Railway with persistent SQLite volume.
#
# Prerequisites:
#   - Railway CLI installed: npm install -g @railway/cli
#   - Logged in: railway login
#   - Git repo pushed to GitHub (ghanahmada/ta-statute-law-retrieval)
#
# This script:
#   1. Creates a Railway project (or links existing)
#   2. Configures Dockerfile builder with root directory
#   3. Attaches a persistent volume at /app/db for SQLite
#   4. Sets environment variables
#   5. Deploys
#   6. Runs generate_tokens.py inside the container
#
# Usage:
#   cd annotation-tool
#   bash deploy_railway.sh
#
set -euo pipefail

PROJECT_NAME="annotation-tool"
SERVICE_NAME="annotation-tool"
VOLUME_MOUNT="/app/db"
VOLUME_SIZE="1"  # 1 GB, more than enough for SQLite

echo "=== Railway Annotation Tool Deployment ==="
echo ""

# ── Step 0: Check Railway CLI ──────────────────────────────────────
if ! command -v railway &> /dev/null; then
    echo "ERROR: Railway CLI not found. Install with:"
    echo "  npm install -g @railway/cli"
    exit 1
fi

# Check login
if ! railway whoami &> /dev/null 2>&1; then
    echo "Not logged in to Railway. Running 'railway login'..."
    railway login --browserless
fi

echo "Logged in as: $(railway whoami 2>/dev/null || echo 'unknown')"
echo ""

# ── Step 1: Create or link project ────────────────────────────────
echo "=== Step 1: Project setup ==="

# Check if already linked
if railway status &> /dev/null 2>&1; then
    echo "Already linked to a Railway project."
    railway status
else
    echo "Creating new Railway project: ${PROJECT_NAME}"
    railway init --name "${PROJECT_NAME}"
fi
echo ""

# ── Step 2: Link GitHub repo ─────────────────────────────────────
echo "=== Step 2: Connect GitHub repo ==="
echo "MANUAL STEP REQUIRED:"
echo "  1. Go to your Railway dashboard: https://railway.app/dashboard"
echo "  2. Open project '${PROJECT_NAME}'"
echo "  3. Click the service → Settings → Source"
echo "  4. Connect GitHub repo: ghanahmada/ta-statute-law-retrieval"
echo "  5. Set Root Directory to: annotation-tool"
echo "  6. Set Builder to: Dockerfile"
echo ""
read -p "Press Enter when done (or Ctrl+C to abort)..."
echo ""

# ── Step 3: Set environment variables ─────────────────────────────
echo "=== Step 3: Setting environment variables ==="
railway variables set \
    PAIRS_TSV=/app/data/pairs.tsv \
    ANNOTATORS_TXT=/app/data/annotators.txt \
    DATABASE_URL=sqlite:////app/db/labels.sqlite

echo "Environment variables set."
echo ""

# ── Step 4: Add persistent volume ─────────────────────────────────
echo "=== Step 4: Persistent volume ==="
echo "MANUAL STEP REQUIRED:"
echo "  1. In Railway dashboard → your service → Settings → Volumes"
echo "  2. Click 'Add Volume'"
echo "  3. Set mount path to: ${VOLUME_MOUNT}"
echo "  4. Size: ${VOLUME_SIZE} GB is plenty"
echo ""
echo "  This ensures labels.sqlite survives redeploys."
echo ""
read -p "Press Enter when the volume is attached..."
echo ""

# ── Step 5: Deploy ────────────────────────────────────────────────
echo "=== Step 5: Deploying ==="
railway up --detach

echo ""
echo "Deployment triggered. Waiting for it to go live..."
echo "Check status at: https://railway.app/dashboard"
echo ""
read -p "Press Enter once the deployment is live and healthy..."
echo ""

# ── Step 6: Generate access tokens ────────────────────────────────
echo "=== Step 6: Generating access tokens ==="
echo "Running generate_tokens.py inside the deployed container..."
echo ""
railway run python generate_tokens.py

echo ""
echo "=== Deployment complete! ==="
echo ""
echo "IMPORTANT: Save the tokens above and share them with your annotators."
echo "Each annotator pastes their token on the login page to start annotating."
echo ""
railway domain 2>/dev/null && echo "" || echo "To get your public URL: railway domain"
echo ""
echo "Notes:"
echo "  - SQLite data persists in the /app/db volume across redeploys"
echo "  - To regenerate tokens: railway run python generate_tokens.py"
echo "  - To check progress:    railway run python -c \"from app.database import SessionLocal; from app.models import Label; db=SessionLocal(); print(f'Labels: {db.query(Label).count()}')\""
echo "  - To download the DB:   railway run cat /app/db/labels.sqlite > labels_backup.sqlite"
