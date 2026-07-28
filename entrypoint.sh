#!/bin/bash
# Entrypoint: download model artifacts from GCS if model/ is empty, then start app.
set -e

MODEL_DIR="/app/model"

if [ -n "$GCS_BUCKET" ] && [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "[entrypoint] model/ is empty - downloading artifacts from gs://$GCS_BUCKET/model/ ..."
    python3 - << 'PYEOF'
import os
from pathlib import Path
from google.cloud import storage

bucket_name = os.environ["GCS_BUCKET"]
client = storage.Client()
bucket = client.bucket(bucket_name)

blobs = list(bucket.list_blobs(prefix="model/"))
count = 0
for blob in blobs:
    if blob.name.endswith("/"):
        continue
    local = Path("/app") / blob.name
    local.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local))
    count += 1

print(f"[entrypoint] Downloaded {count} files from gs://{bucket_name}/model/")
PYEOF
else
    echo "[entrypoint] model/ already populated or GCS_BUCKET not set - skipping download."
fi

exec "$@"
