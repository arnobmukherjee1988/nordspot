#!/bin/bash
# Entrypoint: download model artifacts from S3 if model/ is empty, then start app.
set -e

MODEL_DIR="/app/model"

if [ -n "$S3_BUCKET" ] && [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "[entrypoint] model/ is empty — downloading artifacts from s3://$S3_BUCKET/model/ ..."
    python3 - << 'PYEOF'
import boto3, os
from pathlib import Path

bucket = os.environ["S3_BUCKET"]
s3 = boto3.client("s3")
paginator = s3.get_paginator("list_objects_v2")
count = 0
for page in paginator.paginate(Bucket=bucket, Prefix="model/"):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        local = Path("/app") / key
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local))
        count += 1
print(f"[entrypoint] Downloaded {count} files from S3.")
PYEOF
else
    echo "[entrypoint] model/ already populated or S3_BUCKET not set — skipping download."
fi

exec "$@"
