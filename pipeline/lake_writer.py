"""
Bronze layer writer.

Writes a Pandas DataFrame as a Parquet file to GCS (production)
or local disk (development, when GCS_BUCKET_NAME is not set).

Path convention:
    bronze/{data_type}/{zone}/{year}/{month}/{data_type}_{zone}_{date}.parquet

Usage:
    from pipeline.lake_writer import LakeWriter
    writer = LakeWriter()
    writer.write(df, data_type="prices", zone="SE3", date=date(2026, 6, 25))
"""

import os
from datetime import date
from pathlib import Path

import pandas as pd


class LakeWriter:
    """Writes DataFrames to the Bronze data lake (GCS or local)."""

    def __init__(self) -> None:
        self.bucket = os.environ.get("GCS_BUCKET_NAME")
        self.local_root = Path("data/bronze")
        self._use_gcs = bool(self.bucket)

    def write(self, df: pd.DataFrame, data_type: str, zone: str, date: date) -> str:
        """
        Write df to Bronze layer. Returns the path written.
        Idempotent: overwrites if file already exists.
        """
        path = self._build_path(data_type, zone, date)
        if self._use_gcs:
            return self._write_gcs(df, path)
        return self._write_local(df, path)

    def _build_path(self, data_type: str, zone: str, date: date) -> str:
        filename = f"{data_type}_{zone}_{date.isoformat()}.parquet"
        return f"bronze/{data_type}/{zone}/{date.year}/{date.month:02d}/{filename}"

    def _write_local(self, df: pd.DataFrame, path: str) -> str:
        full_path = self.local_root.parent / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(full_path, index=False)
        return str(full_path)

    def _write_gcs(self, df: pd.DataFrame, path: str) -> str:
        from google.cloud import storage  # lazy import — only needed in production

        client = storage.Client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(path)
        blob.upload_from_string(
            df.to_parquet(index=False), content_type="application/octet-stream"
        )
        return f"gs://{self.bucket}/{path}"
