"""
Bronze layer reader.

Reads one or more Parquet files from GCS (or local disk) into a DataFrame.

Usage:
    from pipeline.lake_reader import LakeReader
    from datetime import date

    reader = LakeReader()
    df = reader.read_range(
        data_type="prices",
        zone="SE3",
        start=date(2026, 6, 1),
        end=date(2026, 6, 25),
    )
"""

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


class LakeReader:
    """Reads DataFrames from the Bronze data lake (GCS or local)."""

    def __init__(self) -> None:
        self.bucket = os.environ.get("GCS_BUCKET_NAME")
        self.local_root = Path("data")
        self._use_gcs = bool(self.bucket)

    def read_range(
        self, data_type: str, zone: str, start: date, end: date
    ) -> pd.DataFrame:
        """Read all Parquet files for a date range. Returns concatenated DataFrame."""
        frames = []
        current = start
        while current <= end:
            try:
                df = self._read_one(data_type, zone, current)
                frames.append(df)
            except FileNotFoundError:
                pass  # missing dates are skipped silently
            current += timedelta(days=1)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _build_path(self, data_type: str, zone: str, date: date) -> str:
        filename = f"{data_type}_{zone}_{date.isoformat()}.parquet"
        return f"bronze/{data_type}/{zone}/{date.year}/{date.month:02d}/{filename}"

    def _read_one(self, data_type: str, zone: str, date: date) -> pd.DataFrame:
        path = self._build_path(data_type, zone, date)
        if self._use_gcs:
            return self._read_gcs(path)
        return self._read_local(path)

    def _read_local(self, path: str) -> pd.DataFrame:
        full_path = self.local_root / path
        if not full_path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(full_path)

    def _read_gcs(self, path: str) -> pd.DataFrame:
        import io

        from google.cloud import storage  # lazy import

        client = storage.Client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(path)
        if not blob.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
