"""Unit tests for Bronze layer writer."""

from datetime import date

import pandas as pd
import pytest

from pipeline.lake_writer import LakeWriter


@pytest.fixture
def writer(tmp_path, monkeypatch):
    """LakeWriter pointed at a temporary directory, no GCS."""
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    w = LakeWriter()
    w.local_root = tmp_path / "bronze"
    return w


@pytest.fixture
def sample_df():
    return pd.DataFrame({"hour": range(24), "price_eur_mwh": [50.0] * 24})


def test_write_creates_parquet(writer, sample_df):
    path = writer.write(
        sample_df, data_type="prices", zone="SE3", date=date(2026, 6, 25)
    )
    assert path.endswith(".parquet")
    result = pd.read_parquet(path)
    assert list(result.columns) == ["hour", "price_eur_mwh"]
    assert len(result) == 24


def test_write_is_idempotent(writer, sample_df):
    """Writing twice should produce one file, not two - 24 rows, not 48."""
    writer.write(sample_df, data_type="prices", zone="SE3", date=date(2026, 6, 25))
    writer.write(sample_df, data_type="prices", zone="SE3", date=date(2026, 6, 25))
    path = writer._build_path("prices", "SE3", date(2026, 6, 25))
    full_path = writer.local_root.parent / path
    result = pd.read_parquet(full_path)
    assert len(result) == 24


def test_path_convention(writer):
    path = writer._build_path("prices", "SE3", date(2026, 6, 25))
    assert path == "bronze/prices/SE3/2026/06/prices_SE3_2026-06-25.parquet"
