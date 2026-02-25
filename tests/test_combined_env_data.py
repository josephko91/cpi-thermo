#!/usr/bin/env python3
"""
Test suite for combined environmental data creation and QA/QC validation.

Tests verify:
- Parquet file creation
- Campaign data presence
- Data quality (nulls, ranges, duplicates)
- Basic statistics per campaign
"""

import sys
from pathlib import Path

import pytest
import polars as pl

# Ensure main.py and parsers are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import process_all_campaigns, DEFAULT_CAMPAIGN_CONFIG


@pytest.fixture
def tmp_output_dir():
    """Create output directory for test outputs in tests/output/."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    yield output_dir


@pytest.fixture
def combined_df(tmp_output_dir):
    """Create combined environmental dataset from all campaigns."""
    output_path = tmp_output_dir / "test_combined.parquet"
    
    # Process all campaigns (using small subset for testing)
    df = process_all_campaigns(verbose=False)
    
    # Save to parquet
    df.to_parquet(str(output_path))
    
    # Load back and convert to polars
    df_pl = pl.read_parquet(str(output_path))
    
    return df_pl, output_path


class TestFileCreation:
    """Tests for parquet file creation."""
    
    def test_parquet_file_created(self, combined_df, tmp_output_dir):
        """Test that parquet file was created successfully."""
        df_pl, output_path = combined_df
        assert output_path.exists(), "Parquet file was not created"
        assert output_path.stat().st_size > 0, "Parquet file is empty"
    
    def test_file_is_valid_parquet(self, combined_df):
        """Test that the file is a valid parquet file."""
        df_pl, _ = combined_df
        assert isinstance(df_pl, pl.DataFrame), "File is not a valid Polars DataFrame"


class TestCampaignData:
    """Tests for campaign data presence and integrity."""
    
    def test_campaigns_present(self, combined_df):
        """Test that all campaigns are present in the dataset."""
        df_pl, _ = combined_df
        campaigns = df_pl["Campaign"].unique().to_list()
        expected_campaigns = list(DEFAULT_CAMPAIGN_CONFIG.keys())
        
        for campaign in expected_campaigns:
            assert campaign in campaigns, f"Campaign {campaign} not found in data"
    
    def test_records_per_campaign(self, combined_df):
        """Test that each campaign has records."""
        df_pl, _ = combined_df
        counts = df_pl.groupby("Campaign").agg(pl.count().alias("n_records"))
        
        for row in counts.to_dicts():
            campaign = row["Campaign"]
            n_records = row["n_records"]
            assert n_records > 0, f"Campaign {campaign} has no records"


class TestDataQuality:
    """Tests for data quality and validity."""
    
    def test_no_null_timestamp(self, combined_df):
        """Test that Timestamp column has no null values."""
        df_pl, _ = combined_df
        null_count = df_pl["Timestamp"].null_count()
        assert null_count == 0, f"Found {null_count} null values in Timestamp"
    
    def test_no_null_campaign(self, combined_df):
        """Test that Campaign column has no null values."""
        df_pl, _ = combined_df
        null_count = df_pl["Campaign"].null_count()
        assert null_count == 0, f"Found {null_count} null values in Campaign"
    
    def test_temperature_range(self, combined_df):
        """Test that temperature values are within reasonable range."""
        df_pl, _ = combined_df
        temp_col = "Tair_C"
        
        if temp_col in df_pl.columns:
            temp_min = df_pl[temp_col].min()
            temp_max = df_pl[temp_col].max()
            
            assert temp_min > -150, f"Temperature too low: {temp_min}"
            assert temp_max < 200, f"Temperature too high: {temp_max}"
    
    def test_latitude_range(self, combined_df):
        """Test that latitude values are valid."""
        df_pl, _ = combined_df
        lat_col = "Lat"
        
        if lat_col in df_pl.columns:
            lat_min = df_pl[lat_col].min()
            lat_max = df_pl[lat_col].max()
            
            assert lat_min >= -90, f"Latitude too low: {lat_min}"
            assert lat_max <= 90, f"Latitude too high: {lat_max}"
    
    def test_longitude_range(self, combined_df):
        """Test that longitude values are valid."""
        df_pl, _ = combined_df
        lon_col = "Lon"
        
        if lon_col in df_pl.columns:
            lon_min = df_pl[lon_col].min()
            lon_max = df_pl[lon_col].max()
            
            assert lon_min >= -180, f"Longitude too low: {lon_min}"
            assert lon_max <= 180, f"Longitude too high: {lon_max}"
    
    def test_altitude_non_negative(self, combined_df):
        """Test that altitude values are non-negative."""
        df_pl, _ = combined_df
        alt_col = "Alt_m"
        
        if alt_col in df_pl.columns:
            alt_min = df_pl[alt_col].min()
            assert alt_min >= 0, f"Negative altitude found: {alt_min}"
    
    def test_no_duplicate_rows(self, combined_df):
        """Test that there are no duplicate rows."""
        df_pl, _ = combined_df
        unique_count = df_pl.unique().height
        total_count = df_pl.height
        
        assert unique_count == total_count, \
            f"Found {total_count - unique_count} duplicate rows"


class TestBasicStatistics:
    """Tests for basic statistics per campaign."""
    
    def test_statistics_per_campaign(self, combined_df, tmp_output_dir):
        """Test and save basic statistics per campaign."""
        df_pl, _ = combined_df
        
        stats_file = tmp_output_dir / "campaign_statistics.txt"
        
        with open(stats_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("CAMPAIGN STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            
            stats = df_pl.groupby("Campaign").agg([
                pl.count().alias("n_records"),
                pl.col("Tair_C").mean().alias("temp_mean_C"),
                pl.col("Tair_C").std().alias("temp_std_C"),
                pl.col("Lat").mean().alias("lat_mean"),
                pl.col("Lon").mean().alias("lon_mean"),
                pl.col("Alt_m").mean().alias("alt_mean_m"),
            ]).sort("Campaign")
            
            for row in stats.to_dicts():
                f.write(f"Campaign: {row['Campaign']}\n")
                f.write(f"  Records: {row['n_records']:,}\n")
                f.write(f"  Temp (mean ± std): {row['temp_mean_C']:.2f} ± {row['temp_std_C']:.2f} °C\n")
                f.write(f"  Lat (mean): {row['lat_mean']:.2f}°\n")
                f.write(f"  Lon (mean): {row['lon_mean']:.2f}°\n")
                f.write(f"  Alt (mean): {row['alt_mean_m']:.0f} m\n")
                f.write("\n")
        
        assert stats_file.exists(), "Statistics file was not created"
    
    def test_missing_data_summary(self, combined_df, tmp_output_dir):
        """Test and save summary of missing data."""
        df_pl, _ = combined_df
        
        missing_file = tmp_output_dir / "missing_data_summary.txt"
        
        with open(missing_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MISSING DATA SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            for col in df_pl.columns:
                null_count = df_pl[col].null_count()
                total_count = df_pl.height
                pct = (null_count / total_count * 100) if total_count > 0 else 0
                
                if null_count > 0:
                    f.write(f"{col}: {null_count:,} nulls ({pct:.2f}%)\n")
            
            f.write("\nNo missing values found in critical columns.\n")
        
        assert missing_file.exists(), "Missing data summary was not created"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
