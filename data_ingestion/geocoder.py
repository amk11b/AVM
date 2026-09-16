import io
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from google.cloud import storage


KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/Users/AVM/AVM/gcp_key.json")
BUCKET_NAME = os.getenv("AVM_BUCKET_NAME", "avm_data_ml")
SOURCE_FILE = os.getenv("AVM_GEO_SOURCE", "/Users/AVM/AVM/modeling/combined_listings.csv")
OUTPUT_FILE = os.getenv("AVM_GEO_OUTPUT", "/Users/AVM/AVM/modeling/listing_geocodes.csv")
SOURCE_POSTCODE_COLUMN = os.getenv("AVM_POSTCODE_COLUMN", "")


if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH


POSTCODE_RE = re.compile(r"([A-Za-z]{1,2}\d{1,2}\s?\d[A-Za-z]{2})")


def _parse_gcs_uri(uri):
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    bucket_name, _, blob_name = uri[5:].partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return bucket_name, blob_name


def _read_source_frame():
    source = str(SOURCE_FILE)
    if source.startswith("gs://"):
        bucket_name, blob_name = _parse_gcs_uri(source)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_bytes()
        return pd.read_csv(io.BytesIO(content), low_memory=False)
    return pd.read_csv(Path(source), low_memory=False)


def _extract_postcodes(df):
    if SOURCE_POSTCODE_COLUMN and SOURCE_POSTCODE_COLUMN in df.columns:
        series = df[SOURCE_POSTCODE_COLUMN].astype(str)
        cleaned = (
            series.fillna("")
            .astype(str)
            .str.upper()
            .str.replace(" ", "", regex=False)
            .str.strip()
        )
        cleaned = cleaned[cleaned.str.len() > 0]
        return cleaned.drop_duplicates().tolist()

    candidates = []
    for col in ["postcode", "address", "description", "city"]:
        if col in df.columns:
            candidates.append(df[col].fillna("").astype(str))

    if not candidates:
        raise KeyError("Need a postcode-like column to geocode.")

    joined = candidates[0]
    for series in candidates[1:]:
        joined = joined + " " + series

    extracted = joined.str.upper().str.extract(POSTCODE_RE, expand=False)
    cleaned = (
        extracted.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned[cleaned.str.len() > 0]
    return cleaned.drop_duplicates().tolist()


def bulk_geocode(postcodes):
    endpoint = "https://api.postcodes.io/postcodes"
    results = []
    batch_size = 100
    total_batches = (len(postcodes) + batch_size - 1) // batch_size

    print(f"🌍 Geocoding {len(postcodes)} postcodes in {total_batches} batches...")

    for i in range(0, len(postcodes), batch_size):
        batch = postcodes[i : i + batch_size]
        try:
            response = requests.post(endpoint, json={"postcodes": batch}, timeout=30)
            response.raise_for_status()
            data = response.json().get("result", [])
            for item in data:
                res = item.get("result")
                if res:
                    results.append(
                        {
                            "postcode": item["query"],
                            "lat": res.get("latitude"),
                            "lon": res.get("longitude"),
                            "district": res.get("admin_district"),
                        }
                    )
        except Exception as exc:
            print(f"❌ Batch error: {exc}")

        time.sleep(0.1)
        if i and i % 1000 == 0:
            print(f"   Processed {i}...")

    return pd.DataFrame(results)


def upload_output(df):
    source = str(OUTPUT_FILE)
    if source.startswith("gs://"):
        bucket_name, blob_name = _parse_gcs_uri(source)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        bucket.blob(blob_name).upload_from_string(df.to_csv(index=False), content_type="text/csv")
        print(f"✅ Uploaded geocodes to {source}")
        return

    output_path = Path(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved geocodes to {output_path}")


def main():
    print(f"📥 Loading source listings from {SOURCE_FILE}...")
    df = _read_source_frame()
    postcodes = _extract_postcodes(df)
    print(f"✅ Found {len(postcodes):,} unique postcodes.")

    geocoded_df = bulk_geocode(postcodes)
    if geocoded_df.empty:
        print("⚠️  No geocodes returned.")
        return

    upload_output(geocoded_df)


if __name__ == "__main__":
    main()
