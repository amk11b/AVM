import os
from io import StringIO
import hashlib
from pathlib import Path

import pandas as pd
from google.cloud import storage

KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
BUCKET_NAME = "avm_data_ml"
GCS_PREFIX = os.getenv("AVM_GCS_PREFIX", "raw/listings_processed/")
LOCAL_INPUT_DIR = os.getenv("AVM_LOCAL_INPUT_DIR", "").strip()
OUTPUT_FILE = Path(os.getenv("AVM_OUTPUT_CSV", "/Users/AVM/AVM/modeling/combined_listings.csv"))


def _clean_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _build_listing_key(row):
    image_url = _clean_text(row.get("main_image_url"))
    if image_url:
        return f"img:{image_url}"

    parts = [
        _clean_text(row.get("city")),
        _clean_text(row.get("price")),
        _clean_text(row.get("property_type")),
        _clean_text(row.get("bedrooms")),
        _clean_text(row.get("address")),
        _clean_text(row.get("description"))[:240],
    ]
    raw = "|".join(parts)
    return "row:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _quality_score(df):
    score = pd.Series(0, index=df.index, dtype=float)
    for col in df.columns:
        score = score + df[col].notna().astype(int)
    if "description" in df.columns:
        score = score + df["description"].astype(str).str.len().fillna(0).clip(upper=2000) / 2000.0
    if "main_image_url" in df.columns:
        score = score + df["main_image_url"].notna().astype(int) * 2
    if "scraped_at" in df.columns:
        scraped = pd.to_datetime(df["scraped_at"], errors="coerce")
        score = score + scraped.notna().astype(int) * 0.5
    return score


def merge_listings():
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

    all_dfs = []
    file_count = 0

    if LOCAL_INPUT_DIR:
        print(f"⬇️  Loading listing files from local dir: {LOCAL_INPUT_DIR}")
        for path in sorted(Path(LOCAL_INPUT_DIR).glob("*.csv")):
            print(f"   Found: {path}")
            file_count += 1
            df = pd.read_csv(path, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            df["source_file"] = str(path)
            all_dfs.append(df)
    else:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        print("⬇️  Downloading all listing files from GCS...")
        blobs = bucket.list_blobs(prefix=GCS_PREFIX)

        for blob in blobs:
            if blob.name.endswith(".csv") and not blob.name.endswith("/"):
                print(f"   Found: {blob.name}")
                file_count += 1
                data = blob.download_as_text()
                df = pd.read_csv(StringIO(data), low_memory=False)
                df.columns = [c.strip() for c in df.columns]
                df["source_file"] = blob.name
                all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)

        # Standardize missing values before deduplication.
        for col in final_df.columns:
            if final_df[col].dtype == object:
                final_df[col] = final_df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

        final_df["listing_key"] = final_df.apply(_build_listing_key, axis=1)
        final_df["quality_score"] = _quality_score(final_df)

        before = len(final_df)
        final_df = final_df.sort_values(["listing_key", "quality_score"], ascending=[True, False])
        final_df = final_df.drop_duplicates(subset=["listing_key"], keep="first")

        # Remove exact duplicates that survive the fallback key.
        exact_subset = [c for c in ["city", "price", "address", "property_type", "bedrooms", "description", "main_image_url", "scraped_at"] if c in final_df.columns]
        if exact_subset:
            final_df = final_df.drop_duplicates(subset=exact_subset, keep="first")

        final_df = final_df.drop(columns=["listing_key", "quality_score"], errors="ignore")
        final_df = final_df.reset_index(drop=True)

        print(f"✅ Merged {file_count} files.")
        print(f"📊 Rows before dedupe: {before:,}")
        print(f"📊 Rows after dedupe:  {len(final_df):,}")

        # Save locally
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Saved to {OUTPUT_FILE}")
    else:
        print("⚠️  No files found in bucket.")


if __name__ == "__main__":
    merge_listings()
