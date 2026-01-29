import pandas as pd
import requests
import os
import time
import io
from google.cloud import storage

# --- CONFIGURATION ---
KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
BUCKET_NAME = "avm_data_ml"
INPUT_FILE = "raw/land_registry_2024.csv"  # File to read postcodes from
OUTPUT_FILE = "processed/geocodes_2024.csv"


# ---------------------

def get_unique_postcodes_from_gcs():
    """Reads the Land Registry CSV directly from GCS to find postcodes."""
    print(f"📥 Downloading {INPUT_FILE} from GCS to extract postcodes...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(INPUT_FILE)

    content = blob.download_as_bytes()

    # FIX: The file has headers, so we just ask for the "postcode" column directly
    df = pd.read_csv(io.BytesIO(content), usecols=["postcode"])

    unique_postcodes = df['postcode'].dropna().unique().tolist()
    print(f"✅ Found {len(unique_postcodes):,} unique postcodes in the file.")
    return unique_postcodes


def bulk_geocode(postcodes):
    """Hits Postcodes.io API in batches of 100."""
    endpoint = "https://api.postcodes.io/postcodes"
    results = []

    # Batch size of 100 (API limit)
    batch_size = 100
    total_batches = (len(postcodes) + batch_size - 1) // batch_size

    print(f"🌍 Geocoding {len(postcodes)} postcodes in {total_batches} batches...")

    for i in range(0, len(postcodes), batch_size):
        batch = postcodes[i:i + batch_size]

        try:
            response = requests.post(endpoint, json={"postcodes": batch})
            if response.status_code == 200:
                data = response.json()['result']
                for item in data:
                    res = item['result']
                    if res:
                        results.append({
                            'postcode': item['query'],
                            'lat': res['latitude'],
                            'lon': res['longitude'],
                            'district': res['admin_district']
                        })
        except Exception as e:
            print(f"❌ Batch error: {e}")

        time.sleep(0.1)  # Be nice to the API

        if i % 1000 == 0 and i > 0:
            print(f"   Processed {i}...")

    return pd.DataFrame(results)


def upload_to_gcs(df):
    """Uploads the result back to GCS."""
    print(f"☁️  Uploading results to gs://{BUCKET_NAME}/{OUTPUT_FILE}...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(OUTPUT_FILE)

    blob.upload_from_string(df.to_csv(index=False), content_type='text/csv')
    print("✅ Upload successful!")


if __name__ == "__main__":
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

    # 1. Get Postcodes from the file sitting in your bucket
    postcodes = get_unique_postcodes_from_gcs()

    # 2. Run Geocoder
    # For testing, we limit to the first 200.
    # REMOVE [:200] to run the full dataset!
    print("⚠️  Running on first 200 postcodes only. Remove slice in code to run all.")
    geocoded_df = bulk_geocode(postcodes[:200])

    # 3. Upload Result
    if not geocoded_df.empty:
        print(geocoded_df.head())
        upload_to_gcs(geocoded_df)