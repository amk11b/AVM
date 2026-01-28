import pandas as pd
import requests
import io
import os
from google.cloud import storage

# Land Registry Open Data URL pattern
BASE_URL = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}.csv"

COLUMNS = [
    "transaction_id", "price", "transfer_date", "postcode", "property_type",
    "old_new", "duration", "paon", "saon", "street", "locality", "town_city",
    "district", "county", "ppd_category_type", "record_status"
]


def download_and_load_year(year=2024):
    """Downloads data and returns a DataFrame."""
    url = BASE_URL.format(year=year)
    print("📥 Downloading data for {year} from {url}...")

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download: {response.status_code}")

    print("✅ Download complete. Parsing...")

    df = pd.read_csv(
        io.BytesIO(response.content),
        names=COLUMNS,
        header=None,
        dtype={'price': 'int64', 'postcode': 'str', 'property_type': 'category', 'old_new': 'category'},
        parse_dates=['transfer_date']
    )
    return df


def upload_to_gcs(df, bucket_name, destination_blob_name):
    """
    Uploads a DataFrame to a GCS bucket as a CSV file.
    """
    print("☁️  Uploading to gs://{bucket_name}/{destination_blob_name}...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Convert DataFrame to CSV string in memory
    csv_data = df.to_csv(index=False)

    # Upload
    blob.upload_from_string(csv_data, content_type='text/csv')

    print("✅ Upload successful!")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    BUCKET_NAME = "avm_data_ml"  # Your specific bucket name
    YEAR = 2024

    # POINT THIS TO YOUR JSON KEY FILE
    # Make sure this path matches exactly where your key is in your project
    KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
    # ---------------------

    try:
        # 1. Setup Auth explicitly
        if os.path.exists(KEY_PATH):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
            print(f"🔑 Loaded credentials from {KEY_PATH}")
        else:
            print(f"⚠️  WARNING: Key file not found at {KEY_PATH}. Ensure the path is correct.")

        # 2. Download
        df = download_and_load_year(YEAR)

        # 3. Preview
        print(f"\nLoaded {len(df):,} rows.")

        # 4. Upload to GCS
        file_name = f"raw/land_registry_{YEAR}.csv"
        upload_to_gcs(df, BUCKET_NAME, file_name)

    except Exception as e:
        print(f"\n❌ Error Details: {e}")
        print("Tip: If the error is '403 Forbidden', your service account needs the 'Storage Object Admin' role.")
        print("Tip: If the error is '404 Not Found', check if the bucket name 'avm_data_ml' is correct.")