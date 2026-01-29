import pandas as pd
import requests
import os
import base64
import time
import io
from google.cloud import storage

# --- CONFIGURATION ---
KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
BUCKET_NAME = "avm_data_ml"
INPUT_FILE = "raw/land_registry_2024.csv"

# EPC API CREDENTIALS
# Register at: https://epc.opendatacommunities.org/
EPC_USER = "ashleigh.m.jacobs1@gmail.com"  # <--- REPLACE THIS
EPC_KEY = "9cc2e3ea761487fd608a58600ea8bea9f49040d6"  # <--- REPLACE THIS


# ---------------------

def get_auth_header():
    """Creates the Basic Auth header for the EPC API."""
    auth_str = f"{EPC_USER}:{EPC_KEY}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    return {"Authorization": f"Basic {b64_auth}", "Accept": "application/json"}


def get_postcodes_from_gcs():
    """Reads postcodes from the Land Registry file in GCS."""
    print("📥 Reading postcodes from GCS...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(INPUT_FILE)
    content = blob.download_as_bytes()

    # FIX: Use the header name "postcode" directly
    df = pd.read_csv(io.BytesIO(content), usecols=["postcode"])

    unique_postcodes = df['postcode'].dropna().unique().tolist()
    print(f"✅ Found {len(unique_postcodes):,} unique postcodes.")
    return unique_postcodes


def fetch_epc_data(postcodes):
    """
    Fetches EPC data for a list of postcodes.
    """
    base_url = "https://epc.opendatacommunities.org/api/v1/domestic/search"
    headers = get_auth_header()
    results = []

    print(f"⚡ Fetching EPC data for {len(postcodes)} postcodes...")

    for i, pcode in enumerate(postcodes):
        # We limit size to 1 to get the most recent certificate for that postcode
        params = {"postcode": pcode, "size": 100}

        try:
            r = requests.get(base_url, headers=headers, params=params)

            if r.status_code == 200:
                data = r.json()
                if 'rows' in data:
                    for row in data['rows']:
                        results.append({
                            "postcode": row.get("postcode"),
                            "address": row.get("address"),
                            "total_floor_area": row.get("total-floor-area"),
                            "number_habitable_rooms": row.get("number-habitable-rooms"),
                            "energy_rating": row.get("current-energy-rating"),
                            "construction_age": row.get("construction-age-band")
                        })
            elif r.status_code == 429:
                print("⚠️ Rate limit hit. Sleeping for 10s...")
                time.sleep(10)
            elif r.status_code == 401:
                print("❌ Auth Failed. Check your Email/Key.")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Error on {pcode}: {e}")

        time.sleep(0.1)

        if i % 10 == 0:
            print(f"   Processed {i}...")

    return pd.DataFrame(results)


def upload_to_gcs(df):
    filename = "raw/epc_data_2024.csv"
    print(f"☁️  Uploading {len(df)} rows to gs://{BUCKET_NAME}/{filename}...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_string(df.to_csv(index=False), content_type='text/csv')
    print("✅ Upload successful!")


if __name__ == "__main__":
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

    # 1. Get Postcodes
    postcodes = get_postcodes_from_gcs()

    # 2. Fetch EPC Data
    # IMPORTANT: The API is slow. We test with 10 records first.
    # Change [:10] to [:100] or remove it entirely when confident.
    print("⚠️  Running on first 10 postcodes only.")
    epc_df = fetch_epc_data(postcodes[:10])

    # 3. Upload
    if not epc_df.empty:
        print(epc_df.head())
        upload_to_gcs(epc_df)
    else:
        print("❌ No data found.")