import os
from google.cloud import bigquery, storage

# --- CONFIGURATION ---
KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
PROJECT_ID = "avm-ml-485715"  # <--- REPLACE THIS WITH YOUR PROJECT ID
BUCKET_NAME = "avm_data_ml"
DATASET_NAME = "property_data"


# ---------------------

def get_all_epc_uris():
    """
    Scans the GCS bucket to find ONLY the 'certificates.csv' files.
    """
    print("🔎 Scanning bucket for EPC Certificates...")

    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
    storage_client = storage.Client()

    # Scan the folder
    blobs = storage_client.list_blobs(BUCKET_NAME, prefix="all-domestic-certificates")

    uris = []
    for blob in blobs:
        # FIX: Check if the filename ends with 'certificates.csv' specifically
        if blob.name.endswith("certificates.csv"):
            uris.append(f"gs://{BUCKET_NAME}/{blob.name}")

    print(f"   found {len(uris)} Certificate files (ignoring recommendations).")
    return uris


def create_external_table(client, table_name, uris, autodetect=True, skip_rows=1):
    table_id = f"{PROJECT_ID}.{DATASET_NAME}.{table_name}"

    print(f"🔨 Configuring table: {table_name}...")

    # Ensure uris is a list
    if isinstance(uris, str):
        uris = [uris]

    table = bigquery.Table(table_id)
    external_config = bigquery.ExternalConfig("CSV")
    external_config.source_uris = uris
    external_config.skip_leading_rows = skip_rows
    external_config.autodetect = autodetect
    external_config.max_bad_records = 100

    table.external_data_configuration = external_config

    try:
        client.create_table(table, exists_ok=True)
        print(f"   ✅ Created/Updated: {table_id}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

    client = bigquery.Client()

    # 1. Create Dataset
    dataset_id = f"{PROJECT_ID}.{DATASET_NAME}"
    try:
        client.create_dataset(bigquery.Dataset(dataset_id), exists_ok=True)
        print(f"📂 Dataset '{DATASET_NAME}' is ready.\n")
    except Exception as e:
        print(f"❌ Dataset Error: {e}")

    # 2. Link Land Registry
    create_external_table(
        client,
        "land_registry",
        f"gs://{BUCKET_NAME}/raw/land_registry_*.csv"
    )

    # 3. Link Geocodes
    create_external_table(
        client,
        "geocodes",
        f"gs://{BUCKET_NAME}/processed/geocodes_*.csv"
    )

    # 4. Link EPC (The Smart Fix)
    # We get the list of 695 files explicitly
    epc_file_list = get_all_epc_uris()
    if epc_file_list:
        create_external_table(client, "epc_raw", epc_file_list)
    else:
        print("⚠️  No EPC files found. Check the folder path in the script.")

    # 5. Link Listings
    create_external_table(
        client,
        "listings",
        f"gs://{BUCKET_NAME}/raw/listings_processed/*.csv"
    )