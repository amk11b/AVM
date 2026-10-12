import os
from google.cloud import bigquery, storage

# --- CONFIGURATION ---
KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
PROJECT_ID = "avm-ml-485715"
BUCKET_NAME = "avm_data_ml"
DATASET_NAME = "avm_dataset"
LOCATION = "EU"


# ---------------------

def get_all_epc_uris():
    """Scans the GCS bucket to find ONLY the 'certificates.csv' files."""
    print("🔎 Scanning bucket for EPC Certificates...")

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(BUCKET_NAME, prefix="all-domestic-certificates")

    uris = [f"gs://{BUCKET_NAME}/{blob.name}"
            for blob in blobs
            if blob.name.endswith("certificates.csv")]

    print(f"   Found {len(uris)} certificate files.")
    return uris


def load_epc_flexible(client, uris):
    """
    Load EPC with autodetect and generous error tolerance.
    """
    table_id = f"{PROJECT_ID}.{DATASET_NAME}.epc_raw"

    print(f"\n📤 Loading: epc_raw (flexible mode)...")

    client.delete_table(table_id, not_found_ok=True)
    print(f"   🗑️  Cleared existing table (if any)")

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        allow_quoted_newlines=True,
        allow_jagged_rows=True,  # Allow rows with fewer columns
        ignore_unknown_values=True,  # Allow rows with MORE columns (key fix!)
        max_bad_records=100000,  # Skip up to 100k bad rows
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    try:
        load_job = client.load_table_from_uri(uris, table_id, job_config=job_config)
        load_job.result()

        table = client.get_table(table_id)
        print(f"   ✅ Loaded {table.num_rows:,} rows into epc_raw")

        # Report skipped rows
        if load_job.errors:
            print(f"   ⚠️  Skipped some rows due to errors")

    except Exception as e:
        print(f"   ❌ Error loading epc_raw: {e}")
        if hasattr(e, 'errors'):
            for error in e.errors[:5]:
                print(f"      - {error}")


if __name__ == "__main__":
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

    client = bigquery.Client(project=PROJECT_ID)

    # Create Dataset
    dataset_id = f"{PROJECT_ID}.{DATASET_NAME}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = LOCATION
    client.create_dataset(dataset, exists_ok=True)
    print(f"📂 Dataset '{DATASET_NAME}' is ready.")

    # Load EPC
    epc_uris = get_all_epc_uris()
    if epc_uris:
        load_epc_flexible(client, epc_uris)

    print("\n" + "=" * 50)
    print("🎉 Done!")