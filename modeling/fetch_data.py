import os
import pandas as pd
from google.cloud import bigquery

# --- CONFIGURATION ---
KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
PROJECT_ID = "avm-ml-485715"  # Your Project ID
OUTPUT_FILE = "/Users/AVM/AVM/modeling/training_data.csv"


# ---------------------

def fetch_training_data():
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

    client = bigquery.Client()

    print("🚀 Querying BigQuery for Training Data...")

    # The "Golden Record" Query
    # We select specific columns needed for the model
    query = """
    SELECT 
        -- TARGET
        lr.price,

        -- FEATURES
        lr.property_type,
        lr.old_new,
        lr.duration, -- Freehold/Leasehold

        epc.total_floor_area,
        epc.number_habitable_rooms,
        epc.current_energy_rating,
        epc.construction_age_band,

        geo.lat,
        geo.lon,
        geo.district

    FROM `property_data.land_registry` AS lr

    -- INNER JOIN Geocodes (We MUST have location)
    INNER JOIN `property_data.geocodes` AS geo
        ON lr.postcode = geo.postcode

    -- INNER JOIN EPC (We MUST have physical features for a good model)
    -- Using INNER JOIN here reduces row count but ensures high quality data
    INNER JOIN `property_data.epc_raw` AS epc
        ON lr.postcode = epc.postcode

    WHERE 
        lr.transfer_date >= '2023-01-01' -- Let's take last ~1-2 years
        AND lr.price BETWEEN 50000 AND 2000000 -- Remove outliers/errors
        AND epc.total_floor_area > 30 -- Remove bad data (e.g. 1 sq meter houses)
    """

    df = client.query(query).to_dataframe()

    print(f"✅ Data fetched! Shape: {df.shape}")
    print("   (Rows, Columns)")

    # Save to local CSV for fast loading later
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_training_data()