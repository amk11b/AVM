import os
import pandas as pd
from apify_client import ApifyClient
from google.cloud import storage
from datetime import datetime
import json

# --- CONFIGURATION ---
KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
BUCKET_NAME = "avm_data_ml"

# APIFY SETUP
# 1. Get token from: https://console.apify.com/account/integrations
APIFY_TOKEN = "apify_api_lXf3LB2Z89DPqqNYvoZFWxW37x9dbG3d9lE1"

# 2. We use a popular Rightmove Scraper (check Apify store for pricing)
# This one is usually reliable: 'epctex/rightmove-scraper'
ACTOR_ID = "epctex/rightmove-scraper"


# ---------------------

def run_scraper(location, max_items=50):
    """
    Runs the Apify scraper using a direct Rightmove Search URL.
    """
    # 1. Rightmove URL for BIRMINGHAM
    rightmove_url = "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E162&sortType=6&includeSSTC=false"

    print(f"🕷️  Starting scrape for URL: {rightmove_url} (Limit: {max_items})")

    client = ApifyClient(APIFY_TOKEN)

    # 2. Input Schema (FIXED)
    # We pass the URL string directly in the list, NOT wrapped in { "url": ... }
    run_input = {
        "startUrls": [rightmove_url],
        "maxItems": max_items,
        "listingType": "sale",
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"]
        },
        "endPage": 1
    }

    print("⏳ Waiting for Apify to finish (this takes ~30-60 seconds)...")
    run = client.actor(ACTOR_ID).call(run_input=run_input)

    if not run:
        raise Exception("Apify run failed to start.")

    dataset_id = run["defaultDatasetId"]
    print(f"✅ Scrape complete! Fetching results from Dataset: {dataset_id}")

    items = []
    for item in client.dataset(dataset_id).iterate_items():
        items.append(item)

    return items


def upload_listings_to_gcs(data, location):
    """
    Saves the complex JSON data to GCS.
    """
    if not data:
        print("⚠️  No data scraped.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # 1. Save RAW JSON (Already worked!)
    filename_json = f"raw/listings/{location}_{timestamp}.json"
    blob = bucket.blob(filename_json)
    blob.upload_from_string(json.dumps(data), content_type='application/json')
    print(f"☁️  Uploaded Raw JSON: gs://{BUCKET_NAME}/{filename_json}")

    # 2. Save Simplified CSV (The "Silver" Layer)
    simplified_data = []
    for item in data:
        # SAFETY CHECK: Handle different image formats
        main_image = None
        images = item.get("images")

        if images and isinstance(images, list) and len(images) > 0:
            first_img = images[0]
            if isinstance(first_img, dict):
                main_image = first_img.get('url')  # It's a dict
            elif isinstance(first_img, str):
                main_image = first_img  # It's already a URL string

        simplified_data.append({
            "url": item.get("url"),
            "price": item.get("price"),
            "address": item.get("displayAddress"),
            "property_type": item.get("propertyType"),
            "bedrooms": item.get("bedrooms"),
            "description": item.get("summary"),
            "main_image_url": main_image,  # Use our safe variable
            "scraped_at": timestamp
        })

    df = pd.DataFrame(simplified_data)
    filename_csv = f"raw/listings_processed/{location}_{timestamp}.csv"
    blob_csv = bucket.blob(filename_csv)
    blob_csv.upload_from_string(df.to_csv(index=False), content_type='text/csv')
    print(f"☁️  Uploaded Processed CSV: gs://{BUCKET_NAME}/{filename_csv}")


if __name__ == "__main__":
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

    # TEST RUN
    # Pick a location that exists in your Land Registry data (e.g., "Bristol")
    LOCATION = "Birmingham"

    try:
        results = run_scraper(LOCATION, max_items=20)  # Keep low to save credits
        upload_listings_to_gcs(results, LOCATION)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tip: Check your APIFY_TOKEN and ensure you have credits.")