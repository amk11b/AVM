import os
import pandas as pd
from apify_client import ApifyClient
from google.cloud import storage
from datetime import datetime, time
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

TARGET_LOCATIONS = {
    "Bristol": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E219&sortType=6&includeSSTC=false",
    "Birmingham": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E162&sortType=6&includeSSTC=false",
    "Manchester": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E904&sortType=6&includeSSTC=false",
    "Leeds": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E787&sortType=6&includeSSTC=false",
    "Liverpool": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E814&sortType=6&includeSSTC=false",
    "Sheffield": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E1195&sortType=6&includeSSTC=false",
    "Nottingham": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E1019&sortType=6&includeSSTC=false",
    "Newcastle": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E984&sortType=6&includeSSTC=false",
    "Leicester": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E813&sortType=6&includeSSTC=false",
    "Cardiff": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E297&sortType=6&includeSSTC=false",
    "Southampton": "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E1231&sortType=6&includeSSTC=false"
}


# ---------------------

def run_scraper(location_name, url, max_items=100):
    print(f"\n🕷️  Starting scrape for: {location_name} (Target: {max_items})...")
    client = ApifyClient(APIFY_TOKEN)

    run_input = {
        "startUrls": [url],
        "maxItems": max_items,
        "listingType": "sale",
        "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
        # Removed "endPage" so it actually reaches the maxItems limit
    }

    run = client.actor(ACTOR_ID).call(run_input=run_input)

    if not run:
        print(f"   ❌ Failed to start scrape for {location_name}")
        return []

    dataset_id = run["defaultDatasetId"]
    print(f"   ✅ Scrape complete! Fetching results...")

    items = []
    # Loop through pages of results to ensure we get everything
    for item in client.dataset(dataset_id).iterate_items():
        items.append(item)

    return items


def upload_listings_to_gcs(data, location_name):
    if not data:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    simplified_data = []
    for item in data:
        # Robust Image Extraction
        main_image = None
        images = item.get("images")
        if images and isinstance(images, list) and len(images) > 0:
            first_img = images[0]
            if isinstance(first_img, dict):
                main_image = first_img.get('url')
            elif isinstance(first_img, str):
                main_image = first_img

        # Robust Description Extraction
        description = item.get("summary") or item.get("description") or ""

        simplified_data.append({
            "city": location_name,
            "price": item.get("price"),
            "address": item.get("displayAddress"),
            "property_type": item.get("propertyType"),
            "bedrooms": item.get("beds"),
            "description": description,
            "main_image_url": main_image,
            "scraped_at": timestamp
        })

    df = pd.DataFrame(simplified_data)

    # Save as CSV
    filename = f"raw/listings_processed/{location_name}_{timestamp}.csv"
    bucket.blob(filename).upload_from_string(df.to_csv(index=False), content_type='text/csv')
    print(f"   ☁️  Uploaded {len(df)} rows to: gs://{BUCKET_NAME}/{filename}")


if __name__ == "__main__":
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

    # Loop through all cities
    for city, url in TARGET_LOCATIONS.items():
        try:
            results = run_scraper(city, url, max_items=1000)
            upload_listings_to_gcs(results, city)







        except Exception as e:
            print(f"❌ Error scraping {city}: {e}")

    print("\n🎉 MEGA SCRAPE COMPLETE!")