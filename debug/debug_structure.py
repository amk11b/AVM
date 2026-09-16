from apify_client import ApifyClient
import json

# --- CONFIG ---
APIFY_TOKEN = "apify_api_lXf3LB2Z89DPqqNYvoZFWxW37x9dbG3d9lE1"  # <--- Paste your token again
ACTOR_ID = "epctex/rightmove-scraper"
TEST_URL = "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E219&sortType=6&includeSSTC=false"  # Bristol


def debug_one_item():
    print("🕵️‍♀️ Fetching ONE listing to inspect data structure...")
    client = ApifyClient(APIFY_TOKEN)

    run_input = {
        "startUrls": [TEST_URL],
        "maxItems": 1,
        "listingType": "sale",
        # --- THE MISSING PART IS BACK ---
        "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
    }

    try:
        run = client.actor(ACTOR_ID).call(run_input=run_input)
    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    if not run:
        print("❌ Failed to start run.")
        return

    dataset_id = run["defaultDatasetId"]

    # Get the result
    items = client.dataset(dataset_id).list_items(limit=1).items

    if items:
        item = items[0]
        print("\n" + "=" * 40)
        print("🔎 RAW ITEM DATA (Look for bedroom info):")
        print("=" * 40)
        print(json.dumps(item, indent=4))
        print("=" * 40)

        # Check common bedroom keys
        print("\n---- DIAGNOSTIC CHECK ----")
        print(f"Key 'beds':      {item.get('bedrooms')}")
        print(f"Key 'bedroomCount':  {item.get('bedroomCount')}")
        print(f"Key 'numberOfBedrooms': {item.get('numberOfBedrooms')}")
        print(f"Key 'features':      {item.get('features')}")
    else:
        print("❌ No items found.")


if __name__ == "__main__":
    debug_one_item()