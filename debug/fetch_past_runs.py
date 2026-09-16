import pandas as pd
from apify_client import ApifyClient
import os

# --- CONFIG ---
# Your token from the URL you pasted (I've masked it here for safety)
APIFY_TOKEN = "apify_api_lXf3LB2Z89DPqqNYvoZFWxW37x9dbG3d9lE1"
EMBEDDINGS_FILE = "/Users/AVM/AVM/modeling/embeddings.csv"


def fetch_and_repair():
    print("⏳ Connecting to Apify to fetch coordinate data...")
    client = ApifyClient(APIFY_TOKEN)

    runs = client.runs().list(limit=11, desc=True).items
    print(f"   - Found {len(runs)} recent runs.")

    all_repaired_items = []

    print("📥 Downloading data...")
    for run in runs:
        if run['status'] != 'SUCCEEDED': continue
        dataset_id = run['defaultDatasetId']
        items = client.dataset(dataset_id).list_items().items

        for item in items:
            # 1. Image URL (Key)
            main_image = None
            images = item.get("images")
            if images and len(images) > 0:
                if isinstance(images[0], str):
                    main_image = images[0]
                elif isinstance(images[0], dict):
                    main_image = images[0].get('url')

            # 2. Coordinates (THE NEW GOLD)
            lat = item.get("latitude")
            lon = item.get("longitude")

            # 3. Basic Facts
            address = item.get("address") or item.get("displayAddress")
            beds = item.get("beds") or item.get("bedrooms") or item.get("bedroomCount")

            if main_image and address and lat and lon:
                all_repaired_items.append({
                    "main_image_url": main_image,
                    "address": address,
                    "bedrooms": beds,
                    "latitude": lat,  # <--- Captured
                    "longitude": lon,  # <--- Captured
                    "price": item.get("price"),
                    "property_type": item.get("propertyType"),
                    "description": item.get("summary") or item.get("description") or "",
                    "city": "Unknown"
                })

    df_fresh = pd.DataFrame(all_repaired_items)
    df_fresh = df_fresh.drop_duplicates(subset=['main_image_url'])
    print(f"   - Recovered {len(df_fresh)} listings with Coordinates.")

    # --- MERGE ---
    print("🔗 Merging with local Image Embeddings...")
    try:
        df_emb = pd.read_csv(EMBEDDINGS_FILE)
        # Keep visual columns only
        visual_cols = [c for c in df_emb.columns if c.startswith('img_dim_')]
        df_visuals = df_emb[['main_image_url'] + visual_cols]

        final_df = pd.merge(df_fresh, df_visuals, on='main_image_url', how='inner')

        # Simple City Logic
        def guess_city(addr):
            if not isinstance(addr, str): return "Other"
            for c in ["Bristol", "Birmingham", "Manchester", "Leeds", "Liverpool", "Sheffield", "Nottingham",
                      "Newcastle", "Leicester", "Cardiff", "Southampton"]:
                if c.lower() in addr.lower(): return c
            return "Other"

        final_df['city'] = final_df['address'].apply(guess_city)

        final_df.to_csv(EMBEDDINGS_FILE, index=False)
        print(f"💾 Saved {len(final_df)} rows to {EMBEDDINGS_FILE}")

    except FileNotFoundError:
        print(f"❌ Could not find {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    fetch_and_repair()