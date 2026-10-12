import pandas as pd
from google.cloud import storage
import io
import os

# --- CONFIG ---
KEY_PATH = "/Users/AVM/AVM/gcp_key.json"  # <--- Make sure this file is in your project folder!
BUCKET_NAME = "avm_data_ml"
EMBEDDINGS_FILE = "/Users/AVM/AVM/modeling/embeddings.csv"


def smart_merge():
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
    else:
        print(f"❌ Error: Could not find key file at {KEY_PATH}")
        return

    print("🔄 STARTING FULL METADATA REFRESH...")

    # 1. Load your existing Embeddings (Pre-calculated images)
    print("📂 Loading existing embeddings...")
    try:
        df_emb = pd.read_csv(EMBEDDINGS_FILE)
        # We ONLY want the visual data from this file (columns starting with img_dim_)
        # plus the join key (main_image_url)
        visual_cols = [c for c in df_emb.columns if c.startswith('img_dim_')]
        df_emb_visuals = df_emb[['main_image_url'] + visual_cols]

        print(f"   - Kept {len(visual_cols)} visual features for {len(df_emb)} rows.")
    except FileNotFoundError:
        print("❌ No embeddings.csv found.")
        return

    # 2. Download the NEW Scraped Data (Fresh Facts)
    print("☁️  Downloading fresh scrape data...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix="raw/listings_processed/")

        scraped_dfs = []
        for blob in blobs:
            if blob.name.endswith(".csv"):
                content = blob.download_as_text()
                temp_df = pd.read_csv(io.StringIO(content))

                # Check for essential columns
                required = ['main_image_url', 'price', 'address', 'bedrooms', 'property_type', 'description']
                available = [c for c in required if c in temp_df.columns]

                if 'main_image_url' in temp_df.columns:
                    scraped_dfs.append(temp_df[available])

        if not scraped_dfs:
            print("❌ No data found in bucket.")
            return

        df_facts = pd.concat(scraped_dfs, ignore_index=True)
        # Drop duplicates on image URL so we don't have double rows
        df_facts = df_facts.drop_duplicates(subset=['main_image_url'])
        print(f"   - Found {len(df_facts)} fresh rows with addresses & prices.")

    except Exception as e:
        print(f"❌ Google Cloud Error: {e}")
        return

    # 3. The Merge (Join Facts + Vision)
    print("🔗 Merging Facts + Vision...")

    # Inner join: We only keep rows where we have BOTH a scraped fact AND an image vector
    merged_df = pd.merge(df_facts, df_emb_visuals, on='main_image_url', how='inner')

    print(f"   ✅ Merged! Final dataset size: {len(merged_df)} rows.")

    # Quick Diagnostic
    sample_address = merged_df['address'].iloc[0] if len(merged_df) > 0 else "None"
    print(f"   👀 Sample Address check: {sample_address}")

    # Save
    merged_df.to_csv(EMBEDDINGS_FILE, index=False)
    print(f"💾 Saved updated file to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    smart_merge()