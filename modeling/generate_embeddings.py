import os
from pathlib import Path
import pandas as pd
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import numpy as np
import tempfile
from google.cloud import storage

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = os.getenv("AVM_INPUT_CSV", str(BASE_DIR / "combined_listings.csv"))
OUTPUT_FILE = Path(os.getenv("AVM_OUTPUT_CSV", str(BASE_DIR / "embeddings.csv")))

# --- 1. SETUP THE VISION MODEL (ResNet) ---
print("🧠 Loading ResNet50 model...")
# ResNet50 is a famous model trained on millions of images
# We remove the last layer because we want the "features" (vectors), not a classification ("Dog/Cat")
base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
model.eval()  # Set to evaluation mode (not training)

# Define how to prepare the image (Resize -> Normalize)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_image_embedding(url):
    """
    Downloads image -> Converts to Tensor -> Runs through ResNet -> Returns Vector
    """
    if not isinstance(url, str) or not url.startswith("http"):
        return None

    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None

        img = Image.open(BytesIO(response.content)).convert('RGB')

        # Preprocess for AI
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Run Inference
        with torch.no_grad():
            embedding = model(img_tensor)

        # Flatten to a simple list of numbers
        return embedding.flatten().numpy()

    except Exception:
        return None


def _parse_gcs_uri(uri):
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got: {uri}")
    bucket_name, _, blob_name = uri[5:].partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return bucket_name, blob_name


def _download_input(source):
    source_str = str(source)
    if source_str.startswith("gs://"):
        bucket_name, blob_name = _parse_gcs_uri(source_str)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"GCS object does not exist: {source_str}")
        tmp_dir = Path(tempfile.mkdtemp(prefix="avm-embeddings-"))
        local_path = tmp_dir / Path(blob_name).name
        blob.download_to_filename(local_path)
        return local_path
    return Path(source_str)


def _drop_all_null_columns(df):
    drop_cols = [col for col in df.columns if df[col].isna().all()]
    if drop_cols:
        print(f"🧽 Dropping all-null columns: {', '.join(drop_cols)}")
        df = df.drop(columns=drop_cols)
    return df


# --- 2. MAIN LOOP ---
def main():
    # You might need to download the file from GCS first if it's not local!
    local_csv_path = _download_input(INPUT_CSV)

    if not local_csv_path.exists():
        print(f"❌ Input CSV not found: {local_csv_path}")
        return

    df = pd.read_csv(local_csv_path)
    print(f"📂 Loaded {len(df)} listings.")

    embeddings = []
    image_flags = []

    print("🖼️  Processing images (this takes time)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = row.get('main_image_url')
        vector = get_image_embedding(url)
        if vector is None:
            embeddings.append(np.zeros(2048, dtype=np.float32))
            image_flags.append(0)
        else:
            embeddings.append(vector.astype(np.float32))
            image_flags.append(1)

    # Create a DataFrame of just the embeddings
    # ResNet50 outputs 2048 dimensions
    cols = [f"img_dim_{i}" for i in range(2048)]
    emb_df = pd.DataFrame(embeddings, columns=cols, index=df.index)
    emb_df["has_image_embedding"] = image_flags

    # Merge back with the original Price/Address data
    final_df = df.join(emb_df)
    final_df = _drop_all_null_columns(final_df)

    print(f"✅ Generated embeddings for {int(sum(image_flags))} images.")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved rich data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
