import os
import runpy
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("AVM_MODEL_DIR", str(BASE_DIR.parent / "models" / "local")))
ENRICHED_CSV = os.getenv("AVM_ENRICHED_CSV", "gs://avm_data_ml/staging/master_training_data_v4.csv")
EMBEDDINGS_CSV = Path(os.getenv("AVM_EMBEDDINGS_CSV", str(BASE_DIR / "embeddings.csv")))
KEY_PATH = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", str(BASE_DIR.parent / "gcp_key.json")))


if KEY_PATH.exists() and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)


def main():
    print("1/2 Generating image embeddings from enriched GCS data...")
    os.environ["AVM_INPUT_CSV"] = ENRICHED_CSV
    os.environ["AVM_OUTPUT_CSV"] = str(EMBEDDINGS_CSV)
    runpy.run_path(str(BASE_DIR / "generate_embeddings.py"), run_name="__main__")

    print("2/2 Training model locally...")
    os.environ["AVM_INPUT_CSV"] = str(EMBEDDINGS_CSV)
    os.environ["AVM_MODEL_DIR"] = str(MODEL_DIR)
    os.environ.pop("AVM_MODEL_GCS_URI", None)
    runpy.run_path(str(BASE_DIR / "final_training.py"), run_name="__main__")

    print("Done.")
    print(f"Enriched CSV: {ENRICHED_CSV}")
    print(f"Embeddings CSV: {EMBEDDINGS_CSV}")
    print(f"Model dir: {MODEL_DIR}")


if __name__ == "__main__":
    main()
