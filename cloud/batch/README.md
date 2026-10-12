# AVM Batch Pipeline

This submits the training pipeline to Google Cloud Batch using a Spot VM.

What it runs:

1. `data_ingestion/create_enriched_training_data.py`
2. `modeling/generate_embeddings.py`
3. `modeling/final_training.py`

Run it from the repo root:

```bash
bash cloud/batch/submit_avm_pipeline.sh
```

Useful overrides:

```bash
PROJECT_ID=avm-ml-485715
REGION=us-central1
MACHINE_TYPE=n2-standard-16
MODEL_TAG=v5
```

The pipeline reads the enriched CSV from:

`gs://avm_data_ml/staging/master_training_data_v4.csv`

and writes model artifacts to:

`gs://avm_data_ml/models/<MODEL_TAG>`
