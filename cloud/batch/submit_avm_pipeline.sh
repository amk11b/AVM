#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-avm-ml-485715}"
REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-avm-models}"
IMAGE_NAME="${IMAGE_NAME:-avm-pipeline}"
MODEL_TAG="${MODEL_TAG:-v5}"
MACHINE_TYPE="${MACHINE_TYPE:-n2-standard-16}"
JOB_NAME="${JOB_NAME:-avm-pipeline-$(date +%Y%m%d-%H%M%S)}"

ENRICHED_CSV_URI="${ENRICHED_CSV_URI:-gs://avm_data_ml/staging/master_training_data_v4.csv}"
EMBEDDINGS_CSV_URI="${EMBEDDINGS_CSV_URI:-gs://avm_data_ml/staging/embeddings_${MODEL_TAG}.csv}"
MODEL_GCS_URI="${MODEL_GCS_URI:-gs://avm_data_ml/models/${MODEL_TAG}}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${MODEL_TAG}"
JOB_RESOURCE="projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB_NAME}"

echo "Building ${IMAGE_URI} ..."
gcloud artifacts repositories describe "${REPOSITORY}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}"

gcloud builds submit \
  --project="${PROJECT_ID}" \
  --tag "${IMAGE_URI}" \
  .

JOB_CONFIG="$(mktemp)"
cat > "${JOB_CONFIG}" <<EOF
{
  "taskGroups": [
    {
      "taskCount": "1",
      "taskSpec": {
        "maxRetryCount": 1,
        "computeResource": {
          "cpuMilli": "16000",
          "memoryMib": "64000",
          "bootDiskMib": "204800"
        },
        "runnables": [
          {
            "container": {
              "imageUri": "${IMAGE_URI}",
              "enableImageStreaming": true,
              "commands": [
                "/bin/bash",
                "-lc",
                "set -euo pipefail; export AVM_INPUT_CSV='${ENRICHED_CSV_URI}'; export AVM_OUTPUT_CSV='/tmp/avm-embeddings/${MODEL_TAG}/embeddings.csv'; mkdir -p /tmp/avm-embeddings/${MODEL_TAG}; python3 modeling/generate_embeddings.py && export AVM_INPUT_CSV='/tmp/avm-embeddings/${MODEL_TAG}/embeddings.csv'; export AVM_MODEL_DIR='/tmp/avm-models/${MODEL_TAG}'; export AVM_MODEL_GCS_URI='${MODEL_GCS_URI}'; mkdir -p /tmp/avm-models/${MODEL_TAG}; python3 modeling/final_training.py"
              ]
            }
          }
        ]
      }
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "policy": {
          "machineType": "${MACHINE_TYPE}",
          "provisioningModel": "SPOT"
        }
      }
    ],
    "location": {
      "allowedLocations": [
        "regions/${REGION}"
      ]
    }
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
EOF

echo "Submitting Batch job ${JOB_RESOURCE} ..."
gcloud batch jobs submit "${JOB_RESOURCE}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --config="${JOB_CONFIG}"

rm -f "${JOB_CONFIG}"
echo "Submitted ${JOB_RESOURCE}"
