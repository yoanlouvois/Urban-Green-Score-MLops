import os
import datetime
import boto3
from dotenv import load_dotenv

load_dotenv()

ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("AWS_REGION")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")

IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/urban-green-score:latest"

MODEL_PACKAGE_GROUP_NAME = "urban-green-score-model-group"


def get_latest_completed_training_model(sm_client):
    response = sm_client.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        StatusEquals="Completed",
        MaxResults=10,
        NameContains="urban-green-train",
    )

    jobs = response.get("TrainingJobSummaries", [])

    if not jobs:
        raise ValueError("No completed training job found.")

    latest_job_name = jobs[0]["TrainingJobName"]

    details = sm_client.describe_training_job(
        TrainingJobName=latest_job_name
    )

    model_artifact = details["ModelArtifacts"]["S3ModelArtifacts"]

    print(f"Latest completed training job: {latest_job_name}")
    print(f"Model artifact: {model_artifact}")

    return latest_job_name, model_artifact


def get_latest_completed_evaluation_metrics(sm_client):
    response = sm_client.list_processing_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        StatusEquals="Completed",
        MaxResults=20,
        NameContains="urban-green-evaluate",
    )

    jobs = response.get("ProcessingJobSummaries", [])

    if not jobs:
        raise ValueError("No completed evaluation job found.")

    latest_job_name = jobs[0]["ProcessingJobName"]

    metrics_s3_uri = (
        f"s3://{BUCKET}/artifacts/evaluation/"
        f"{latest_job_name}/metrics.json"
    )

    print(f"Latest completed evaluation job: {latest_job_name}")
    print(f"Metrics artifact: {metrics_s3_uri}")

    return latest_job_name, metrics_s3_uri


def ensure_model_package_group(sm_client):
    try:
        sm_client.describe_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME
        )
        print(f"Model package group already exists: {MODEL_PACKAGE_GROUP_NAME}")

    except sm_client.exceptions.ClientError:
        print(f"Creating model package group: {MODEL_PACKAGE_GROUP_NAME}")

        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelPackageGroupDescription=(
                "Urban Green Score semantic segmentation model registry"
            ),
        )


def register_model(sm_client, model_artifact_s3, metrics_s3_uri):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    response = sm_client.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelPackageDescription=f"Urban Green Score U-Net model registered at {timestamp}",
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": IMAGE_URI,
                    "ModelDataUrl": model_artifact_s3,
                }
            ],
            "SupportedContentTypes": [
                "application/json",
                "image/png",
                "image/jpeg",
            ],
            "SupportedResponseMIMETypes": [
                "application/json",
            ],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": metrics_s3_uri,
                }
            }
        },
        CustomerMetadataProperties={
            "project": "urban-green-score",
            "framework": "pytorch",
            "task": "semantic-segmentation",
            "model": "unet",
        },
    )

    model_package_arn = response["ModelPackageArn"]

    print("\nModel registered successfully.")
    print(f"Model package ARN: {model_package_arn}")

    print("\nOpen in AWS Console:")
    print(
        f"https://{REGION}.console.aws.amazon.com/sagemaker/home"
        f"?region={REGION}#/model-registry/model-group/{MODEL_PACKAGE_GROUP_NAME}"
    )

    return model_package_arn


def main():
    sm_client = boto3.client("sagemaker", region_name=REGION)

    ensure_model_package_group(sm_client)

    training_job_name, model_artifact_s3 = get_latest_completed_training_model(sm_client)
    evaluation_job_name, metrics_s3_uri = get_latest_completed_evaluation_metrics(sm_client)

    print("\nRegistering model with:")
    print(f"Training job:   {training_job_name}")
    print(f"Evaluation job: {evaluation_job_name}")
    print(f"Model:          {model_artifact_s3}")
    print(f"Metrics:        {metrics_s3_uri}")
    print(f"Image:          {IMAGE_URI}")

    register_model(
        sm_client=sm_client,
        model_artifact_s3=model_artifact_s3,
        metrics_s3_uri=metrics_s3_uri,
    )


if __name__ == "__main__":
    main()