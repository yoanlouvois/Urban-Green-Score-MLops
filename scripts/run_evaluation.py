import boto3
import os
from dotenv import load_dotenv
import datetime

load_dotenv()

ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
BUCKET = os.getenv("S3_BUCKET").strip()
REGION = os.getenv("AWS_REGION")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
MODEL_ARTIFACT_S3 = os.getenv("MODEL_ARTIFACT_S3")

IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/urban-green-score:latest"

def get_latest_completed_training_model(sm_client):
    response = sm_client.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        StatusEquals="Completed",
        MaxResults=10,
        NameContains="urban-green-train",
    )

    jobs = response.get("TrainingJobSummaries", [])

    if len(jobs) == 0:
        raise ValueError("No completed training job found.")

    latest_job_name = jobs[0]["TrainingJobName"]

    details = sm_client.describe_training_job(
        TrainingJobName=latest_job_name
    )

    model_artifact = details["ModelArtifacts"]["S3ModelArtifacts"]

    print(f"Latest completed training job: {latest_job_name}")
    print(f"Resolved model artifact: {model_artifact}")

    return model_artifact


def main():
    sm_client = boto3.client("sagemaker", region_name=REGION)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"urban-green-evaluate-{timestamp}"

    print("--- Envoi du Evaluation Job via Boto3 ---")
    print(f"Job Name: {job_name}")

    model_artifact_s3 = MODEL_ARTIFACT_S3

    if not model_artifact_s3:
        model_artifact_s3 = get_latest_completed_training_model(sm_client)

    print(f"Model artifact: {model_artifact_s3}")

    try:
        response = sm_client.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=ROLE_ARN,
            StoppingCondition={"MaxRuntimeInSeconds": 10800},
            AppSpecification={
                "ImageUri": IMAGE_URI,
                "ContainerEntrypoint": ["python3", "src/evaluation/evaluate.py"],
                "ContainerArguments": [
                    "--model-path", "/opt/ml/processing/input/model/model.tar.gz",
                    "--val-dir", "/opt/ml/processing/input/val",
                    "--output-dir", "/opt/ml/processing/output/evaluation",
                    "--batch-size", "2",
                    "--num-workers", "2",
                    # Mets cette ligne seulement si tu veux utiliser le subset de config.py
                    # "--use-subset",
                ],
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.large",
                    "VolumeSizeInGB": 20,
                }
            },
            ProcessingInputs=[
                {
                    "InputName": "model-artifact",
                    "S3Input": {
                        "S3Uri": model_artifact_s3,
                        "LocalPath": "/opt/ml/processing/input/model",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                },
                {
                    "InputName": "validation-data",
                    "S3Input": {
                        "S3Uri": f"s3://{BUCKET}/data/processed/val",
                        "LocalPath": "/opt/ml/processing/input/val",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                },
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "evaluation-output",
                        "S3Output": {
                            "S3Uri": f"s3://{BUCKET}/artifacts/evaluation/{job_name}",
                            "LocalPath": "/opt/ml/processing/output/evaluation",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
        )

        print("\nSuccès ! Le job d'évaluation a été créé.")
        print(f"ARN du Job: {response['ProcessingJobArn']}")
        print("\nTu peux suivre l'avancement ici :")
        print(
            f"https://{REGION}.console.aws.amazon.com/sagemaker/home"
            f"?region={REGION}#/processing-jobs/{job_name}"
        )

    except Exception as e:
        print(f"\nErreur lors de la création du job : {e}")


if __name__ == "__main__":
    main()