import boto3
import os
from dotenv import load_dotenv
import datetime

load_dotenv()

ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
BUCKET = os.getenv("S3_BUCKET").strip()
REGION = os.getenv("AWS_REGION")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")

IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/urban-green-score:latest"


def main():
    sm_client = boto3.client("sagemaker", region_name=REGION)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"urban-green-train-{timestamp}"

    print("--- Envoi du Training Job via Boto3 ---")
    print(f"Job Name: {job_name}")
    print(f"Image URI: {IMAGE_URI}")

    try:
        response = sm_client.create_training_job(
            TrainingJobName=job_name,
            RoleArn=ROLE_ARN,
            StoppingCondition={
                "MaxRuntimeInSeconds": 21600
            },
            AlgorithmSpecification={
                "TrainingImage": IMAGE_URI,
                "TrainingInputMode": "File",
                "ContainerEntrypoint": [
                    "python3",
                    "src/training/train.py",
                ],
                "ContainerArguments": [
                    "--train-dir", "/opt/ml/input/data/train",
                    "--val-dir", "/opt/ml/input/data/val",
                    "--model-dir", "/opt/ml/model",
                    "--batch-size", "2",
                    "--epochs", "10",
                    "--learning-rate", "0.0001",
                    "--num-workers", "4",

                    # Mets cette ligne seulement si tu veux utiliser le subset de config.py
                    # "--use-subset",
                ],
                "MetricDefinitions": [
                    {
                        "Name": "train_loss",
                        "Regex": "Train Loss: ([0-9\\.]+)"
                    },
                    {
                        "Name": "val_loss",
                        "Regex": "Val Loss: ([0-9\\.]+)"
                    },
                ],
            },
            ResourceConfig={
                # Pour tester moins cher :
                # "InstanceType": "ml.m5.large",

                # Pour vrai entraînement GPU :
                "InstanceType": "ml.g4dn.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            InputDataConfig=[
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{BUCKET}/data/processed/train",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "InputMode": "File",
                },
                {
                    "ChannelName": "val",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{BUCKET}/data/processed/val",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "InputMode": "File",
                },
            ],
            OutputDataConfig={
                "S3OutputPath": f"s3://{BUCKET}/artifacts/models"
            },
        )

        print("\nSuccès ! Le training job a été créé.")
        print(f"ARN du Job: {response['TrainingJobArn']}")
        print("\nTu peux suivre l'avancement ici :")
        print(
            f"https://{REGION}.console.aws.amazon.com/sagemaker/home"
            f"?region={REGION}#/jobs/{job_name}"
        )

    except Exception as e:
        print(f"\nErreur lors de la création du training job : {e}")


if __name__ == "__main__":
    main()