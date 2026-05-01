import boto3
import os
from dotenv import load_dotenv
import datetime

load_dotenv()

# Configuration
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
BUCKET = os.getenv("S3_BUCKET").strip()
REGION = os.getenv("AWS_REGION")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/urban-green-score:latest"

def main():
    # Initialisation du client de bas niveau
    sm_client = boto3.client("sagemaker", region_name=REGION)
    
    # Création d'un nom de job unique
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"urban-green-preprocess-{timestamp}"
    
    print(f"--- Envoi du Job via Boto3 (Low-Level) ---")
    print(f"Job Name: {job_name}")

    try:
        response = sm_client.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=ROLE_ARN,
            StoppingCondition={'MaxRuntimeInSeconds': 3600},
            AppSpecification={
                'ImageUri': IMAGE_URI,
                'ContainerEntrypoint': ["python3", "src/preprocessing/preprocess.py"],
                'ContainerArguments': [
                    "--raw-data-dir", "/opt/ml/processing/input/raw",
                    "--output-dir", "/opt/ml/processing/output/processed",
                    "--num-workers", "4"
                ]
            },
            ProcessingResources={
                'ClusterConfig': {
                    'InstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'VolumeSizeInGB': 20
                }
            },
            ProcessingInputs=[
                {
                    'InputName': 'input-1',
                    'S3Input': {
                        'S3Uri': f"s3://{BUCKET}/data/raw",
                        'LocalPath': '/opt/ml/processing/input/raw',
                        'S3DataType': 'S3Prefix',
                        'S3InputMode': 'File'
                    }
                }
            ],
            ProcessingOutputConfig={
                'Outputs': [
                    {
                        'OutputName': 'output-1',
                        'S3Output': {
                            'S3Uri': f"s3://{BUCKET}/data/processed",
                            'LocalPath': '/opt/ml/processing/output/processed',
                            'S3UploadMode': 'EndOfJob'
                        }
                    }
                ]
            }
        )
        
        print("\n Succès ! Le job a été créé.")
        print(f"ARN du Job: {response['ProcessingJobArn']}")
        print(f"\nTu peux suivre l'avancement ici :")
        print(f"https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/processing-jobs/{job_name}")

    except Exception as e:
        print(f"\n Erreur lors de la création du job : {e}")

if __name__ == "__main__":
    main()