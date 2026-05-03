import os
import time
import datetime
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
REGION = os.getenv("AWS_REGION")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")

MODEL_PACKAGE_GROUP_NAME = os.getenv(
    "MODEL_PACKAGE_GROUP_NAME",
    "urban-green-score-model-group",
)

ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "urban-green-score-endpoint",
)

INSTANCE_TYPE = os.getenv(
    "SAGEMAKER_ENDPOINT_INSTANCE_TYPE",
    "ml.m5.large",
)

INITIAL_INSTANCE_COUNT = int(os.getenv("SAGEMAKER_ENDPOINT_INSTANCE_COUNT", "1"))

IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/urban-green-score:latest"


def get_latest_approved_model_package(sm_client):
    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )

    packages = response.get("ModelPackageSummaryList", [])

    if not packages:
        raise ValueError(
            f"No Approved model found in model package group: {MODEL_PACKAGE_GROUP_NAME}"
        )

    model_package_arn = packages[0]["ModelPackageArn"]

    details = sm_client.describe_model_package(
        ModelPackageName=model_package_arn
    )

    container = details["InferenceSpecification"]["Containers"][0]
    model_data_url = container["ModelDataUrl"]

    print(f"Latest approved model package: {model_package_arn}")
    print(f"Model data URL: {model_data_url}")

    return model_package_arn, model_data_url


def endpoint_exists(sm_client, endpoint_name):
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def wait_for_endpoint(sm_client, endpoint_name):
    print(f"\nWaiting for endpoint: {endpoint_name}")

    while True:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]

        print(f"Endpoint status: {status}", flush=True)

        if status == "InService":
            print("\nEndpoint is InService.")
            return

        if status in ["Failed", "OutOfService"]:
            reason = response.get("FailureReason", "Unknown")
            raise RuntimeError(f"Endpoint failed: {reason}")

        time.sleep(30)


def main():
    sm_client = boto3.client("sagemaker", region_name=REGION)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    model_name = f"urban-green-model-{timestamp}"
    endpoint_config_name = f"urban-green-endpoint-config-{timestamp}"

    print("--- Deploy SageMaker Endpoint ---")
    print(f"Region: {REGION}")
    print(f"Model package group: {MODEL_PACKAGE_GROUP_NAME}")
    print(f"Endpoint name: {ENDPOINT_NAME}")
    print(f"Instance type: {INSTANCE_TYPE}")
    print(f"Image URI: {IMAGE_URI}")

    model_package_arn, model_data_url = get_latest_approved_model_package(sm_client)

    print("\nCreating SageMaker Model...")
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "MODEL_DIR": "/opt/ml/model",
            },
        },
    )

    print(f"Model created: {model_name}")

    print("\nCreating Endpoint Config...")
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": INITIAL_INSTANCE_COUNT,
                "InstanceType": INSTANCE_TYPE,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    print(f"Endpoint config created: {endpoint_config_name}")

    if endpoint_exists(sm_client, ENDPOINT_NAME):
        print("\nEndpoint already exists. Updating endpoint...")
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name,
        )
    else:
        print("\nCreating endpoint...")
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name,
        )

    print("\nDeployment started.")
    print(
        f"https://{REGION}.console.aws.amazon.com/sagemaker/home"
        f"?region={REGION}#/endpoints/{ENDPOINT_NAME}"
    )

    wait_for_endpoint(sm_client, ENDPOINT_NAME)


if __name__ == "__main__":
    main()