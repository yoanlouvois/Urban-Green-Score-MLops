import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION")
ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "urban-green-score-endpoint",
)


def safe_call(description, func, **kwargs):
    try:
        print(f"{description}...")
        return func(**kwargs)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        message = e.response["Error"]["Message"]

        if error_code == "ValidationException":
            print(f"Skipped: {message}")
            return None

        raise


def main():
    sm_client = boto3.client("sagemaker", region_name=REGION)

    print("--- Delete SageMaker Endpoint ---")
    print(f"Region: {REGION}")
    print(f"Endpoint: {ENDPOINT_NAME}")

    endpoint = safe_call(
        "Reading endpoint",
        sm_client.describe_endpoint,
        EndpointName=ENDPOINT_NAME,
    )

    if endpoint is None:
        print("Endpoint does not exist. Nothing to delete.")
        return

    endpoint_config_name = endpoint["EndpointConfigName"]

    endpoint_config = safe_call(
        "Reading endpoint config",
        sm_client.describe_endpoint_config,
        EndpointConfigName=endpoint_config_name,
    )

    model_names = []

    if endpoint_config:
        for variant in endpoint_config["ProductionVariants"]:
            model_names.append(variant["ModelName"])

    safe_call(
        "Deleting endpoint",
        sm_client.delete_endpoint,
        EndpointName=ENDPOINT_NAME,
    )

    safe_call(
        "Deleting endpoint config",
        sm_client.delete_endpoint_config,
        EndpointConfigName=endpoint_config_name,
    )

    for model_name in model_names:
        safe_call(
            f"Deleting model {model_name}",
            sm_client.delete_model,
            ModelName=model_name,
        )

    print("\nCleanup completed.")
    print("Endpoint deleted. Billing for this endpoint should stop.")


if __name__ == "__main__":
    main()