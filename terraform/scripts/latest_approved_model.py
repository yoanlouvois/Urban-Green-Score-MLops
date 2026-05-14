import sys
import json
import boto3

query = json.load(sys.stdin)

region = query["region"]
model_package_group_name = query["model_package_group_name"]

sm = boto3.client("sagemaker", region_name=region)

response = sm.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)

packages = response.get("ModelPackageSummaryList", [])

if not packages:
    raise RuntimeError(f"No Approved model found in {model_package_group_name}")

model_package_arn = packages[0]["ModelPackageArn"]

details = sm.describe_model_package(
    ModelPackageName=model_package_arn
)

container = details["InferenceSpecification"]["Containers"][0]
model_data_url = container["ModelDataUrl"]

print(json.dumps({
    "model_package_arn": model_package_arn,
    "model_data_url": model_data_url
}))