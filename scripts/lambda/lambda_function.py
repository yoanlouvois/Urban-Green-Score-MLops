import os 
import json 
import base64
import boto3

runtime = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "urban-green-score-endpoint"
)

def lambda_handler(event, context):
    try:
        body = event.get("body")

        # API Gateway envoie souvent le body encodé
        if event.get("isBase64Encoded", False):
            image_bytes = base64.b64decode(body)
        else:
            image_bytes = body.encode()
        
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="image/png",
            Body=image_bytes,
        )

        result = response["Body"].read().decode("utf-8")

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": result,
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
    
