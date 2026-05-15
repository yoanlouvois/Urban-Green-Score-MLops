aws_region               = "eu-west-3"
project_name             = "urban-green-score"
sagemaker_endpoint_name  = "urban-green-score-endpoint"
lambda_zip_path          = "../scripts/lambda/lambda.zip"
deploy_endpoint          = true
endpoint_instance_type   = "ml.m5.large"
model_package_group_name = "urban-green-score-model-group"