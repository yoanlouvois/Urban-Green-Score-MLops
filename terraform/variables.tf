variable "aws_region" {
    type = string
    default = "eu-west-3"
}

variable "project_name" {
    type = string
    default = "urban-green-score"
}

variable "sagemaker_endpoint_name" {
    type = string
    default = "urban-green-score-endpoint"
}

variable "lambda_zip_path" {
    type = string
    default = "../scripts/lambda/lambda.zip"
}

variable "deploy_endpoint" {
    type = bool
    default = false 
}

variable "sagemaker_execution_role_arn" {
    type = string 
}

variable "endpoint_instance_type"  {
    type = string 
    default = "ml.m5.large"
}

variable "model_package_group_name" {
    type    = string
    default = "urban-green-score-model-group"
}