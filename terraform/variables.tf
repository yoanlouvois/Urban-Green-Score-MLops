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
