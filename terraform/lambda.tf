data "aws_caller_identity" "current" {}

resource "aws_iam_role" "lambda_role" {
    name = "${var.project_name}-lambda-role"

    assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Effect = "Allow"
                Principal = {
                    Service = "lambda.amazonaws.com"
                }
                Action = "sts:AssumeRole"
            }
        ]
    })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_logs" {
    role = aws_iam_role.lambda_role.name
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_policy" "lambda_invoke_sagemaker" {
  name = "${var.project_name}-lambda-invoke-sagemaker"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint"
        ]
        Resource = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:endpoint/${var.sagemaker_endpoint_name}"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_invoke_sagemaker_attach" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_invoke_sagemaker.arn
}

resource "aws_lambda_function" "predict_lambda" {
  function_name = "${var.project_name}-lambda"

  role    = aws_iam_role.lambda_role.arn
  runtime = "python3.11"
  handler = "lambda_function.lambda_handler"

  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)

  timeout     = 60
  memory_size = 512

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME = var.sagemaker_endpoint_name
    }
  }
}