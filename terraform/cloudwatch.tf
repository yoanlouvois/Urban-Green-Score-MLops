resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", aws_lambda_function.predict_lambda.function_name],
            [".", "Errors", ".", "."],
            [".", "Duration", ".", "."]
          ]
          period = 60
          stat   = "Sum"
          region = var.aws_region
          title  = "Lambda metrics"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApiGateway", "Count", "ApiId", aws_apigatewayv2_api.http_api.id],
            [".", "4xx", ".", "."],
            [".", "5xx", ".", "."]
          ]
          period = 60
          stat   = "Sum"
          region = var.aws_region
          title  = "API Gateway metrics"
        }
      }
    ]
  })
}