output "api_gateway_url" {
  value = "${aws_apigatewayv2_api.http_api.api_endpoint}/predict"
}

output "lambda_name" {
  value = aws_lambda_function.predict_lambda.function_name
}

output "cloudwatch_dashboard_name" {
  value = aws_cloudwatch_dashboard.main.dashboard_name
}