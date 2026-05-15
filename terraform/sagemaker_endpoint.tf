data "external" "latest_approved_model" {
  count = var.deploy_endpoint ? 1 : 0

  program = ["python", "${path.module}/scripts/latest_approved_model.py"]

  query = {
    region                   = var.aws_region
    model_package_group_name = var.model_package_group_name
  }
}

resource "aws_sagemaker_model" "urban_green_model" {
  count = var.deploy_endpoint ? 1 : 0

  name               = "${var.project_name}-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image          = "${aws_ecr_repository.app_repo.repository_url}:latest"
    model_data_url = data.external.latest_approved_model[0].result.model_data_url

    environment = {
      MODEL_DIR = "/opt/ml/model"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "urban_green_endpoint_config" {
  count = var.deploy_endpoint ? 1 : 0

  name = "${var.project_name}-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.urban_green_model[0].name
    initial_instance_count = 1
    instance_type          = var.endpoint_instance_type
  }
}

resource "aws_sagemaker_endpoint" "urban_green_endpoint" {
  count = var.deploy_endpoint ? 1 : 0

  name                 = var.sagemaker_endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.urban_green_endpoint_config[0].name
}