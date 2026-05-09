terraform {
    required_version ">= 1.6.9"

    required_providers {
        aws = {
            source = "hashicorp/aws"
            version = "~> 6.0"
        }
    }
}

provicer "aws" {
    region = var.aws_region
}