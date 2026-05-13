resource "aws_s3_bucket" "ml_bucket" {
  bucket = "urban-green-score"

  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name = "urban-green-score"
  }
}