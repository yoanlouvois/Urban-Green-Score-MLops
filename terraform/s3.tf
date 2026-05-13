resource "aws_s3_bucket" "ml_bucket" {
  bucket = "urban-green-score"

  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name = "urban-green-score"
  }
}

resource "aws_s3_bucket_public_access_block" "ml_bucket_block" {
  bucket = aws_s3_bucket.ml_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}