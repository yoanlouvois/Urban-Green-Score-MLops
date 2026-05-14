# Urban Green Score - MLOps Pipeline

End-to-end MLOps pipeline to compute an **Urban Green Score** from satellite imagery using:

- **PyTorch** → semantic segmentation model (**U-Net**)
- **Docker** → reproducible training/inference environments
- **AWS**
  - S3 → dataset & model artifact storage
  - ECR → container registry
  - SageMaker → processing, training, evaluation, deployment
  - CloudWatch → logs & monitoring
  - Lambda → serverless inference proxy
  - API Gateway → HTTP endpoint exposure
- **Terraform** → infrastructure as code
- **Github Actions** → CI/CD pipelines

The following diagram illustrates the system architecture : 

<img width="1530" height="566" alt="Diagramme sans nom (9)" src="https://github.com/user-attachments/assets/951a5c2d-4be9-4d34-810c-5b5ac552c287" />



## Project Overview

This project takes satellite images and:

1. Preprocesses raw data
2. Trains a segmentation model (U-Net)
3. Evaluates predictions using segmentation metrics
4. Computes a **Green Score** based on land usage (forest, agriculture, etc.)


## Setup

First, clone the repository and navigate into the project directory

### 1. AWS Authentication

Configure your AWS credentials locally:

```bash
aws configure
```

You will need:

- AWS Access Key
- AWS Secret Key
- Region: `eu-west-3`
- Output format: `json`

---

### 2. Local Python Environment

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
.\venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install local dependencies:

```bash
pip install -r requirements-local.txt
```

This installs:

- SageMaker SDK
- boto3
- Streamlit
- requests
- local utilities

---

### 3. Docker Authentication (AWS ECR)

Login to your private ECR repository:

```bash
aws ecr get-login-password --region eu-west-3 | docker login --username AWS --password-stdin 147914447581.dkr.ecr.eu-west-3.amazonaws.com
```

---

# Build & Push Docker Image

Build the Docker image used by SageMaker:

```bash
docker build --platform linux/amd64 --provenance=false -t urban-green-score .
```

Tag the image:

```bash
docker tag urban-green-score:latest 147914447581.dkr.ecr.eu-west-3.amazonaws.com/urban-green-score:latest
```

Push the image to AWS ECR:

```bash
docker push 147914447581.dkr.ecr.eu-west-3.amazonaws.com/urban-green-score:latest
```

---

# SageMaker Pipeline

## 1. Data Processing

Run SageMaker Processing job:

```bash
python src/scripts/run_processing.py
```

This step:

- downloads raw data
- preprocesses satellite images
- resizes images
- prepares masks
- uploads processed dataset to S3

---

## 2. Model Training

Launch SageMaker Training job:

```bash
python src/scripts/run_training.py
```

This step:

- trains the U-Net segmentation model
- stores model artifacts in S3
- logs training metrics in CloudWatch

---

## 3. Model Evaluation

Launch SageMaker Evaluation job:

```bash
python src/scripts/run_evaluation.py
```

This step:

- evaluates model performance
- computes IoU / accuracy metrics
- saves evaluation artifacts

---

# Model Registry

Register the trained model:

```bash
python src/scripts/model_registry/register_model.py
```

This creates a new version inside SageMaker Model Registry.

---

# Deploy Endpoint

Deploy the latest approved model to SageMaker Endpoint:

```bash
python src/scripts/endpoint/deploy_endpoint.py
```

This creates:

- SageMaker Model
- Endpoint Configuration
- Real-time Endpoint

---

# Test Endpoint Inference

Test endpoint directly:

```bash
python src/scripts/endpoint/invoke_endpoint.py
```

This returns:

- green score
- class proportions
- segmentation mask

---

# Lambda Deployment

Package Lambda function:

```bash
Compress-Archive -Path lambda/lambda_function.py -DestinationPath lambda/lambda.zip -Force
```

Then upload `lambda.zip` manually in AWS Lambda.

This Lambda acts as a proxy between API Gateway and SageMaker Endpoint.

---

# API Gateway Inference

Architecture:

Client → API Gateway → Lambda → SageMaker Endpoint

Test API Gateway:

```bash
python src/scripts/api/invoke_api.py
```

---

# Streamlit Demo (Optional)

Run local UI:

```bash
streamlit run src/app/streamlit_app.py
```

Features:

- upload satellite image
- local inference mode
- cloud inference mode
- segmentation mask visualization
- green score display

---

# Cleanup (Important)

Delete the SageMaker endpoint after testing to avoid unnecessary costs:

```bash
python src/scripts/endpoint/delete_endpoint.py
```

