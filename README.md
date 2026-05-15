# Urban Green Score - MLOps Pipeline

## Project Overview

This project implements a production-grade MLOps pipeline designed to automate the calculation of an Urban Green Score from satellite imagery. By leveraging deep learning and cloud infrastructure, the system transforms raw geospatial data into actionable environmental insights.
The core workflow includes:
- Semantic Segmentation: Utilizing a U-Net architecture to identify land cover.
- Green Scoring: Computing an environmental index based on detected vegetation (forest, agriculture, etc.).
- Automated Lifecycle: From data preprocessing to serverless inference deployment.
- The final model is accessible via a custom-built Streamlit interface, allowing users to upload satellite images and receive instant Green Score results : 

<p align="center">
  <img src="https://github.com/user-attachments/assets/667bfe54-9d4a-49d8-87b8-a6ea8d2ae855" width="45%" />
  <img src="https://github.com/user-attachments/assets/ecd3227a-e16d-443e-a5c5-137cf2e30208" width="45%" />
</p>



## MLOps Pipeline

The architecture is built for scalability and reproducibility, using industry-standard tools:

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

<img width="1530" height="566" alt="Diagramme sans nom (11)" src="https://github.com/user-attachments/assets/1c66d635-7819-43e6-b22f-e521035d0c03" />


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

