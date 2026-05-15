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

### Prerequisites

To run this project, you will need:

- An AWS account with permissions to use SageMaker, Lambda, API Gateway, IAM, CloudWatch, ECR, and S3
- Docker installed on your local machine
- Terraform installed for infrastructure provisioning
- Python 3.11+
- AWS CLI installed and configured
- GitHub account (optional, for CI/CD workflows)


### Clone the Repository

First, clone the repository and navigate into the project folder:
```bash
git clone https://github.com/<your-username>/Urban-Green-Score-MLops.git
cd Urban-Green-Score-MLops
```


### Configure AWS CLI

Authenticate your AWS account locally:
```bash
aws configure
```


### Terraform Infrastructure Setup

Navigate to the Terraform folder:
```bash
cd terraform
```
Update the `terraform.tfvars` file with your configuration.

Then initialize Terraform:
```bash
terraform init
```
Preview infrastructure changes:
```bash
terraform plan
```
Deploy infrastructure:
```bash
terraform apply
```
This will provision the following AWS resources:

- S3 bucket for datasets and model artifacts
- ECR repository for Docker images
- IAM roles and policies
- SageMaker model + endpoint configuration
- SageMaker endpoint (optional via deploy_endpoint flag)
- Lambda function
- API Gateway
- CloudWatch dashboard


### Environment Variables

Create a `.env` file at the root of the project using the provided template:
```bash
cp .env.example .env
```
Then configure your AWS and project variables inside `.env`.


### Local Python Environment

Create and activate a local virtual environment to run the SageMaker scripts, Streamlit app, and utility scripts locally.

Create the virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-local.txt
```
This installs SageMaker SDK, boto3, Streamlit, pytest and other local development utilities


### Dataset Setup

Download the dataset manually and place it inside the data/raw/ folder.
Current project structure expects something similar to:
```
data/
  raw/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
```
Then upload the raw dataset to your S3 bucket:
```
aws s3 cp data/raw s3://your-bucket-name/data/raw --recursive
```

### Docker Authentication (AWS ECR)

Authenticate Docker with your private ECR repository:
```
aws ecr get-login-password --region eu-west-3 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.eu-west-3.amazonaws.com
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

