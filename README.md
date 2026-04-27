# Urban Green Score - MLOps Pipeline

End-to-end MLOps pipeline to compute an **Urban Green Score** from satellite imagery using:

- PyTorch (semantic segmentation - U-Net)
- Docker (reproducible environments)
- AWS (S3, SageMaker, CloudWatch)
- Terraform (infrastructure as code)

## Project Overview

This project takes satellite images and:

1. Preprocesses raw data
2. Trains a segmentation model (U-Net)
3. Evaluates predictions using segmentation metrics
4. Computes a **Green Score** based on land usage (forest, agriculture, etc.)


## Setup

```bash
docker build -t urban-green .
```

### Run preprocessing

```bash
docker run --rm -v ${PWD}:/app urban-green python src/preprocessing/preprocess.py
```

### Run training

```bash
docker run --rm -v ${PWD}:/app urban-green python src/training/train.py
```

### Run evaluation 

```bash
docker run --rm -v ${PWD}:/app urban-green python src/evaluation/evaluate.py
```

### Test Green Score

```bash
docker run --rm -v ${PWD}:/app urban-green python src/inference/predict.py --image-path data/raw/test/<your image>
```
