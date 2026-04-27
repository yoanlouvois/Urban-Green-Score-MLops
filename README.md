# urban-green-score-mlops
End-to-end MLOps pipeline to compute an Urban Green Score from satellite imagery using Pytorch, Docker, AWS Sagemaker, S3, Terraform and Cloudwatch

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
