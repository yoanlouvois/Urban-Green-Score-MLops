# urban-green-score-mlops
End-to-end MLOps pipeline to compute an Urban Green Score from satellite imagery using AWS

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run preprocessing
```bash
python src/preprocessing/preprocess.py
```
