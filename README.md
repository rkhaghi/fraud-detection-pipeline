# Fraud Detection ML Pipeline on AWS

An end-to-end, production-grade fraud detection system built on AWS — covering data ingestion, feature engineering, model training, real-time serving, monitoring, and CI/CD.

📖 **Full walkthrough:** [Building an End-to-End Fraud Detection ML Pipeline on AWS](https://medium.com/@rkhaghi/building-an-end-to-end-fraud-detection-ml-pipeline-on-aws-2f434108316b)

---

## Architecture

```
Raw CSV (S3)
    ↓
Feature Engineering    (SageMaker Processing Job)
    ↓
XGBoost Training       (SageMaker Training Job, spot instances)
    ↓
Evaluation Gate        (PR-AUC ≥ 0.85)
    ↓
Model Registry         (PendingManualApproval)
    ↓
SageMaker Endpoint     (real-time, <100ms)
    ↓
SQS → Lambda           (real-time scoring stream)
    ↓
DynamoDB + SNS         (audit trail + fraud alerts)
    ↓
Model Monitor          (daily drift detection)
    ↓
GitHub Actions         (CI/CD → automated retraining)
```

---

## Dataset

[Synthetic Card Transactions for Fraud Detection](https://www.kaggle.com/datasets/larangetiwari/synthetic-card-transactions-for-fraud-detection) — 500,000 transactions across UK, US, and India with 0.42% fraud rate and 5 fraud patterns.

---

## Project Structure

```
src/
├── engineering.py          # Feature engineering (shared across jobs)
├── sm_processing.py        # SageMaker Processing script (feature eng + split)
├── sm_train.py             # SageMaker Training script (XGBoost)
├── sm_evaluate.py          # SageMaker Evaluation script (PR-AUC on test set)
├── serving/
│   ├── inference.py        # Feature vector builder for Lambda
│   └── lambda_handler.py   # SQS → score → DynamoDB + SNS
└── streaming/
    └── producer.py         # CSV replay to SQS (simulation)

notebook/
├── pipeline.ipynb          # Define + run SageMaker Pipeline
├── deploy.ipynb            # Deploy endpoint
└── monitoring.ipynb        # Model Monitor + CloudWatch dashboard

infrastructure/
└── setup_phase3.py         # Provision SQS, DynamoDB, SNS, Lambda

tests/
└── test_inference.py       # Unit tests for inference feature vector

.github/workflows/
├── ci.yml                  # Lint (ruff) + test (pytest) on push
└── cd.yml                  # Upload scripts to S3 + start pipeline
```

---

## Engineered Features

| Feature | Detects |
|---------|---------|
| `time_since_last_txn_seconds` | Card Testing |
| `txn_count_last_1h / 6h / 24h` | Card Testing, Account Takeover |
| `amount_vs_cardholder_avg` | Account Takeover |
| `geo_velocity_kmh` | Geographic Impossibility |
| `is_new_merchant / is_new_city` | Account Takeover, CNP |
| `online_ratio_last_7d` | Card-Not-Present Fraud |
| `spending_acceleration_7d` | Bust-Out Fraud |

---

## AWS Services

`S3` · `SageMaker Pipelines` · `SageMaker Processing` · `SageMaker Training` · `SageMaker Endpoint` · `SageMaker Model Registry` · `SageMaker Model Monitor` · `Lambda` · `SQS` · `DynamoDB` · `SNS` · `CloudWatch` · `IAM`

---

## CI/CD

Pushes to the `development` branch trigger:
1. **CI** — `ruff` lint + `pytest` unit tests
2. **CD** (on CI pass) — upload scripts to S3 + start SageMaker Pipeline

---

## Running Locally

```bash
pip install -r requirements.txt -r requirements-dev.txt

# Run feature engineering smoke test
python src/engineering.py

# Run unit tests
pytest tests/ -v

# Replay transactions to SQS
python src/streaming/producer.py --limit 100 --rate 5
```

---

## Key Design Decisions

- **PR-AUC over ROC-AUC** — more honest metric for 0.42% fraud rate
- **`scale_pos_weight`** — handles class imbalance natively in XGBoost (238:1 ratio)
- **Time-based train/val/test split** — prevents data leakage in financial data
- **`feature_columns.json`** saved with model — ensures identical column order at inference time
- **Quality gate** — model only registers if PR-AUC ≥ 0.85 on held-out test set
