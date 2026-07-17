# End-to-End Fraud Detection ML Pipeline on AWS

## Project Goal
Build a production-grade, end-to-end ML pipeline for real-time and batch fraud detection on AWS — covering data ingestion, feature engineering, model training, serving, monitoring, CI/CD, and infrastructure as code.

---

## Dataset

**Synthetic Card Transactions for Fraud Detection (2026)**
- **Link:** https://www.kaggle.com/datasets/larangetiwari/synthetic-card-transactions-for-fraud-detection
- **Size:** 500,000 transactions, 64MB
- **License:** MIT
- **Fraud Rate:** 0.42% (2,088 fraudulent out of 500K)
- **Cardholders:** 5,000 (5 spending profiles: commuter, family, young professional, business traveler, student)
- **Merchants:** 104 across 15 categories
- **Geography:** 24 cities across UK, US, India

### Columns (16)
| Column | Description |
|---|---|
| transaction_id | Unique transaction identifier |
| cardholder_id | Anonymous cardholder hash (CH-XXXXXXXX) |
| timestamp | Transaction datetime |
| merchant_name | Fictional merchant name (104 unique) |
| merchant_category | 15 types: grocery, restaurant, coffee, transport_rail, rideshare, retail_clothing, retail_discount, online_shopping, subscription, hotel, telecom, fuel, pharmacy, entertainment, jewellery |
| mcc | Merchant Category Code |
| merchant_city | Transaction city (24 cities) |
| merchant_state | State/region |
| merchant_country | Country code (GB, US, IN) |
| amount | Transaction amount in local currency |
| currency | GBP, USD, or INR |
| is_online | True if online transaction |
| is_recurring | True if recurring/subscription payment |
| is_international | True if cross-border transaction |
| is_fraud | 1 = fraudulent, 0 = legitimate |
| fraud_type | Type of fraud (if fraudulent) |

### 5 Fraud Patterns
1. **Card Testing** — Small rapid transactions at random merchants (avg £2.75)
2. **Account Takeover** — Sudden spending spikes in unusual categories (avg £1,082)
3. **Geographic Impossibility** — In-store transactions in two countries within 30 minutes
4. **Bust-Out** — Gradual spending escalation then max-out over 14 days
5. **Card-Not-Present (CNP)** — Online purchases from unusual international locations (avg £428)

---

## Architecture Overview

```
Kaggle CSV → S3 Raw → Kinesis (stream simulation) → Lambda (processor) → S3 Processed
                                                                              ↓
                                                              Glue/SageMaker Processing
                                                                              ↓
                                                                    SageMaker Feature Store
                                                                     (Online + Offline)
                                                                              ↓
                                                                    SageMaker Pipeline
                                                              (Validate → Train → Evaluate)
                                                                              ↓
                                                                    Model Registry
                                                                     (Approval Gate)
                                                                              ↓
                                                    ┌───────────────────────────┴───────────────────────────┐
                                              SageMaker Endpoint                              Batch Transform
                                              (Real-time, <100ms)                             (Nightly rescore)
                                                      ↓                                              ↓
                                              API Gateway + Lambda                              S3 Results
                                                      ↓                                              ↓
                                              DynamoDB (decisions)                            Athena (analysis)
                                                      ↓
                                              SNS (fraud alerts)

                                              Model Monitor → EventBridge → Auto-Retrain
                                              CloudWatch → Dashboards + Alarms
```

---

## Component Details

### Layer 1: Data Ingestion
| Component | AWS Service | Purpose |
|---|---|---|
| Raw storage | S3 | Store raw CSV, partitioned by date |
| Stream simulation | Kinesis Data Streams (1 shard) | Replay CSV rows as real-time events |
| Stream consumer | Lambda | Parse, validate, write to processed S3 |
| Schema validation | Glue Schema Registry | Enforce transaction schema |

### Layer 2: Feature Engineering
| Component | AWS Service | Purpose |
|---|---|---|
| Batch features | SageMaker Processing or Glue | Compute aggregated features |
| Feature store | SageMaker Feature Store | Online (real-time lookup) + Offline (training) |

#### Features to Engineer
```
- txn_amount_vs_cardholder_avg       (ratio)
- txn_count_last_1h / 6h / 24h      (velocity)
- time_since_last_txn_seconds        (recency)
- is_new_merchant                    (boolean)
- is_new_city                        (boolean)
- category_frequency_deviation       (z-score)
- geo_velocity_kmh                   (impossible travel)
- online_ratio_last_7d               (channel shift)
- spending_acceleration_7d           (bust-out signal)
```

### Layer 3: Training Pipeline
| Component | AWS Service | Purpose |
|---|---|---|
| Orchestration | SageMaker Pipelines | DAG of training steps |
| Training | SageMaker Training Job | Managed compute (spot instances) |
| Experiment tracking | SageMaker Experiments or MLflow on ECS | Log params, metrics, artifacts |
| Model registry | SageMaker Model Registry | Version control, approval workflow |
| Data validation | Great Expectations | Schema + distribution checks |

**Training flow:**
```
Fetch Features → Data Validation → Train/Val/Test Split → Train XGBoost → Evaluate (PR-AUC ≥ 0.85?) → Register Model → Approve → Deploy
```

### Layer 4: Model Serving
| Component | AWS Service | Purpose |
|---|---|---|
| Real-time endpoint | SageMaker Endpoint (ml.m5.large) | Auto-scaling 1-4 instances |
| Batch scoring | SageMaker Batch Transform | Nightly full-table rescore |
| API layer | API Gateway + Lambda | Auth, rate limiting, request validation |
| Results store | DynamoDB | Low-latency fraud decision lookup |

### Layer 5: Monitoring & Retraining
| What to Monitor | Tool | Threshold |
|---|---|---|
| Data drift | SageMaker Model Monitor | PSI > 0.2 on any feature |
| Prediction drift | Model Monitor | Fraud rate deviates > 2σ |
| Latency P99 | CloudWatch | > 200ms |
| Error rate | CloudWatch | > 1% 5xx |
| Model staleness | EventBridge | No retrain in 7 days |

**Auto-retraining:** Model Monitor detects drift → EventBridge → SageMaker Pipeline → New model → Pending approval

### Layer 6: CI/CD & Infrastructure
| Component | Tool | Purpose |
|---|---|---|
| CI | GitHub Actions | Lint (ruff), test (pytest), build Docker, push ECR |
| CD | GitHub Actions | Deploy infra (Terraform), update endpoint, smoke test |
| IaC | Terraform or CDK | All AWS resources as code |
| Containerization | Docker + ECR | Reproducible training and serving environments |

---

## Project Structure
```
fraud-detection-pipeline/
├── infrastructure/              # Terraform or CDK
│   ├── s3.tf
│   ├── kinesis.tf
│   ├── sagemaker.tf
│   ├── lambda.tf
│   ├── api_gateway.tf
│   └── monitoring.tf
├── src/
│   ├── ingestion/               # Kinesis consumer Lambda
│   ├── features/                # Feature engineering code
│   ├── training/                # Training script + pipeline definition
│   ├── serving/                 # Inference script + API handler
│   └── monitoring/              # Drift detection + alerting
├── tests/
│   ├── unit/
│   └── integration/
├── notebooks/                   # EDA and experimentation
├── Dockerfile
├── Makefile
├── requirements.txt
└── .github/workflows/
    ├── ci.yml                   # lint, test, build
    └── cd.yml                   # deploy to dev/staging/prod
```

---

## AWS Cost Estimate (Dev/Learning)

| Service | Monthly Cost |
|---|---|
| S3 (< 1GB) | ~$0.02 |
| Kinesis (1 shard) | ~$15 |
| Lambda (low volume) | Free tier |
| SageMaker Training (spot) | ~$5-10/run |
| SageMaker Endpoint (hours only) | ~$2-5 |
| DynamoDB (on-demand) | ~$1 |
| Glue (on-demand) | ~$1-2/run |
| **Total** | **~$25-35/month** |

---

## Build Phases

| Phase | What to Build | Focus Area |
|---|---|---|
| **Phase 1** | S3 ingestion + feature engineering + local training script | Data engineering, feature stores |
| **Phase 2** | SageMaker Pipeline + Model Registry + Endpoint | MLOps, orchestration |
| **Phase 3** | Kinesis streaming + real-time scoring + DynamoDB | Streaming, event-driven architecture |
| **Phase 4** | Monitoring, drift detection, auto-retraining | Production monitoring |
| **Phase 5** | Terraform IaC + CI/CD + API Gateway | DevOps, infrastructure as code |
| **Phase 6** | Dashboards, documentation, load testing | Production readiness |

---

## Skills Covered

| DS Already Knows | ML Engineer Skills Learned |
|---|---|
| Model building | Containerization (Docker) |
| Jupyter notebooks | Production code (modular, tested, typed) |
| Local experimentation | Infrastructure as Code (Terraform/CDK) |
| Accuracy metrics | Latency, throughput, cost metrics |
| One-off training | Automated retraining pipelines |
| pandas/sklearn | Feature Store patterns |
| Manual deployment | CI/CD (GitHub Actions) |
| print() debugging | Logging, monitoring, alerting (CloudWatch) |

---

## Alternative/Supplementary Dataset

**PaySim** (for volume/scale testing)
- **Link:** https://www.kaggle.com/datasets/ealaxi/paysim1
- **Size:** 6.3M transactions
- **Use case:** Combine with Synthetic Card Txns if you want to test at higher scale
