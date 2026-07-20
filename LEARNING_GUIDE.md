# Fraud Detection ML Pipeline — Learning Guide

A complete reference for everything built across Phase 1, 2, and 3.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1 — Local Pipeline](#2-phase-1--local-pipeline)
   - [Feature Engineering](#21-feature-engineering-engineeringpy)
   - [Training](#22-training-trainpy)
   - [Interpretation](#23-interpretation-interpretpy)
3. [Phase 2 — SageMaker Pipeline](#3-phase-2--sagemaker-pipeline)
   - [Key Concepts](#31-key-concepts)
   - [Processing Script](#32-processing-script-sm_processingpy)
   - [Training Script](#33-training-script-sm_trainpy)
   - [Evaluation Script](#34-evaluation-script-sm_evaluatepy)
   - [Pipeline Notebook](#35-pipeline-notebook)
4. [Phase 3 — Streaming Architecture](#4-phase-3--streaming-architecture)
   - [Key Concepts](#41-key-concepts)
   - [Lambda Handler](#42-lambda-handler)
   - [Inference Helper](#43-inference-helper)
5. [AWS Services Used](#5-aws-services-used)
6. [Python Concepts](#6-python-concepts)
7. [Key Commands Reference](#7-key-commands-reference)

---

## 1. Project Overview

We built a production-grade fraud detection system that:
- Trains an XGBoost model on 500K card transactions (PR-AUC = 0.91)
- Runs as a managed AWS pipeline (SageMaker)
- Scores transactions in real time (<100ms) via REST endpoint
- Streams transactions through SQS → Lambda → DynamoDB
- Sends fraud alerts via SNS email

```
CSV Data → Feature Engineering → XGBoost Training → Model Registry
                                                           ↓
Real-time Transaction → SQS → Lambda → SageMaker Endpoint → DynamoDB → SNS Alert
```

---

## 2. Phase 1 — Local Pipeline

### 2.1 Feature Engineering (`engineering.py`)

**Purpose:** Transform raw transaction columns into predictive signals.

---

#### `df.sort_values(['cardholder_id', 'timestamp'])`

Sorts the entire DataFrame by cardholder, then by time within each cardholder.

**Why:** All rolling window calculations depend on correct time ordering. If rows are out of order, `txn_count_last_1h` would count wrong transactions.

```python
df = df.sort_values(['cardholder_id', 'timestamp']).reset_index(drop=True)
```

`reset_index(drop=True)` — after sorting, the row numbers (index) are scrambled. This resets them to 0, 1, 2, 3... `drop=True` means don't save the old index as a column.

---

#### `df.groupby('cardholder_id').diff()`

Computes the difference between consecutive rows **within each cardholder group**.

```python
df['time_since_last_txn_seconds'] = (
    df.groupby('cardholder_id')['timestamp']
    .diff()                    # None for first row, timedelta for rest
    .dt.total_seconds()        # convert timedelta to float seconds
    .fillna(-1)                # -1 = "this is the cardholder's first transaction"
)
```

**What `groupby().diff()` does:**
- Groups rows by `cardholder_id`
- For each row, subtracts the previous row's value within that group
- Returns `NaN` for the first row of each group (no previous row)

**Example:**
```
cardholder  timestamp          → diff result
CH-001      2024-01-01 09:00   → NaN (first tx)
CH-001      2024-01-01 09:02   → 2 minutes
CH-001      2024-01-01 14:30   → 5.5 hours
CH-002      2024-01-01 10:00   → NaN (new cardholder, reset)
```

---

#### `df.set_index('timestamp')` + `x.rolling('1h').count()`

Time-based rolling windows require the timestamp to be the DataFrame index.

```python
df = df.set_index('timestamp')   # timestamp becomes the row label

df['txn_count_last_1h'] = (
    df.groupby('cardholder_id')['amount']
    .transform(lambda x: x.rolling('1h').count() - 1)
    # - 1 excludes the current transaction from its own count
)

df = df.reset_index()  # move timestamp back to a column
```

**`rolling('1h')`** — a time-based window that looks back 1 hour from each row.

**`transform()`** vs **`apply()`:**
- `apply()` returns one result per group
- `transform()` returns one result per **row**, same shape as input — this is what we need to add a column back to the DataFrame

---

#### `x.expanding().mean().shift(1)`

Expanding mean = average of ALL previous rows for this cardholder.
`shift(1)` = exclude the current transaction from its own average.

```python
cardholder_avg = (
    df.groupby('cardholder_id')['amount']
    .transform(lambda x: x.expanding().mean().shift(1))
)
df['amount_vs_cardholder_avg'] = df['amount'] / cardholder_avg.fillna(df['amount'])
```

**Why `shift(1)`:** Without it, the model would know the current transaction's amount when computing its own "unusual amount" signal — data leakage.

**`fillna(df['amount'])`:** For the very first transaction of a cardholder, the expanding mean is NaN (no history). Filling with `df['amount']` makes the ratio = 1.0 (perfectly average).

---

#### `df.groupby(['cardholder_id', 'merchant_name']).cumcount()`

Counts how many times each cardholder has visited each merchant, cumulatively.

```python
df['merchant_cumcount'] = df.groupby(['cardholder_id', 'merchant_name']).cumcount()
df['is_new_merchant'] = (df['merchant_cumcount'] == 0).astype(int)
```

`cumcount()` returns 0 for the first occurrence, 1 for the second, etc. So `== 0` means "first time at this merchant" = new merchant.

---

#### `geopy.distance.geodesic`

Computes the real-world distance between two lat/lon coordinates using the geodesic (shortest path over Earth's surface).

```python
from geopy.distance import geodesic

dist_km = geodesic(
    (row['lat'], row['lon']),
    (row['prev_lat'], row['prev_lon'])
).km
```

**Geographic impossibility detection:** London → Mumbai = ~7,200 km. If the previous transaction was 30 minutes ago, the implied speed is 7200 / 0.5 = 14,400 km/h — impossible, therefore likely fraud.

---

### 2.2 Training (`train.py`)

#### Time-based train/val/test split

```python
train = df[df['timestamp'] <= '2024-04-30']
val   = df[(df['timestamp'] > '2024-04-30') & (df['timestamp'] <= '2024-05-31')]
test  = df[df['timestamp'] > '2024-05-31']
```

**Why not random split:** In production, you always predict the future from the past. A random split would allow future transaction data to appear in the training set, making the model look artificially better (data leakage).

---

#### `pd.get_dummies()`

One-hot encodes categorical columns.

```python
X = pd.get_dummies(X, columns=['merchant_category', 'merchant_country', 'currency'])
```

**Before:**
```
merchant_category
grocery
coffee
restaurant
```

**After:**
```
merchant_category_grocery  merchant_category_coffee  merchant_category_restaurant
1                          0                         0
0                          1                         0
0                          0                         1
```

**Why:** XGBoost needs numbers, not strings.

---

#### `XGBClassifier` key parameters

```python
model = XGBClassifier(
    n_estimators=500,          # number of trees (more = more accurate but slower)
    max_depth=6,               # how deep each tree can grow (deeper = more complex patterns)
    learning_rate=0.05,        # how much each tree corrects the previous (lower = slower but better)
    subsample=0.8,             # use 80% of rows per tree (reduces overfitting)
    colsample_bytree=0.8,      # use 80% of features per tree (reduces overfitting)
    scale_pos_weight=238,      # weight for fraud class: neg/pos ratio (handles 0.42% fraud rate)
    eval_metric='aucpr',       # optimise for Precision-Recall AUC (correct for imbalanced data)
    early_stopping_rounds=20,  # stop if val score doesn't improve for 20 rounds
)
```

**`scale_pos_weight`** is the most important parameter for fraud detection.
- Without it: model learns "always predict legit" → 99.58% accuracy, 0% fraud caught
- With it (238): fraud cases are weighted 238× more heavily in the loss function

**PR-AUC vs ROC-AUC:**
- ROC-AUC is misleading with extreme class imbalance (0.42%)
- PR-AUC measures Precision-Recall tradeoff — the right metric for fraud

---

#### `precision_recall_curve` + optimal threshold

```python
precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_threshold = thresholds[f1_scores.argmax()]
```

The model outputs a probability (0.0 → 1.0). We need to choose a **threshold** to convert that to a binary fraud/legit decision.

- High threshold (0.9): very precise (few false alarms) but misses many frauds
- Low threshold (0.1): catches most fraud but many false alarms
- Optimal: threshold that maximises F1 = balance of precision and recall

---

### 2.3 Interpretation (`interpret.py`)

#### SHAP values

```python
import shap
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, plot_type='bar')
```

SHAP (SHapley Additive exPlanations) answers: **"Why did the model predict fraud for this transaction?"**

Each feature gets a SHAP value = how much that feature pushed the prediction toward fraud (positive) or toward legit (negative).

**Example:** For a transaction with geo_velocity_kmh = 15,000:
- SHAP value for geo_velocity_kmh might be +2.3 (strongly pushed toward fraud)
- SHAP value for amount might be -0.1 (slightly pushed toward legit)

---

## 3. Phase 2 — SageMaker Pipeline

### 3.1 Key Concepts

#### SageMaker Session vs PipelineSession

```python
session          = sagemaker.Session()      # regular session — runs jobs immediately
pipeline_session = PipelineSession()        # pipeline session — captures args, doesn't run
```

When you call `processor.run()` with a regular `Session`, it **immediately executes** the processing job. With `PipelineSession`, it **captures the arguments** as a pipeline step definition without running anything.

**Rule:** Use `PipelineSession` for anything that becomes a pipeline step. Use `Session` only for the final `pipeline.upsert()` and `pipeline.start()` calls.

---

#### SageMaker container paths (the "contract")

Every SageMaker container has fixed paths for inputs and outputs:

| Path | Purpose |
|---|---|
| `/opt/ml/input/data/<channel>/` | Training input data |
| `/opt/ml/model/` | Where training saves the model |
| `/opt/ml/processing/input/<name>/` | Processing job inputs |
| `/opt/ml/processing/output/<name>/` | Processing job outputs |

These never change. Your script just reads/writes to these paths and SageMaker handles the S3 ↔ container data movement automatically.

---

#### Pipeline property references

```python
step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
```

This is a **lazy reference** — it doesn't know the actual S3 path until the pipeline runs. SageMaker resolves it at execution time.

**Why:** Each pipeline execution creates a new unique folder in S3. You can't hardcode the path when defining the pipeline — it doesn't exist yet.

---

#### `pipeline.upsert()` vs `pipeline.start()`

```python
pipeline.upsert(role_arn=role)   # creates/updates the pipeline definition in AWS
execution = pipeline.start()      # kicks off one execution of the pipeline
```

- `upsert()` = "save the blueprint"
- `start()` = "run the blueprint now"

You can call `upsert()` many times to update the pipeline, then `start()` as many times as you want to run it.

---

### 3.2 Processing Script (`sm_processing.py`)

```python
subprocess.run(['pip', 'install', 'geopy', '--quiet'], check=True)
```

Installing packages at script start. The SageMaker sklearn container doesn't include `geopy` by default. `check=True` raises an exception if pip fails.

```python
sys.path.insert(0, '/opt/ml/processing/input/scripts')
from engineering import compute_features
```

`sys.path` is Python's list of directories to search when importing modules. By inserting our scripts directory at position 0 (highest priority), we can import `engineering.py` as if it were a standard library.

---

### 3.3 Training Script (`sm_train.py`)

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n-estimators', type=int, default=500)
args = parser.parse_args()
```

`argparse` lets SageMaker pass hyperparameters as command-line arguments:

```
python sm_train.py --n-estimators 500 --max-depth 6 --learning-rate 0.05
```

SageMaker does this automatically based on the `hyperparameters` dict in the `XGBoost` estimator.

```python
MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
```

SageMaker sets environment variables automatically. `SM_MODEL_DIR` points to where the model should be saved. `os.environ.get(key, default)` returns the default if the variable isn't set (useful for local testing).

---

### 3.4 Evaluation Script (`sm_evaluate.py`)

```python
import tarfile
with tarfile.open(model_tar) as tar:
    tar.extractall(MODEL_DIR)
```

SageMaker training packages `/opt/ml/model/` into `model.tar.gz` before uploading to S3. When this is mounted in the evaluation step, we must extract it first to access `model.json`.

```python
evaluation = {
    'binary_classification_metrics': {
        'average_precision': {
            'value': round(pr_auc, 4),
            'standard_deviation': 'NaN',
        }
    }
}
```

This exact JSON structure is required by SageMaker's `ConditionStep` + `JsonGet` to read the PR-AUC value. The key `binary_classification_metrics` is part of SageMaker's evaluation report schema.

---

### 3.5 Pipeline Notebook

#### `CacheConfig`

```python
cache_config = CacheConfig(enable_caching=True, expire_after='7d')
step_process = ProcessingStep(..., cache_config=cache_config)
```

When caching is enabled, SageMaker stores the output of each step. If you re-run the pipeline with identical inputs, it skips cached steps and reuses their output.

**Benefit:** If FeatureEngineering takes 10 minutes and you're debugging the training step, you don't want to re-run feature engineering every time. Caching skips it.

---

#### `ConditionStep` + `JsonGet`

```python
cond = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_evaluate.name,
        property_file=evaluation_report,
        json_path='binary_classification_metrics.average_precision.value',
    ),
    right=0.85,
)
step_condition = ConditionStep(
    name='CheckPRAUC',
    conditions=[cond],
    if_steps=[step_register],    # run these if condition is True
    else_steps=[],               # run these if condition is False
)
```

`JsonGet` reads a value from a JSON file output by a previous step. `json_path` uses dot notation to navigate the JSON structure.

**The gate:** If PR-AUC ≥ 0.85, register the model. If not, the pipeline ends without registering — preventing a bad model from reaching production.

---

#### `ModelStep` + `PendingManualApproval`

```python
step_register = ModelStep(
    step_args=model.register(
        approval_status='PendingManualApproval',
    )
)
```

After the model passes the PR-AUC check, it's registered in the **Model Registry** with status `PendingManualApproval`. A human must review and approve it before it can be deployed to an endpoint.

This is the **approval gate** — a safety check before code reaches production.

---

## 4. Phase 3 — Streaming Architecture

### 4.1 Key Concepts

#### SQS (Simple Queue Service)

A message queue — producers drop messages in, consumers pick them up.

```
Producer (sends transaction JSON)
    ↓
SQS Queue
    ↓
Consumer (Lambda processes each message)
```

**Key settings:**
- `VisibilityTimeout`: How long a message is hidden after a consumer picks it up. If Lambda takes >120 seconds and fails, the message becomes visible again for another Lambda invocation to retry.
- `MessageRetentionPeriod`: How long unprocessed messages stay in the queue before being deleted (86400 = 24 hours).

---

#### Lambda trigger (event source mapping)

```python
lambda_client.create_event_source_mapping(
    EventSourceArn=queue_arn,
    FunctionName='fraud-score-handler',
    BatchSize=1,   # process one transaction at a time
    Enabled=True,
)
```

This wires SQS to Lambda. When a message arrives in the queue, Lambda is automatically invoked with the message as the `event` parameter.

`BatchSize=1` — we process one transaction per Lambda invocation to keep latency low and make error handling simple (one failure doesn't block other transactions).

---

#### Lambda cold start

The first invocation of a Lambda function is a **cold start** — AWS must provision a container, install dependencies, and initialise the code. This can take 500ms-2s.

Subsequent invocations reuse the warm container — much faster (~50ms).

**This is why we initialise AWS clients at module level (outside the handler):**

```python
# ✅ Initialised once at cold start
_runtime  = boto3.client('sagemaker-runtime', region_name=REGION)
_dynamodb = boto3.resource('dynamodb', region_name=REGION)
_table    = _dynamodb.Table(TABLE_NAME)

def handler(event, context):
    # Uses pre-initialised clients — fast!
    score = _runtime.invoke_endpoint(...)
```

vs

```python
def handler(event, context):
    # ❌ Creates new client on every invocation — slow!
    runtime = boto3.client('sagemaker-runtime')
    score = runtime.invoke_endpoint(...)
```

---

#### DynamoDB `put_item`

```python
table.put_item(Item={
    'transaction_id': 'TXN-001',    # partition key — must be unique
    'fraud_score':    '0.9821',
    'is_fraud_flag':  'True',
})
```

`put_item` writes one item. If an item with the same `transaction_id` already exists, it's overwritten. DynamoDB stores all values as strings, numbers, or binary — we convert to strings for simplicity.

---

#### SNS publish

```python
sns.publish(
    TopicArn='arn:aws:sns:us-east-1:...',
    Subject='Fraud Alert',
    Message='Transaction TXN-001 scored 0.98...',
)
```

SNS fans the message out to all subscribers. In our case, one subscriber: an email address. In production, you'd also subscribe other Lambda functions, Slack webhooks, PagerDuty, etc.

---

### 4.2 Lambda Handler

#### `event['Records']`

SQS sends messages as a list of records even with BatchSize=1:

```python
def handler(event, context):
    for record in event['Records']:   # loop (handles BatchSize > 1 too)
        body = json.loads(record['body'])
```

Each `record['body']` is the raw string of the SQS message body — we `json.loads()` it to get the transaction dict.

#### Re-raising exceptions

```python
except Exception as e:
    logger.error(f'Error processing record: {e}', exc_info=True)
    raise   # re-raise so SQS retries the message
```

If we swallow the exception (don't re-raise), SQS thinks the message was processed successfully and deletes it — the transaction is lost. By re-raising, SQS keeps the message and retries it (up to the configured retry limit).

---

### 4.3 Inference Helper

#### Why behavioural features default to 0

```python
row['txn_count_last_1h'] = float(txn.get('txn_count_last_1h', 0))
```

Real-time inference has no access to historical transaction data. The rolling window features (`txn_count_last_1h`, `time_since_last_txn_seconds`, etc.) require knowing all previous transactions for each cardholder.

In production, these would come from **SageMaker Feature Store** (online store) — a low-latency key-value store where each cardholder's running feature values are updated with every transaction.

For Phase 3 we default to 0/1.0 — the model still works but is less accurate for the behavioural features.

---

## 5. AWS Services Used

| Service | Role in our system | Cost |
|---|---|---|
| **S3** | Store data, model artifacts, pipeline outputs | ~$0.02/month |
| **SageMaker Processing** | Run feature engineering at scale | ~$0.01/run |
| **SageMaker Training** | Managed XGBoost training | ~$0.02/run |
| **SageMaker Endpoint** | Real-time fraud scoring (<100ms) | ~$0.10/hr while running |
| **SageMaker Pipelines** | Orchestrate ML workflow as DAG | Free |
| **SageMaker Model Registry** | Version control for models + approval gate | Free |
| **SQS** | Message queue for transaction stream | Free tier |
| **Lambda** | Serverless function to score transactions | Free tier |
| **DynamoDB** | Store fraud decisions (fast lookups) | Free tier |
| **SNS** | Email alerts for detected fraud | Free tier |
| **CloudWatch** | Logs, metrics, dashboards | Free tier |
| **IAM** | Access control (who can do what) | Free |

---

## 6. Python Concepts

### `if __name__ == '__main__':`

```python
def compute_features(df): ...

if __name__ == '__main__':
    # Only runs when file is executed directly
    # Does NOT run when file is imported
    raw = pd.read_csv(...)
    featured = compute_features(raw)
```

When Python runs a file directly: `__name__ = '__main__'`
When a file is imported: `__name__ = 'module_name'`

This prevents side effects (loading data, printing) when the module is imported elsewhere.

---

### Underscore prefix `_function_name`

```python
def _add_recency_features(df):   # private — don't call from outside
    ...

def compute_features(df):        # public API
    df = _add_recency_features(df)
```

Single underscore = convention for "internal use only". Not enforced by Python but signals intent to other developers.

---

### Type hints

```python
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
```

`: pd.DataFrame` = parameter type
`-> pd.DataFrame` = return type

Not enforced at runtime but enables IDE autocomplete, catches bugs, and documents intent.

---

### `os.environ.get(key, default)`

```python
MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
```

`os.environ` is a dictionary of environment variables. `.get(key, default)` returns `default` if the variable isn't set — safer than `os.environ['key']` which raises `KeyError` if missing.

---

### `argparse`

```python
parser = argparse.ArgumentParser()
parser.add_argument('--n-estimators', type=int, default=500)
args = parser.parse_args()
model = XGBClassifier(n_estimators=args.n_estimators)
```

Standard library for parsing command-line arguments. SageMaker uses this to pass hyperparameters to training scripts.

---

## 7. Key Commands Reference

### Git

```bash
git init                          # initialise a new repo
git add .                         # stage all changes
git commit -m "message"           # commit staged changes
git checkout -b branch-name       # create and switch to new branch
git push -u origin branch-name    # push branch to GitHub (first time)
git push                          # push (after -u is set)
git status                        # see what's changed
git log --oneline                 # see commit history
```

### SageMaker (Python SDK)

```python
# Session
session = sagemaker.Session()
pipeline_session = PipelineSession()

# Pipeline
pipeline.upsert(role_arn=role)    # save/update pipeline definition
execution = pipeline.start()       # run the pipeline
execution.list_steps()            # check step status

# Endpoint
sm.create_endpoint(...)           # deploy model
sm.describe_endpoint(...)         # check status
sm.delete_endpoint(...)           # delete (stop charges)
```

### boto3 (AWS SDK)

```python
# Common pattern
client = boto3.client('sagemaker', region_name='us-east-1')
resource = boto3.resource('dynamodb', region_name='us-east-1')

# S3
s3.upload_file(local_path, bucket, key)
s3.get_object(Bucket=bucket, Key=key)

# SQS
sqs.send_message(QueueUrl=url, MessageBody=json.dumps(data))

# DynamoDB
table.put_item(Item={...})
table.scan()

# SNS
sns.publish(TopicArn=arn, Subject='...', Message='...')
```

### pandas key operations

```python
df.sort_values(['col1', 'col2'])         # sort
df.groupby('col').transform(func)        # apply func per group, return same shape
df.groupby('col').diff()                 # difference between consecutive rows per group
df.groupby('col').cumcount()             # cumulative count per group
df.set_index('col') / df.reset_index()  # move column to/from index
df.rolling('1h').count()                # time-based rolling window
df.expanding().mean()                   # expanding mean (all previous rows)
df.shift(1)                             # shift values down by 1 row
pd.get_dummies(df, columns=['cat'])     # one-hot encode
df.reindex(columns=cols, fill_value=0)  # align columns to expected order
```

---

## What to Learn Next

| Topic | Why | Resource |
|---|---|---|
| **Feature Store** | Replace default-0 behavioural features with real-time lookups | SageMaker Feature Store docs |
| **Terraform** | Define all AWS infrastructure as code (Phase 5) | terraform.io/learn |
| **GitHub Actions** | Automate testing and deployment (Phase 5) | docs.github.com/actions |
| **Docker** | Package training code into containers | docs.docker.com/get-started |
| **MLflow** | Experiment tracking (log params, metrics, artifacts) | mlflow.org |
| **Great Expectations** | Data validation before training | docs.greatexpectations.io |
| **pandas advanced** | Window functions, MultiIndex, memory optimisation | pandas.pydata.org |
| **XGBoost tuning** | Hyperparameter optimisation (Optuna, SageMaker HPO) | xgboost.readthedocs.io |
