"""
Lambda handler — triggered by SQS.

For each transaction message:
  1. Parse transaction JSON from SQS record
  2. Build feature vector (inference.py)
  3. Call SageMaker endpoint → fraud probability
  4. Write decision to DynamoDB
  5. If fraud detected → publish SNS alert
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import boto3

from inference import build_feature_vector, is_fraud

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── AWS clients (initialised once at cold start) ─────────────────────────
REGION        = os.environ.get("REGION", "us-east-1")
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
TABLE_NAME    = os.environ["DYNAMODB_TABLE"]
SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]

_runtime   = boto3.client("sagemaker-runtime", region_name=REGION)
_dynamodb  = boto3.resource("dynamodb",        region_name=REGION)
_sns       = boto3.client("sns",               region_name=REGION)
_cw        = boto3.client("cloudwatch",        region_name=REGION)
_table     = _dynamodb.Table(TABLE_NAME)


def _score_transaction(txn: dict[str, Any]) -> float:
    """Call SageMaker endpoint and return fraud probability."""
    csv_row = build_feature_vector(txn)
    response = _runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=csv_row,
    )
    score = float(response["Body"].read().decode().strip())
    return score


def _save_decision(txn: dict[str, Any], score: float) -> None:
    """Write fraud decision to DynamoDB."""
    _table.put_item(Item={
        "transaction_id":  txn.get("transaction_id", "unknown"),
        "cardholder_id":   txn.get("cardholder_id", "unknown"),
        "timestamp":       txn.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "amount":          str(txn.get("amount", 0)),
        "merchant_name":   txn.get("merchant_name", "unknown"),
        "fraud_score":     str(round(score, 4)),
        "is_fraud_flag":   str(is_fraud(score)),
        "processed_at":    datetime.now(timezone.utc).isoformat(),
    })


def _send_alert(txn: dict[str, Any], score: float) -> None:
    """Publish SNS alert for detected fraud."""
    message = (
        f"🚨 FRAUD DETECTED\n\n"
        f"Transaction ID : {txn.get('transaction_id')}\n"
        f"Cardholder     : {txn.get('cardholder_id')}\n"
        f"Amount         : {txn.get('amount')} {txn.get('currency', '')}\n"
        f"Merchant       : {txn.get('merchant_name')} ({txn.get('merchant_category')})\n"
        f"Fraud score    : {score:.4f}\n"
        f"Timestamp      : {txn.get('timestamp')}\n"
    )
    _sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="Fraud Alert",
        Message=message,
    )
    logger.info(f"Alert sent for {txn.get('transaction_id')}")


def _publish_fraud_metric(is_fraud_flag: bool) -> None:
    """Publish custom CloudWatch metric for fraud rate tracking."""
    _cw.put_metric_data(
        Namespace="FraudDetection/Custom",
        MetricData=[{
            "MetricName": "FraudDetected",
            "Value": 1.0 if is_fraud_flag else 0.0,
            "Unit": "Count",
        }],
    )


def handler(event: dict, context: Any) -> dict:
    """
    Lambda entry point.
    event["Records"] contains 1 SQS message (BatchSize=1).
    """
    results = []

    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            txn  = body if isinstance(body, dict) else json.loads(body)

            txn_id = txn.get("transaction_id", "unknown")
            logger.info(f"Processing {txn_id}")

            score = _score_transaction(txn)
            _save_decision(txn, score)

            fraud_flag = is_fraud(score)
            _publish_fraud_metric(fraud_flag)

            if fraud_flag:
                logger.warning(f"FRAUD detected: {txn_id}  score={score:.4f}")
                _send_alert(txn, score)
            else:
                logger.info(f"Legit: {txn_id}  score={score:.4f}")

            results.append({"transaction_id": txn_id, "score": score, "fraud": fraud_flag})

        except Exception as e:
            logger.error(f"Error processing record: {e}", exc_info=True)
            raise   # re-raise so SQS retries the message

    return {"statusCode": 200, "body": json.dumps(results)}
