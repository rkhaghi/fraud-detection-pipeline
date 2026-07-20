"""
Transaction producer — replays CSV rows into SQS.

Simulates a real-time card transaction stream by reading
synthetic_transactions.csv and sending each row as a JSON
message to SQS, respecting the original timestamps.

Usage:
    python src/streaming/producer.py                  # replay all rows
    python src/streaming/producer.py --limit 100      # first 100 rows
    python src/streaming/producer.py --fraud-only     # only fraud rows
    python src/streaming/producer.py --rate 10        # 10 tx/sec
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import boto3
import pandas as pd

REGION     = "us-east-1"
QUEUE_NAME = "fraud-transactions-queue"
DATA_PATH  = Path(__file__).resolve().parent.parent.parent / "Data" / "synthetic_transactions.csv"


def get_queue_url(sqs) -> str:
    return sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]


def send_transaction(sqs, queue_url: str, txn: dict) -> None:
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(txn),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Send transactions to SQS")
    parser.add_argument("--limit",      type=int,   default=None,  help="Max rows to send")
    parser.add_argument("--rate",       type=float, default=5.0,   help="Transactions per second")
    parser.add_argument("--fraud-only", action="store_true",       help="Send only fraud transactions")
    parser.add_argument("--data",       type=str,   default=str(DATA_PATH), help="CSV path")
    args = parser.parse_args()

    sqs       = boto3.client("sqs", region_name=REGION)
    queue_url = get_queue_url(sqs)
    print(f"Queue: {queue_url}")

    df = pd.read_csv(args.data)
    if args.fraud_only:
        df = df[df["is_fraud"] == 1]
        print(f"Fraud-only mode: {len(df):,} fraud transactions")
    if args.limit:
        df = df.head(args.limit)

    delay = 1.0 / args.rate
    sent = fraud_sent = 0

    print(f"Sending {len(df):,} transactions at {args.rate}/sec...\n")

    for _, row in df.iterrows():
        txn = row.to_dict()
        # Convert numpy types to Python native for JSON serialisation
        txn = {k: (bool(v) if str(v) in ("True", "False") else
                   int(v) if isinstance(v, (int,)) else
                   float(v) if isinstance(v, float) else str(v))
               for k, v in txn.items()}

        send_transaction(sqs, queue_url, txn)
        sent += 1
        if txn.get("is_fraud") == 1:
            fraud_sent += 1

        if sent % 50 == 0:
            print(f"  Sent {sent:,} transactions ({fraud_sent} fraud)")

        time.sleep(delay)

    print(f"\nDone. Sent {sent:,} transactions ({fraud_sent} fraud) to SQS.")
    print("Check DynamoDB 'fraud-decisions' table for results.")


if __name__ == "__main__":
    main()
