"""
Phase 3 Infrastructure Setup
Creates: SQS queue, DynamoDB table, SNS topic, Lambda function

Run once to provision all Phase 3 AWS resources.
Usage: python infrastructure/setup_phase3.py
"""
import boto3
import json
import time

REGION         = "us-east-1"
QUEUE_NAME     = "fraud-transactions-queue"
TABLE_NAME     = "fraud-decisions"
SNS_TOPIC_NAME = "fraud-alerts"
LAMBDA_NAME    = "fraud-score-handler"
ROLE_NAME      = "AmazonSageMaker-ExecutionRole-20260717T132288"


def get_role_arn() -> str:
    iam = boto3.client("iam", region_name=REGION)
    return iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]


def create_sqs_queue(sqs) -> str:
    """Create SQS queue. Returns queue URL."""
    try:
        resp = sqs.create_queue(
            QueueName=QUEUE_NAME,
            Attributes={
                "VisibilityTimeout": "120",       # 2 min — Lambda processing time
                "MessageRetentionPeriod": "86400", # 24 hrs
            },
        )
        url = resp["QueueUrl"]
        print(f"SQS queue created: {url}")
        return url
    except sqs.exceptions.QueueNameExists:
        url = sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
        print(f"SQS queue already exists: {url}")
        return url


def get_queue_arn(sqs, queue_url: str) -> str:
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["QueueArn"],
    )
    return attrs["Attributes"]["QueueArn"]


def create_dynamodb_table(dynamodb) -> None:
    """Create DynamoDB table for fraud decisions."""
    try:
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "transaction_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "transaction_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"DynamoDB table created: {TABLE_NAME}")
    except dynamodb.exceptions.ResourceInUseException:
        print(f"DynamoDB table already exists: {TABLE_NAME}")


def create_sns_topic(sns) -> str:
    """Create SNS topic for fraud alerts. Returns topic ARN."""
    resp = sns.create_topic(Name=SNS_TOPIC_NAME)
    arn  = resp["TopicArn"]
    print(f"SNS topic created: {arn}")
    return arn


def subscribe_email(sns, topic_arn: str, email: str) -> None:
    sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=email)
    print(f"Subscription confirmation sent to {email} — check your inbox")


def create_lambda(lambda_client, role_arn: str, queue_arn: str, topic_arn: str, endpoint_name: str) -> None:
    """Package and deploy the Lambda function."""
    import zipfile, io, os

    # Zip lambda_handler.py + inference.py
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "serving")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["lambda_handler.py", "inference.py"]:
            fpath = os.path.join(src_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)
                print(f"  Added {fname} to zip")
    buf.seek(0)

    env_vars = {
        "ENDPOINT_NAME": endpoint_name,
        "DYNAMODB_TABLE": TABLE_NAME,
        "SNS_TOPIC_ARN": topic_arn,
        "FRAUD_THRESHOLD": "0.5",
        "REGION": REGION,
    }

    try:
        lambda_client.create_function(
            FunctionName=LAMBDA_NAME,
            Runtime="python3.11",
            Role=role_arn,
            Handler="lambda_handler.handler",
            Code={"ZipFile": buf.read()},
            Timeout=60,
            MemorySize=512,
            Environment={"Variables": env_vars},
        )
        print(f"Lambda function created: {LAMBDA_NAME}")
    except lambda_client.exceptions.ResourceConflictException:
        buf.seek(0)
        lambda_client.update_function_code(
            FunctionName=LAMBDA_NAME,
            ZipFile=buf.read(),
        )
        lambda_client.update_function_configuration(
            FunctionName=LAMBDA_NAME,
            Environment={"Variables": env_vars},
        )
        print(f"Lambda function updated: {LAMBDA_NAME}")

    # Wire SQS → Lambda
    time.sleep(5)
    try:
        lambda_client.create_event_source_mapping(
            EventSourceArn=queue_arn,
            FunctionName=LAMBDA_NAME,
            BatchSize=1,                  # process one transaction at a time
            Enabled=True,
        )
        print(f"SQS → Lambda trigger created ✅")
    except lambda_client.exceptions.ResourceConflictException:
        print(f"SQS → Lambda trigger already exists ✅")


def main():
    print("=== Phase 3 Infrastructure Setup ===\n")
    role_arn = get_role_arn()

    sqs      = boto3.client("sqs",      region_name=REGION)
    dynamodb = boto3.client("dynamodb", region_name=REGION)
    sns      = boto3.client("sns",      region_name=REGION)
    lam      = boto3.client("lambda",   region_name=REGION)

    queue_url = create_sqs_queue(sqs)
    queue_arn = get_queue_arn(sqs, queue_url)
    create_dynamodb_table(dynamodb)
    topic_arn = create_sns_topic(sns)

    # ── Configure these before running ──────────────────────────────────
    ALERT_EMAIL   = "your.email@example.com"   # ← change this
    ENDPOINT_NAME = "fraud-detection-endpoint"  # ← must be InService
    # ────────────────────────────────────────────────────────────────────

    subscribe_email(sns, topic_arn, ALERT_EMAIL)
    create_lambda(lam, role_arn, queue_arn, topic_arn, ENDPOINT_NAME)

    print("\n=== Setup Complete ===")
    print(f"Queue URL : {queue_url}")
    print(f"Table     : {TABLE_NAME}")
    print(f"SNS Topic : {topic_arn}")
    print(f"Lambda    : {LAMBDA_NAME}")
    print(f"\nNext: run src/streaming/producer.py to send transactions")


if __name__ == "__main__":
    main()
