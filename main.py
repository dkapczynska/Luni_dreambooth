import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
#AWS_S3_ENDPOINT = os.getenv('AWS_S3_ENDPOINT')  # Optional for AWS, required for S3-compatible services
# AWS_REGION = os.getenv('AWS_REGION')            # Optional for default region
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Create S3 client
s3 = boto3.client(
    's3',
    #endpoint_url=AWS_S3_ENDPOINT,                # Set to None for normal AWS S3 use
    # region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

try:
    # Attempt to list objects in the bucket (lightweight call, shows if access is OK)
    s3.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=1)
    print(f"Access to bucket {BUCKET_NAME} succeeded!")
except ClientError as e:
    print(f"Access to bucket {BUCKET_NAME} failed: {e}")
