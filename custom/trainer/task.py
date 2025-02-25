from dotenv import load_dotenv
from os import environ
from google.cloud import aiplatform

load_dotenv()

PROJECT_ID = environ["PROJECT_ID"]
LOCATION = environ["LOCATION"]
BUCKET_URI = environ["BUCKET_URI"]

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)