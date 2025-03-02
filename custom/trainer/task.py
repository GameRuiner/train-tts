from dotenv import load_dotenv
from os import environ
from google.cloud import aiplatform

from trainer.create_dataset import preprocess_dataset

load_dotenv()

PROJECT_ID = environ["PROJECT_ID"]
LOCATION = environ["LOCATION"]
BUCKET_URI = environ["BUCKET_URI"]

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

preprocess_dataset(
  dataset_name="mozilla-common_voice_15-23",
  output_dir="/app/xcodec2_dataset/",
  sample_rate=16000,
  max_length=4096,
  debug=True
)

