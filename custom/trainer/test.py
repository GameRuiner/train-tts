from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "trainer/text-to-speech-451918-8309307a12ed.json"

bucket_uri = os.getenv("BUCKET_URI") 
bucket_name = bucket_uri.replace("gs://", "")
model_dir = "llasa_finetuned_model"

os.makedirs(model_dir, exist_ok=True)

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

blobs = list(storage_client.list_blobs(bucket_name))
for blob in blobs:
    if "/" not in blob.name or blob.name.startswith("/checkpoint-"):
        continue
    blob_name = blob.name.replace("/", "")
    file_path = os.path.join(model_dir, blob_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Downloading {blob_name} to {file_path}...")
    blob.download_to_filename(file_path)