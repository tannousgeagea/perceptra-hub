import os
import requests
from fastapi import HTTPException
from common_utils.azure_manager.core import AzureManager
from django.core.files.storage import default_storage

REMOTE_INFERENCE_URL = os.getenv("REMOTE_INFERENCE_URL")

def run_inference(image:str, model_version_id:str, confidence_threshold:float=0.25):
    with default_storage.open(image, "rb") as f:
        files = {
            "file": f
        }

        params = {
            "confidence_threshold": confidence_threshold,
        }
        response = requests.post(f"{REMOTE_INFERENCE_URL}/api/v1/infer/{model_version_id}", files=files, params=params)
   
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Error in inference service: {response.text}")

    result = response.json()
    return result.get("predictions", [])


if __name__ == "__main__":

    import django
    django.setup()

    image = "images/AGR_gate01_left_2025-03-24_06-37-27_3ca7e74e-4ed5-4f95-8afa-75863a7cbb85_EVopA3L.jpg"
    model_version_id = 5

    result = run_inference(
        image=image,
        model_version_id=model_version_id,
    )

    print(result)