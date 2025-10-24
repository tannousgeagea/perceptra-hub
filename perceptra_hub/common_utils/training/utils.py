import requests
from requests.exceptions import HTTPError
from typing import Optional

def trigger_trainml_training(
    trainml_url: str,
    payload: dict,
    token: Optional[str] = None,
    timeout: int = 10
) -> dict:
    """
    Sends a POST request to TrainML to trigger training.

    Args:
        trainml_url (str): Full URL to the TrainML /api/v1/train endpoint.
        payload (dict): JSON payload to send to TrainML.
        token (Optional[str]): Bearer token for authentication, if needed.
        timeout (int): Request timeout in seconds.

    Returns:
        dict: JSON response from TrainML if successful.

    Raises:
        Exception: If the request fails or TrainML responds with an error.
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(trainml_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    
    except HTTPError as err:
        raise err
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to trigger training on TrainML: {str(e)} (payload: {payload})")
