import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_predict_endpoint():
    url = f"{BASE_URL}/predict"
    test_data = {
        "age": 39,
        "sex": "female",
        "bmi": 27.9,
        "children": 0,
        "smoker": "yes",
        "region": "southwest"
    }
    response = requests.post(url, json=test_data)
    assert response.status_code == 200, f"Predict failed: {response.text}"
    result = response.json()
    assert "prediction" in result, "No prediction in response"
    assert isinstance(result["prediction"], float) or isinstance(result["prediction"], int)


def test_update_model_endpoint():
    url = f"{BASE_URL}/update-model"
    update_data = {
        "model_name": "random_forest_model",
        "version": "latest"
    }
    try:
        response = requests.post(url, json=update_data, timeout=60)
        assert response.status_code == 200, f"Update failed: {response.text}"
        result = response.json()
        assert "message" in result, "No message in response"
    except requests.exceptions.Timeout:
        assert False, "Model update timed out (>60s) - this is expected with slow MLFlow artifact downloads"


def wait_for_service(max_retries=30):
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

if __name__ == "__main__":
    print("=== MLFlow Model Service Test ===")
    if not wait_for_service():
        print("Service not ready")
        exit(1)
    test_predict_endpoint()
    test_update_model_endpoint()
    print("All tests passed.")
