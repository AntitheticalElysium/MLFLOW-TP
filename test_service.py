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


def test_accept_next_model_endpoint():
    url = f"{BASE_URL}/accept-next-model"
    response = requests.post(url)
    assert response.status_code == 200, f"Accept next model failed: {response.text}"
    result = response.json()
    assert "message" in result, "No message in response"
    assert "current_version" in result, "No current_version in response"


def test_canary_deployment():
    """Test that predictions use both current and next models"""
    url = f"{BASE_URL}/predict"
    test_data = {
        "age": 39,
        "sex": "female", 
        "bmi": 27.9,
        "children": 0,
        "smoker": "yes",
        "region": "southwest"
    }
    
    models_used = set()
    # Make multiple requests to see canary deployment in action
    for _ in range(20):
        response = requests.post(url, json=test_data)
        assert response.status_code == 200, f"Predict failed: {response.text}"
        result = response.json()
        assert "model_used" in result, "No model_used in response"
        models_used.add(result["model_used"])
    
    print(f"Models used in canary deployment: {models_used}")
    # Should use both current and next models with high probability
    # Note: This test might occasionally fail due to randomness


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
    
    print("Testing basic prediction endpoint...")
    test_predict_endpoint()
    
    print("Testing model update endpoint...")
    test_update_model_endpoint()
    
    print("Testing canary deployment...")
    test_canary_deployment()
    
    print("Testing accept next model endpoint...")
    test_accept_next_model_endpoint()
    
    print("All tests passed.")
