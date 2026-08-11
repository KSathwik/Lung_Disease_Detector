"""
End-to-End Live API Verification Script
Tests all live endpoints on http://localhost:8000
"""

import requests
from pathlib import Path
import json
import sys
import io

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=== STARTING LIVE API VERIFICATION ===")

    # 1. Health Check
    res = requests.get(f"{BASE_URL}/api/v1/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print(f"✅ GET /api/v1/health -> 200 OK: {res.json()}")

    # 2. Patient Registration
    patient_payload = {
        "name": "Jane Smith",
        "age": 38,
        "gender": "Female",
        "medical_history": "No prior respiratory issues"
    }
    res = requests.post(f"{BASE_URL}/api/v1/patients", json=patient_payload)
    assert res.status_code == 200, f"Patient registration failed: {res.text}"
    patient_data = res.json()
    patient_id_str = patient_data["patient_id"]
    print(f"✅ POST /api/v1/patients -> 200 OK (Patient ID: {patient_id_str})")

    # 3. Model Inference with Valid Radiograph (Pneumonia)
    sample_img_path = list(Path("data/raw/Pneumonia").glob("*.*"))[0]
    with open(sample_img_path, "rb") as f:
        files = {"file": (sample_img_path.name, f, "image/jpeg")}
        data = {"patient_id": patient_id_str, "scan_type": "X-Ray"}
        res = requests.post(f"{BASE_URL}/api/v1/predict", files=files, data=data)
    
    assert res.status_code == 200, f"Prediction failed: {res.text}"
    pred_res = res.json()
    assert "prediction_id" in pred_res
    assert "resnet" in pred_res
    assert pred_res["resnet"]["condition"] in ["Pneumonia", "COVID-19", "Normal", "Tuberculosis", "Lung Cancer"]
    print(f"✅ POST /api/v1/predict (Pneumonia scan) -> 200 OK")
    print(f"   Selected Model: {pred_res['selected_model']}")
    print(f"   Predicted Condition: {pred_res['resnet']['condition']} ({pred_res['resnet']['confidence']:.1f}%)")
    print(f"   Urgency Level: {pred_res['final']['urgency']}")


    # 4. Input Validation (Invalid File Type)
    invalid_files = {"file": ("test.txt", b"This is not an image", "text/plain")}
    res = requests.post(f"{BASE_URL}/api/v1/predict", files=invalid_files)
    assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
    print(f"✅ POST /api/v1/predict (Invalid text file) -> 400 Bad Request: {res.json()['detail']}")

    # 5. List Predictions
    res = requests.get(f"{BASE_URL}/api/v1/predictions")
    assert res.status_code == 200
    preds_list = res.json()
    print(f"✅ GET /api/v1/predictions -> 200 OK ({len(preds_list)} records found)")

    # 6. Model Metrics
    res = requests.get(f"{BASE_URL}/api/v1/model-metrics")
    assert res.status_code == 200
    metrics_res = res.json()
    print(f"✅ GET /api/v1/model-metrics -> 200 OK (Selected: {metrics_res['selected_model']})")

    print("\n🎉 ALL LIVE ENDPOINTS PASSED VERIFICATION 100% SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
