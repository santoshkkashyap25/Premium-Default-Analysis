# test_predict_api.py
import requests
import json

API_URL = "http://127.0.0.1:5000/predict"

# Example customer data
customer_data = {
    "id": 110936,
    "perc_premium_paid_by_cash_credit": 0.429,
    "age_in_days": 12058,
    "Income": 355060,
    "Count_3-6_months_late": 0,
    "Count_6-12_months_late": 0,
    "Count_more_than_12_months_late": 0,
    "application_underwriting_score": 99.02,
    "no_of_premiums_paid": 13,
    "sourcing_channel": "C",
    "residence_area_type": "Urban"
}

def call_prediction_api(payload):
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Result:")
        print(json.dumps(result, indent=4))
    else:
        print(f"\nError {response.status_code}: {response.text}")

if __name__ == "__main__":
    call_prediction_api(customer_data)
