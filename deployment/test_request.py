import requests

# Define the URL and the JSON data
url = "http://localhost:8989/predict"
data = {
    "gender": 'Female',
    "age": 33,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": 'Yes',
    "work_type": 'Self-employed',
    "Residence_type": 'Urban',
    "avg_glucose_level": 85,
    "bmi": 20.8,
    "smoking_status": 'never smoked'
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())