import requests
import time
import json
import statistics

BASE_URL = "https://gliner-benchmark-32547059082.us-central1.run.app"
PREDICT_URL = f"{BASE_URL}/predict"

# 30 Test Cases with PII
TEST_CASES = [
    {"text": "My name is John Doe and I work at Microsoft.", "labels": ["person", "organization"]},
    {"text": "Contact Alice at alice@example.com for more info.", "labels": ["person", "email"]},
    {"text": "The headquarters of Google is in Mountain View.", "labels": ["organization", "city"]},
    {"text": "Please call 555-0199 to reach customer support.", "labels": ["phone_number"]},
    {"text": "Dr. Smith scheduled an appointment for Monday.", "labels": ["person", "date"]},
    {"text": "My IP address is 192.168.1.1.", "labels": ["ip_address"]},
    {"text": "I bought this from Amazon using my Visa card.", "labels": ["organization", "credit_card"]},
    {"text": "Sarah visited Paris last summer.", "labels": ["person", "city"]},
    {"text": "The meeting is at 10:00 AM in Room 302.", "labels": ["time", "location"]},
    {"text": "Send the package to 123 Main St, New York, NY.", "labels": ["address"]},
    {"text": "My social security number is 123-45-6789.", "labels": ["ssn"]},
    {"text": "Robert Downey Jr. is a famous actor.", "labels": ["person"]},
    {"text": "I use an iPhone 14 Pro Max.", "labels": ["product"]},
    {"text": "The code is stored on GitHub.", "labels": ["organization"]},
    {"text": "Contact info@openai.com for API access.", "labels": ["email"]},
    {"text": "My birthdate is January 1st, 1990.", "labels": ["date"]},
    {"text": "The server is running on AWS us-east-1.", "labels": ["organization", "location"]},
    {"text": "Payment of $50.00 was received.", "labels": ["currency"]},
    {"text": "Mr. Bond drives an Aston Martin.", "labels": ["person", "product"]},
    {"text": "The event is in London, UK.", "labels": ["city", "country"]},
    {"text": "Call me at +1 (415) 555-2671.", "labels": ["phone_number"]},
    {"text": "I have a meeting with CEO Satya Nadella.", "labels": ["person", "job_title"]},
    {"text": "My passport number is A1234567.", "labels": ["passport_number"]},
    {"text": "The driver's license number is D98765432.", "labels": ["drivers_license"]},
    {"text": "Login with username admin and password secret.", "labels": ["username", "password"]},
    {"text": "The patient id is 99887766.", "labels": ["id_number"]},
    {"text": "I work for the FBI in Washington DC.", "labels": ["organization", "city"]},
    {"text": "My Bitcoin wallet is 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa.", "labels": ["cryptocurrency_wallet"]},
    {"text": "The flight AA100 departs from JFK.", "labels": ["flight_number", "airport"]},
    {"text": "Please refer to ticket #90210 for support.", "labels": ["ticket_number"]}
]

def run_tests():
    print(f"Running 30 PII Test Cases against {PREDICT_URL}...\n")
    
    results = []
    latencies = []
    
    for i, case in enumerate(TEST_CASES):
        print(f"Test Case #{i+1}: {case['text'][:50]}...")
        
        try:
            start_req = time.perf_counter()
            response = requests.post(PREDICT_URL, json=case)
            req_latency = (time.perf_counter() - start_req) * 1000
            
            if response.status_code == 200:
                data = response.json()
                entities = data.get("entities", [])
                model_time = data.get("inference_time_ms", 0)
                
                print(f"  ✅ Status: 200 OK")
                print(f"  ⏱️  API Latency: {req_latency:.2f} ms")
                print(f"  🧠 Model Time: {model_time:.2f} ms")
                print(f"  🔍 Entities Found: {len(entities)}")
                for e in entities:
                    print(f"     - {e['text']} ({e['label']})")
                
                results.append({
                    "id": i+1,
                    "text": case["text"],
                    "latency_api": req_latency,
                    "latency_model": model_time,
                    "entities": len(entities),
                    "details": entities
                })
                latencies.append(req_latency)
            else:
                print(f"  ❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  ❌ Exception: {e}")
        
        print("-" * 50)
        time.sleep(0.1) # Small delay to be nice

    # Summary
    if latencies:
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print("\n=== SUMMARY ===")
        print(f"Total Tests: {len(TEST_CASES)}")
        print(f"Successful: {len(results)}")
        print(f"Avg API Latency: {avg_latency:.2f} ms")
        print(f"Min API Latency: {min_latency:.2f} ms")
        print(f"Max API Latency: {max_latency:.2f} ms")
        
        return results, avg_latency, min_latency, max_latency
    return [], 0, 0, 0

if __name__ == "__main__":
    results, avg, min_lat, max_lat = run_tests()
    
    # Save results to JSON for the report generator
    with open("test_results.json", "w") as f:
        json.dump({
            "summary": {
                "avg_latency": avg,
                "min_latency": min_lat,
                "max_latency": max_lat
            },
            "results": results
        }, f, indent=2)
    print("\nResults saved to test_results.json")
