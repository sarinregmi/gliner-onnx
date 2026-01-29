import requests
import time
import json
import sys

# URL from the deployment output
BASE_URL = "https://gliner-benchmark-32547059082.us-central1.run.app"

def test_endpoint(name, endpoint, payload=None):
    url = f"{BASE_URL}{endpoint}"
    print(f"\n--- Testing {name} ({url}) ---")
    
    start_time = time.time()
    try:
        if payload:
            response = requests.post(url, json=payload)
        else:
            response = requests.get(url)
        
        latency = (time.time() - start_time) * 1000
        print(f"Status Code: {response.status_code}")
        print(f"API Latency: {latency:.2f} ms")
        
        if response.status_code == 200:
            try:
                data = response.json()
                # Print a snippet of the response
                json_str = json.dumps(data, indent=2)
                if len(json_str) > 1000:
                    print(f"Response Snippet:\n{json_str[:1000]}...\n(truncated)")
                else:
                    print(f"Response:\n{json_str}")
                return data
            except:
                print("Response is not JSON")
                print(response.text[:200])
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def main():
    print(f"Testing GLiNER Benchmark API at {BASE_URL}")
    
    # 1. Health Check
    # This checks if the container is running and responsive
    test_endpoint("Health Check", "/")
    
    # 2. Warmup / Detection Test
    # This endpoint loads the ONNX model (if not loaded) and runs a test prediction
    # Text: "John Doe works at Microsoft."
    # Expected Entities: "John Doe" (person), "Microsoft" (organization) -> Count: 2
    print("\n[Warmup & PII Detection Test]")
    data = test_endpoint("Warmup (Predict 'John Doe works at Microsoft')", "/warmup")
    
    if data and "test_entities_found" in data:
        count = data['test_entities_found']
        if count == 2:
            print(f"✅ PII Detection Verification: SUCCESS. Found {count} entities (Expected 2).")
        else:
            print(f"⚠️ PII Detection Verification: UNEXPECTED. Found {count} entities (Expected 2).")
    
    if data and "device" in data:
        print(f"✅ Device Verification: Running on {data['device']}")

    # 3. Predict Endpoint (Test Inference Latency)
    print("\n[Predict Endpoint - Testing Real-Time Inference]")
    predict_payload = {
        "text": "My name is Alice and I live in Paris.",
        "labels": ["person", "city"]
    }
    predict_data = test_endpoint("Predict ('Alice', 'Paris')", "/predict", predict_payload)
    
    if predict_data and "entities" in predict_data:
        entities = predict_data["entities"]
        print(f"✅ Entities Found: {len(entities)}")
        for e in entities:
            print(f"   - {e['text']} ({e['label']})")
        
        if "inference_time_ms" in predict_data:
            print(f"⏱️ Model Inference Time: {predict_data['inference_time_ms']:.2f} ms")

    # 4. Run Full Benchmark
    # This runs the benchmark.py script on the server
    # It compares PyTorch CPU vs ONNX CPU performance
    print("\n[Running Full Benchmark Suite - Please wait...]")
    print("This will test: Single Inference Latency & Batched Inference Throughput")
    benchmark_data = test_endpoint("Benchmark", "/run_benchmark")
    
    if benchmark_data and "stdout" in benchmark_data:
        print("\n" + "="*40)
        print("=== SERVER-SIDE BENCHMARK RESULTS ===")
        print("="*40)
        print(benchmark_data["stdout"])
        print("="*40)

if __name__ == "__main__":
    main()
