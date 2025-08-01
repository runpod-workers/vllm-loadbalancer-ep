import requests
import json
import time
import os
import sys

# Configuration
ENDPOINT_ID = os.getenv("ENDPOINT_ID", "your-endpoint-id")
API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_URL = f"https://{ENDPOINT_ID}.api.runpod.ai"

if not API_KEY:
    print("Error: Please set RUNPOD_API_KEY environment variable")
    sys.exit(1)

def test_streaming():
    """Test streaming endpoint with real-time output"""
    print("🔄 Testing Streaming Endpoint")
    print("=" * 50)
    
    payload = {
        "prompt": "Write a long story about a brave knight who discovers a magical forest",
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": True
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"📡 Making streaming request to: {ENDPOINT_URL}/v1/completions")
        print(f"📝 Prompt: {payload['prompt'][:50]}...")
        print("\n🎬 Streaming output:")
        print("-" * 30)
        
        response = requests.post(
            f"{ENDPOINT_URL}/v1/completions",
            json=payload,
            headers=headers,
            stream=True,  # Enable streaming
            timeout=300
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return
        
        # Process streaming response
        generated_text = ""
        chunk_count = 0
        start_time = time.time()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_part = line[6:]  # Remove 'data: ' prefix
                    
                    if data_part == '[DONE]':
                        print("\n\n✅ Stream completed!")
                        break
                    
                    try:
                        chunk_data = json.loads(data_part)
                        if 'text' in chunk_data:
                            new_text = chunk_data['text']
                            print(new_text, end='', flush=True)
                            generated_text += new_text
                            chunk_count += 1
                            
                    except json.JSONDecodeError:
                        # Skip malformed JSON chunks
                        continue
        
        end_time = time.time()
        
        print(f"\n\n📊 Streaming Statistics:")
        print(f"   • Total chunks: {chunk_count}")
        print(f"   • Total characters: {len(generated_text)}")
        print(f"   • Time taken: {end_time - start_time:.2f} seconds")
        print(f"   • Average chars/second: {len(generated_text) / (end_time - start_time):.1f}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_non_streaming():
    """Test non-streaming endpoint for comparison"""
    print("\n🔄 Testing Non-Streaming Endpoint")
    print("=" * 50)
    
    payload = {
        "prompt": "Write a short story about a robot",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"📡 Making non-streaming request...")
        start_time = time.time()
        
        response = requests.post(
            f"{ENDPOINT_URL}/v1/completions",
            json=payload,
            headers=headers,
            timeout=300
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received!")
            print(f"📝 Generated text: {result.get('text', 'No text found')}")
            print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def compare_streaming_vs_non_streaming():
    """Compare streaming vs non-streaming with same prompt"""
    print("\n🔄 Comparing Streaming vs Non-Streaming")
    print("=" * 50)
    
    prompt = "Tell me about the history of artificial intelligence"
    
    # Test streaming
    print("1️⃣ Streaming version:")
    start_time = time.time()
    
    payload = {
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": True
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=300
        )
        
        if response.status_code == 200:
            first_chunk_time = None
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_part = line[6:]
                        if data_part != '[DONE]':
                            try:
                                chunk_data = json.loads(data_part)
                                if 'text' in chunk_data and first_chunk_time is None:
                                    first_chunk_time = time.time()
                                    print(f"   ⚡ First chunk received in: {first_chunk_time - start_time:.2f}s")
                                    break
                            except:
                                continue
        
        streaming_time = time.time() - start_time
        
    except Exception as e:
        print(f"   ❌ Streaming failed: {e}")
        return
    
    # Test non-streaming
    print("\n2️⃣ Non-streaming version:")
    start_time = time.time()
    
    payload["stream"] = False
    
    try:
        response = requests.post(
            f"{ENDPOINT_URL}/v1/completions",
            json=payload,
            headers=headers,
            timeout=300
        )
        
        non_streaming_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"   ⚡ Complete response in: {non_streaming_time:.2f}s")
        
        print(f"\n📊 Comparison:")
        print(f"   • Streaming first chunk: {first_chunk_time - start_time:.2f}s" if first_chunk_time else "   • Streaming: Failed")
        print(f"   • Non-streaming total: {non_streaming_time:.2f}s")
        
        if first_chunk_time:
            improvement = non_streaming_time - (first_chunk_time - start_time)
            print(f"   • Time to first response improved by: {improvement:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Non-streaming failed: {e}")

def main():
    print("🧪 vLLM Streaming Test Suite")
    print("=" * 50)
    print(f"🎯 Endpoint: {ENDPOINT_URL}")
    print(f"🔑 API Key: {API_KEY[:10]}...")
    
    while True:
        print("\n" + "="*50)
        print("Choose a test:")
        print("1. Test Streaming (real-time output)")
        print("2. Test Non-Streaming")
        print("3. Compare Streaming vs Non-Streaming")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            test_streaming()
        elif choice == '2':
            test_non_streaming()
        elif choice == '3':
            compare_streaming_vs_non_streaming()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()