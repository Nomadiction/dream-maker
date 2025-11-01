"""
Example test script for Dream Maker API
Run this after starting the server to test the endpoints
"""
import requests
import json
from typing import Dict, Any


# API base URL
BASE_URL = "http://localhost:8000"


def print_response(title: str, response: requests.Response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse:")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except:
        print(response.text)
    print('='*60)


def test_health_check():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print_response("Health Check", response)
    return response.status_code == 200


def test_text_to_prompt_with_development():
    """Test text-to-prompt with development mode enabled"""
    data = {
        "user_text": "Кошка играет с мячиком на солнечной лужайке",
        "develop_mode": True
    }
    
    response = requests.post(
        f"{BASE_URL}/api/text-to-prompt",
        json=data
    )
    
    print_response("Text to Prompt (Development Mode ON)", response)
    return response.status_code == 200


def test_text_to_prompt_without_development():
    """Test text-to-prompt without development mode"""
    data = {
        "user_text": "A cyberpunk city at night with neon lights reflecting on wet streets",
        "develop_mode": False
    }
    
    response = requests.post(
        f"{BASE_URL}/api/text-to-prompt",
        json=data
    )
    
    print_response("Text to Prompt (Development Mode OFF)", response)
    return response.status_code == 200


def test_direct_sora_prompt():
    """Test direct Sora prompt generation"""
    data = {
        "user_text": "A majestic eagle soaring through a dramatic sunset sky, with clouds painted in vibrant orange and purple hues. The camera follows the eagle in a smooth tracking shot, capturing the graceful wing movements and the vast landscape below."
    }
    
    response = requests.post(
        f"{BASE_URL}/api/direct-sora-prompt",
        json=data
    )
    
    print_response("Direct Sora Prompt", response)
    return response.status_code == 200


def test_custom_prompt():
    """Test custom prompt endpoint"""
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a creative assistant specialized in video descriptions."
            },
            {
                "role": "user",
                "content": "Describe a short video idea about nature."
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    response = requests.post(
        f"{BASE_URL}/api/custom-prompt",
        json=data
    )
    
    print_response("Custom Prompt", response)
    return response.status_code == 200


def main():
    """Run all tests"""
    print("\n" + "🚀" * 30)
    print("  Dream Maker API - Test Suite")
    print("🚀" * 30)
    
    tests = [
        ("Health Check", test_health_check),
        ("Text to Prompt (Dev Mode)", test_text_to_prompt_with_development),
        ("Text to Prompt (No Dev)", test_text_to_prompt_without_development),
        ("Direct Sora Prompt", test_direct_sora_prompt),
        ("Custom Prompt", test_custom_prompt),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            results.append((test_name, f"❌ ERROR: {str(e)}"))
    
    # Print summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    for test_name, result in results:
        print(f"{result} - {test_name}")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")

